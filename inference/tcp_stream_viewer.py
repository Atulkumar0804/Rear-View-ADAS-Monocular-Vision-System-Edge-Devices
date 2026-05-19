#!/usr/bin/env python3
"""
TCP stream viewer that exposes frames in a web browser (MJPEG).

Run on the laptop:
  python tcp_stream_viewer.py --host <pi-ip> --port 5001 --web-port 8000

Then open:
  http://localhost:8000
"""

import argparse
import socket
import struct
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, request


MIN_STREAM_SIZE = 240
MAX_STREAM_SIZE = 4096


def recv_all(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


class TcpFrameClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_seq = 0
        self.running = False

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"✅ Connected to {self.host}:{self.port}")

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    def _loop(self):
        while self.running:
            header = recv_all(self.sock, 4)
            if header is None:
                break
            (length,) = struct.unpack("!I", header)
            payload = recv_all(self.sock, length)
            if payload is None:
                break
            with self.lock:
                self.latest_jpeg = payload
                self.latest_seq += 1

        self.running = False

    def get_latest(self):
        with self.lock:
            return self.latest_jpeg, self.latest_seq


def parse_dimension(value):
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return None

    return max(MIN_STREAM_SIZE, min(MAX_STREAM_SIZE, dimension))


def resize_jpeg_to_viewport(jpeg_bytes, target_width, target_height):
    if not target_width or not target_height:
        return jpeg_bytes

    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return jpeg_bytes

    source_height, source_width = frame.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    if scale <= 0:
        return jpeg_bytes

    output_width = max(1, int(source_width * scale))
    output_height = max(1, int(source_height * scale))
    if output_width == source_width and output_height == source_height:
        return jpeg_bytes

    resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
    ok, output = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return jpeg_bytes

    return output.tobytes()


def main():
    parser = argparse.ArgumentParser(description="TCP stream viewer (web)")
    parser.add_argument("--host", type=str, required=True, help="Server IP")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web bind host")
    parser.add_argument("--web-port", type=int, default=8000, help="Web bind port")
    args = parser.parse_args()

    client = TcpFrameClient(args.host, args.port)
    client.connect()
    client.start()

    app = Flask(__name__)

    @app.route("/")
    def index():
        html = (
            "<!doctype html>"
            "<html lang='en'>"
            "<head>"
            "<meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>ADAS Stream</title>"
            "<style>"
            "*{box-sizing:border-box;}"
            "html,body{margin:0;width:100vw;height:100vh;overflow:hidden;background:#000;}"
            "body{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;}"
            "#stream{position:fixed;inset:0;width:100vw;height:100vh;max-width:100vw;"
            "max-height:100vh;object-fit:contain;display:block;background:#000;}"
            "</style>"
            "</head>"
            "<body>"
            "<img id='stream' alt='ADAS camera stream'>"
            "<script>"
            "const stream=document.getElementById('stream');"
            "function setStream(){"
            "const w=Math.max(240,Math.floor(window.innerWidth*window.devicePixelRatio));"
            "const h=Math.max(240,Math.floor(window.innerHeight*window.devicePixelRatio));"
            "stream.src=`/stream?w=${w}&h=${h}&t=${Date.now()}`;"
            "}"
            "setStream();"
            "let timer;"
            "window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(setStream,250);});"
            "</script>"
            "</body>"
            "</html>"
        )
        return Response(html, headers={"Cache-Control": "no-store, max-age=0"})

    @app.route("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.route("/stream")
    def stream():
        target_width = parse_dimension(request.args.get("w"))
        target_height = parse_dimension(request.args.get("h"))

        def generate():
            last_seq = -1
            last_output = None
            while True:
                frame, seq = client.get_latest()
                if frame is None:
                    time.sleep(0.01)
                    continue
                if seq != last_seq:
                    last_output = resize_jpeg_to_viewport(frame, target_width, target_height)
                    last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store, max-age=0\r\n\r\n"
                    + last_output
                    + b"\r\n"
                )
                time.sleep(0.001)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    print(f"✅ Web viewer on http://{args.web_host}:{args.web_port}")
    try:
        app.run(host=args.web_host, port=args.web_port, threaded=True)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
