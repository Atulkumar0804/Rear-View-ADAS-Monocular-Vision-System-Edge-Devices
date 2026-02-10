# Rear-View ADAS Monocular System (CNN Module)

This module provides vehicle detection and classification for a rear-view ADAS system using a Two-Stage Pipeline.

## 📂 Structure

```
CNN/
├── main.sh                 # Main launcher script
├── inference/              # Inference scripts
│   ├── camera_inference.py # Real-time camera detection
│   └── video_inference.py  # Video file processing
├── models/                 # Trained models
│   ├── yolo/               # Stage 1: Detection
│   │   └── yolo11x-seg.pt
│   └── classifier/         # Stage 2: Classification
│       ├── weights/
│       │   └── best.pt     # Fine-tuned YOLOv11m-cls
│       └── results/        # Training metrics
├── training/               # Training tools
│   ├── train_classifier.py
│   └── prepare_classification_data.py
└── dataset/                # Datasets
    └── uvh26_cls/          # Classification dataset
```

## 🚀 Usage

Run the main launcher:
```bash
./main.sh
```

Or run scripts individually:

**Camera Inference:**
```bash
python inference/camera_inference.py --camera 0
```

**Video Inference:**
```bash
python inference/video_inference.py --input video.mp4 --output result.mp4
```

## 🧠 Pipeline Logic

1.  **Stage 1 (Detection):** `yolo11x-seg.pt` detects Persons and generic Vehicles.
2.  **Stage 2 (Classification):** Detected vehicles are cropped and passed to `models/classifier/weights/best.pt` (YOLOv11m-cls) to identify the specific vehicle type (14 classes).
3.  **Logic:** Includes "Rider + Vehicle" merging and distance estimation.

## 🏋️ Training

To retrain the classifier:
1.  Ensure dataset is in `dataset/uvh26_cls`.
2.  Run:
    ```bash
    python training/train_classifier.py
    ```
