#!/bin/bash

# Mobile ADAS Web Server Launcher
# Quick start script to run web server and display network info

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to display header
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║        Rear-View ADAS - Mobile Web Server                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function to get IP address
get_ip_address() {
    hostname -I | awk '{print $1}'
}

# Function to check Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 not found${NC}"
        echo "Install with: sudo apt install python3"
        exit 1
    fi
    echo -e "${GREEN}✓ Python3 found: $(python3 --version)${NC}"
}

# Function to check dependencies
check_dependencies() {
    echo -e "\n${BLUE}Checking dependencies...${NC}"
    
    python3 -c "import flask" 2>/dev/null && echo -e "${GREEN}✓ Flask${NC}" || {
        echo -e "${YELLOW}⚠ Installing Flask...${NC}"
        pip3 install -q flask flask-cors
        echo -e "${GREEN}✓ Flask installed${NC}"
    }
    
    python3 -c "import cv2" 2>/dev/null && echo -e "${GREEN}✓ OpenCV${NC}" || {
        echo -e "${RED}✗ OpenCV not found${NC}"
        exit 1
    }
}

# Function to display network info
show_network_info() {
    echo -e "\n${BLUE}╔ Network Information ╗${NC}"
    
    local ip=$(get_ip_address)
    
    if [ -z "$ip" ]; then
        echo -e "${YELLOW}⚠ No network connection detected${NC}"
        echo -e "   Localhost: http://127.0.0.1:5000"
    else
        echo -e "${GREEN}Your PC IP Address: $ip${NC}"
        echo -e "${GREEN}Mobile Access URL:  http://$ip:5000${NC}"
        echo ""
        echo -e "${YELLOW}To access from mobile:${NC}"
        echo -e "  1. Ensure mobile is on same WiFi network"
        echo -e "  2. Open browser on mobile"
        echo -e "  3. Type: http://$ip:5000"
    fi
}

# Function to display usage
show_usage() {
    echo -e "\n${BLUE}╔ Usage ╗${NC}"
    echo "  Start server:    bash start_web_server.sh"
    echo "  Custom port:     PORT=8080 bash start_web_server.sh"
    echo "  Debug mode:      DEBUG=1 bash start_web_server.sh"
}

# Function to start server
start_server() {
    local port="${PORT:-5000}"
    local debug="${DEBUG:-0}"
    
    echo -e "\n${GREEN}Starting web server on port $port...${NC}"
    echo -e "${YELLOW}Press CTRL+C to stop${NC}\n"
    
    # Start Python server
    if [ "$debug" = "1" ]; then
        python3 inference/web_server.py --port "$port" --debug
    else
        python3 inference/web_server.py --port "$port"
    fi
}

# Function to handle cleanup
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    exit 0
}

# Trap SIGINT for clean shutdown
trap cleanup SIGINT

# Main execution
main() {
    print_header
    check_python
    check_dependencies
    show_network_info
    show_usage
    start_server
}

# Run main
main
