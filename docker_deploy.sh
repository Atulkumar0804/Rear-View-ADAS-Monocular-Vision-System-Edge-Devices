#!/bin/bash

# Docker deployment helper for ADAS Web Server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║        ADAS Web Server - Docker Deployment Helper        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  bash docker_deploy.sh build     - Build Docker image"
    echo "  bash docker_deploy.sh run       - Run container"
    echo "  bash docker_deploy.sh stop      - Stop container"
    echo "  bash docker_deploy.sh logs      - View logs"
    echo "  bash docker_deploy.sh compose   - Use docker-compose"
}

case "${1:-help}" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker build -f Dockerfile.web -t adas-web:latest .
        echo -e "${GREEN}✓ Image built successfully${NC}"
        ;;
    run)
        echo -e "${GREEN}Starting ADAS web container...${NC}"
        docker run -d -p 5000:5000 \
            --device /dev/video0 \
            -v $(pwd)/logs:/app/logs \
            --name adas-web \
            adas-web:latest
        echo -e "${GREEN}✓ Container started${NC}"
        echo -e "${YELLOW}Access at: http://localhost:5000${NC}"
        ;;
    stop)
        echo -e "${YELLOW}Stopping container...${NC}"
        docker stop adas-web || true
        docker rm adas-web || true
        echo -e "${GREEN}✓ Container stopped${NC}"
        ;;
    logs)
        docker logs -f adas-web
        ;;
    compose)
        echo -e "${GREEN}Starting with docker-compose...${NC}"
        docker-compose -f docker-compose.web.yml up -d
        echo -e "${YELLOW}Access at: http://localhost:5000${NC}"
        ;;
    *)
        print_header
        show_usage
        ;;
esac
