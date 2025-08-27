#!/bin/bash

# Semantic Segmentation Platform - Auto-Detection Startup Script
# This script automatically detects GPU availability and chooses the right configuration

echo "🚀 Semantic Segmentation Platform - Auto-Detection Startup"
echo "=========================================================="

# Detect which docker compose command to use
if command -v docker &> /dev/null; then
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
        echo "✅ Using: docker compose (modern)"
    elif docker-compose --version &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
        echo "✅ Using: docker-compose (legacy fallback)"
    else
        echo "❌ Error: Neither 'docker compose' nor 'docker-compose' found"
        exit 1
    fi
else
    echo "❌ Error: Docker not found"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
echo "🔍 Checking for NVIDIA Docker runtime..."

if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA Docker runtime detected!"
    echo "🚀 Starting with GPU acceleration..."
    echo "   Using: $DOCKER_COMPOSE -f docker-compose.gpu.yml up"
    echo ""
    $DOCKER_COMPOSE -f docker-compose.gpu.yml up
else
    echo "ℹ️  NVIDIA Docker runtime not available"
    echo "🔄 Starting with CPU fallback (GPU will be detected at runtime if available)..."
    echo "   Using: $DOCKER_COMPOSE up"
    echo ""
    $DOCKER_COMPOSE up
fi
