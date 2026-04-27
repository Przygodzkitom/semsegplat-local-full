#!/bin/bash

# Semantic Segmentation Platform - Auto-Detection Startup Script
# This script automatically detects GPU availability and chooses the right configuration

echo "🚀 Semantic Segmentation Platform - Auto-Detection Startup"
echo "=========================================================="

# Create required directories if they don't exist
echo "📁 Ensuring directory structure..."
mkdir -p label-studio-data
mkdir -p minio-data
mkdir -p models/checkpoints
echo "✅ Directory structure ready"
echo ""

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

if docker info 2>/dev/null | grep -i "Runtimes" | grep -q "nvidia"; then
    echo "✅ NVIDIA Docker runtime detected!"
    echo "🚀 Starting with GPU acceleration..."
    # Prefer docker-compose (V1) — it correctly honors "runtime: nvidia"
    if command -v docker-compose &> /dev/null; then
        echo "   Using: docker-compose -f docker-compose.gpu.yml up"
        echo ""
        docker-compose -f docker-compose.gpu.yml up
    else
        echo "   Using: $DOCKER_COMPOSE -f docker-compose.gpu.yml up"
        echo ""
        $DOCKER_COMPOSE -f docker-compose.gpu.yml up
    fi
else
    echo "ℹ️  NVIDIA Docker runtime not available"
    echo "🔄 Starting with CPU fallback (GPU will be detected at runtime if available)..."
    echo "   Using: $DOCKER_COMPOSE up"
    echo ""
    $DOCKER_COMPOSE up
fi
