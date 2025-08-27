#!/bin/bash

# Semantic Segmentation Platform - Deployment Script
# This script sets up the complete project on any new machine

set -e  # Exit on any error

echo "🚀 Semantic Segmentation Platform - Deployment"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create required directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p label-studio-data
mkdir -p models/checkpoints
mkdir -p models/saved_models
mkdir -p models/utils
mkdir -p app
mkdir -p docker
mkdir -p logs
mkdir -p backups
mkdir -p temp

echo "✅ Directory structure created"

# Start MinIO first
echo ""
echo "🚀 Starting MinIO..."
docker compose up -d minio

# Wait for MinIO to be ready
echo "⏳ Waiting for MinIO to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:9000/minio/health/live > /dev/null; then
        echo "✅ MinIO is ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

# Start all services
echo ""
echo "🚀 Starting all services..."
docker compose up -d

echo "✅ All services started"

# Wait a moment for services to initialize
echo "⏳ Waiting for services to initialize..."
sleep 10

# Test connectivity
echo ""
echo "🔗 Testing connectivity..."

# Test MinIO
if curl -s http://localhost:9000/minio/health/live > /dev/null; then
    echo "✅ MinIO API: http://localhost:9000"
else
    echo "❌ MinIO API not responding"
fi

# Test MinIO Console
if curl -s http://localhost:9001 > /dev/null; then
    echo "✅ MinIO Console: http://localhost:9001"
else
    echo "❌ MinIO Console not responding"
fi

# Test Label Studio
if curl -s http://localhost:8080 > /dev/null; then
    echo "✅ Label Studio: http://localhost:8080"
else
    echo "❌ Label Studio not responding"
fi

# Test Streamlit App
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Streamlit App: http://localhost:8501"
else
    echo "❌ Streamlit App not responding"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Access Information:"
echo "  • Label Studio: http://localhost:8080"
echo "    Username: admin@example.com"
echo "    Password: admin"
echo ""
echo "  • MinIO Console: http://localhost:9001"
echo "    Username: minioadmin"
echo "    Password: minioadmin123"
echo ""
echo "  • Streamlit App: http://localhost:8501"
echo ""
echo "📚 Next Steps:"
echo "1. Configure Label Studio storage (see LABEL_STUDIO_MINIO_SETTINGS.md)"
echo "2. Upload images via Streamlit app"
echo "3. Start annotating in Label Studio"
echo ""
echo "🛠️ Useful Commands:"
echo "  • View logs: docker compose logs -f"
echo "  • Stop services: docker compose down"
echo "  • Restart services: docker compose restart"
echo "  • Backup data: cp -r label-studio-data label-studio-data-backup-\$(date +%Y%m%d)"

