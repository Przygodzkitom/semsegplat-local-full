# 🚀 Quick Start Guide - Pure Docker Approach

Get your semantic segmentation platform running with true Docker portability!

## 🐳 **Docker Philosophy: Build Once, Run Anywhere**

This project follows Docker's core principle:
- **Single Dockerfile** that works on Windows, macOS, Linux
- **No platform-specific scripts** needed
- **Automatic GPU detection** at runtime
- **Pure containerized approach**

## 📋 Prerequisites

- **Docker Desktop** installed and running
- **Docker and Docker Compose** for local MinIO storage
- **8GB+ RAM** available
- **10GB+ free disk space**
- **NVIDIA GPU** (optional - will use CPU if not available)

## ⚡ Quick Setup (3 steps!)

### 1. Add Google Cloud Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin > Service Accounts**
3. Create a new service account or select existing
4. Add roles: **Storage Object Admin** and **Storage Object Viewer**
5. Create a new key (JSON format)
6. Download and save as `label-studio-key.json` in project root

### 2. Build and Run (That's it!)
```bash
# Build the container (works on any platform)
docker build -f docker/Dockerfile -t semsegplat-app .

# Start the application (auto-detects GPU availability)
# On Linux/macOS:
./start.sh

# On Windows:
start.bat

# Or manually (modern Docker):
docker compose up  # Works on any machine, auto-detects GPU

# Or manually (legacy Docker):
docker-compose up  # Works on any machine, auto-detects GPU
```

### 3. Access the Applications
- **Main App**: http://localhost:8501
- **Label Studio**: http://localhost:8080 (admin@example.com / admin)

## 🎯 First Steps

1. **Upload Images**: Go to "Upload Images" tab and upload some test images
2. **Annotate**: Open Label Studio and create segmentation masks
3. **Train**: Go to "Train Model" tab, detect classes, and start training
4. **Inference**: Test your trained model on new images

## 🔍 Automatic GPU Detection

The container automatically detects GPU availability:
- **GPU detected**: Uses ResNet101 + ImageNet weights + gradient checkpointing
- **CPU only**: Uses ResNet50 + no pre-trained weights + optimized for CPU
- **Batch sizes**: Automatically adjusted based on available memory
- **Training epochs**: 100 for GPU, 50 for CPU

## 🐛 Common Issues

**Docker not running:**
- Start Docker Desktop
- Wait for Docker to fully initialize

**Port conflicts:**
- Check if ports 8501 or 8080 are in use
- Change ports in `docker-compose.yml` if needed

**MinIO connection failed:**
- Verify `label-studio-key.json` is in project root
- Check service account permissions
- Ensure bucket exists and is accessible

**Out of memory:**
- Close other applications
- Increase Docker memory limit in Docker Desktop settings

**GPU not detected:**
- Check if NVIDIA Docker runtime is installed
- Run `docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi`
- Container will automatically fall back to CPU if GPU not available

**"no nvidia runtime" error:**
- Use the default configuration: `docker compose up` (or `docker-compose up`)
- The app will automatically detect GPU availability and fall back to CPU
- For GPU machines, use: `docker compose -f docker-compose.gpu.yml up`

## 📞 Need Help?

- Check logs: `docker compose logs -f semseg-app` (or `docker-compose logs -f semseg-app`)
- Check GPU detection: Look at container logs for device information
- See full documentation in `GPU_DETECTION_SETUP.md`
- For Docker setup issues, see `DOCKER_SETUP.md`

## ⏱️ Expected Times

- **First build**: 10-15 minutes (downloading Docker images)
- **Training (GPU)**: 15-30 minutes (depending on dataset size)
- **Training (CPU)**: 2-4 hours (depending on dataset size)
- **Inference**: 5-10 seconds per image

## 🚀 **That's it!**

No platform-specific scripts, no complex setup - just pure Docker portability! 🎉

**The same commands work on Windows, macOS, and Linux.** 