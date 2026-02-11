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

## ⚡ Quick Setup (2 steps!)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd semsegplat-full_local_version
```

### 2. Start the Platform (Choose One)

**Option A: Quick Start (Recommended)**
```bash
# Linux/macOS - Automatic directory creation & GPU detection
chmod +x start.sh
./start.sh

# Windows - Automatic directory creation & GPU detection
start.bat
```

**Option B: Full Deployment (First-time setup)**
```bash
# Linux/macOS - Includes validation and health checks
chmod +x deploy.sh
./deploy.sh
```

**Option C: Manual Start**
```bash
# CPU-only or automatic GPU fallback
docker compose up -d

# Force GPU configuration (requires NVIDIA Docker)
docker compose -f docker-compose.gpu.yml up -d
```

### 3. Access the Applications
- **Streamlit App**: http://localhost:8501
- **Label Studio**: http://localhost:8080 (admin@example.com / admin)
- **MinIO Console**: http://localhost:9001 (minioadmin / minioadmin123)

## 🎯 First Steps

1. **Upload Images**: Go to "Upload Images" tab and upload some test images
2. **Annotate**: Open Label Studio and create segmentation masks
3. **Train**: Go to "Train Model" tab, detect classes, and start training
4. **Inference**: Test your trained model on new images

## 🔧 Understanding Startup Scripts

### start.sh / start.bat (Quick Start)
**What it does:**
- ✅ Creates required data directories automatically
- ✅ Auto-detects GPU availability (start.sh only)
- ✅ Starts containers immediately
- ⚡ Fast startup for daily use

**Use when:**
- Restarting after `docker compose down`
- Daily development work
- You want quick startup

### deploy.sh (Full Deployment)
**What it does:**
- ✅ Checks Docker installation
- ✅ Creates all directories (including redundant safety checks)
- ✅ Starts MinIO first and waits for readiness
- ✅ Tests connectivity to all services
- ✅ Displays access URLs and credentials
- ✅ Shows next steps and useful commands

**Use when:**
- First-time setup on a new machine
- Troubleshooting deployment issues
- You want comprehensive validation

### Data Directory Creation

Both scripts automatically create:
```
label-studio-data/      ← Label Studio database
minio-data/             ← S3 storage for images/annotations
models/checkpoints/     ← Trained model storage
```

**No manual directory creation needed!** 🎉

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