# Docker Setup Guide

This project supports both GPU and CPU-only machines with automatic runtime detection. Choose the appropriate setup based on your hardware.

## 🚀 Quick Start

### For CPU-Only Machines (Default)
```bash
# This will work on any machine, automatically detecting GPU availability
docker compose up  # Modern Docker
# OR
docker-compose up  # Legacy Docker
```

### For GPU-Enabled Machines
```bash
# Use the GPU-specific configuration
docker compose -f docker-compose.gpu.yml up  # Modern Docker
# OR
docker-compose -f docker-compose.gpu.yml up  # Legacy Docker
```

## 🔧 Configuration Options

### 1. CPU-Only Setup (`docker-compose.yml`)
- **Use case**: Any machine (CPU-only or GPU-enabled)
- **Runtime detection**: Automatically detects GPU availability at runtime
- **Fallback**: Gracefully falls back to CPU if no GPU is available
- **Command**: `docker compose up` (or `docker-compose up`)

### 2. GPU-Optimized Setup (`docker-compose.gpu.yml`)
- **Use case**: Machines with NVIDIA Docker runtime installed
- **GPU passthrough**: Explicitly enables GPU acceleration
- **Performance**: Optimal for GPU training and inference
- **Command**: `docker compose -f docker-compose.gpu.yml up` (or `docker-compose -f docker-compose.gpu.yml up`)

## 🐳 Docker Runtime Detection

The application includes sophisticated runtime GPU detection:

### Automatic Detection Features
- **Hardware Detection**: Detects NVIDIA GPUs at container startup
- **Memory Analysis**: Determines optimal batch sizes based on GPU memory
- **Model Optimization**: Selects optimal model architecture (ResNet101 for GPU, ResNet50 for CPU)
- **Environment Configuration**: Sets optimal environment variables for each device type

### Detection Process
1. **Container Starts**: Single container with PyTorch CUDA support
2. **GPU Check**: `gpu_detector.py` checks if CUDA is available inside container
3. **Configuration**: Automatically configures model, batch size, and training parameters
4. **Fallback**: If no GPU, automatically switches to CPU-optimized settings

## 📋 Usage Examples

### Scenario 1: CPU-Only Machine
```bash
# Build and run on CPU-only machine
docker compose up  # Modern Docker
# OR
docker-compose up  # Legacy Docker

# The app will automatically detect no GPU and use CPU optimization
# Training will be slower but fully functional
```

### Scenario 2: GPU Machine with NVIDIA Docker Runtime
```bash
# Option A: Use GPU-optimized configuration
docker compose -f docker-compose.gpu.yml up  # Modern Docker
# OR
docker-compose -f docker-compose.gpu.yml up  # Legacy Docker

# Option B: Use default configuration (will still detect GPU)
docker compose up  # Modern Docker
# OR
docker-compose up  # Legacy Docker
```

### Scenario 3: GPU Machine without NVIDIA Docker Runtime
```bash
# Use default configuration - will fall back to CPU
docker compose up  # Modern Docker
# OR
docker-compose up  # Legacy Docker

# The app will detect no GPU access and use CPU optimization
```

## 🔍 Verification

### Check GPU Detection
After starting the container, check the logs:
```bash
docker compose logs semseg-app  # Modern Docker
# OR
docker-compose logs semseg-app  # Legacy Docker
```

Look for these messages:
- **GPU detected**: `✅ GPU detected: [GPU Name]`
- **CPU fallback**: `ℹ️ No GPU detected, using CPU`

### Web Interface
The Streamlit interface shows device information in the sidebar:
- **GPU Available**: Shows GPU name, memory, and CUDA version
- **CPU Mode**: Shows "Running on CPU" message

## 🛠️ Troubleshooting

### Error: "no nvidia runtime"
**Problem**: Using `docker-compose.gpu.yml` on a machine without NVIDIA Docker runtime.

**Solution**: Use the default configuration instead:
```bash
# Instead of: docker compose -f docker-compose.gpu.yml up
# Use: docker compose up
# OR
# Instead of: docker-compose -f docker-compose.gpu.yml up
# Use: docker-compose up
```

### GPU Not Detected in Container
**Problem**: GPU is available on host but not detected in container.

**Solutions**:
1. **Check NVIDIA Docker runtime**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Use GPU configuration**:
   ```bash
   docker compose -f docker-compose.gpu.yml up  # Modern Docker
   # OR
   docker-compose -f docker-compose.gpu.yml up  # Legacy Docker
   ```

3. **Check container logs**:
   ```bash
   docker compose logs semseg-app  # Modern Docker
   # OR
   docker-compose logs semseg-app  # Legacy Docker
   ```

### Performance Issues
**Problem**: Training is very slow.

**Solutions**:
1. **Check device detection**: Look at container logs for GPU detection
2. **Verify GPU usage**: Run `nvidia-smi` on host during training
3. **Use GPU configuration**: Ensure you're using `docker-compose.gpu.yml` on GPU machines

## 📊 Performance Expectations

| Configuration | Device | Training Speed | Training Time (50 epochs) |
|---------------|--------|----------------|---------------------------|
| `docker-compose.yml` | GPU | 20x faster | 15-30 minutes |
| `docker-compose.yml` | CPU | 1x (baseline) | 2-4 hours |
| `docker-compose.gpu.yml` | GPU | 20x faster | 15-30 minutes |

## 🔄 Migration Guide

### From CPU-Only to GPU
1. **Install NVIDIA drivers and CUDA**
2. **Install NVIDIA Docker runtime**
3. **Use GPU configuration**:
   ```bash
   docker compose -f docker-compose.gpu.yml up  # Modern Docker
   # OR
   docker-compose -f docker-compose.gpu.yml up  # Legacy Docker
   ```

### From GPU to CPU-Only
1. **Use default configuration**:
   ```bash
   docker compose up  # Modern Docker
   # OR
   docker-compose up  # Legacy Docker
   ```
2. **The app will automatically detect no GPU and use CPU**

## 🎯 Best Practices

1. **Start with default**: Always try `docker compose up` (or `docker-compose up`) first
2. **Check logs**: Monitor container logs for device detection
3. **Use GPU config on GPU machines**: Use `docker-compose.gpu.yml` for optimal performance
4. **Monitor resources**: Check GPU usage with `nvidia-smi`
5. **Keep drivers updated**: For best GPU performance

## 📞 Support

If you encounter issues:

1. **Check runtime detection**: Look at container logs
2. **Verify PyTorch installation**: `docker exec -it <container> python -c "import torch; print(torch.cuda.is_available())"`
3. **Check system requirements**: Ensure sufficient RAM and storage
4. **Use appropriate configuration**: Match docker-compose file to your hardware

## 🚀 **That's it!**

The same container works everywhere - just choose the right configuration for your hardware! 🎉
