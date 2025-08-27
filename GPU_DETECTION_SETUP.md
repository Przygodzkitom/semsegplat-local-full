# Runtime GPU Detection Setup

This project uses **runtime GPU detection** with pure Docker portability - the container automatically detects if a GPU is available and configures itself for optimal performance.

## 🐳 **Docker Philosophy: Build Once, Run Anywhere**

This project follows Docker's core principle:
- **Single Dockerfile** that works on Windows, macOS, Linux
- **No platform-specific scripts** needed
- **Automatic GPU detection** at runtime
- **Pure containerized approach**

## 🚀 Key Features

### Runtime GPU Detection
- **Single Dockerfile**: One container that works on both GPU and CPU
- **Runtime Detection**: GPU detection happens when the container starts
- **Automatic Fallback**: If no GPU is available, automatically uses CPU
- **No Rebuilding**: Same container works everywhere

### Automatic Optimizations
- **Hardware Detection**: Automatically detects NVIDIA GPUs at runtime
- **Memory Analysis**: Determines optimal batch sizes based on available GPU memory
- **Model Optimization**: Selects optimal model architecture (ResNet101 for GPU, ResNet50 for CPU)
- **Environment Configuration**: Sets optimal environment variables for each device type

## 📋 How It Works

### Runtime Detection Process
1. **Container Starts**: Single container starts with PyTorch CUDA support
2. **GPU Check**: `gpu_detector.py` checks if CUDA is available inside container
3. **Configuration**: Automatically configures model, batch size, and training parameters
4. **Fallback**: If no GPU, automatically switches to CPU-optimized settings

### Configuration Logic

#### GPU Configuration (when detected):
- **Model**: ResNet101 with ImageNet weights
- **Batch Size**: 2-8 based on memory (8GB+ = 8, 16GB+ = 6, etc.)
- **Epochs**: 100 (more training for better performance)
- **Gradient Checkpointing**: Enabled
- **DataLoader**: 4 workers, pin_memory=True

#### CPU Configuration (fallback):
- **Model**: ResNet50 without pre-trained weights
- **Batch Size**: 1 (memory efficient)
- **Epochs**: 50 (faster training)
- **Gradient Checkpointing**: Disabled
- **DataLoader**: 0 workers, pin_memory=False

## 🛠️ Installation

### Docker (Recommended)

#### Build the Container:
```bash
docker build -f docker/Dockerfile -t semsegplat-app .
```

#### Run with GPU Passthrough (if you have NVIDIA Docker runtime):
```bash
# GPU passthrough is enabled by default in docker-compose.yml
docker-compose up
```

#### Run without GPU Passthrough:
```bash
# Comment out the deploy section in docker-compose.yml
docker-compose up
```

### Direct Python Installation

#### Manual Setup:
```bash
# Install PyTorch with CUDA support (will fall back to CPU)
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Docker Usage

#### Single Container for All Environments:
```bash
# Build once, run anywhere
docker build -f docker/Dockerfile -t semsegplat-app .

# Run on any machine (GPU or CPU)
docker-compose up
```

#### Manual Docker Run:
```bash
# Run on GPU machine (with GPU passthrough)
docker run --gpus all -p 8501:8501 semsegplat-app

# Run on CPU-only machine
docker run -p 8501:8501 semsegplat-app
```

### Direct Python Usage

#### Training:
```python
# Training automatically detects GPU and configures itself
python models/training.py
```

#### Inference:
```python
# Inference automatically uses the best available device
from models.inferencer import Inferencer
inferencer = Inferencer(model_path, config)
```

#### Web Interface:
```bash
# Shows device information in the sidebar
streamlit run app/main.py
```

## 📊 Performance Expectations

### Training Speed Comparison:

| Environment | Device Type | Relative Speed | Training Time (50 epochs) |
|-------------|-------------|----------------|---------------------------|
| Docker GPU  | GPU (24GB)  | 20x faster     | 15-30 minutes            |
| Docker GPU  | GPU (16GB)  | 15x faster     | 20-40 minutes            |
| Docker GPU  | GPU (8GB)   | 10x faster     | 30-60 minutes            |
| Docker CPU  | CPU         | 1x (baseline)  | 2-4 hours                |
| Direct GPU  | GPU (any)   | 20x faster     | 15-30 minutes            |
| Direct CPU  | CPU         | 1x (baseline)  | 2-4 hours                |

## 🔧 Configuration Files

### Runtime Detection (`models/utils/gpu_detector.py`)
- **`detect_gpu()`**: Detects GPU availability at runtime
- **`get_optimal_batch_size()`**: Batch size optimization
- **`get_optimal_model_config()`**: Model configuration
- **`setup_environment_for_device()`**: Environment setup

### Docker Configuration
- **`docker/Dockerfile`**: Single container with CUDA support
- **`docker-compose.yml`**: GPU passthrough enabled by default

## 🔍 Monitoring

### Device Information Display
The web interface shows:
- **Device Type**: GPU name or CPU
- **Memory**: Available GPU memory
- **CUDA Version**: Installed CUDA version
- **Status**: Available/Not available

### Training Progress
Training logs include:
- **Device**: GPU/CPU being used
- **Batch Size**: Current batch size
- **Memory Usage**: GPU and system memory
- **Performance**: Training speed metrics

## 🐛 Troubleshooting

### Docker GPU Issues:

#### GPU Not Available in Container:
```bash
# Check if NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### Container Falls Back to CPU:
```bash
# Check container logs
docker-compose logs semseg-app

# Verify GPU detection
docker exec -it <container_name> python -c "import torch; print(torch.cuda.is_available())"
```

### Direct Python Issues:

#### GPU Not Detected:
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with GPU support
pip uninstall torch torchvision
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Performance Tips

### For Docker Users:
1. **GPU passthrough is enabled by default** in docker-compose.yml
2. **Monitor container logs**: `docker-compose logs -f semseg-app`
3. **Check GPU usage**: `nvidia-smi` on host
4. **Use appropriate resources**: Allocate sufficient RAM to Docker

### For Direct Python Users:
1. **Close other GPU applications** during training
2. **Monitor GPU memory** with `nvidia-smi`
3. **Use appropriate batch sizes** for your GPU memory
4. **Enable gradient checkpointing** for large models

## 🔄 Migration Guide

### From CPU-Only to GPU:
1. **Install NVIDIA drivers and CUDA**
2. **Install PyTorch with GPU support**: `pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu118`
3. **For Docker**: GPU passthrough is enabled by default
4. **Verify detection**: Check web interface

## 📝 Environment Variables

The system automatically sets these environment variables at runtime:

### For GPU:
```bash
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### For CPU:
```bash
OMP_NUM_THREADS=<cpu_count>
MKL_NUM_THREADS=<cpu_count>
```

## 🎯 Best Practices

1. **Use runtime detection**: Single container for all environments
2. **Monitor resource usage**: Check logs and GPU usage
3. **GPU passthrough is enabled by default**: No additional configuration needed
4. **Keep GPU drivers updated**: For best performance
5. **Monitor GPU temperature**: During long training sessions

## 📞 Support

If you encounter issues:

1. **Check runtime detection**: Look at container logs
2. **Verify PyTorch installation**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check system requirements**: Ensure sufficient RAM and storage
4. **Review logs**: Check training logs for error messages
