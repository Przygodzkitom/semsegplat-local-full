import torch
import os
import subprocess
import sys

def detect_gpu():
    """
    Detect GPU availability and return device configuration
    Returns:
        dict: Device configuration with keys:
            - available: bool
            - device: torch.device
            - name: str
            - memory_gb: float
            - cuda_version: str
    """
    config = {
        'available': False,
        'device': torch.device('cpu'),
        'name': 'CPU',
        'memory_gb': 0.0,
        'cuda_version': 'N/A'
    }
    
    # Check if CUDA is available in PyTorch
    if torch.cuda.is_available():
        try:
            # Get GPU device
            device = torch.device('cuda:0')
            
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(0)
            
            # Get GPU memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Get CUDA version
            cuda_version = torch.version.cuda
            
            config.update({
                'available': True,
                'device': device,
                'name': gpu_name,
                'memory_gb': memory_gb,
                'cuda_version': cuda_version
            })
            
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Memory: {memory_gb:.1f} GB")
            print(f"   CUDA Version: {cuda_version}")
            
        except Exception as e:
            print(f"⚠️ GPU detection error: {e}")
            config['available'] = False
    else:
        print("ℹ️ No GPU detected, using CPU")
    
    return config

def get_optimal_batch_size(gpu_config):
    """
    Determine optimal batch size based on GPU memory
    """
    if not gpu_config['available']:
        return 1  # CPU training
    
    memory_gb = gpu_config['memory_gb']
    
    # Conservative batch size estimation based on memory
    if memory_gb >= 24:
        return 8
    elif memory_gb >= 16:
        return 6
    elif memory_gb >= 12:
        return 4
    elif memory_gb >= 8:
        return 2
    else:
        return 1

def get_optimal_model_config(gpu_config):
    """
    Determine optimal model configuration based on GPU
    """
    if gpu_config['available']:
        # Use ResNet101 for GPU training (better performance)
        return {
            'encoder': 'resnet101',
            'encoder_weights': 'imagenet',
            'use_checkpoint': True  # Enable gradient checkpointing for GPU
        }
    else:
        # Use ResNet50 for CPU training (faster)
        return {
            'encoder': 'resnet50',
            'encoder_weights': None,
            'use_checkpoint': False
        }

def setup_environment_for_device(gpu_config):
    """
    Set up environment variables for optimal device performance
    """
    if gpu_config['available']:
        # GPU optimizations
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("🚀 Environment configured for GPU training")
    else:
        # CPU optimizations
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
        print("🚀 Environment configured for CPU training")

def print_device_info(gpu_config):
    """
    Print detailed device information
    """
    print("=" * 60)
    print("🔍 DEVICE CONFIGURATION")
    print("=" * 60)
    print(f"Device: {gpu_config['device']}")
    print(f"Name: {gpu_config['name']}")
    print(f"Memory: {gpu_config['memory_gb']:.1f} GB")
    print(f"CUDA Version: {gpu_config['cuda_version']}")
    print(f"Available: {gpu_config['available']}")
    print("=" * 60)
