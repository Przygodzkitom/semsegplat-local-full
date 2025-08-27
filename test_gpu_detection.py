#!/usr/bin/env python3
"""
Simple GPU detection test script
"""

import torch
import os

def test_gpu_detection():
    print("🔍 GPU Detection Test")
    print("=" * 50)
    
    # Check PyTorch CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU computation
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅ GPU computation successful: {z.shape}")
    else:
        print("❌ CUDA not available - using CPU")
        print("This could be due to:")
        print("1. NVIDIA drivers not installed")
        print("2. NVIDIA Container Toolkit not installed")
        print("3. Docker GPU passthrough not working")
        print("4. PyTorch installed without CUDA support")
    
    # Check environment variables
    print("\n🔧 Environment Variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")

if __name__ == "__main__":
    test_gpu_detection()
