#!/usr/bin/env python3
"""
Test script to check data loader memory usage
"""

import sys
import os
import gc
import psutil
import torch

# Add the models/utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'utils'))

from simple_brush_dataloader import SimpleBrushDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_data_loader_memory():
    """Test data loader memory usage step by step"""
    
    print("🔍 Testing Data Loader Memory Usage")
    print("=" * 50)
    
    # Configuration
    bucket_name = "segmentation-platform"
    annotation_prefix = "annotations/"
    img_prefix = "images/"
    
    print(f"Initial memory: {get_memory_usage():.1f} MB")
    
    # Step 1: Create transform
    print("\nStep 1: Creating transform...")
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ], is_check_shapes=False)
    print(f"Memory after transform: {get_memory_usage():.1f} MB")
    
    # Step 2: Create dataset
    print("\nStep 2: Creating dataset...")
    try:
        dataset = SimpleBrushDataset(
            bucket_name=bucket_name,
            img_prefix=img_prefix,
            annotation_prefix=annotation_prefix,
            transform=transform,
            multilabel=True,
            class_names=['Background', 'Object'],
            has_explicit_background=True
        )
        print(f"✅ Dataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        print(f"Memory after dataset: {get_memory_usage():.1f} MB")
        
        # Step 3: Test loading first sample
        print("\nStep 3: Loading first sample...")
        image, mask = dataset[0]
        print(f"✅ Sample loaded successfully!")
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Memory after first sample: {get_memory_usage():.1f} MB")
        
        # Step 4: Test loading multiple samples
        print("\nStep 4: Loading multiple samples...")
        for i in range(min(5, len(dataset))):
            image, mask = dataset[i]
            print(f"  Sample {i}: Memory: {get_memory_usage():.1f} MB")
            
            # Clear variables
            del image, mask
            gc.collect()
        
        print(f"Memory after multiple samples: {get_memory_usage():.1f} MB")
        
        # Step 5: Test with DataLoader
        print("\nStep 5: Testing with DataLoader...")
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        print(f"Memory after DataLoader: {get_memory_usage():.1f} MB")
        
        # Test one batch
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"  Batch {batch_idx}: Memory: {get_memory_usage():.1f} MB")
            print(f"    Images shape: {images.shape}")
            print(f"    Masks shape: {masks.shape}")
            
            # Clear batch
            del images, masks
            gc.collect()
            
            if batch_idx >= 2:  # Test only first 3 batches
                break
        
        print(f"Memory after DataLoader test: {get_memory_usage():.1f} MB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Data loader memory test completed!")

if __name__ == "__main__":
    test_data_loader_memory()

