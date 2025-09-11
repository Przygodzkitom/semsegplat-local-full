#!/usr/bin/env python3
"""
Minimal brush training script with extreme memory optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.simple_brush_dataloader import SimpleBrushDataset
from utils.gpu_detector import detect_gpu, get_optimal_batch_size, get_optimal_model_config, setup_environment_for_device, print_device_info
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import gc
import psutil
import shutil
import json
import sys
from datetime import datetime

# 🔍 DEBUG: Show Environment Variables at startup
print("=" * 60)
print("🔍 MINIMAL BRUSH TRAINING SCRIPT - ENVIRONMENT VARIABLES DEBUG:")
print("=" * 60)
print(f"BUCKET_NAME env var: '{os.getenv('BUCKET_NAME', 'NOT SET')}'")
print(f"ANNOTATION_PREFIX env var: '{os.getenv('ANNOTATION_PREFIX', 'NOT SET')}'")
print(f"GCS_BUCKET_NAME env var: '{os.getenv('GCS_BUCKET_NAME', 'NOT SET')}'")
print("=" * 60)

# 🔍 GPU Detection and Configuration
print("🔍 Detecting GPU availability...")
gpu_config = detect_gpu()
setup_environment_for_device(gpu_config)
print_device_info(gpu_config)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_disk_usage():
    """Get current disk usage percentage"""
    total, used, free = shutil.disk_usage("/")
    return (used / total) * 100

def update_progress(epoch, total_epochs, status="running", log_message=""):
    """Update progress file for external monitoring"""
    progress_data = {
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'progress': (epoch / total_epochs) * 100,
        'status': status,
        'memory_usage': get_memory_usage(),
        'gpu_memory_usage': 0,
        'disk_usage': get_disk_usage(),
        'annotation_type': 'brush'
    }
    
    if log_message:
        progress_data['log'] = [log_message]
    
    try:
        with open("/app/training_progress.json", 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"🔍 ERROR: Could not update progress file: {e}")

def load_class_configuration():
    """Load class configuration from file"""
    config_file = "/app/class_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get('class_names', ["Background"])
        except Exception as e:
            print(f"Error loading class configuration: {e}")
    
    return ["Background"]

# --- Configuration ---
BUCKET_NAME = os.getenv('BUCKET_NAME', 'segmentation-platform')
IMG_PREFIX = "images/"
ANNOTATION_PREFIX = os.getenv('ANNOTATION_PREFIX', 'masks/')

print("🔍 Loading class configuration for BRUSH annotations...")
class_names = load_class_configuration()
print(f"✅ Classes: {class_names}")

# --- Transforms (minimal) ---
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
], is_check_shapes=False)

# --- Dataset and Dataloader ---
print("🔗 Creating dataset...")
update_progress(0, 100, "initializing", "Creating dataset...")

try:
    # Detect if background is explicitly defined in annotations
    from utils.annotation_type_detector import AnnotationTypeDetector
    detector = AnnotationTypeDetector(BUCKET_NAME, ANNOTATION_PREFIX)
    detection = detector.detect_annotation_type()
    has_explicit_background = detection.get('has_explicit_background', False)
    
    print(f"🎨 Background handling: {'Explicit' if has_explicit_background else 'Implicit'}")
    
    dataset = SimpleBrushDataset(
        bucket_name=BUCKET_NAME,
        img_prefix=IMG_PREFIX,
        annotation_prefix=ANNOTATION_PREFIX,
        transform=transform, 
        multilabel=True,
        class_names=class_names,
        has_explicit_background=has_explicit_background
    )
    print("✅ Dataset created successfully!")
    update_progress(0, 100, "initializing", "Dataset created successfully!")
    
    # Use batch size 1 for minimal memory usage
    batch_size = 1
    print(f"📊 Using batch size: {batch_size} (minimal)")
    
    # Create data loader with minimal workers
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    print("✅ Data loader created successfully!")
    update_progress(0, 100, "initializing", "Data loader created successfully!")
    
except Exception as e:
    error_msg = f"❌ Error creating dataset: {e}"
    print(error_msg)
    update_progress(0, 100, "failed", error_msg)
    import traceback
    traceback.print_exc()
    exit(1)

# --- Model (minimal) ---
print("🏗️ Initializing model...")
update_progress(0, 100, "initializing", "Initializing model...")

device = gpu_config['device']
print(f"🚀 Using device: {device}")

# Use smaller model for memory efficiency
model = smp.Unet(
    encoder_name='resnet18',  # Use smaller encoder for memory efficiency
    classes=len(class_names), 
    activation=None, 
    encoder_weights=None  # No pretrained weights to save memory
).to(device)

print("✅ Model initialized successfully!")
update_progress(0, 100, "initializing", "Model initialized successfully!")

# --- Simple Loss Function ---
print("⚖️ Setting up simple loss function...")
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Reduce LR every 20 epochs
print("✅ Simple loss function ready!")

# --- Training Loop (minimal) ---
epochs = 20  # Reduced epochs for memory stability
os.makedirs("models/checkpoints", exist_ok=True)

print("🚀 Starting MINIMAL BRUSH training...")
update_progress(0, 100, "running", "Starting MINIMAL BRUSH training...")
print(f"📊 Total epochs: {epochs}")
print(f"📊 Training batches per epoch: {len(train_loader)}")

try:
    for epoch in range(epochs):
        epoch_msg = f"🔄 Epoch {epoch+1}/{epochs} - Memory: {get_memory_usage():.1f} MB (MINIMAL BRUSH)"
        print(epoch_msg)
        update_progress(epoch + 1, epochs, "running", epoch_msg)
        
        # Check memory before starting epoch
        current_memory = get_memory_usage()
        if current_memory > 1500:  # Conservative memory limit for laptop stability
            error_msg = f"❌ Memory usage too high ({current_memory:.1f} MB), stopping training"
            print(error_msg)
            update_progress(epoch + 1, epochs, "failed", error_msg)
            break
            
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                print(f"🔍 DEBUG: Starting batch {batch_idx}/{len(train_loader)}")
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Move to device
                images = images.to(device)
                masks = masks.to(device).float()
                
                # Forward pass
                preds = model(images)
                
                # Calculate loss
                loss = loss_fn(preds, masks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                running_loss += loss_value
                batch_count += 1
                
                # Aggressive memory cleanup
                del preds, loss, images, masks
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Extra cleanup every batch for laptop stability
                if batch_idx % 1 == 0:  # Every batch
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                batch_msg = f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss_value:.4f}, Memory: {get_memory_usage():.1f} MB"
                print(batch_msg)
                update_progress(epoch + 1, epochs, "running", batch_msg)
                
                # Stop if memory gets too high
                if get_memory_usage() > 1500:
                    print("❌ Memory limit reached, stopping training")
                    break
                    
            except Exception as e:
                error_msg = f"❌ Error in training batch {batch_idx}: {e}"
                print(error_msg)
                update_progress(epoch + 1, epochs, "running", error_msg)
                import traceback
                traceback.print_exc()
                break

        # Calculate average loss
        avg_train_loss = running_loss / max(batch_count, 1)
        epoch_summary = f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} (MINIMAL BRUSH)"
        print(epoch_summary)
        update_progress(epoch + 1, epochs, "running", epoch_summary)
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📈 Learning rate: {current_lr:.6f}")
        
        # Aggressive cleanup after each epoch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save final model
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"final_model_brush_minimal_{timestamp}.pth"
        torch.save(model.state_dict(), f"models/checkpoints/{model_filename}")
        
        # Save config
        config_filename = model_filename.replace('.pth', '_config.json')
        config = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'model_filename': model_filename,
            'annotation_type': 'brush',
            'encoder_name': 'resnet18',  # Use the actual encoder used in training
            'created_at': datetime.now().isoformat(),
            'training_info': {
                'bucket_name': BUCKET_NAME,
                'img_prefix': IMG_PREFIX,
                'annotation_prefix': ANNOTATION_PREFIX
            }
        }
        
        with open(f"models/checkpoints/{config_filename}", 'w') as f:
            json.dump(config, f, indent=2)
        
        final_save_msg = f"✅ Final MINIMAL BRUSH model saved as {model_filename}!"
        print(final_save_msg)
        update_progress(epochs, epochs, "completed", final_save_msg)
        
    except Exception as e:
        error_msg = f"❌ Error saving final model: {e}"
        print(error_msg)
        update_progress(epochs, epochs, "failed", error_msg)

    completion_msg = f"\n✅ MINIMAL BRUSH Training complete! Final model saved as {model_filename} in models/checkpoints/"
    print(completion_msg)
    update_progress(epochs, epochs, "completed", completion_msg)

except KeyboardInterrupt:
    interrupt_msg = "\n⚠️ Training interrupted by user"
    print(interrupt_msg)
    update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "interrupted", interrupt_msg)
    
except Exception as e:
    error_msg = f"\n❌ Training failed with error: {e}"
    print(error_msg)
    update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "failed", error_msg)
    import traceback
    traceback.print_exc()
