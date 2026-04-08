import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.labelstudio_dataloader import LabelStudioMinIODatasetNumeric
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

# Clear memory state at startup
def clear_memory_state():
    """Clear memory state at training startup"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Memory state cleared at training startup")
    except Exception as e:
        print(f"Warning: Could not clear memory state: {e}")

# Clear memory before starting
clear_memory_state()

# 🔍 DEBUG: Show Environment Variables at startup
print("=" * 60)
print("🔍 POLYGON TRAINING SCRIPT - ENVIRONMENT VARIABLES DEBUG:")
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

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_disk_usage():
    """Get current disk usage percentage"""
    total, used, free = shutil.disk_usage("/")
    return (used / total) * 100

def check_disk_space():
    """Check if we have enough disk space"""
    usage = get_disk_usage()
    if usage > 95:
        print(f"⚠️ WARNING: Disk usage is {usage:.1f}% - very high!")
        return False
    elif usage > 90:
        print(f"⚠️ WARNING: Disk usage is {usage:.1f}% - high!")
    return True

def save_model_config(model_filename, class_names, annotation_type="polygon", encoder_name=None):
    """Save model configuration alongside the model file"""
    try:
        config_filename = model_filename.replace('.pth', '_config.json')
        config = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'model_filename': model_filename,
            'annotation_type': annotation_type,
            'created_at': datetime.now().isoformat(),
            'training_info': {
                'bucket_name': BUCKET_NAME,
                'img_prefix': IMG_PREFIX,
                'annotation_prefix': ANNOTATION_PREFIX
            }
        }
        
        # Add encoder information if provided
        if encoder_name:
            config['encoder_name'] = encoder_name
        
        with open(f"models/checkpoints/{config_filename}", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Verify the config was saved correctly
        if encoder_name:
            print(f"✅ Model config saved as {config_filename}")
            print(f"🔑 VERIFIED: encoder_name = {encoder_name} saved in config")
        else:
            print(f"⚠️ Model config saved as {config_filename}")
            print(f"⚠️ WARNING: No encoder_name saved in config!")
        
        return True
    except Exception as e:
        print(f"❌ Error saving model config: {e}")
        return False

loss_history = {'train_losses': [], 'val_losses': []}

def update_progress(epoch, total_epochs, status="running", log_message=""):
    """Update progress file for external monitoring"""
    print(f"🔍 DEBUG: update_progress called - epoch={epoch}, total={total_epochs}, status={status}")

    progress_data = {
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'progress': (epoch / total_epochs) * 100,
        'status': status,
        'memory_usage': get_memory_usage(),
        'gpu_memory_usage': get_gpu_memory_usage(),
        'disk_usage': get_disk_usage(),
        'annotation_type': 'polygon',
        'loss_history': loss_history,
    }
    
    if log_message:
        # Read existing log
        log_file = "/app/training.log"
        existing_log = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_log = f.readlines()
            except:
                pass
        
        # Add new message and keep only last 100 lines
        existing_log.append(log_message + "\n")
        if len(existing_log) > 100:
            existing_log = existing_log[-100:]
        
        # Write back to log file
        try:
            with open(log_file, 'w') as f:
                f.writelines(existing_log)
        except:
            pass
        
        progress_data['log'] = [line.strip() for line in existing_log[-20:]]  # Last 20 lines
    
    # Write progress to JSON file
    try:
        progress_file_path = "/app/training_progress.json"
        print(f"🔍 DEBUG: Writing progress to {progress_file_path}")
        with open(progress_file_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"🔍 DEBUG: Progress file updated successfully")
    except Exception as e:
        print(f"🔍 ERROR: Could not update progress file: {e}")
        import traceback
        traceback.print_exc()

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
    
    # Default fallback
    return ["Background"]

# --- 1. GCS Configuration ---
BUCKET_NAME = os.getenv('BUCKET_NAME', 'segmentation-platform')  # Your GCS bucket name
IMG_PREFIX = "images/"  # Prefix for images in GCS
ANNOTATION_PREFIX = os.getenv('ANNOTATION_PREFIX', 'masks/')  # Prefix for annotations in GCS

# 🔍 DEBUG: Show GCS Configuration
print("=" * 60)
print("🔍 POLYGON TRAINING - GCS CONFIGURATION DEBUG INFO:")
print("=" * 60)
print(f"📦 BUCKET_NAME: '{BUCKET_NAME}'")
print(f"🖼️  IMG_PREFIX: '{IMG_PREFIX}'")
print(f"🎯 ANNOTATION_PREFIX: '{ANNOTATION_PREFIX}'")
print(f"🔗 Full annotation path: gs://{BUCKET_NAME}/{ANNOTATION_PREFIX}")
print(f"🖼️  Full image path: gs://{BUCKET_NAME}/{IMG_PREFIX}")
print("=" * 60)

# --- 2. Load Class Configuration ---
print("🔍 Loading class configuration for POLYGON annotations...")
update_progress(0, 100, "initializing", "Loading class configuration for POLYGON annotations...")
class_names = load_class_configuration()
print(f"✅ Classes: {class_names}")
print(f"🎯 POLYGON MODE: Background is automatically class 0 (unlabeled areas)")
update_progress(0, 100, "initializing", f"Classes: {class_names} (POLYGON mode)")

# --- 3. Transforms ---
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2(),
], is_check_shapes=False)  # Disable shape checking to prevent issues

# --- 4. Dataset and Dataloader ---
print("🔗 Connecting to GCS and loading dataset for POLYGON annotations...")
update_progress(0, 100, "initializing", "Connecting to GCS and loading dataset for POLYGON annotations...")
print(f"Initial memory usage: {get_memory_usage():.1f} MB")

# 🔍 DEBUG: Show dataset configuration
print("=" * 60)
print("🔍 POLYGON TRAINING - DATASET CONFIGURATION:")
print("=" * 60)
print(f"Dataset will load from:")
print(f"  📦 Bucket: '{BUCKET_NAME}'")
print(f"  🖼️  Images: '{IMG_PREFIX}'")
print(f"  🎯 Annotations: '{ANNOTATION_PREFIX}'")
print(f"  🏷️  Classes: {class_names}")
print(f"  🎨 Annotation Type: POLYGON (Background = class 0)")
print("=" * 60)

try:
    dataset = LabelStudioMinIODatasetNumeric(
        bucket_name=BUCKET_NAME,
        img_prefix=IMG_PREFIX,
        annotation_prefix=ANNOTATION_PREFIX,
        transform=transform, 
        multilabel=True,
        class_names=class_names  # Use dynamic class configuration
    )
    print("✅ Dataset created successfully for POLYGON annotations!")
    update_progress(0, 100, "initializing", "Dataset created successfully for POLYGON annotations!")
    print(f"Memory after dataset creation: {get_memory_usage():.1f} MB")
except Exception as e:
    error_msg = f"❌ Error creating dataset: {e}"
    print(error_msg)
    update_progress(0, 100, "failed", error_msg)
    import traceback
    traceback.print_exc()
    exit(1)

# Filter out test-set images before training
test_split_file = os.getenv('TEST_SPLIT_FILE')
if test_split_file and os.path.exists(test_split_file):
    with open(test_split_file) as f:
        split_data = json.load(f)
    test_keys = set(split_data.get('test_image_keys', []))
    original_size = len(dataset.image_annotation_pairs)
    dataset.image_annotation_pairs = [
        p for p in dataset.image_annotation_pairs if p[0] not in test_keys
    ]
    excluded = original_size - len(dataset.image_annotation_pairs)
    print(f"📊 Excluded {excluded} test-set images from training (reserved for evaluation)")
    print(f"📊 Training pool: {len(dataset.image_annotation_pairs)} images")
else:
    print("⚠️ No test split file found — training on all annotated images")

# Split dataset into train/val (80/20 split)
from torch.utils.data import random_split
total_size = len(dataset)
if total_size < 2:
    train_size = total_size
    val_size = 0
else:
    train_size = max(1, int(0.8 * total_size))
    val_size = total_size - train_size

print(f"📊 Dataset size: {total_size}")
print(f"📊 Training samples: {train_size}")
print(f"📊 Validation samples: {val_size}")

try:
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("✅ Dataset split successful!")
    update_progress(0, 100, "initializing", "Dataset split successful!")
    print(f"Memory after dataset split: {get_memory_usage():.1f} MB")
except Exception as e:
    error_msg = f"❌ Error splitting dataset: {e}"
    print(error_msg)
    update_progress(0, 100, "failed", error_msg)
    import traceback
    traceback.print_exc()
    exit(1)

print("📊 Creating data loaders...")
# Get optimal batch size based on device
batch_size = get_optimal_batch_size(gpu_config)
print(f"📊 Using batch size: {batch_size}")

# Configure DataLoader based on device
num_workers = 4 if gpu_config['available'] else 0
pin_memory = gpu_config['available']

try:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print("✅ Data loaders created successfully!")
    update_progress(0, 100, "initializing", "Data loaders created successfully!")
    print(f"Memory after dataloader creation: {get_memory_usage():.1f} MB")
except Exception as e:
    error_msg = f"❌ Error creating data loaders: {e}"
    print(error_msg)
    update_progress(0, 100, "failed", error_msg)
    import traceback
    traceback.print_exc()
    exit(1)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Test loading first batch
print("🧪 Testing first batch loading...")
try:
    test_batch = next(iter(train_loader))
    print(f"✅ First batch loaded successfully! Shape: {test_batch[0].shape}, {test_batch[1].shape}")
    update_progress(0, 100, "initializing", "First batch loaded successfully!")
    print(f"Memory after first batch: {get_memory_usage():.1f} MB")
    # Clear test batch
    del test_batch
    gc.collect()
    print(f"Memory after clearing test batch: {get_memory_usage():.1f} MB")
except Exception as e:
    error_msg = f"❌ Error loading first batch: {str(e)}"
    print(error_msg)
    update_progress(0, 100, "failed", error_msg)
    import traceback
    traceback.print_exc()
    exit(1)

# --- 4. Model ---
print("🏗️ Initializing model...")
update_progress(0, 100, "initializing", "Initializing model...")

# Use detected device
device = gpu_config['device']
print(f"🚀 Using device: {device}")

# Get optimal model configuration based on device
model_config = get_optimal_model_config(gpu_config)
print(f"🏗️ Model config: {model_config}")
print(f"🔍 GPU Detection Results:")
print(f"   GPU Available: {gpu_config['available']}")
print(f"   Device: {gpu_config['device']}")
print(f"   GPU Name: {gpu_config['name']}")
print(f"   Selected Encoder: {model_config['encoder']}")
print(f"   Encoder Weights: {model_config['encoder_weights']}")

# CRITICAL: Log the exact encoder being used
encoder_name = model_config['encoder']
print(f"🔑 CRITICAL: Using encoder: {encoder_name}")
print(f"🔑 CRITICAL: This encoder will be saved in the config file")
print(f"🔑 CRITICAL: Make sure this matches what you expect!")

# Initialize model with optimal configuration
model = smp.Unet(
    encoder_name, 
    classes=len(class_names), 
    activation=None, 
    encoder_weights=model_config['encoder_weights']
).to(device)

# Configure gradient checkpointing based on device
model.use_checkpoint = model_config['use_checkpoint']
print("✅ Model initialized successfully!")
update_progress(0, 100, "initializing", "Model initialized successfully!")
print(f"Memory after model creation: {get_memory_usage():.1f} MB")

# --- 5. Universal Loss Function ---
print("⚖️ Setting up universal loss function...")
update_progress(0, 100, "initializing", "Setting up universal loss function...")

class FocalLoss(nn.Module):
    """Universal focal loss that automatically handles class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        pt = (targets * probs) + ((1 - targets) * (1 - probs))
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice loss for better handling of small objects and class imbalance"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply sigmoid
        probs = torch.sigmoid(inputs)

        # Compute Dice per class channel (dim=1), then average across classes
        # inputs/targets shape: (batch, classes, H, W)
        probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)      # (B, C, H*W)
        targets_flat = targets.reshape(targets.shape[0], targets.shape[1], -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)               # (B, C)
        dice_per_class = (2. * intersection + self.smooth) / (
            probs_flat.sum(dim=2) + targets_flat.sum(dim=2) + self.smooth   # (B, C)
        )

        return 1 - dice_per_class.mean()

class CombinedLoss(nn.Module):
    """Combined focal + dice loss for maximum universality"""
    def __init__(self, focal_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

# Use combined loss for maximum universality
loss_fn = CombinedLoss(focal_weight=0.7, dice_weight=0.3)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("✅ Universal combined loss function ready!")

# --- 6. Training Loop ---
epochs = int(os.environ.get('NUM_EPOCHS', 100))
os.makedirs("models/checkpoints", exist_ok=True)

print("🚀 Starting POLYGON training...")
update_progress(0, epochs, "running", "Starting POLYGON training...")
print(f"📊 Total epochs: {epochs}")
print(f"📊 Training batches per epoch: {len(train_loader)}")
print(f"📊 Validation batches per epoch: {len(val_loader)}")
print(f"⏱️ Estimated time: ~30-60 minutes on CPU (runs in background)")

try:
    for epoch in range(epochs):
        epoch_msg = f"🔄 Epoch {epoch+1}/{epochs} - Memory: {get_memory_usage():.1f} MB (POLYGON)"
        print(epoch_msg)
        update_progress(epoch + 1, epochs, "running", epoch_msg)
        
        disk_msg = f"💾 Disk usage: {get_disk_usage():.1f}%"
        print(disk_msg)
        update_progress(epoch + 1, epochs, "running", disk_msg)
        
        # Check disk space before starting epoch
        if not check_disk_space():
            error_msg = "❌ Disk space too low, stopping training"
            print(error_msg)
            update_progress(epoch + 1, epochs, "failed", error_msg)
            break
            
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                print(f"🔍 DEBUG: Starting batch {batch_idx}/{len(train_loader)}")
                
                # Clear gradients before each batch
                optimizer.zero_grad()
                print(f"🔍 DEBUG: Gradients cleared for batch {batch_idx}")
                
                images = images.to(device)
                masks = masks.to(device).float()
                print(f"🔍 DEBUG: Data moved to device for batch {batch_idx}")

                preds = model(images)
                print(f"🔍 DEBUG: Forward pass completed for batch {batch_idx}")
                
                loss_unreduced = loss_fn(preds, masks)
                loss = loss_unreduced # Focal loss is already reduced
                print(f"🔍 DEBUG: Loss calculated for batch {batch_idx}: {loss.item():.4f}")

                loss.backward()
                print(f"🔍 DEBUG: Backward pass completed for batch {batch_idx}")
                
                optimizer.step()
                print(f"🔍 DEBUG: Optimizer step completed for batch {batch_idx}")

                loss_value = loss.item()  # Store the loss value before deleting the tensor
                running_loss += loss_value
                batch_count += 1

                # Clear intermediate tensors immediately
                del preds, loss_unreduced, loss, images, masks
                gc.collect()  # Garbage collect after every batch
                print(f"🔍 DEBUG: Memory cleared for batch {batch_idx}")

                # Print progress every batch for debugging
                batch_msg = f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss_value:.4f}, Memory: {get_memory_usage():.1f} MB"
                print(batch_msg)
                update_progress(epoch + 1, epochs, "running", batch_msg)
                    
            except Exception as e:
                error_msg = f"❌ Error in training batch {batch_idx}: {e}"
                print(error_msg)
                update_progress(epoch + 1, epochs, "running", error_msg)
                continue

        after_train_msg = f"After training - Memory: {get_memory_usage():.1f} MB"
        print(after_train_msg)
        update_progress(epoch + 1, epochs, "running", after_train_msg)

        # Validation
        val_msg = f"🔍 Starting validation..."
        print(val_msg)
        update_progress(epoch + 1, epochs, "running", val_msg)
        
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for val_batch_idx, (images, masks) in enumerate(val_loader):
                try:
                    images = images.to(device)
                    masks = masks.to(device).float()
                    
                    preds = model(images)
                    loss_unreduced = loss_fn(preds, masks)
                    loss = loss_unreduced # Focal loss is already reduced
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    # Clear validation tensors immediately
                    del preds, loss_unreduced, loss, images, masks
                    gc.collect()
                    
                except Exception as e:
                    error_msg = f"❌ Error in validation batch {val_batch_idx}: {e}"
                    print(error_msg)
                    update_progress(epoch + 1, epochs, "running", error_msg)
                    continue

        # Clear validation data
        gc.collect()
        after_val_msg = f"After validation - Memory: {get_memory_usage():.1f} MB"
        print(after_val_msg)
        update_progress(epoch + 1, epochs, "running", after_val_msg)

        # Calculate average losses
        avg_train_loss = running_loss / max(batch_count, 1)
        avg_val_loss = val_loss / max(val_batch_count, 1)
        
        epoch_summary = f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (POLYGON)"
        print(epoch_summary)
        loss_history['train_losses'].append(avg_train_loss)
        loss_history['val_losses'].append(avg_val_loss)
        try:
            with open("/app/loss_history.json", 'w') as f:
                json.dump(loss_history, f)
        except Exception as e:
            print(f"Warning: could not write loss_history.json: {e}")
        update_progress(epoch + 1, epochs, "running", epoch_summary)
        
        # No checkpoint saving during training - only save final model
        # This saves significant disk space for Streamlit platform

    # Save final model only
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"final_model_polygon_{timestamp}.pth"
        torch.save(model.state_dict(), f"models/checkpoints/{model_filename}")
        
        # CRITICAL: Save config with the exact encoder used
        saved_encoder = model_config['encoder']
        print(f"🔑 CRITICAL: Saving model with encoder: {saved_encoder}")
        save_model_config(model_filename, class_names, "polygon", saved_encoder)
        
        final_save_msg = f"✅ Final POLYGON model saved as {model_filename} with encoder: {saved_encoder}!"
        print(final_save_msg)
        update_progress(epochs, epochs, "running", final_save_msg)
        
        disk_final_msg = f"💾 Final disk usage: {get_disk_usage():.1f}%"
        print(disk_final_msg)
        update_progress(epochs, epochs, "running", disk_final_msg)
    except Exception as e:
        error_msg = f"❌ Error saving final model: {e}"
        print(error_msg)
        update_progress(epochs, epochs, "failed", error_msg)

    completion_msg = f"\n✅ POLYGON Training complete! Final model saved as {model_filename} in models/checkpoints/"
    print(completion_msg)
    print(f"🔑 FINAL SUMMARY:")
    print(f"   Model file: {model_filename}")
    print(f"   Config file: {model_filename.replace('.pth', '_config.json')}")
    print(f"   Encoder used: {saved_encoder}")
    print(f"   Classes: {class_names}")
    print(f"   Device: {device}")
    update_progress(epochs, epochs, "completed", completion_msg)

except KeyboardInterrupt:
    interrupt_msg = "\n⚠️ Training interrupted by user"
    print(interrupt_msg)
    update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "interrupted", interrupt_msg)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"interrupted_model_polygon_{timestamp}.pth"
        torch.save(model.state_dict(), f"models/checkpoints/{model_filename}")
        save_model_config(model_filename, class_names, "polygon", model_config['encoder'])  # Save config alongside model
        save_msg = f"✅ Interrupted POLYGON model saved as {model_filename}!"
        print(save_msg)
        update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "interrupted", save_msg)
        
        disk_msg = f"💾 Disk usage after save: {get_disk_usage():.1f}%"
        print(disk_msg)
        update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "interrupted", disk_msg)
    except Exception as e:
        error_msg = f"❌ Error saving interrupted model: {e}"
        print(error_msg)
        update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "interrupted", error_msg)
        
except Exception as e:
    error_msg = f"\n❌ Training failed with error: {e}"
    print(error_msg)
    update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "failed", error_msg)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"error_model_polygon_{timestamp}.pth"
        torch.save(model.state_dict(), f"models/checkpoints/{model_filename}")
        save_model_config(model_filename, class_names, "polygon", model_config['encoder'])  # Save config alongside model
        save_msg = f"✅ Error POLYGON model saved as {model_filename}!"
        print(save_msg)
        update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "failed", save_msg)
        
        disk_msg = f"💾 Disk usage after save: {get_disk_usage():.1f}%"
        print(disk_msg)
        update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "failed", disk_msg)
    except Exception as save_error:
        save_error_msg = f"❌ Error saving error model: {save_error}"
        print(save_error_msg)
        update_progress(epoch + 1 if 'epoch' in locals() else 0, epochs, "failed", save_error_msg)

