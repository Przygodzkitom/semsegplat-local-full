import torch
import cv2
import numpy as np
import streamlit as st
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.base import BaseSegmentation
from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
import gc
import psutil
import os
from scipy.ndimage import label as connected_components
import json

MIN_OBJECT_AREA = 100  # Minimum area to consider an object (increased to filter fragments)

def calculate_iou(pred_mask, gt_mask, num_classes):
    print(f"    🔍 DEBUG: calculate_iou called with pred_mask shape: {pred_mask.shape}, gt_mask shape: {gt_mask.shape}")
    print(f"    🔍 DEBUG: calculate_iou num_classes: {num_classes}")
    ious = []
    for cls in range(num_classes):
        pred = (pred_mask == cls)
        gt = (gt_mask == cls)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        print(f"    🔍 DEBUG: Class {cls} - pred sum: {pred.sum()}, gt sum: {gt.sum()}, intersection: {intersection}, union: {union}")
        if union == 0:
            ious.append(float('nan'))
            print(f"    🔍 DEBUG: Class {cls} IoU = NaN (no union)")
        else:
            iou = intersection / union
            ious.append(iou)
            print(f"    🔍 DEBUG: Class {cls} IoU = {iou:.4f}")
    print(f"    🔍 DEBUG: Final IoUs: {ious}")
    return ious

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return {
        'ram': process.memory_info().rss / 1024**2,  # MB
        'gpu': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0  # MB
    }

def log_memory(message):
    """Log memory usage with a message"""
    mem = get_memory_usage()
    st.sidebar.text(f"{message}\nRAM: {mem['ram']:.1f}MB\nGPU: {mem['gpu']:.1f}MB")

def iou_coef(pred_obj, gt_obj):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(pred_obj, gt_obj).sum()
    union = np.logical_or(pred_obj, gt_obj).sum()
    return intersection / (union + 1e-6)

def compute_objectwise_metrics(pred_mask, gt_mask, iou_threshold=0.1):
    """Compute pixel-level metrics (precision, recall, F1) for semantic segmentation"""
    print(f"    🔍 DEBUG: compute_objectwise_metrics called with pred_mask shape: {pred_mask.shape}, gt_mask shape: {gt_mask.shape}")
    print(f"    🔍 DEBUG: pred_mask unique values: {np.unique(pred_mask)}, gt_mask unique values: {np.unique(gt_mask)}")
    print(f"    🔍 DEBUG: pred_mask dtype: {pred_mask.dtype}, gt_mask dtype: {gt_mask.dtype}")
    
    # Convert masks to binary uint8 if they're not already
    # Handle both boolean masks and multi-class masks
    if pred_mask.dtype == bool:
        pred_mask = pred_mask.astype(np.uint8)
    elif pred_mask.max() > 1:
        pred_mask = (pred_mask > 0).astype(np.uint8)
    else:
        pred_mask = pred_mask.astype(np.uint8)
        
    if gt_mask.dtype == bool:
        gt_mask = gt_mask.astype(np.uint8)
    elif gt_mask.max() > 1:
        gt_mask = (gt_mask > 0).astype(np.uint8)
    else:
        gt_mask = gt_mask.astype(np.uint8)
    
    print(f"    🔍 DEBUG: After binary conversion - pred_mask unique: {np.unique(pred_mask)}, gt_mask unique: {np.unique(gt_mask)}")
    print(f"    🔍 DEBUG: pred_mask sum: {pred_mask.sum()}, gt_mask sum: {gt_mask.sum()}")
    
    # Use the exact same logic as debug_visualization.py
    pred_class = (pred_mask == 1)
    gt_class = (gt_mask == 1)
    
    tp = np.sum(pred_class & gt_class)
    fp = np.sum(pred_class & ~gt_class)
    fn = np.sum(~pred_class & gt_class)
    tn = np.sum(~pred_class & ~gt_class)
    
    print(f"    🔍 DEBUG: Using debug_visualization.py logic:")
    print(f"    🔍 DEBUG: tp={tp}, fp={fp}, fn={fn}, tn={tn}")
    
    # Calculate metrics using the same logic as debug_visualization.py
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"    🔍 DEBUG: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'n_pred': int(pred_mask.sum()),  # Total predicted pixels
        'n_gt': int(gt_mask.sum())       # Total ground truth pixels
    }

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
    else:
        print("No class configuration found. Using default.")
        return ["Background"]

class Inferencer(BaseSegmentation):
    def __init__(self, model_path, config, threshold=0.3, max_size=1024):
        super().__init__()
        self.threshold = threshold
        self.config = config
        self.num_classes = config.num_classes
        self.class_names = config.class_names
        
        # Detect GPU and configure device
        gpu_config = detect_gpu()
        self.device = gpu_config['device']
        
        # Get optimal model configuration
        model_config = get_optimal_model_config(gpu_config)
        
        # Check if config file specifies encoder (for compatibility with different training scripts)
        encoder_name = model_config['encoder']  # Default to optimal encoder
        if hasattr(config, 'encoder_name') and getattr(config, 'encoder_name', None):
            encoder_name = config.encoder_name
            print(f"Using encoder from config: {encoder_name}")
        else:
            print(f"⚠️ No encoder_name in config, using optimal encoder: {encoder_name}")
            print(f"⚠️ This may cause model loading errors if the model was trained with a different encoder")
        
        # Load model with correct encoder and weights
        # Try different encoder weight combinations to find the right one
        encoder_weights_options = ['imagenet', None]
        self.model = None
        
        for encoder_weights in encoder_weights_options:
            print(f"🔄 Trying encoder: {encoder_name} with weights: {encoder_weights}")
            try:
                test_model = smp.Unet(encoder_name, classes=self.num_classes, activation=None, encoder_weights=encoder_weights)
                
                # Try to load the state dict with strict=False to check compatibility
                state_dict = torch.load(model_path, map_location=self.device)
                missing_keys, unexpected_keys = test_model.load_state_dict(state_dict, strict=False)
                
                # If we have very few missing keys, this is probably the right combination
                if len(missing_keys) < 10:
                    print(f"✅ Good match! Missing keys: {len(missing_keys)}")
                    self.model = test_model
                    print(f"✅ Using encoder: {encoder_name} with weights: {encoder_weights}")
                    break
                else:
                    print(f"⚠️ Some missing keys ({len(missing_keys)}), trying next option...")
                    
            except Exception as e:
                print(f"❌ Error with {encoder_name} + {encoder_weights}: {str(e)[:100]}...")
                continue
        
        if self.model is None:
            # Fallback: use the first option and let strict=False handle it
            print(f"⚠️ No perfect match found, using fallback: {encoder_name} with imagenet weights")
            self.model = smp.Unet(encoder_name, classes=self.num_classes, activation=None, encoder_weights='imagenet')
        
        # Model is already loaded with the best matching encoder/weights combination
        print(f"✅ Model loaded successfully")
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform - match training exactly
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)

    def overlay_mask(self, image, mask, color, alpha=0.4):
        """Overlay mask on image"""
        overlay = image.copy()
        overlay[mask == 1] = color
        return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    def predict(self, image):
        """Prediction function"""
        print(f"    🔍 DEBUG: predict() called with image shape: {image.shape}")
        
        # Get original dimensions
        h, w = image.shape[:2]
        print(f"    🔍 DEBUG: Original image dimensions: {w}x{h} (WxH)")
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"    🔍 DEBUG: Converted BGR to RGB")
        
        # Create dummy mask for consistent transform
        dummy_mask = np.zeros((h, w, self.num_classes), dtype=np.uint8)
        print(f"    🔍 DEBUG: Created dummy mask with shape: {dummy_mask.shape}")
        
        # Transform image
        augmented = self.transform(image=image, mask=dummy_mask)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)
        print(f"    🔍 DEBUG: Input tensor shape: {input_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            print(f"    🔍 DEBUG: Model output probs shape: {probs.shape}")
            print(f"    🔍 DEBUG: Probs min/max: {probs.min():.4f}/{probs.max():.4f}")
            
            # Apply threshold and keep at original size
            preds = (probs > self.threshold).astype(np.uint8)
            print(f"    🔍 DEBUG: After threshold {self.threshold}, preds shape: {preds.shape}")
            print(f"    🔍 DEBUG: Preds unique values: {np.unique(preds)}")
            print(f"    🔍 DEBUG: Preds sum per class: {[preds[i].sum() for i in range(self.num_classes)]}")
            
            # Resize predictions back to original size using INTER_NEAREST
            masks = []
            for i in range(self.num_classes):
                mask = cv2.resize(preds[i], (w, h), interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
                print(f"    🔍 DEBUG: Class {i} mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                print(f"    🔍 DEBUG: Class {i} mask sum: {mask.sum()}")
        
        print(f"    🔍 DEBUG: Returning {len(masks)} masks")
        return masks

    def create_visualization(self, image, pred_masks, gt_mask=None):
        """Create visualization"""
        # Create overlays
        overlayed = image.copy()
        
        # Define colors for each class (excluding background)
        colors = [
            (0, 255, 0),   # Green
            (255, 0, 0),   # Red
            (0, 0, 255),   # Blue
            (255, 255, 0), # Yellow
            (255, 0, 255), # Magenta
            (0, 255, 255), # Cyan
            (128, 0, 128), # Purple
            (255, 165, 0), # Orange
        ]
        
        # Apply overlays for each class (skip background)
        for i in range(1, self.num_classes):
            if i-1 < len(colors):
                overlayed = self.overlay_mask(overlayed, pred_masks[i], colors[i-1])
        
        # Display class-wise preview with probability threshold info
        st.subheader(f"Class-wise Mask Preview (threshold={self.threshold})")
        
        # Create columns for class masks
        cols = st.columns(min(3, self.num_classes))
        for i in range(self.num_classes):
            col_idx = i % 3
            with cols[col_idx]:
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                st.image(pred_masks[i] * 255, caption=f"{class_name}")
        
        # Create GT overlay if provided
        overlayed_gt = None
        if gt_mask is not None:
            overlayed_gt = image.copy()
            
            # Apply GT overlays for each class (skip background)
            for i in range(1, self.num_classes):
                gt_mask_class = (gt_mask == i).astype(np.uint8)
                if i-1 < len(colors):
                    # Use slightly different colors for GT
                    gt_color = tuple(int(c * 0.7) for c in colors[i-1])
                    overlayed_gt = self.overlay_mask(overlayed_gt, gt_mask_class, gt_color)
            
            st.subheader("GT Mask Preview")
            gt_cols = st.columns(min(3, self.num_classes))
            for i in range(self.num_classes):
                col_idx = i % 3
                with gt_cols[col_idx]:
                    class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                    gt_mask_class = (gt_mask == i).astype(np.uint8)
                    st.image(gt_mask_class * 255, caption=f"GT {class_name}")
        
        return overlayed, overlayed_gt

    def infer(self, data):
        """Main inference method"""
        image = data['image']
        gt_mask = data.get('gt_mask', None)
        
        # Get predictions
        pred_masks = self.predict(image)
        
        # Create visualizations
        overlayed_pred, overlayed_gt = self.create_visualization(image, pred_masks, gt_mask)
        
        # Display results
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(overlayed_pred, caption="Overlayed Prediction", use_column_width=True)
        
        if overlayed_gt is not None:
            st.image(overlayed_gt, caption="GT Mask Overlay", use_column_width=True)
        
        st.success("Inference complete!")
        
        return {
            'orig_image': image,
            'pred_masks': pred_masks,
            'overlayed_pred': overlayed_pred,
            'overlayed_gt': overlayed_gt
        }

    def predict_and_compare(self, image, gt_mask):
        """Predict masks and compare with ground truth"""
        print("=" * 40)
        print("🔍 PREDICT_AND_COMPARE CALLED - DEBUG VERSION 2024")
        print("=" * 40)
        print(f"  🔍 predict_and_compare called with image shape: {image.shape}, gt_mask shape: {gt_mask.shape}")
        
        # Get predictions
        pred_masks = self.predict(image)
        print(f"  🔍 predict() returned {len(pred_masks)} masks")
        
        # Use the exact same logic as debug_visualization.py
        pred_mask = np.zeros_like(pred_masks[0])
        print(f"  🔍 DEBUG: Starting with empty pred_mask shape: {pred_mask.shape}")
        
        for i in range(self.num_classes):
            class_mask = pred_masks[i]
            print(f"  🔍 DEBUG: Class {i} mask shape: {class_mask.shape}, unique values: {np.unique(class_mask)}")
            print(f"  🔍 DEBUG: Class {i} mask sum: {class_mask.sum()}, max: {class_mask.max()}, min: {class_mask.min()}")
            pred_mask[class_mask == 1] = i
            print(f"  🔍 DEBUG: After adding class {i}, pred_mask unique values: {np.unique(pred_mask)}")
            print(f"  🔍 DEBUG: After adding class {i}, pred_mask sum: {pred_mask.sum()}")
        
        # Ensure background class (0) is properly assigned to remaining areas
        # This is important for correct IoU calculation
        pred_mask[pred_mask == 0] = 0  # Explicitly set background areas to class 0
        print(f"  🔍 DEBUG: Final pred_mask unique values: {np.unique(pred_mask)}")
        print(f"  🔍 DEBUG: Final pred_mask value distribution: {dict(zip(*np.unique(pred_mask, return_counts=True)))}")
        
        print(f"  🔍 Debug - Predicted mask shape: {pred_mask.shape}")
        print(f"  🔍 Debug - Ground truth mask shape: {gt_mask.shape}")
        print(f"  🔍 Debug - Predicted mask unique values: {np.unique(pred_mask)}")
        print(f"  🔍 Debug - Ground truth mask unique values: {np.unique(gt_mask)}")
        
        # Debug: Show mask statistics
        if len(np.unique(pred_mask)) > 1:
            unique_vals, counts = np.unique(pred_mask, return_counts=True)
            print(f"  🔍 DEBUG: Predicted mask value distribution: {dict(zip(unique_vals, counts))}")
            print(f"  🔍 DEBUG: Predicted mask percentage of class 1: {np.sum(pred_mask == 1) / pred_mask.size * 100:.2f}%")
        
        if len(np.unique(gt_mask)) > 1:
            unique_vals, counts = np.unique(gt_mask, return_counts=True)
            print(f"  🔍 DEBUG: Ground truth mask value distribution: {dict(zip(unique_vals, counts))}")
            print(f"  🔍 DEBUG: Ground truth mask percentage of class 1: {np.sum(gt_mask == 1) / gt_mask.size * 100:.2f}%")
        
        # Resize ground truth mask to match predicted mask size for comparison
        # NOTE: This should no longer be needed since GT masks are now created at original image size
        if gt_mask is not None and pred_mask.shape != gt_mask.shape:
            print(f"  ⚠️  WARNING: Resizing ground truth from {gt_mask.shape} to {pred_mask.shape}")
            print(f"  ⚠️  This indicates a dimension mismatch that should be fixed!")
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            print(f"  🔍 Debug - Resized GT mask unique values: {np.unique(gt_mask)}")
        else:
            print(f"  ✅ No resizing needed - both masks are same size: {pred_mask.shape}")
        
        # Calculate IoU for each class
        print(f"  🔍 DEBUG: About to calculate IoU for {self.num_classes} classes")
        ious = calculate_iou(pred_mask, gt_mask, num_classes=self.num_classes)
        print(f"  🔍 DEBUG: Calculated IoUs: {ious}")
        
        # Calculate object-wise metrics for each class (excluding background)
        metrics = {}
        for i in range(1, self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            print(f"  🔍 DEBUG: Calculating object-wise metrics for class {i} ({class_name})")
            
            pred_class_mask = (pred_mask == i)
            gt_class_mask = (gt_mask == i)
            print(f"  🔍 DEBUG: Class {i} pred mask unique values: {np.unique(pred_class_mask)}")
            print(f"  🔍 DEBUG: Class {i} gt mask unique values: {np.unique(gt_class_mask)}")
            print(f"  🔍 DEBUG: Class {i} pred mask sum: {pred_class_mask.sum()}")
            print(f"  🔍 DEBUG: Class {i} gt mask sum: {gt_class_mask.sum()}")
            
            class_metrics = compute_objectwise_metrics(pred_class_mask, gt_class_mask)
            metrics[class_name] = class_metrics
            print(f"  🔍 DEBUG: Class {i} metrics: {class_metrics}")
        
        return pred_mask, ious, metrics

    def __del__(self):
        """Cleanup when destroyed"""
        try:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
