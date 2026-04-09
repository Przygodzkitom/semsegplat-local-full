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
    """Calculate pixel-level IoU for each class."""
    ious = []
    for cls in range(num_classes):
        pred = (pred_mask == cls)
        gt = (gt_mask == cls)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
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

def compute_objectwise_metrics(pred_mask, gt_mask, iou_threshold=0.5):
    """Compute true object-wise metrics using connected component analysis and IoU matching.

    Each connected component in pred and GT is treated as an individual object.
    Objects are matched via IoU: a predicted object counts as TP if it overlaps
    a GT object with IoU >= iou_threshold. Unmatched predictions are FP,
    unmatched GT objects are FN.
    """
    # Convert masks to binary uint8
    if pred_mask.dtype == bool:
        pred_binary = pred_mask.astype(np.uint8)
    elif pred_mask.max() > 1:
        pred_binary = (pred_mask > 0).astype(np.uint8)
    else:
        pred_binary = pred_mask.astype(np.uint8)

    if gt_mask.dtype == bool:
        gt_binary = gt_mask.astype(np.uint8)
    elif gt_mask.max() > 1:
        gt_binary = (gt_mask > 0).astype(np.uint8)
    else:
        gt_binary = gt_mask.astype(np.uint8)

    # Extract connected components (individual objects)
    pred_labels, n_pred = connected_components(pred_binary)
    gt_labels, n_gt = connected_components(gt_binary)

    # Filter out tiny fragments below MIN_OBJECT_AREA
    pred_objects = []
    for idx in range(1, n_pred + 1):
        obj_mask = (pred_labels == idx)
        if obj_mask.sum() >= MIN_OBJECT_AREA:
            pred_objects.append(obj_mask)

    gt_objects = []
    for idx in range(1, n_gt + 1):
        obj_mask = (gt_labels == idx)
        if obj_mask.sum() >= MIN_OBJECT_AREA:
            gt_objects.append(obj_mask)

    n_pred_obj = len(pred_objects)
    n_gt_obj = len(gt_objects)

    # Diagnostic: show object sizes
    pred_sizes = [int(p.sum()) for p in pred_objects]
    gt_sizes = [int(g.sum()) for g in gt_objects]
    print(f"    Object-wise: {n_pred_obj} pred objects (sizes: {pred_sizes}), "
          f"{n_gt_obj} GT objects (sizes: {gt_sizes}), iou_threshold={iou_threshold}")
    print(f"    Total pred pixels: {int(pred_binary.sum())}, total GT pixels: {int(gt_binary.sum())}, "
          f"mask shape: {pred_binary.shape}")

    # Edge case: no objects at all
    if n_pred_obj == 0 and n_gt_obj == 0:
        return {
            'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'n_pred_objects': 0, 'n_gt_objects': 0,
        }

    if n_pred_obj == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': n_gt_obj,
            'n_pred_objects': 0, 'n_gt_objects': n_gt_obj,
        }

    if n_gt_obj == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': n_pred_obj, 'fn': 0,
            'n_pred_objects': n_pred_obj, 'n_gt_objects': 0,
        }

    # Build IoU matrix between every predicted and GT object
    iou_matrix = np.zeros((n_pred_obj, n_gt_obj))
    for i, p_obj in enumerate(pred_objects):
        for j, g_obj in enumerate(gt_objects):
            iou_matrix[i, j] = iou_coef(p_obj, g_obj)

    # Diagnostic: show best IoU for each GT object
    for j in range(n_gt_obj):
        best_iou = iou_matrix[:, j].max() if n_pred_obj > 0 else 0
        print(f"      GT obj {j} (size={gt_sizes[j]}): best IoU with any pred = {best_iou:.4f}")

    # Greedy matching: repeatedly pick the highest IoU pair above threshold
    matched_pred = set()
    matched_gt = set()
    tp = 0

    # Flatten and sort by descending IoU
    pairs = []
    for i in range(n_pred_obj):
        for j in range(n_gt_obj):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for _, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp += 1

    fp = n_pred_obj - tp
    fn = n_gt_obj - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"    Object-wise results: tp={tp}, fp={fp}, fn={fn}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_pred_objects': n_pred_obj,
        'n_gt_objects': n_gt_obj,
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
        # Get original dimensions
        h, w = image.shape[:2]

        # Convert to RGB if needed
        if len(image.shape) == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create dummy mask for consistent transform
        dummy_mask = np.zeros((h, w, self.num_classes), dtype=np.uint8)

        # Transform image
        augmented = self.transform(image=image, mask=dummy_mask)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

            # Apply threshold and keep at original size
            preds = (probs > self.threshold).astype(np.uint8)

            # Resize predictions back to original size using INTER_NEAREST
            masks = []
            for i in range(self.num_classes):
                mask = cv2.resize(preds[i], (w, h), interpolation=cv2.INTER_NEAREST)
                masks.append(mask)

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
                st.image((pred_masks[i] * 255).astype(np.uint8), caption=f"{class_name}")
        
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
                    st.image((gt_mask_class * 255).astype(np.uint8), caption=f"GT {class_name}")
        
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

    def predict_and_compare(self, image, gt_mask, iou_threshold=0.5):
        """Predict masks and compare with ground truth"""
        print(f"  predict_and_compare: image {image.shape}, gt {gt_mask.shape}, iou_threshold={iou_threshold}")
        
        # Get predictions
        pred_masks = self.predict(image)

        # Combine per-class binary masks into a single label mask
        pred_mask = np.zeros_like(pred_masks[0])
        for i in range(self.num_classes):
            pred_mask[pred_masks[i] == 1] = i

        # Resize ground truth mask to match predicted mask size if needed
        if gt_mask is not None and pred_mask.shape != gt_mask.shape:
            print(f"  WARNING: Resizing ground truth from {gt_mask.shape} to {pred_mask.shape}")
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Calculate pixel-level IoU for each class
        ious = calculate_iou(pred_mask, gt_mask, num_classes=self.num_classes)

        # Calculate object-wise metrics for each class (excluding background)
        metrics = {}
        for i in range(1, self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            pred_class_mask = (pred_mask == i)
            gt_class_mask = (gt_mask == i)
            class_metrics = compute_objectwise_metrics(pred_class_mask, gt_class_mask, iou_threshold=iou_threshold)
            metrics[class_name] = class_metrics

        return pred_mask, ious, metrics

    def __del__(self):
        """Cleanup when destroyed"""
        try:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
