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

MIN_OBJECT_AREA = 1  # Minimum area to consider an object (reduced from 100)

def calculate_iou(pred_mask, gt_mask, num_classes):
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

def compute_objectwise_metrics(pred_mask, gt_mask, iou_threshold=0.1):
    """Compute object-wise metrics (precision, recall, F1)"""
    # Convert masks to binary if they're not already
    if pred_mask.max() > 1:
        pred_mask = (pred_mask > 0).astype(np.uint8)
    if gt_mask.max() > 1:
        gt_mask = (gt_mask > 0).astype(np.uint8)
    
    labeled_pred, n_pred = connected_components(pred_mask)
    labeled_gt, n_gt = connected_components(gt_mask)

    matched_gt = set()
    tp = 0

    for i in range(1, n_pred + 1):
        pred_obj = (labeled_pred == i).astype(np.uint8)
        pred_area = pred_obj.sum()
        
        if pred_area < MIN_OBJECT_AREA:
            continue
            
        best_iou = 0
        best_gt = -1
        for j in range(1, n_gt + 1):
            if j in matched_gt:
                continue
            gt_obj = (labeled_gt == j).astype(np.uint8)
            gt_area = gt_obj.sum()
            
            if gt_area < MIN_OBJECT_AREA:
                continue
                
            iou = iou_coef(pred_obj, gt_obj)
            
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)

    # Count objects after filtering by MIN_OBJECT_AREA
    valid_pred = sum(1 for i in range(1, n_pred + 1) 
                    if ((labeled_pred == i).sum() >= MIN_OBJECT_AREA))
    valid_gt = sum(1 for i in range(1, n_gt + 1) 
                   if ((labeled_gt == i).sum() >= MIN_OBJECT_AREA))

    fp = valid_pred - tp
    fn = valid_gt - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_pred': valid_pred,
        'n_gt': valid_gt
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
            print(f"Using optimal encoder: {encoder_name}")
        
        # Load model with correct encoder
        self.model = smp.Unet(encoder_name, classes=self.num_classes, activation=None)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
        # Get predictions
        pred_masks = self.predict(image)
        
        # Combine predictions into a single mask where each class has its own value
        pred_mask = np.zeros_like(pred_masks[0])
        for i in range(self.num_classes):
            pred_mask[pred_masks[i] == 1] = i
        
        # Resize predicted mask to match ground truth size (512x512)
        if gt_mask is not None and pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate IoU for each class
        ious = calculate_iou(pred_mask, gt_mask, num_classes=self.num_classes)
        
        # Calculate object-wise metrics for each class (excluding background)
        metrics = {}
        for i in range(1, self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            metrics[class_name] = compute_objectwise_metrics(pred_mask == i, gt_mask == i)
        
        return pred_mask, ious, metrics

    def __del__(self):
        """Cleanup when destroyed"""
        try:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
