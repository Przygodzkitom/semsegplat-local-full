#!/usr/bin/env python3
"""
Debug visualization script to compare model predictions vs ground truth annotations
"""

import sys
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.inferencer import Inferencer
from models.inference import parse_labelstudio_annotation
import boto3
from botocore.exceptions import ClientError

def setup_s3_client():
    """Setup S3 client for MinIO"""
    return boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123',
        region_name='us-east-1'
    )

def load_image_from_s3(s3_client, bucket_name, image_key):
    """Load image from S3/MinIO"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_data = response['Body'].read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except ClientError as e:
        print(f"Error loading image {image_key}: {e}")
        return None

def create_mask_visualization(mask, class_names, title="Mask"):
    """Create a colored visualization of a mask"""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Define colors for each class
    colors = [
        (0, 0, 0),      # Background - black
        (0, 255, 0),    # Class 1 - green
        (255, 0, 0),    # Class 2 - red
        (0, 0, 255),    # Class 3 - blue
        (255, 255, 0),  # Class 4 - yellow
        (255, 0, 255),  # Class 5 - magenta
    ]
    
    for class_id in range(len(class_names)):
        if class_id < len(colors):
            vis[mask == class_id] = colors[class_id]
    
    return vis

def create_overlay_visualization(image, pred_mask, gt_mask, class_names):
    """Create overlay visualization showing differences"""
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Create difference mask
    diff_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # True positives (both pred and gt have class 1) - green
    tp = (pred_mask == 1) & (gt_mask == 1)
    diff_mask[tp] = [0, 255, 0]
    
    # False positives (pred has class 1, gt has class 0) - red
    fp = (pred_mask == 1) & (gt_mask == 0)
    diff_mask[fp] = [255, 0, 0]
    
    # False negatives (pred has class 0, gt has class 1) - blue
    fn = (pred_mask == 0) & (gt_mask == 1)
    diff_mask[fn] = [0, 0, 255]
    
    # True negatives (both pred and gt have class 0) - transparent (keep original)
    
    # Blend with original image
    alpha = 0.6
    overlay = cv2.addWeighted(overlay, 1-alpha, diff_mask, alpha, 0)
    
    return overlay, diff_mask

def calculate_detailed_metrics(pred_mask, gt_mask, class_names):
    """Calculate detailed metrics"""
    metrics = {}
    
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        
        pred_class = (pred_mask == class_id)
        gt_class = (gt_mask == class_id)
        
        tp = np.sum(pred_class & gt_class)
        fp = np.sum(pred_class & ~gt_class)
        fn = np.sum(~pred_class & gt_class)
        tn = np.sum(~pred_class & ~gt_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        metrics[class_name] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        }
    
    return metrics

def create_comparison_visualization(image, pred_mask, gt_mask, class_names, image_name, metrics):
    """Create comprehensive comparison visualization"""
    
    # Create individual visualizations
    pred_vis = create_mask_visualization(pred_mask, class_names, "Prediction")
    gt_vis = create_mask_visualization(gt_mask, class_names, "Ground Truth")
    overlay, diff_mask = create_overlay_visualization(image, pred_mask, gt_mask, class_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Debug Visualization: {image_name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Prediction mask
    axes[0, 1].imshow(pred_vis)
    axes[0, 1].set_title('Model Prediction')
    axes[0, 1].axis('off')
    
    # Ground truth mask
    axes[0, 2].imshow(gt_vis)
    axes[0, 2].set_title('Ground Truth Annotation')
    axes[0, 2].axis('off')
    
    # Overlay comparison
    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Overlay (Green=TP, Red=FP, Blue=FN)')
    axes[1, 0].axis('off')
    
    # Difference mask
    axes[1, 1].imshow(diff_mask)
    axes[1, 1].set_title('Difference Mask')
    axes[1, 1].axis('off')
    
    # Metrics table
    axes[1, 2].axis('off')
    metrics_text = "Detailed Metrics:\n\n"
    for class_name, class_metrics in metrics.items():
        metrics_text += f"{class_name}:\n"
        metrics_text += f"  IoU: {class_metrics['iou']:.4f}\n"
        metrics_text += f"  Precision: {class_metrics['precision']:.4f}\n"
        metrics_text += f"  Recall: {class_metrics['recall']:.4f}\n"
        metrics_text += f"  F1: {class_metrics['f1']:.4f}\n"
        metrics_text += f"  TP: {class_metrics['tp']}, FP: {class_metrics['fp']}, FN: {class_metrics['fn']}\n\n"
    
    axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    if len(sys.argv) != 4:
        print("Usage: python debug_visualization.py <export_file> <model_path> <image_index>")
        print("Example: python debug_visualization.py label-studio-data/export/project-1-at-YYYY-MM-DD-HH-MM-XXXXXXXX.json models/checkpoints/final_model_polygon_20250910_110111.pth 0")
        sys.exit(1)
    
    export_file = sys.argv[1]
    model_path = sys.argv[2]
    image_index = int(sys.argv[3])
    
    print(f"🔍 Debug Visualization Script")
    print(f"Export file: {export_file}")
    print(f"Model path: {model_path}")
    print(f"Image index: {image_index}")
    
    # Load export data
    with open(export_file, 'r') as f:
        export_data = json.load(f)
    
    if image_index >= len(export_data):
        print(f"❌ Image index {image_index} out of range. Available: 0-{len(export_data)-1}")
        sys.exit(1)
    
    task = export_data[image_index]
    image_path = task['data']['image']
    image_key = image_path.replace('s3://segmentation-platform/', '')
    
    print(f"📸 Processing image: {image_key}")
    
    # Load model
    config_path = model_path.replace('.pth', '_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    class_names = config.get('class_names', ['Background', 'wbc'])
    print(f"🏷️ Class names: {class_names}")
    
    # Initialize inferencer with proper config
    from models.config import ModelConfig
    model_config = ModelConfig(
        num_classes=len(class_names),
        class_names=class_names
    )
    # Set encoder name if available
    if 'encoder_name' in config:
        model_config.encoder_name = config['encoder_name']
    inferencer = Inferencer(model_path, model_config)
    
    # Setup S3 client
    s3_client = setup_s3_client()
    bucket_name = "segmentation-platform"
    
    # Load image
    image = load_image_from_s3(s3_client, bucket_name, image_key)
    if image is None:
        print(f"❌ Failed to load image: {image_key}")
        sys.exit(1)
    
    print(f"📸 Image loaded: {image.shape}")
    
    # Get model prediction
    print("🤖 Running model prediction...")
    pred_masks = inferencer.predict(image)
    
    # Combine predictions into single mask
    pred_mask = np.zeros_like(pred_masks[0])
    for i in range(len(class_names)):
        pred_mask[pred_masks[i] == 1] = i
    
    print(f"🎯 Prediction mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")
    
    # Parse ground truth annotation
    print("📝 Parsing ground truth annotation...")
    annotation_data = task['annotations'][0]  # Get first annotation
    gt_mask = parse_labelstudio_annotation(annotation_data, class_names)
    
    if gt_mask is None:
        print("❌ Failed to parse ground truth annotation")
        sys.exit(1)
    
    print(f"🎯 Ground truth mask shape: {gt_mask.shape}, unique values: {np.unique(gt_mask)}")
    
    # Calculate detailed metrics
    metrics = calculate_detailed_metrics(pred_mask, gt_mask, class_names)
    
    # Create visualization
    print("🎨 Creating visualization...")
    fig = create_comparison_visualization(image, pred_mask, gt_mask, class_names, image_key, metrics)
    
    # Save visualization
    output_file = f"debug_visualization_{image_index}_{image_key.replace('/', '_').replace('.png', '')}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_file}")
    
    # Print summary
    print("\n📊 Summary:")
    for class_name, class_metrics in metrics.items():
        print(f"  {class_name}: IoU={class_metrics['iou']:.4f}, F1={class_metrics['f1']:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()
