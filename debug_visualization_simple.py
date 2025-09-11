#!/usr/bin/env python3
"""
Simple debug visualization script for Docker container
"""

import sys
import os
import json
import numpy as np
import cv2
from models.inferencer import Inferencer
from models.inference import parse_labelstudio_annotation
from models.config import ModelConfig
import boto3
from botocore.exceptions import ClientError

def setup_minio_client():
    """Setup MinIO client using boto3"""
    endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
    
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )

def load_image_from_minio(s3_client, bucket_name, image_key):
    """Load image from MinIO using boto3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_data = response['Body'].read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except ClientError as e:
        print(f"Error loading image {image_key}: {e}")
        return None

def save_mask_as_image(mask, filename, class_names):
    """Save mask as colored image"""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Colors: Background=black, Class1=green, Class2=red, etc.
    colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for class_id in range(len(class_names)):
        if class_id < len(colors):
            vis[mask == class_id] = colors[class_id]
    
    cv2.imwrite(filename, vis)
    print(f"💾 Saved {filename}")

def create_overlay_image(image, pred_mask, gt_mask, filename):
    """Create overlay showing differences"""
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # True positives (both pred and gt have class 1) - green
    tp = (pred_mask == 1) & (gt_mask == 1)
    overlay[tp] = [0, 255, 0]
    
    # False positives (pred has class 1, gt has class 0) - red
    fp = (pred_mask == 1) & (gt_mask == 0)
    overlay[fp] = [255, 0, 0]
    
    # False negatives (pred has class 0, gt has class 1) - blue
    fn = (pred_mask == 0) & (gt_mask == 1)
    overlay[fn] = [0, 0, 255]
    
    cv2.imwrite(filename, overlay)
    print(f"💾 Saved {filename}")

def calculate_metrics(pred_mask, gt_mask, class_names):
    """Calculate detailed metrics"""
    print("\n📊 Detailed Metrics:")
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        
        pred_class = (pred_mask == class_id)
        gt_class = (gt_mask == class_id)
        
        tp = np.sum(pred_class & gt_class)
        fp = np.sum(pred_class & ~gt_class)
        fn = np.sum(~pred_class & gt_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        print(f"  {class_name}:")
        print(f"    IoU: {iou:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1: {f1:.4f}")
        print(f"    TP: {tp}, FP: {fp}, FN: {fn}")

def main():
    """Main function"""
    if len(sys.argv) != 4:
        print("Usage: python debug_visualization_simple.py <export_file> <model_path> <image_index>")
        sys.exit(1)
    
    export_file = sys.argv[1]
    model_path = sys.argv[2]
    image_index = int(sys.argv[3])
    
    print(f"🔍 Simple Debug Visualization")
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
    
    # Load model config
    config_path = model_path.replace('.pth', '_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    class_names = config.get('class_names', ['Background', 'wbc'])
    num_classes = len(class_names)
    print(f"🏷️ Class names: {class_names}")
    
    # Create config object
    model_config = ModelConfig(num_classes=num_classes, class_names=class_names)
    if 'encoder_name' in config:
        model_config.encoder_name = config['encoder_name']
    
    # Initialize inferencer
    inferencer = Inferencer(model_path, model_config)
    
    # Setup MinIO client
    s3_client = setup_minio_client()
    bucket_name = "segmentation-platform"
    
    # Load image
    image = load_image_from_minio(s3_client, bucket_name, image_key)
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
    annotation_data = task['annotations'][0]
    gt_mask = parse_labelstudio_annotation(annotation_data, class_names)
    
    if gt_mask is None:
        print("❌ Failed to parse ground truth annotation")
        sys.exit(1)
    
    print(f"🎯 Ground truth mask shape: {gt_mask.shape}, unique values: {np.unique(gt_mask)}")
    
    # Save visualizations
    base_name = f"debug_{image_index}_{image_key.replace('/', '_').replace('.png', '')}"
    
    # Save original image
    cv2.imwrite(f"{base_name}_original.png", image)
    print(f"💾 Saved {base_name}_original.png")
    
    # Save prediction mask
    save_mask_as_image(pred_mask, f"{base_name}_prediction.png", class_names)
    
    # Save ground truth mask
    save_mask_as_image(gt_mask, f"{base_name}_ground_truth.png", class_names)
    
    # Save overlay
    create_overlay_image(image, pred_mask, gt_mask, f"{base_name}_overlay.png")
    
    # Calculate and print metrics
    calculate_metrics(pred_mask, gt_mask, class_names)
    
    print(f"\n✅ Debug visualization complete! Check the generated images:")
    print(f"  - {base_name}_original.png")
    print(f"  - {base_name}_prediction.png")
    print(f"  - {base_name}_ground_truth.png")
    print(f"  - {base_name}_overlay.png")

if __name__ == "__main__":
    main()
