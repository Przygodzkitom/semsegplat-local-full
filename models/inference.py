import os
import cv2
import numpy as np
from tqdm import tqdm
from models.inferencer import Inferencer
from models.config import ModelConfig
import json
from google.cloud import storage

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

def parse_labelstudio_annotation(annotation_data, class_names, target_size=(512, 512)):
    """Parse Label Studio annotation and convert to mask"""
    try:
        result = annotation_data.get('result', [])
        target_width, target_height = target_size
        
        # Create empty mask at target size
        mask = np.zeros((target_height, target_width), dtype=np.uint8)
        
        # Process polygon annotations
        for item in result:
            if item.get('type') == 'polygonlabels':
                value = item.get('value', {})
                points = value.get('points', [])
                
                if points:
                    # Convert points to polygon coordinates at target size
                    polygon_points = []
                    for point in points:
                        if len(point) == 2:
                            # Convert percentage to pixels at target size
                            x = int(point[0] * target_width / 100)
                            y = int(point[1] * target_height / 100)
                            polygon_points.append((x, y))
                    
                    # Get class label
                    labels = value.get('polygonlabels', [])
                    if labels:
                        class_label = labels[0]
                        class_id = class_names.index(class_label) if class_label in class_names else 1
                        
                        # Convert polygon to mask
                        from PIL import Image, ImageDraw
                        img = Image.new('L', (target_width, target_height), 0)
                        draw = ImageDraw.Draw(img)
                        
                        if len(polygon_points) >= 3:
                            draw.polygon(polygon_points, fill=1)
                        
                        polygon_mask = np.array(img)
                        mask = np.maximum(mask, polygon_mask * class_id)
            
            elif item.get('type') == 'brushlabels':
                # Handle brush annotations if they exist
                value = item.get('value', {})
                brush_data = value.get('rle', [])
                
                if brush_data:
                    # Convert RLE to mask
                    rle_mask = np.zeros(target_height * target_width, dtype=np.uint8)
                    for i in range(0, len(brush_data), 2):
                        start = brush_data[i]
                        length = brush_data[i + 1]
                        rle_mask[start:start + length] = 1
                    rle_mask = rle_mask.reshape(target_height, target_width)
                    
                    # Get class label
                    labels = value.get('brushlabels', [])
                    if labels:
                        class_label = labels[0]
                        class_id = class_names.index(class_label) if class_label in class_names else 1
                        mask = np.maximum(mask, rle_mask * class_id)
        
        return mask
        
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return None

def batch_evaluate_with_labelstudio(image_dir, annotation_dir, model_path, bucket_name="segmentation-platform", num_classes=None, threshold=0.3):
    """Evaluate model using Label Studio annotations from MinIO"""
    # Load class configuration if num_classes not provided
    if num_classes is None:
        class_names = load_class_configuration()
        num_classes = len(class_names)
        print(f"Loaded {num_classes} classes: {class_names}")
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    config = ModelConfig(num_classes=num_classes, class_names=class_names)
    inferencer = Inferencer(model_path, config, threshold=threshold)

    # Initialize MinIO client
    import boto3
    from botocore.exceptions import ClientError
    
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000',  # MinIO endpoint (internal Docker hostname)
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123',
        region_name='us-east-1'
    )

    # Debug: List all objects in bucket to see structure
    print(f"🔍 Exploring MinIO bucket: {bucket_name}")
    # List all objects in bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    all_objects = response.get('Contents', [])
    print(f"Total objects in bucket: {len(all_objects)}")
    
    # Show bucket structure
    prefixes = set()
    for obj in all_objects[:20]:  # Show first 20 objects
        prefix = obj['Key'].split('/')[0] if '/' in obj['Key'] else obj['Key']
        prefixes.add(prefix)
        print(f"  {obj['Key']} (size: {obj['Size']})")
    
    print(f"Found prefixes: {list(prefixes)}")

    # Get all image files from MinIO
    image_files = []
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=image_dir)
    objects = response.get('Contents', [])
    for obj in objects:
        if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_files.append(obj['Key'])
    
    print(f"📸 Found {len(image_files)} images in {image_dir}")
    for img in image_files[:5]:  # Show first 5 images
        print(f"  {img}")
    
    # Try multiple possible annotation directories
    possible_annotation_dirs = [
        "masks/_/",      # User's specific structure (primary)
        annotation_dir,  # Original: "masks/"
        "annotations/",  # Common Label Studio location
        "masks/",       # Alternative
        "labels/",      # Another possibility
    ]
    
    annotation_files = []
    used_annotation_dir = None
    
    for test_dir in possible_annotation_dirs:
        print(f"🔍 Checking for annotations in: {test_dir}")
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=test_dir)
        objects = response.get('Contents', [])
        test_files = []
        for obj in objects:
            if obj['Key'].endswith('/') or obj['Size'] == 0:
                continue
            # Accept numeric files (Label Studio annotation IDs)
            filename = obj['Key'].split('/')[-1]
            if filename.isdigit():
                test_files.append(obj['Key'])
        
        print(f"  Found {len(test_files)} annotation files")
        if test_files:
            annotation_files = test_files
            used_annotation_dir = test_dir
            print(f"✅ Using annotation directory: {used_annotation_dir}")
            break
    
    if not annotation_files:
        print("❌ No annotation files found in any expected location")
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'no_data',
            'error': 'No annotation files found'
        }
    
    # Create mapping between images and annotations (using same logic as training script)
    image_annotation_pairs = []
    
    print(f"🔍 Matching {len(image_files)} images with {len(annotation_files)} annotations...")
    
    for annotation_blob_name in annotation_files:
        try:
            # Load annotation from MinIO
            response = s3_client.get_object(Bucket=bucket_name, Key=annotation_blob_name)
            annotation_data = json.loads(response['Body'].read().decode('utf-8'))
            
            if annotation_data:
                # Extract image reference from JSON
                image_url = annotation_data.get('task', {}).get('data', {}).get('image', '')
                
                if image_url:
                    # Convert S3 URL to object key
                    if image_url.startswith('s3://'):
                        # Remove s3://bucket-name/ prefix
                        image_blob_name = image_url.replace(f's3://{bucket_name}/', '')
                    else:
                        image_blob_name = image_url
                    
                    # Check if this image exists in our image list
                    if image_blob_name in image_files:
                        image_annotation_pairs.append((image_blob_name, annotation_blob_name))
                        print(f"✅ Matched: {os.path.basename(image_blob_name)} -> {os.path.basename(annotation_blob_name)}")
                    else:
                        print(f"❌ Image not found: {image_blob_name}")
                else:
                    print(f"❌ No image reference in annotation: {annotation_blob_name}")
                    
        except Exception as e:
            print(f"❌ Error processing annotation {annotation_blob_name}: {e}")
            continue
    
    print(f"Found {len(image_annotation_pairs)} image-annotation pairs for evaluation")
    
    # Show first few pairs for debugging
    for i, (img, ann) in enumerate(image_annotation_pairs[:3]):
        print(f"Pair {i+1}: {os.path.basename(img)} -> {os.path.basename(ann)}")
    
    if len(image_annotation_pairs) == 0:
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'no_data',
            'error': 'No valid image-annotation pairs found'
        }
    
    all_ious = []
    all_metrics = []

    for img_blob_name, annotation_blob_name in tqdm(image_annotation_pairs, desc="Evaluating"):
        try:
            print(f"🔄 Processing: {os.path.basename(img_blob_name)} -> {os.path.basename(annotation_blob_name)}")
            
            # Load image from MinIO
            response = s3_client.get_object(Bucket=bucket_name, Key=img_blob_name)
            image_data = response['Body'].read()
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"  📸 Image loaded: {img.shape}")
            
            # Load and parse annotation from MinIO
            response = s3_client.get_object(Bucket=bucket_name, Key=annotation_blob_name)
            annotation_data = json.loads(response['Body'].read().decode('utf-8'))
            gt_mask = parse_labelstudio_annotation(annotation_data, class_names)
            
            print(f"  🎯 Annotation parsed: {gt_mask.shape if gt_mask is not None else 'None'}")
            
            if img is None or gt_mask is None:
                print(f"❌ Skipping {img_blob_name}: missing image or annotation.")
                continue

            print(f"  🤖 Running inference...")
            pred_mask, ious, metrics = inferencer.predict_and_compare(img, gt_mask)
            print(f"  ✅ Inference complete: IoU={ious}")
            
            all_ious.append(ious)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"❌ Error evaluating {img_blob_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate and display results
    results = {
        'num_classes': num_classes,
        'class_names': class_names,
        'images_evaluated': 0,
        'mean_ious': [],
        'overall_mean_iou': 0.0,
        'avg_metrics': {},
        'status': 'no_data'
    }
    
    if all_ious:
        all_ious = np.array(all_ious)
        mean_ious = np.nanmean(all_ious, axis=0)
        
        results.update({
            'images_evaluated': len(all_ious),
            'mean_ious': mean_ious.tolist(),
            'overall_mean_iou': float(np.nanmean(mean_ious)),
            'status': 'success'
        })
        
        print("\n--- Evaluation Report ---")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        print(f"Images evaluated: {len(all_ious)}")
        
        for i, miou in enumerate(mean_ious):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"{class_name}: Mean IoU = {miou:.4f}")
        
        print(f"Mean IoU (all classes): {np.nanmean(mean_ious):.4f}")
        
        # Calculate average object-wise metrics
        if all_metrics:
            print("\n--- Object-wise Metrics ---")
            avg_metrics = {}
            for class_name in all_metrics[0].keys():
                avg_metrics[class_name] = {
                    'precision': np.mean([m[class_name]['precision'] for m in all_metrics]),
                    'recall': np.mean([m[class_name]['recall'] for m in all_metrics]),
                    'f1': np.mean([m[class_name]['f1'] for m in all_metrics])
                }
                print(f"{class_name}:")
                print(f"  Precision: {avg_metrics[class_name]['precision']:.4f}")
                print(f"  Recall: {avg_metrics[class_name]['recall']:.4f}")
                print(f"  F1-Score: {avg_metrics[class_name]['f1']:.4f}")
            
            results['avg_metrics'] = avg_metrics
    else:
        print("No valid image-annotation pairs found for evaluation.")
        results['status'] = 'no_data'
    
    return results

def batch_evaluate(image_dir, mask_dir, model_path, num_classes=None, threshold=0.3):
    """Original function for TIFF mask evaluation - kept for backward compatibility"""
    # Load class configuration if num_classes not provided
    if num_classes is None:
        class_names = load_class_configuration()
        num_classes = len(class_names)
        print(f"Loaded {num_classes} classes: {class_names}")
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    config = ModelConfig(num_classes=num_classes, class_names=class_names)
    inferencer = Inferencer(model_path, config, threshold=threshold)

    image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".tif"))]
    all_ious = []
    all_metrics = []

    for img_name in tqdm(image_files, desc="Evaluating"):
        img_path = os.path.join(image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.tif"
        mask_path = os.path.join(mask_dir, mask_name)

        img = cv2.imread(img_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt_mask is None:
            print(f"Skipping {img_name}: missing image or mask.")
            continue

        pred_mask, ious, metrics = inferencer.predict_and_compare(img, gt_mask)
        all_ious.append(ious)
        all_metrics.append(metrics)

    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)
    
    print("\n--- Evaluation Report ---")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    for i, miou in enumerate(mean_ious):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{class_name}: Mean IoU = {miou:.4f}")
    
    print(f"Mean IoU (all classes): {np.nanmean(mean_ious):.4f}")
    
    # Calculate average object-wise metrics
    if all_metrics:
        print("\n--- Object-wise Metrics ---")
        avg_metrics = {}
        for class_name in all_metrics[0].keys():
            avg_metrics[class_name] = {
                'precision': np.mean([m[class_name]['precision'] for m in all_metrics]),
                'recall': np.mean([m[class_name]['recall'] for m in all_metrics]),
                'f1': np.mean([m[class_name]['f1'] for m in all_metrics])
            }
            print(f"{class_name}:")
            print(f"  Precision: {avg_metrics[class_name]['precision']:.4f}")
            print(f"  Recall: {avg_metrics[class_name]['recall']:.4f}")
            print(f"  F1-Score: {avg_metrics[class_name]['f1']:.4f}")

if __name__ == "__main__":
    # Example usage for Label Studio annotations
    batch_evaluate_with_labelstudio(
        image_dir="images/",
        annotation_dir="masks/",
        model_path="models/checkpoints/final_model.pth",
        bucket_name="segmentation-platform",
        num_classes=None,  # Will be auto-detected from class_config.json
        threshold=0.3
    )
