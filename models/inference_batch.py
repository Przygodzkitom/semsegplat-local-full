import os
import cv2
import numpy as np
from tqdm import tqdm
from models.inference_single import Inferencer
from models.config import ModelConfig
from label_studio_sdk.converter.brush import decode_rle
import json

def _rle_to_mask(rle_data, height, width):
    """Decode Label Studio brush RLE to a binary mask.

    Uses the canonical decoder from label_studio_sdk (same as training).
    The RLE is a @thi.ng/rle-pack bitstream that decodes to a flat RGBA array.
    """
    try:
        flat = decode_rle(rle_data)
        image = np.reshape(flat, [height, width, 4])
        mask = (image[:, :, 3] > 0).astype(np.uint8)
        return mask
    except Exception as e:
        print(f"Error in RLE conversion: {e}")
        return np.zeros((height, width), dtype=np.uint8)

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

def _polygon_to_mask(polygon_points, height, width):
    """Convert polygon points to binary mask"""
    from PIL import Image, ImageDraw
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    if len(polygon_points) >= 3:
        draw.polygon(polygon_points, fill=1)
    
    return np.array(img)

def parse_labelstudio_annotation(annotation_data, class_names):
    """Parse Label Studio annotation and convert to mask at original image dimensions."""
    try:
        result = annotation_data.get('result', [])
        if not result:
            return None

        # Get image dimensions from the first annotation
        first_item = result[0]
        target_width = first_item.get('original_width', 512)
        target_height = first_item.get('original_height', 512)

        mask = np.zeros((target_height, target_width), dtype=np.uint8)

        for item in result:
            if item.get('type') == 'polygonlabels':
                value = item.get('value', {})
                points = value.get('points', [])

                if points:
                    # Convert percentage coordinates to pixel coordinates
                    polygon_points = []
                    for point in points:
                        if len(point) == 2:
                            x = int(point[0] * target_width / 100)
                            y = int(point[1] * target_height / 100)
                            polygon_points.append((x, y))

                    labels = value.get('polygonlabels', [])
                    if labels:
                        class_label = labels[0]
                        if class_label in class_names:
                            class_id = class_names.index(class_label)
                        elif len(class_names) > 1:
                            class_id = 1
                        else:
                            class_id = 1

                        polygon_mask = _polygon_to_mask(polygon_points, target_height, target_width)

                        if class_id > 0:
                            polygon_class_mask = polygon_mask * class_id
                            mask = np.maximum(mask, polygon_class_mask)
                        else:
                            mask = np.maximum(mask, polygon_mask)

            elif item.get('type') == 'brushlabels':
                value = item.get('value', {})
                brush_data = value.get('rle', [])
                labels = value.get('brushlabels', [])

                if brush_data and labels:
                    rle_mask = _rle_to_mask(brush_data, target_height, target_width)

                    class_label = labels[0]
                    if class_label in class_names:
                        class_id = class_names.index(class_label)
                    elif len(class_names) > 1:
                        class_id = 1
                    else:
                        class_id = 1

                    if class_id > 0:
                        mask[rle_mask == 1] = class_id
                    else:
                        mask = np.maximum(mask, rle_mask)

        return mask

    except Exception as e:
        print(f"Error parsing annotation: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_evaluate_with_minio_annotations(bucket_name="segmentation-platform", model_path=None, num_classes=None, threshold=0.3):
    """Batch evaluate using annotations directly from Label Studio database"""
    
    # Load model configuration to get both encoder and class information
    config_file = model_path.replace('.pth', '_config.json')
    class_names = None
    num_classes = None
    
    # Try to read from model's config file first
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            # Get class information from model config
            if 'class_names' in model_config:
                class_names = model_config['class_names']
                num_classes = len(class_names)
                print(f"✅ Loaded classes from model config: {class_names}")
            else:
                print(f"⚠️ No class_names in model config")
            
            # Get encoder information from model config
            if 'encoder_name' in model_config:
                encoder_name = model_config['encoder_name']
                print(f"✅ Loaded encoder from model config: {encoder_name}")
            else:
                print(f"⚠️ No encoder_name in model config")
                
        except Exception as e:
            print(f"⚠️ Error reading model config: {e}")
    
    # Fallback to default class configuration if not found in model config
    if class_names is None:
        if num_classes is None:
            class_names = load_class_configuration()
            num_classes = len(class_names)
            print(f"⚠️ Using fallback class configuration: {class_names}")
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]

    
    # Create config object
    config = ModelConfig(num_classes=num_classes, class_names=class_names)
    
    # Set encoder name if we found it
    if 'encoder_name' in locals() and encoder_name:
        config.encoder_name = encoder_name
        print(f"🔑 DEBUG: Set config.encoder_name to: {encoder_name}")
    else:
        print(f"⚠️ DEBUG: encoder_name not found or empty, using default: {config.encoder_name}")
    
    debug_info = []
    debug_info.append(f"🔍 DEBUG: Creating inferencer with model_path: {model_path}")
    debug_info.append(f"🔍 DEBUG: Model config class_names: {class_names}")
    debug_info.append(f"🔍 DEBUG: Encoder name: {config.encoder_name}")
    debug_info.append(f"🔍 DEBUG: Threshold: {threshold}")
    print(f"🔍 DEBUG: Creating inferencer with model_path: {model_path}")
    print(f"🔍 DEBUG: Model config class_names: {class_names}")
    print(f"🔍 DEBUG: Threshold: {threshold}")
    inferencer = Inferencer(model_path, config, threshold=threshold)
    debug_info.append(f"🔍 DEBUG: Inferencer created successfully")
    print(f"🔍 DEBUG: Inferencer created successfully")
    
    # Read annotations directly from Label Studio database
    print(f"🔍 DEBUG: Reading annotations from Label Studio database")
    debug_info.append(f"🔍 DEBUG: Reading annotations from Label Studio database")
    
    try:
        import sqlite3
        
        # Path to Label Studio database
        db_path = "label-studio-data/label_studio.sqlite3"
        
        if not os.path.exists(db_path):
            print(f"❌ Label Studio database not found: {db_path}")
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': f'Label Studio database not found: {db_path}',
                'debug_info': debug_info
            }
        
        # Connect to Label Studio database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to get tasks with completions
        query = """
        SELECT 
            t.id as task_id,
            t.data as task_data,
            tc.id as completion_id,
            tc.result as completion_result
        FROM task t
        LEFT JOIN task_completion tc ON t.id = tc.task_id
        WHERE tc.result IS NOT NULL
        ORDER BY t.id, tc.id
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ No annotations found in Label Studio database")
            conn.close()
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': 'No annotations found in Label Studio database',
                'debug_info': debug_info
            }
        
        print(f"✅ Found {len(rows)} annotation records in database")
        debug_info.append(f"✅ Found {len(rows)} annotation records in database")
        
        # Group by task_id
        tasks = {}
        for row in rows:
            task_id, task_data, completion_id, completion_result = row
            
            if task_id not in tasks:
                tasks[task_id] = {
                    'id': task_id,
                    'data': json.loads(task_data) if task_data else {},
                    'annotations': []
                }
            
            if completion_result:
                tasks[task_id]['annotations'].append({
                    'id': completion_id,
                    'result': json.loads(completion_result),
                    'created_at': None  # Not available in this query
                })
        
        # Convert to Label Studio export format
        export_data = []
        for task in tasks.values():
            if task['annotations']:  # Only include tasks with annotations
                export_data.append(task)
        
        if not export_data:
            print("❌ No valid annotations found")
            conn.close()
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': 'No valid annotations found',
                'debug_info': debug_info
            }
        
        print(f"✅ Successfully parsed {len(export_data)} image-annotation pairs from database")
        debug_info.append(f"✅ Successfully parsed {len(export_data)} image-annotation pairs from database")
        
        conn.close()
        
        # Load test split to restrict evaluation to held-out images only
        test_split_file = "/app/models/checkpoints/test_split.json"
        if os.path.exists(test_split_file):
            with open(test_split_file) as f:
                split_data = json.load(f)
            test_keys = set(split_data.get('test_image_keys', []))
            n_total = split_data.get('total_annotated', '?')
            print(f"✅ Test split found: evaluating on {len(test_keys)}/{n_total} held-out images")
            debug_info.append(f"Test split: {len(test_keys)}/{n_total} images reserved for evaluation")
        else:
            test_keys = None
            print("⚠️ No test split file found — evaluating on ALL annotated images (train data included!)")
            debug_info.append("WARNING: No test split found — evaluation includes training images")

        # Now process the export data and load images
        image_annotation_pairs = []

        # Initialize MinIO client for image access
        import boto3
        from botocore.exceptions import ClientError
        
        # MinIO configuration (same as training scripts)
        endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        
        for task_data in export_data:
            try:
                # Extract image path from task data
                data = task_data.get('data', {})
                image_path = data.get('image', '') or data.get('$undefined$', '')
                
                if not image_path:
                    print(f"⚠️ No image path in task {task_data.get('id', 'unknown')}")
                    continue
                
                # Convert S3 URL to object key
                if image_path.startswith('s3://'):
                    image_key = image_path.replace(f's3://{bucket_name}/', '')
                else:
                    image_key = image_path
                
                # Check if image exists in MinIO
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=image_key)
                except:
                    print(f"⚠️ Image not found in MinIO: {image_key}")
                    continue
                
                # Load image from MinIO
                response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
                image_data = response['Body'].read()
                img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    print(f"⚠️ Failed to load image: {image_key}")
                    continue
                
                # Parse the annotation data
                # Get annotations from task_data
                annotations = task_data.get('annotations', [])
                if not annotations:
                    print(f"⚠️ No annotations for task {task_data.get('id', 'unknown')}")
                    continue
                
                # Get the result from the first annotation
                annotation = annotations[0]
                result = annotation.get('result', [])
                # Create annotation data in the format expected by parse_labelstudio_annotation
                annotation_data = {
                    'result': result
                }
                # Skip images not in the test split
                if test_keys is not None and image_key not in test_keys:
                    continue

                gt_mask = parse_labelstudio_annotation(annotation_data, class_names)
                if gt_mask is not None:
                    # Return format: (image_path, gt_mask, image)
                    image_annotation_pairs.append((image_key, gt_mask, img))
                    print(f"✅ Processed: {os.path.basename(image_key)}")
                else:
                    print(f"⚠️ Failed to parse annotation for task {task_data.get('id', 'unknown')}")
                    
            except Exception as e:
                print(f"⚠️ Error processing task {task_data.get('id', 'unknown')}: {e}")
                debug_info.append(f"⚠️ Error processing task {task_data.get('id', 'unknown')}: {e}")
                continue
        
        if not image_annotation_pairs:
            print(f"❌ No valid image-annotation pairs found")
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': 'No valid image-annotation pairs found',
                'debug_info': debug_info
            }
        
        print(f"✅ Successfully processed {len(image_annotation_pairs)} image-annotation pairs")
        debug_info.append(f"✅ Successfully processed {len(image_annotation_pairs)} image-annotation pairs")
        
        # Now run the actual evaluation
        all_ious = []
        all_metrics = []
        
        for i, (image_path, gt_mask, image) in enumerate(image_annotation_pairs):
            try:
                print(f"Processing image {i+1}/{len(image_annotation_pairs)}: {os.path.basename(image_path)}")
                
                # Run inference using predict_and_compare (like the old working version)
                pred_mask, ious, metrics = inferencer.predict_and_compare(image, gt_mask, iou_threshold=0.1)
                
                print(f"  ✅ Inference complete: IoU={ious}")
                
                all_ious.append(ious)
                all_metrics.append(metrics)
                    
            except Exception as e:
                print(f"⚠️ Error processing image {os.path.basename(image_path)}: {e}")
                debug_info.append(f"⚠️ Error processing image {os.path.basename(image_path)}: {e}")
                continue
        
        # Calculate final results
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
            
            # Calculate object-wise metrics (micro-averaged: sum TP/FP/FN across images)
            if all_metrics:
                print("\n--- Object-wise Metrics ---")
                avg_metrics = {}
                for class_name in all_metrics[0].keys():
                    total_tp = sum(m[class_name]['tp'] for m in all_metrics)
                    total_fp = sum(m[class_name]['fp'] for m in all_metrics)
                    total_fn = sum(m[class_name]['fn'] for m in all_metrics)
                    total_pred = sum(m[class_name].get('n_pred_objects', 0) for m in all_metrics)
                    total_gt = sum(m[class_name].get('n_gt_objects', 0) for m in all_metrics)

                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                    avg_metrics[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    }
                    print(f"{class_name}: ({total_pred} predicted, {total_gt} GT objects)")
                    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall: {recall:.4f}")
                    print(f"  F1-Score: {f1:.4f}")

                results['avg_metrics'] = avg_metrics
        else:
            print("No valid image-annotation pairs found for evaluation.")
            results['status'] = 'no_data'

        # Add debug info to results
        results['debug_info'] = debug_info
        
        return results
        
    except Exception as e:
        print(f"❌ Error reading from database: {e}")
        debug_info.append(f"❌ Error reading from database: {e}")
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'error',
            'error': f'Database read error: {e}',
            'debug_info': debug_info
        }

def batch_evaluate_with_labelstudio_export(export_file_path, model_path, bucket_name="segmentation-platform", num_classes=None, threshold=0.3):
    """Evaluate model using Label Studio annotations from export file or MinIO storage"""
    
    # Check if we should use MinIO annotations directly
    if export_file_path == "minio://annotations":
        print("🔍 MinIO annotations requested but not available in correct format")
        print("🔍 Please use the export file creation script to generate proper export files")
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'no_data',
            'error': 'MinIO annotations not in correct format. Please create export file first.',
            'debug_info': ['MinIO annotations requested but not available in correct format']
        }
    
    # Load model configuration to get both encoder and class information
    config_file = model_path.replace('.pth', '_config.json')
    class_names = None
    num_classes = None
    
    # Try to read from model's config file first
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            # Get class information from model config
            if 'class_names' in model_config:
                class_names = model_config['class_names']
                num_classes = len(class_names)
                print(f"✅ Loaded classes from model config: {class_names}")
            else:
                print(f"⚠️ No class_names in model config")
            
            # Get encoder information from model config
            if 'encoder_name' in model_config:
                encoder_name = model_config['encoder_name']
                print(f"✅ Loaded encoder from model config: {encoder_name}")
            else:
                print(f"⚠️ No encoder_name in model config")
                
        except Exception as e:
            print(f"⚠️ Error reading model config: {e}")
    
    # Fallback to default class configuration if not found in model config
    if class_names is None:
        if num_classes is None:
            class_names = load_class_configuration()
            num_classes = len(class_names)
            print(f"⚠️ Using fallback class configuration: {class_names}")
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]
            print(f"⚠️ Using generic class names: {class_names}")
    
    # Create config object
    config = ModelConfig(num_classes=num_classes, class_names=class_names)
    
    # Set encoder name if we found it
    if 'encoder_name' in locals() and encoder_name:
        config.encoder_name = encoder_name
        print(f"🔑 DEBUG: Set config.encoder_name to: {encoder_name}")
    else:
        print(f"⚠️ DEBUG: encoder_name not found or empty, using default: {config.encoder_name}")
    
    debug_info = []
    debug_info.append(f"🔍 DEBUG: Creating inferencer with model_path: {model_path}")
    debug_info.append(f"🔍 DEBUG: Model config class_names: {class_names}")
    debug_info.append(f"🔍 DEBUG: Encoder name: {config.encoder_name}")
    debug_info.append(f"🔍 DEBUG: Threshold: {threshold}")
    print(f"🔍 DEBUG: Creating inferencer with model_path: {model_path}")
    print(f"🔍 DEBUG: Model config class_names: {class_names}")
    print(f"🔍 DEBUG: Threshold: {threshold}")
    inferencer = Inferencer(model_path, config, threshold=threshold)
    debug_info.append(f"🔍 DEBUG: Inferencer created successfully")
    print(f"🔍 DEBUG: Inferencer created successfully")
    
    # Load Label Studio export file
    print(f"🔍 DEBUG: About to load Label Studio export file: {export_file_path}")
    print(f"🔍 Loading Label Studio export file: {export_file_path}")
    if not os.path.exists(export_file_path):
        print(f"❌ Export file not found: {export_file_path}")
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'no_data',
            'error': f'Export file not found: {export_file_path}'
        }
    
    try:
        with open(export_file_path, 'r') as f:
            export_data = json.load(f)
        print(f"✅ Loaded export file with {len(export_data)} tasks")
        print(f"🔍 DEBUG: Export file loaded successfully, processing tasks...")
    except Exception as e:
        print(f"❌ Error loading export file: {e}")
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'no_data',
            'error': f'Error loading export file: {e}'
        }

    # Initialize MinIO client for image access
    import boto3
    from botocore.exceptions import ClientError
    
    # MinIO configuration (same as training scripts)
    endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )

    # Process export data to create image-annotation pairs
    image_annotation_pairs = []
    
    print(f"🔍 Processing {len(export_data)} tasks from export file...")
    
    for task_data in export_data:
        try:
            # Extract image path from task data
            # Try both 'image' and '$undefined$' keys (Label Studio can use either)
            data = task_data.get('data', {})
            image_path = data.get('image', '') or data.get('$undefined$', '')
            if not image_path:
                print(f"⚠️ No image path in task {task_data.get('id', 'unknown')}")
                print(f"   Available data keys: {list(data.keys())}")
                continue
            
            # Convert S3 URL to object key
            if image_path.startswith('s3://'):
                # Remove s3://bucket-name/ prefix
                image_key = image_path.replace(f's3://{bucket_name}/', '')
            else:
                image_key = image_path
            
            # Check if image exists in MinIO
            try:
                s3_client.head_object(Bucket=bucket_name, Key=image_key)
            except:
                print(f"⚠️ Image not found in MinIO: {image_key}")
                continue
            
            # Get annotations for this task
            annotations = task_data.get('annotations', [])
            if not annotations:
                print(f"⚠️ No annotations for task {task_data.get('id', 'unknown')}")
                continue
            
            # Use the first annotation (assuming there's only one per task)
            annotation = annotations[0]
            result = annotation.get('result', [])
            
            if not result:
                print(f"⚠️ No annotation result for task {task_data.get('id', 'unknown')}")
                continue
            
            # Create annotation data in the format expected by parse_labelstudio_annotation
            annotation_data = {
                'result': result
            }
            
            image_annotation_pairs.append((image_key, annotation_data))
            print(f"✅ Matched: {os.path.basename(image_key)} -> task {task_data.get('id', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error processing task {task_data.get('id', 'unknown')}: {e}")
            continue
    
    print(f"Found {len(image_annotation_pairs)} image-annotation pairs for evaluation")
    
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

    for img_key, annotation_data in tqdm(image_annotation_pairs, desc="Evaluating"):
        try:
            # Load image from MinIO
            response = s3_client.get_object(Bucket=bucket_name, Key=img_key)
            image_data = response['Body'].read()
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            gt_mask = parse_labelstudio_annotation(annotation_data, class_names)

            if img is None or gt_mask is None:
                print(f"Skipping {img_key}: missing image or annotation.")
                continue

            pred_mask, ious, metrics = inferencer.predict_and_compare(img, gt_mask, iou_threshold=0.1)
            
            all_ious.append(ious)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"❌ Error evaluating {img_key}: {e}")
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
        
        # Calculate object-wise metrics (micro-averaged: sum TP/FP/FN across images)
        if all_metrics:
            print("\n--- Object-wise Metrics ---")
            avg_metrics = {}
            for class_name in all_metrics[0].keys():
                total_tp = sum(m[class_name]['tp'] for m in all_metrics)
                total_fp = sum(m[class_name]['fp'] for m in all_metrics)
                total_fn = sum(m[class_name]['fn'] for m in all_metrics)
                total_pred = sum(m[class_name].get('n_pred_objects', 0) for m in all_metrics)
                total_gt = sum(m[class_name].get('n_gt_objects', 0) for m in all_metrics)

                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                avg_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
                print(f"{class_name}: ({total_pred} predicted, {total_gt} GT objects)")
                print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")

            results['avg_metrics'] = avg_metrics
    else:
        print("No valid image-annotation pairs found for evaluation.")
        results['status'] = 'no_data'

    # Add debug info to results
    results['debug_info'] = debug_info
    
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

        pred_mask, ious, metrics = inferencer.predict_and_compare(img, gt_mask, iou_threshold=0.1)
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
    
    # Calculate object-wise metrics (micro-averaged: sum TP/FP/FN across images)
    if all_metrics:
        print("\n--- Object-wise Metrics ---")
        avg_metrics = {}
        for class_name in all_metrics[0].keys():
            total_tp = sum(m[class_name]['tp'] for m in all_metrics)
            total_fp = sum(m[class_name]['fp'] for m in all_metrics)
            total_fn = sum(m[class_name]['fn'] for m in all_metrics)
            total_pred = sum(m[class_name].get('n_pred_objects', 0) for m in all_metrics)
            total_gt = sum(m[class_name].get('n_gt_objects', 0) for m in all_metrics)

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            avg_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
            print(f"{class_name}: ({total_pred} predicted, {total_gt} GT objects)")
            print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")

if __name__ == "__main__":
    # Example usage for Label Studio annotations
    # Note: This will only work if export files exist in label-studio-data/export/
    # Use the Streamlit interface to run batch evaluation with proper export file detection
    
    # Check if export files exist
    export_dir = "label-studio-data/export/"
    if os.path.exists(export_dir):
        export_files = [f for f in os.listdir(export_dir) if f.endswith('.json') and 'project-' in f]
        if export_files:
            # Use the most recent export file
            export_files.sort(reverse=True)
            export_file = os.path.join(export_dir, export_files[0])
            print(f"Using export file: {export_file}")
            
            batch_evaluate_with_labelstudio_export(
                export_file_path=export_file,
                model_path="models/checkpoints/final_model_polygon_20250910_110111.pth",
                bucket_name="segmentation-platform",
                num_classes=None,  # Will be auto-detected from class_config.json
                threshold=0.3
            )
        else:
            print("❌ No Label Studio export files found in label-studio-data/export/")
            print("Please export your Label Studio project first to generate JSON files.")
    else:
        print("❌ Label Studio export directory not found: label-studio-data/export/")
        print("Please ensure Label Studio is properly configured and export files are generated.")
