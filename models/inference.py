import os
import cv2
import numpy as np
from tqdm import tqdm
from models.inferencer import Inferencer
from models.config import ModelConfig
import json

def _simple_rle_to_mask(rle_data, height, width):
    """Convert RLE (Run Length Encoding) to binary mask - same logic as training script"""
    try:
        # LabelStudio uses a different RLE format - it's more like a compressed bitmap
        # The format appears to be: [value, count, value, count, ...] where value is 0 or 1
        mask = np.zeros(height * width, dtype=np.uint8)
        
        if isinstance(rle_data, list) and len(rle_data) > 0:
            # Try different RLE formats
            if len(rle_data) % 2 == 0:
                # Standard RLE format [value, count, value, count, ...]
                pos = 0
                for i in range(0, len(rle_data), 2):
                    value = rle_data[i]
                    count = rle_data[i + 1]
                    
                    if pos + count <= len(mask):
                        mask[pos:pos + count] = value
                        pos += count
                    else:
                        break
            else:
                # Odd-length format - might be a different compression
                # Try to interpret as [start, length, start, length, ...] but handle odd length
                pos = 0
                for i in range(0, len(rle_data) - 1, 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    
                    if start + length <= len(mask):
                        mask[start:start + length] = 1
                    else:
                        break
        
        return mask.reshape(height, width)
        
    except Exception as e:
        print(f"Error converting RLE to mask: {e}")
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
    """Parse Label Studio annotation and convert to mask - using 512x512 like training"""
    try:
        print(f"  🔍 DEBUG: parse_labelstudio_annotation called with class_names: {class_names}")
        result = annotation_data.get('result', [])
        if not result:
            print("  ❌ No result in annotation data")
            return None
        
        print(f"  🔍 DEBUG: Found {len(result)} annotation items")
        
        # Get image dimensions from the first annotation
        first_item = result[0]
        original_width = first_item.get('original_width', 512)
        original_height = first_item.get('original_height', 512)
        
        print(f"  🔍 DEBUG: First annotation item keys: {list(first_item.keys())}")
        print(f"  🔍 DEBUG: original_width from annotation: {original_width}")
        print(f"  🔍 DEBUG: original_height from annotation: {original_height}")
        
        # Use original dimensions to match model output size (no resizing needed)
        target_width = original_width
        target_height = original_height
        
        print(f"  📐 Original image dimensions: {original_width}x{original_height}")
        print(f"  🎯 Target mask dimensions: {target_width}x{target_height}")
        
        # Create empty mask at original dimensions to match model output
        mask = np.zeros((target_height, target_width), dtype=np.uint8)
        print(f"  🔍 DEBUG: Created empty mask with shape: {mask.shape}")
        
        # Process polygon annotations
        for i, item in enumerate(result):
            print(f"  🔍 DEBUG: Processing item {i}: type={item.get('type')}")
            if item.get('type') == 'polygonlabels':
                value = item.get('value', {})
                points = value.get('points', [])
                
                print(f"  🔍 DEBUG: Found {len(points)} polygon points")
                print(f"  🔍 DEBUG: First few points: {points[:3] if len(points) >= 3 else points}")
                
                if points:
                    # Convert points to polygon coordinates at target dimensions
                    polygon_points = []
                    for j, point in enumerate(points):
                        if len(point) == 2:
                            # Convert percentage to pixels at target dimensions
                            x_percent = point[0]
                            y_percent = point[1]
                            x = int(x_percent * target_width / 100)
                            y = int(y_percent * target_height / 100)
                            polygon_points.append((x, y))
                            
                            if j < 3:  # Debug first few points
                                print(f"  🔍 DEBUG: Point {j}: {x_percent}%, {y_percent}% -> ({x}, {y})")
                    
                    print(f"  🔍 DEBUG: Converted to {len(polygon_points)} polygon points")
                    
                    # Get class label
                    labels = value.get('polygonlabels', [])
                    print(f"  🔍 DEBUG: Labels found: {labels}")
                    print(f"  🔍 DEBUG: Raw annotation value: {value}")
                    if labels:
                        class_label = labels[0]
                        # More flexible class name matching
                        if class_label in class_names:
                            class_id = class_names.index(class_label)
                        elif len(class_names) > 1:  # If we have object classes
                            class_id = 1  # Use first object class (skip background)
                        else:
                            class_id = 1
                        
                        print(f"  🏷️ Processing class: {class_label} (ID: {class_id})")
                        print(f"  🔍 DEBUG: Available class_names: {class_names}")
                        print(f"  🔍 DEBUG: Class name matching: '{class_label}' -> class_id {class_id}")
                        print(f"  🔍 DEBUG: Class ID {class_id} corresponds to: '{class_names[class_id] if class_id < len(class_names) else 'INVALID'}'")
                        
                        # Convert polygon to mask at original size
                        polygon_mask = _polygon_to_mask(polygon_points, target_height, target_width)
                        print(f"  🔍 DEBUG: Polygon mask shape: {polygon_mask.shape}, unique values: {np.unique(polygon_mask)}")
                        
                        # Add to main mask with correct class assignment
                        if class_id > 0:
                            # Create a mask with the correct class_id for this polygon
                            polygon_class_mask = polygon_mask * class_id
                            # Use maximum to combine with existing mask (preserves multiple polygons)
                            mask = np.maximum(mask, polygon_class_mask)
                        else:
                            # If class_id is 0 (background), just add the mask as-is
                            mask = np.maximum(mask, polygon_mask)
                        
                        print(f"  🔍 DEBUG: After adding polygon, mask unique values: {np.unique(mask)}")
            
            elif item.get('type') == 'brushlabels':
                # Handle brush annotations if they exist
                value = item.get('value', {})
                brush_data = value.get('rle', [])
                
                if brush_data:
                    # Use the same improved RLE parsing as the training script
                    rle_mask = _simple_rle_to_mask(brush_data, target_height, target_width)
                    
                    # Get class label
                    labels = value.get('brushlabels', [])
                    if labels:
                        class_label = labels[0]
                        # More flexible class name matching
                        if class_label in class_names:
                            class_id = class_names.index(class_label)
                        elif len(class_names) > 1:  # If we have object classes
                            class_id = 1  # Use first object class (skip background)
                        else:
                            class_id = 1
                        
                        print(f"  🖌️ Processing brush class: {class_label} (ID: {class_id})")
                        print(f"  🔍 DEBUG: Class name matching: '{class_label}' -> class_id {class_id}")
                        print(f"  🔍 DEBUG: Class ID {class_id} corresponds to: '{class_names[class_id] if class_id < len(class_names) else 'INVALID'}'")
                        
                        # Add to main mask with correct class assignment
                        if class_id > 0:
                            # Assign class_id to the brush area
                            mask[rle_mask == 1] = class_id
                        else:
                            # If class_id is 0 (background), just add the mask as-is
                            mask = np.maximum(mask, rle_mask)
        
        print(f"  🎯 Final mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        
        # Debug: Show mask statistics
        if len(np.unique(mask)) > 1:
            unique_vals, counts = np.unique(mask, return_counts=True)
            print(f"  🔍 DEBUG: Mask value distribution: {dict(zip(unique_vals, counts))}")
            print(f"  🔍 DEBUG: Mask percentage of class 1: {np.sum(mask == 1) / mask.size * 100:.2f}%")
        
        return mask
        
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_evaluate_with_minio_annotations(bucket_name="segmentation-platform", model_path=None, num_classes=None, threshold=0.3):
    """Batch evaluate using annotations directly from MinIO export storage"""
    
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
    
    # Read annotations directly from MinIO
    print(f"🔍 DEBUG: Reading annotations from MinIO bucket: {bucket_name}")
    debug_info.append(f"🔍 DEBUG: Reading annotations from MinIO bucket: {bucket_name}")
    
    try:
        # Import MinIO client
        from minio import Minio
        from minio.error import S3Error
        
        # Initialize MinIO client
        minio_client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False
        )
        
        # List annotation files in the bucket
        annotation_objects = []
        try:
            for obj in minio_client.list_objects(bucket_name, prefix="annotations/", recursive=True):
                if obj.object_name.endswith('.json') or not obj.object_name.endswith('/'):
                    annotation_objects.append(obj.object_name)
        except S3Error as e:
            print(f"❌ Error listing MinIO objects: {e}")
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': f'MinIO error: {e}',
                'debug_info': debug_info
            }
        
        if not annotation_objects:
            print(f"❌ No annotation files found in MinIO bucket {bucket_name}")
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': f'No annotation files found in MinIO bucket {bucket_name}',
                'debug_info': debug_info
            }
        
        print(f"✅ Found {len(annotation_objects)} annotation files in MinIO")
        debug_info.append(f"✅ Found {len(annotation_objects)} annotation files in MinIO")
        
        # Process each annotation file
        image_annotation_pairs = []
        for obj_name in annotation_objects:
            try:
                # Download annotation data
                response = minio_client.get_object(bucket_name, obj_name)
                annotation_data = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                # Parse the annotation data
                parsed_data = parse_labelstudio_annotation(annotation_data, class_names)
                if parsed_data:
                    image_annotation_pairs.append(parsed_data)
                    
            except Exception as e:
                print(f"⚠️ Error processing annotation {obj_name}: {e}")
                debug_info.append(f"⚠️ Error processing annotation {obj_name}: {e}")
                continue
        
        if not image_annotation_pairs:
            print(f"❌ No valid annotations found in MinIO")
            return {
                'num_classes': num_classes,
                'class_names': class_names,
                'images_evaluated': 0,
                'mean_ious': [],
                'overall_mean_iou': 0.0,
                'avg_metrics': {},
                'status': 'no_data',
                'error': 'No valid annotations found in MinIO',
                'debug_info': debug_info
            }
        
        print(f"✅ Successfully parsed {len(image_annotation_pairs)} image-annotation pairs")
        debug_info.append(f"✅ Successfully parsed {len(image_annotation_pairs)} image-annotation pairs")
        
        # Continue with the rest of the evaluation logic...
        # (The rest of the function would be similar to the original batch_evaluate_with_labelstudio_export)
        
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': len(image_annotation_pairs),
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'success',
            'debug_info': debug_info
        }
        
    except Exception as e:
        print(f"❌ Error reading from MinIO: {e}")
        debug_info.append(f"❌ Error reading from MinIO: {e}")
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'images_evaluated': 0,
            'mean_ious': [],
            'overall_mean_iou': 0.0,
            'avg_metrics': {},
            'status': 'error',
            'error': f'MinIO read error: {e}',
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

    print(f"🔍 DEBUG: About to process {len(image_annotation_pairs)} image-annotation pairs")
    print(f"🔍 DEBUG: First few pairs: {image_annotation_pairs[:2] if image_annotation_pairs else 'No pairs'}")

    for img_key, annotation_data in tqdm(image_annotation_pairs, desc="Evaluating"):
        try:
            print(f"🔄 Processing: {os.path.basename(img_key)}")
            print(f"  🔍 DEBUG: Full image key: {img_key}")
            
            # Load image from MinIO
            print(f"  🔍 DEBUG: Loading image from MinIO: {img_key}")
            response = s3_client.get_object(Bucket=bucket_name, Key=img_key)
            image_data = response['Body'].read()
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"  📸 Image loaded: {img.shape}")
            print(f"  🔍 DEBUG: Image dtype: {img.dtype}, min/max values: {img.min()}/{img.max()}")
            print(f"  🔍 DEBUG: Image filename: {os.path.basename(img_key)}")
            
            # Parse annotation data (already loaded from export file)
            print(f"  🔍 DEBUG: About to parse annotation data...")
            print(f"  🔍 DEBUG: class_names being passed to parse_labelstudio_annotation: {class_names}")
            gt_mask = parse_labelstudio_annotation(annotation_data, class_names)
            print(f"  🔍 DEBUG: Annotation parsing complete, gt_mask: {gt_mask.shape if gt_mask is not None else 'None'}")
            
            if img is None or gt_mask is None:
                print(f"❌ Skipping {img_key}: missing image or annotation.")
                continue

            print(f"  🤖 Running inference...")
            print(f"  🔍 About to call predict_and_compare with img shape: {img.shape}, gt_mask shape: {gt_mask.shape}")
            print(f"  🔍 DEBUG: Image dimensions: {img.shape[1]}x{img.shape[0]} (WxH)")
            print(f"  🔍 DEBUG: GT mask dimensions: {gt_mask.shape[1]}x{gt_mask.shape[0]} (WxH)")
            print(f"  🔍 DEBUG: GT mask unique values: {np.unique(gt_mask)}")
            
            pred_mask, ious, metrics = inferencer.predict_and_compare(img, gt_mask)
            print(f"  ✅ Inference complete: IoU={ious}")
            print(f"  🔍 Returned pred_mask shape: {pred_mask.shape}, ious: {ious}, metrics keys: {list(metrics.keys()) if metrics else 'None'}")
            print(f"  🔍 DEBUG: Final pred_mask unique values: {np.unique(pred_mask)}")
            
            # Print individual image results
            print(f"  📊 Individual Results for {os.path.basename(img_key)}:")
            for i, iou in enumerate(ious):
                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                print(f"    {class_name}: IoU = {iou:.4f}")
            
            if metrics:
                for class_name, class_metrics in metrics.items():
                    print(f"    {class_name}: Precision={class_metrics.get('precision', 0):.4f}, Recall={class_metrics.get('recall', 0):.4f}, F1={class_metrics.get('f1', 0):.4f}")
            
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
