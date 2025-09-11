# Batch Evaluation System Documentation

## Overview

The batch evaluation system (`models/inference.py`) performs automated evaluation of a trained segmentation model against annotated data from Label Studio. It calculates IoU (Intersection over Union) metrics and object-wise precision/recall/F1 scores for each class.

## System Architecture

### File Structure and Storage Locations

| Component | Location | Storage Type | Access Method | Purpose |
|-----------|----------|--------------|---------------|---------|
| **Model Files** | `/app/models/checkpoints/` | Mounted Volume | Direct file system | Trained model weights and configuration |
| **Export Files** | `/app/label-studio-data/export/` | Mounted Volume | Direct file system | Label Studio annotation data in JSON format |
| **Images** | `s3://segmentation-platform/images/` | MinIO Object Storage | boto3 S3 client | Original images for evaluation |
| **Database** | `/app/label-studio-data/label_studio.sqlite3` | Mounted Volume | SQLite direct access | Source of annotation data for export generation |
| **Annotations** | MinIO `annotations/` | MinIO Object Storage | boto3 S3 client | Metadata only (not used for evaluation) |

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BATCH EVALUATION FLOW                    │
└─────────────────────────────────────────────────────────────────┘

1. EXPORT FILE LOADING
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ Export File     │───▶│ /app/label-studio-data/export/      │
   │ (JSON)          │    │ project-1-at-2025-09-11-09-09-     │
   │                 │    │ db-export.json                     │
   └─────────────────┘    └─────────────────────────────────────┘
                                    │
                                    ▼
2. MODEL CONFIGURATION LOADING
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ Model Config    │───▶│ /app/models/checkpoints/            │
   │ (JSON)          │    │ final_model_polygon_20250910_       │
   │                 │    │ 110111_config.json                 │
   └─────────────────┘    └─────────────────────────────────────┘
                                    │
                                    ▼
3. MODEL LOADING
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ Model Weights   │───▶│ /app/models/checkpoints/            │
   │ (.pth)          │    │ final_model_polygon_20250910_       │
   │                 │    │ 110111.pth                         │
   └─────────────────┘    └─────────────────────────────────────┘
                                    │
                                    ▼
4. IMAGE LOADING
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ Images          │───▶│ MinIO Object Storage                │
   │ (PNG/JPG)       │    │ s3://segmentation-platform/images/  │
   │                 │    │ (via boto3 S3 client)              │
   └─────────────────┘    └─────────────────────────────────────┘
                                    │
                                    ▼
5. ANNOTATION PROCESSING
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ Polygon Data    │───▶│ From Export File                    │
   │ (JSON)          │    │ (already loaded in step 1)         │
   │                 │    │                                     │
   └─────────────────┘    └─────────────────────────────────────┘
                                    │
                                    ▼
6. MODEL INFERENCE
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ Bitmap Masks    │───▶│ Generated from Model Output         │
   │ (Multi-class)   │    │ (2D numpy arrays with class IDs)   │
   │                 │    │                                     │
   └─────────────────┘    └─────────────────────────────────────┘
                                    │
                                    ▼
7. METRICS CALCULATION
   ┌─────────────────┐    ┌─────────────────────────────────────┐
   │ IoU, Precision, │───▶│ Computed from Bitmap Masks         │
   │ Recall, F1      │    │ (pred_mask vs gt_mask comparison)  │
   │                 │    │                                     │
   └─────────────────┘    └─────────────────────────────────────┘
```

## Detailed Function Logic

### Main Function: `batch_evaluate_with_labelstudio_export()`

**Input Parameters:**
- `export_file_path`: Path to Label Studio export JSON file
- `model_path`: Path to trained model (.pth file)
- `bucket_name`: MinIO bucket name (default: "segmentation-platform")
- `num_classes`: Number of classes (auto-detected from model config)
- `threshold`: Prediction threshold (default: 0.3)

### Step-by-Step Process

#### Step 1: Model Configuration Loading
```python
# Load model config from JSON file
config_file = model_path.replace('.pth', '_config.json')
# Example: models/checkpoints/final_model_polygon_20250910_110111_config.json

# Extract class information
class_names = model_config['class_names']  # ['Background', 'wbc']
num_classes = len(class_names)  # 2

# Extract encoder information
encoder_name = model_config['encoder_name']  # 'resnet50'
```

#### Step 2: Export File Processing
```python
# Load export file
with open(export_file_path, 'r') as f:
    export_data = json.load(f)

# Process each task in export file
for task_data in export_data:
    # Extract image path (supports both 'image' and '$undefined$' keys)
    image_path = data.get('image', '') or data.get('$undefined$', '')
    # Example: "s3://segmentation-platform/images/1757500723276_Snap-5537.png"
    
    # Convert S3 URL to object key
    image_key = image_path.replace(f's3://{bucket_name}/', '')
    # Example: "images/1757500723276_Snap-5537.png"
    
    # Extract annotation data
    annotations = task_data.get('annotations', [])
    annotation = annotations[0]  # Use first annotation
    result = annotation.get('result', [])
```

#### Step 3: Image Loading from MinIO
```python
# Initialize S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin123',
    region_name='us-east-1'
)

# Download image from MinIO
response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
image_data = response['Body'].read()
img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### Step 4: Annotation Parsing
```python
# Parse polygon annotations
def parse_labelstudio_annotation(annotation_data, class_names):
    # Extract image dimensions from annotation
    original_width = first_item.get('original_width', 512)   # 2056
    original_height = first_item.get('original_height', 512) # 2452
    
    # Process polygon points
    for item in result:
        if item.get('type') == 'polygonlabels':
            points = value.get('points', [])
            # Convert percentage coordinates to pixels
            for point in points:
                x = int(point[0] * target_width / 100)
                y = int(point[1] * target_height / 100)
                polygon_points.append((x, y))
            
            # Get class label
            labels = value.get('polygonlabels', [])  # ['wbc']
            class_label = labels[0]
            class_id = class_names.index(class_label)  # 1
            
            # Convert polygon to mask
            polygon_mask = _polygon_to_mask(polygon_points, target_height, target_width)
            mask = np.maximum(mask, polygon_mask * class_id)
```

#### Step 5: Model Inference

**Data Format Transformation in Model Inference:**

The model inference process involves several data format transformations:

1. **Model Output (Raw)**: 
   - Shape: `(2, 512, 512)` - 2 classes at 512x512 resolution
   - Type: `float32` probability values (0.0 to 1.0)
   - Format: Raw logits converted to probabilities via sigmoid

2. **Threshold Application**:
   ```python
   probs = torch.sigmoid(logits)[0].cpu().numpy()  # (2, 512, 512)
   preds = (probs > self.threshold).astype(np.uint8)  # (2, 512, 512)
   ```
   - Shape: `(2, 512, 512)` - 2 binary masks
   - Type: `uint8` with values `[0, 1]`
   - Format: Binary masks per class

3. **Resize to Original Dimensions**:
   ```python
   for i in range(self.num_classes):
       mask = cv2.resize(preds[i], (w, h), interpolation=cv2.INTER_NEAREST)
       masks.append(mask)  # (2452, 2056) for each class
   ```
   - Shape: `(2452, 2056)` - Original image dimensions
   - Type: `uint8` with values `[0, 1]`
   - Format: Binary masks resized to original image size

4. **Combine into Single Multi-Class Mask**:
   ```python
   pred_mask = np.zeros_like(pred_masks[0])  # (2452, 2056)
   for i in range(self.num_classes):
       pred_mask[class_mask == 1] = i  # Assign class IDs
   ```
   - Shape: `(2452, 2056)` - Original image dimensions
   - Type: `uint8` with values `[0, 1]` (class IDs)
   - Format: **Single multi-class bitmap mask**

**Answer: YES, between steps 5 and 6, the segmentation predictions are already in the form of bitmap masks!**

The final `pred_mask` is a single 2D numpy array where each pixel contains the predicted class ID (0 for background, 1 for 'wbc', etc.).

#### Step 6: Metrics Calculation
```python
# Calculate IoU for each class
for i, iou in enumerate(ious):
    class_name = class_names[i]
    print(f"{class_name}: IoU = {iou:.4f}")

# Calculate object-wise metrics
for class_name, class_metrics in metrics.items():
    print(f"{class_name}: Precision={class_metrics['precision']:.4f}")
    print(f"{class_name}: Recall={class_metrics['recall']:.4f}")
    print(f"{class_name}: F1={class_metrics['f1']:.4f}")
```

## Key Data Structures

### Export File Format
```json
[
  {
    "id": 1,
    "data": {
      "$undefined$": "s3://segmentation-platform/images/1757500723276_Snap-5537.png"
    },
    "annotations": [
      {
        "id": 1,
        "result": [
          {
            "original_width": 2056,
            "original_height": 2452,
            "value": {
              "points": [[56.35, 73.97], [59.47, 76.33], ...],
              "polygonlabels": ["wbc"]
            },
            "type": "polygonlabels"
          }
        ]
      }
    ]
  }
]
```

### Model Configuration Format
```json
{
  "class_names": ["Background", "wbc"],
  "num_classes": 2,
  "model_filename": "final_model_polygon_20250910_110111.pth",
  "annotation_type": "polygon",
  "encoder_name": "resnet50"
}
```

### Bitmap Mask Format
- **Shape**: `(height, width)` - e.g., `(2452, 2056)`
- **Type**: `numpy.ndarray` with `dtype=uint8`
- **Values**: Class IDs where `0=background`, `1=wbc`, etc.
- **Example**: A pixel at position `(100, 200)` with value `1` means that pixel belongs to class 'wbc'

## Performance Characteristics

- **Processing Time**: ~4-5 seconds per image
- **Memory Usage**: Loads images at original resolution (2056x2452)
- **Scalability**: Processes images sequentially with progress bar
- **Storage**: Images stored in MinIO, annotations in local JSON files

## Error Handling

The system includes comprehensive error handling for:
- Missing export files
- Invalid model configurations
- MinIO connection failures
- Image loading errors
- Annotation parsing failures
- Model inference errors

## Dependencies

- **OpenCV**: Image processing and loading
- **NumPy**: Array operations and mask processing
- **boto3**: MinIO/S3 object storage access
- **PIL**: Polygon to mask conversion
- **tqdm**: Progress bar display
- **PyTorch**: Model inference (via Inferencer class)

## Usage Example

```python
from models.inference import batch_evaluate_with_labelstudio_export

# Run batch evaluation
results = batch_evaluate_with_labelstudio_export(
    export_file_path="label-studio-data/export/project-1-at-2025-09-11-09-09-db-export.json",
    model_path="models/checkpoints/final_model_polygon_20250910_110111.pth",
    bucket_name="segmentation-platform",
    num_classes=None,  # Auto-detected from model config
    threshold=0.3
)

# Access results
print(f"Status: {results['status']}")
print(f"Images evaluated: {results['images_evaluated']}")
print(f"Overall mean IoU: {results['overall_mean_iou']}")
print(f"Class-wise IoUs: {results['mean_ious']}")
print(f"Object-wise metrics: {results['avg_metrics']}")
```

## File Generation

The export files are generated using the `create_export_corrected.py` script, which:
1. Connects to the Label Studio SQLite database
2. Queries the `task` and `task_completion` tables
3. Extracts annotation data and image paths
4. Generates a JSON file in Label Studio export format
5. Saves the file to `label-studio-data/export/`

This documentation provides a complete technical reference for understanding and maintaining the batch evaluation system.
