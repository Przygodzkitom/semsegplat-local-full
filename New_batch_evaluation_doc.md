# New Batch Evaluation Documentation

## Where the current program logic stores annotations from LS

The current program logic stores Label Studio annotations in the SQLite database located at `label-studio-data/label_studio.sqlite3`. The annotations are stored in two main tables:

- **`task` table**: Contains task data with image references pointing to MinIO storage
- **`task_completion` table**: Contains the actual annotation data including polygon coordinates, labels, and metadata

The annotation data is stored as JSON fields within these database tables, with the `task_completion.result` field containing the complete annotation information in Label Studio's standard format.

## Database to Function Input Conversion Process

The `batch_evaluate_with_minio_annotations` function in `inference_batch.py` converts database data to the format needed by `parse_labelstudio_annotation`:

1. **Query database** to get task and completion data
2. **Group data by task_id** and create proper structure
3. **Convert completion_result** from JSON string to dictionary with `'result'` key:
   ```python
   tasks[task_id]['annotations'].append({
       'id': completion_id,
       'result': json.loads(completion_result),  # Convert JSON string to dict
       'created_at': None
   })
   ```
4. **Call parse_labelstudio_annotation** with the properly formatted data

## Annotation Dictionary to Bitmap Conversion

The `batch_evaluate_with_minio_annotations` function converts annotation dictionaries to bitmaps by calling `parse_labelstudio_annotation()` for each task:

```python
# Now process the export data similar to the original function
image_annotation_pairs = []
for task_data in export_data:
    try:
        # Parse the annotation data
        parsed_data = parse_labelstudio_annotation(task_data, class_names)
        if parsed_data:
            image_annotation_pairs.append(parsed_data)
```

This converts each annotation dictionary to a bitmap mask (2D numpy array) ready for comparison with model predictions.

## Image-Annotation Pairing and Comparison

The script pairs annotations with model predictions through **task_id matching**:

1. **Database query** gets both task data (with image path) and completion data (with annotations)
2. **Groups by task_id** so each task has its image path and annotations together
3. **For each task**:
   - Loads the image from MinIO using the image path from task data
   - Runs model inference on that same image to get prediction bitmap
   - Converts the annotation to bitmap using `parse_labelstudio_annotation()`
   - Compares the model prediction bitmap with the annotation bitmap

The pairing happens automatically through the **task_id** - each task has one image and one annotation, and they're processed together in the same iteration.

## Direct Database Reading Logic

The batch evaluation system reads annotations directly from the Label Studio SQLite database without requiring export files. Here's how it works:

### Database Query Process

1. **SQL Query**: The system executes a JOIN query to get both task and completion data:
   ```sql
   SELECT 
       t.id as task_id,
       t.data as task_data,
       tc.id as completion_id,
       tc.result as completion_result
   FROM task t
   LEFT JOIN task_completion tc ON t.id = tc.task_id
   WHERE tc.result IS NOT NULL
   ORDER BY t.id, tc.id
   ```

2. **Data Grouping**: Results are grouped by `task_id` to create a proper task structure:
   ```python
   tasks[task_id] = {
       'id': task_id,
       'data': json.loads(task_data),  # Contains image path
       'annotations': []
   }
   ```

3. **Annotation Processing**: Each completion result is parsed and added to the task:
   ```python
   tasks[task_id]['annotations'].append({
       'id': completion_id,
       'result': json.loads(completion_result),  # Label Studio annotation format
       'created_at': None
   })
   ```

### Image Loading and Processing

1. **Image Path Extraction**: The system extracts image paths from the task data:
   - Uses `$undefined$` key (Label Studio's default for image references)
   - Converts S3 URLs to MinIO object keys
   - Example: `s3://segmentation-platform/images/file.png` → `images/file.png`

2. **MinIO Integration**: Images are loaded directly from MinIO storage:
   - Uses boto3 S3 client configured for MinIO
   - Downloads image data as bytes
   - Converts to OpenCV format for processing

3. **Annotation Parsing**: Each annotation is converted to a bitmap mask:
   - Extracts polygon coordinates from Label Studio format
   - Converts percentage coordinates to pixel coordinates
   - Creates multi-class masks using PIL ImageDraw
   - Handles multiple polygons per annotation

### Key Advantages

- **No Export Files Required**: Eliminates the need for manual export file generation
- **Real-time Data**: Always uses the latest annotations from the database
- **Automatic Pairing**: Task ID ensures correct image-annotation matching
- **Direct Integration**: Seamlessly works with the existing MinIO storage system