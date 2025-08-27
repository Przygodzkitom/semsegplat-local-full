# Dynamic Class Configuration System

## Overview

The semantic segmentation platform now supports **dynamic class configuration** instead of hardcoded classes. This allows users to work with any number of classes and any class names from their Label Studio annotations.

## How It Works

### 1. **Automatic Class Detection**
- The system automatically scans all annotations in storage
- Detects all unique class names used in polygon and brush annotations
- Counts the number of annotations per class
- Presents this information to the user in the Streamlit interface

### 2. **User Configuration Interface**
- Users can view detected classes and their annotation counts
- Users can edit class names if needed
- Users can reorder classes (Background is always first)
- Configuration is saved to a JSON file for training

### 3. **Dynamic Training**
- Training script reads class configuration from JSON file
- Model architecture adapts to the number of classes
- Loss function and class weights adjust automatically
- No hardcoded class names or counts

## User Workflow

### Step 1: Detect Classes
1. Go to the "Train Model" step in Streamlit
2. Click "🔍 Detect Classes from Label Studio"
3. System scans all annotations and shows detected classes

### Step 2: Configure Classes
1. Review detected classes and annotation counts
2. Edit class names if needed (e.g., fix typos)
3. Classes are automatically ordered with Background first
4. Click "💾 Save Class Configuration"

### Step 3: Start Training
1. Training button becomes enabled after class configuration
2. System uses the configured classes for training
3. Model adapts to the exact number of classes

## Technical Implementation

### Files Modified

1. **`models/utils/class_detector.py`** (NEW)
   - `ClassDetector` class for scanning Label Studio annotations
   - Extracts class names from polygon and brush annotations
   - Provides statistics about class distribution

2. **`models/training.py`**
   - Added `load_class_configuration()` function
   - Model classes now dynamic: `classes=len(class_names)`
   - Class weights adapt to number of classes
   - Progress tracking includes class information

3. **`app/main.py`**
   - Added class configuration section in training interface
   - Class detection and editing UI
   - Configuration validation before training

4. **`models/utils/labelstudio_dataloader.py`**
   - Updated multilabel conversion to support any number of classes
   - Dynamic channel creation based on class count

### Configuration File

Class configuration is saved to `/app/class_config.json`:

```json
{
  "class_names": ["Background", "blade", "norma"],
  "detected_classes": {
    "blade": 10,
    "norma": 9
  },
  "total_annotations": 19
}
```

## Benefits

### 1. **Flexibility**
- Works with any number of classes (2, 3, 10, etc.)
- Supports any class names (no hardcoded restrictions)
- Easy to add new classes without code changes

### 2. **User Control**
- Users can see exactly what classes were detected
- Users can correct class names if needed
- Users can verify annotation counts before training

### 3. **Robustness**
- Automatic detection prevents configuration errors
- Fallback to default if detection fails
- Clear error messages for troubleshooting

### 4. **Scalability**
- System adapts to different datasets automatically
- No need to modify code for different class sets
- Easy to extend for future annotation types

## Example Usage

### Scenario 1: Medical Imaging
- **Detected Classes**: Background, tumor, cyst, calcification
- **Result**: 4-class model automatically configured

### Scenario 2: Industrial Inspection
- **Detected Classes**: Background, defect, scratch, dent
- **Result**: 4-class model automatically configured

### Scenario 3: Agricultural Analysis
- **Detected Classes**: Background, leaf, stem, fruit, disease
- **Result**: 5-class model automatically configured

## Error Handling

### Detection Failures
- If no annotations found: Shows warning message
- If GCS access fails: Shows error with troubleshooting steps
- If annotation format invalid: Logs error and continues

### Configuration Issues
- Empty class names: Automatically uses detected names
- Missing configuration file: Uses default Background class
- Invalid JSON: Shows error and allows reconfiguration

### Training Issues
- Class mismatch: Training fails with clear error message
- Missing annotations: Continues with available data
- Memory issues: Suggests reducing batch size or image size

## Future Enhancements

1. **Class Validation**: Check for overlapping annotations
2. **Class Merging**: Allow combining similar classes
3. **Class Splitting**: Allow splitting complex classes
4. **Import/Export**: Save/load class configurations
5. **Class Statistics**: Show pixel coverage per class
6. **Visualization**: Show sample annotations per class

## Troubleshooting

### Common Issues

1. **No classes detected**
   - Check if annotations exist in GCS
   - Verify annotation format is correct
   - Check GCS permissions

2. **Wrong class names**
   - Edit class names in the interface
   - Re-save configuration
   - Restart training

3. **Training fails**
   - Check class configuration is saved
   - Verify all classes have annotations
   - Check system resources

### Debug Information

The system provides detailed debug information:
- Raw annotation data
- Class detection results
- Configuration file contents
- Training progress with class details 