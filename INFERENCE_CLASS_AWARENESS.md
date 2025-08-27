# Inference Module Class Awareness

## Overview

The inference module has been updated to be aware of the number of classes the model was trained on, removing hardcoded assumptions about 2 classes. This allows the system to work with any number of classes dynamically.

## Changes Made

### 1. `models/inferencer.py`

**Key Changes:**
- Added `load_class_configuration()` function to read class configuration from `class_config.json`
- Modified `Inferencer.__init__()` to accept and use `ModelConfig` with dynamic class information
- Updated `predict()` method to handle variable number of classes
- Enhanced `create_visualization()` to display masks for all classes with proper colors
- Modified `predict_and_compare()` to return metrics for all classes

**Before:**
```python
# Hardcoded for 2 classes
self.model = smp.Unet("resnet101", classes=2, activation=None)
mask1, mask2 = self.predict(image)  # Fixed 2 masks
```

**After:**
```python
# Dynamic based on config
self.model = smp.Unet("resnet101", classes=self.num_classes, activation=None)
masks = self.predict(image)  # Variable number of masks
```

### 2. `models/inference.py`

**Key Changes:**
- Added `load_class_configuration()` function
- Modified `batch_evaluate()` to auto-detect number of classes from configuration
- Enhanced evaluation reporting to show class names and metrics for all classes
- Added object-wise metrics calculation for each class

**Before:**
```python
def batch_evaluate(image_dir, mask_dir, model_path, num_classes):
    config = ModelConfig(num_classes=num_classes)  # Required num_classes
```

**After:**
```python
def batch_evaluate(image_dir, mask_dir, model_path, num_classes=None):
    if num_classes is None:
        class_names = load_class_configuration()
        num_classes = len(class_names)
    config = ModelConfig(num_classes=num_classes, class_names=class_names)
```

## Features

### 1. Auto-Detection of Classes
The system automatically reads the class configuration from `class_config.json`:
```json
{
  "class_names": ["Background", "platelet"],
  "detected_classes": {"platelet": 10},
  "total_annotations": 10
}
```

### 2. Dynamic Model Loading
The model is loaded with the correct number of classes based on the configuration:
```python
# Automatically uses the right number of classes
inferencer = Inferencer(model_path, config)
```

### 3. Flexible Visualization
- Supports any number of classes with automatic color assignment
- Displays individual masks for each class
- Shows ground truth comparison when available

### 4. Comprehensive Metrics
- IoU calculation for each class
- Object-wise metrics (precision, recall, F1) for each class
- Detailed evaluation reports with class names

## Usage Examples

### Single Image Inference
```python
from models.inferencer import Inferencer
from models.config import ModelConfig

# Auto-detect classes
class_names = load_class_configuration()
config = ModelConfig(num_classes=len(class_names), class_names=class_names)
inferencer = Inferencer("model.pth", config)

# Run inference
pred_mask, ious, metrics = inferencer.predict_and_compare(image, gt_mask)
```

### Batch Evaluation
```python
from models.inference import batch_evaluate

# Auto-detect classes from configuration
batch_evaluate(
    image_dir="/path/to/images",
    mask_dir="/path/to/masks", 
    model_path="/path/to/model.pth",
    num_classes=None  # Auto-detected
)

# Or specify manually
batch_evaluate(
    image_dir="/path/to/images",
    mask_dir="/path/to/masks",
    model_path="/path/to/model.pth", 
    num_classes=3
)
```

## Backward Compatibility

The changes maintain backward compatibility:
- Existing code specifying `num_classes` will continue to work
- Default behavior falls back to 2 classes if no configuration is found
- The `ModelConfig` default remains `num_classes=2` for compatibility

## Benefits

1. **Flexibility**: Works with any number of classes (2, 3, 5, 10, etc.)
2. **Maintainability**: No need to modify code when adding/removing classes
3. **Consistency**: Uses the same class configuration as training
4. **User-Friendly**: Automatic detection reduces configuration errors
5. **Comprehensive**: Provides detailed metrics for all classes

## Testing

The changes have been tested with:
- ✅ Class configuration loading
- ✅ ModelConfig creation with dynamic classes
- ✅ Configuration verification
- ✅ Multiple class configurations (2, 3, 5 classes)

The inference module is now fully class-aware and ready for production use with any number of classes. 