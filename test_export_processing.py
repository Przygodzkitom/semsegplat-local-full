#!/usr/bin/env python3
"""
Test script to verify export file processing works correctly
"""

import json
import sys
import os

# Add the models directory to the path
sys.path.append('models')

def test_export_processing():
    """Test the export file processing logic"""
    
    # Load the export file
    export_file = "label-studio-data/export/project-1-at-2025-09-11-09-09-db-export.json"
    
    if not os.path.exists(export_file):
        print(f"❌ Export file not found: {export_file}")
        return False
    
    print(f"✅ Loading export file: {export_file}")
    
    with open(export_file, 'r') as f:
        export_data = json.load(f)
    
    print(f"✅ Loaded {len(export_data)} tasks")
    
    # Test the image path extraction logic
    for i, task_data in enumerate(export_data):
        print(f"\n--- Task {i+1} ---")
        print(f"Task ID: {task_data.get('id')}")
        
        # Test image path extraction
        data = task_data.get('data', {})
        image_path = data.get('image', '') or data.get('$undefined$', '')
        print(f"Image path: {image_path}")
        
        if not image_path:
            print(f"❌ No image path found")
            print(f"   Available data keys: {list(data.keys())}")
            continue
        
        # Test S3 URL conversion
        bucket_name = "segmentation-platform"
        if image_path.startswith('s3://'):
            image_key = image_path.replace(f's3://{bucket_name}/', '')
        else:
            image_key = image_path
        
        print(f"Image key: {image_key}")
        
        # Test annotation processing
        annotations = task_data.get('annotations', [])
        print(f"Annotations: {len(annotations)}")
        
        if annotations:
            annotation = annotations[0]
            result = annotation.get('result', [])
            print(f"Result items: {len(result)}")
            
            if result:
                first_item = result[0]
                print(f"First item type: {first_item.get('type')}")
                print(f"First item keys: {list(first_item.keys())}")
                
                if 'value' in first_item:
                    value = first_item['value']
                    print(f"Value keys: {list(value.keys())}")
                    
                    if 'polygonlabels' in value:
                        labels = value['polygonlabels']
                        print(f"Polygon labels: {labels}")
                    
                    if 'points' in value:
                        points = value['points']
                        print(f"Points count: {len(points)}")
                        if points:
                            print(f"First point: {points[0]}")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing export file processing...")
    success = test_export_processing()
    
    if success:
        print("\n✅ Export file processing test completed!")
    else:
        print("\n❌ Export file processing test failed!")
