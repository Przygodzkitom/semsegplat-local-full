#!/usr/bin/env python3
"""
Test script to run batch evaluation directly and see what happens
"""

import sys
import os

# Add the models directory to the path
sys.path.append('models')

def test_batch_evaluation():
    """Test the batch evaluation function directly"""
    
    try:
        from models.inference import batch_evaluate_with_labelstudio_export
        
        # Test parameters
        export_file = "label-studio-data/export/project-1-at-2025-09-11-09-09-db-export.json"
        model_path = "models/checkpoints/final_model_polygon_20250910_110111.pth"
        bucket_name = "segmentation-platform"
        
        print("🧪 Testing batch evaluation...")
        print(f"Export file: {export_file}")
        print(f"Model path: {model_path}")
        print(f"Bucket name: {bucket_name}")
        
        # Check if files exist
        if not os.path.exists(export_file):
            print(f"❌ Export file not found: {export_file}")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print("✅ All files exist")
        
        # Run batch evaluation
        print("\n🚀 Running batch evaluation...")
        results = batch_evaluate_with_labelstudio_export(
            export_file_path=export_file,
            model_path=model_path,
            bucket_name=bucket_name,
            num_classes=None,
            threshold=0.3
        )
        
        print(f"\n📊 Results:")
        print(f"Status: {results.get('status')}")
        print(f"Images evaluated: {results.get('images_evaluated')}")
        print(f"Classes: {results.get('class_names')}")
        print(f"Overall mean IoU: {results.get('overall_mean_iou')}")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        
        if 'debug_info' in results:
            print(f"\nDebug info:")
            for info in results['debug_info']:
                print(f"  {info}")
        
        return results.get('status') == 'success'
        
    except Exception as e:
        print(f"❌ Error running batch evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing batch evaluation directly...")
    success = test_batch_evaluation()
    
    if success:
        print("\n✅ Batch evaluation test completed successfully!")
    else:
        print("\n❌ Batch evaluation test failed!")
