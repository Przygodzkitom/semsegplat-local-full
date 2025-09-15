#!/usr/bin/env python3
"""
Test script to debug training silent failure
"""

import os
import sys
import json
import sqlite3

def test_annotation_loading():
    """Test if we can load annotations from database"""
    print("🔍 Testing annotation loading from database...")
    
    try:
        db_path = 'label-studio-data/label_studio.sqlite3'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = '''
        SELECT 
            t.id as task_id,
            t.data as task_data,
            tc.result as completion_result
        FROM task t
        LEFT JOIN task_completion tc ON t.id = tc.task_id
        WHERE tc.result IS NOT NULL
        ORDER BY t.id
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        print(f"✅ Found {len(rows)} annotations in database")
        
        # Test parsing each annotation
        for i, (task_id, task_data, completion_result) in enumerate(rows):
            print(f"\n📝 Testing annotation {i+1} (Task {task_id}):")
            
            try:
                result = json.loads(completion_result)
                print(f"  ✅ JSON parsing successful: {len(result)} items")
                
                # Check for brush annotations
                brush_items = [item for item in result if item.get('type') == 'brushlabels']
                print(f"  🖌️  Brush items: {len(brush_items)}")
                
                for j, item in enumerate(brush_items):
                    value = item.get('value', {})
                    rle_data = value.get('rle', [])
                    labels = value.get('brushlabels', [])
                    
                    print(f"    Brush {j+1}: {labels}, RLE length: {len(rle_data)}")
                    
                    # Check RLE format
                    if len(rle_data) % 2 == 0:
                        print(f"      ✅ RLE format valid (even length)")
                    else:
                        print(f"      ⚠️  RLE format odd length - might cause issues")
                        
                        # Analyze odd-length RLE
                        print(f"      🔍 RLE analysis:")
                        print(f"        - First 10: {rle_data[:10]}")
                        print(f"        - Last 10: {rle_data[-10:]}")
                        print(f"        - Min/Max: {min(rle_data)}/{max(rle_data)}")
                        
                        # Try to understand the pattern
                        if len(rle_data) > 100:
                            # Check if it's actually valid RLE with some extra data
                            valid_pairs = len(rle_data) // 2
                            print(f"        - Valid pairs: {valid_pairs}")
                            print(f"        - Extra values: {len(rle_data) - valid_pairs * 2}")
                            
            except Exception as e:
                print(f"  ❌ Error parsing annotation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return False

def test_environment():
    """Test environment variables and paths"""
    print("\n🔍 Testing environment...")
    
    print(f"BUCKET_NAME: {os.getenv('BUCKET_NAME', 'NOT SET')}")
    print(f"ANNOTATION_PREFIX: {os.getenv('ANNOTATION_PREFIX', 'NOT SET')}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")
    
    # Check if required files exist
    required_files = [
        'models/training_brush_minimal.py',
        'models/utils/simple_brush_dataloader.py',
        'models/utils/annotation_type_detector.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")

def main():
    print("=" * 60)
    print("🔍 TRAINING SILENT FAILURE DEBUG TEST")
    print("=" * 60)
    
    # Test annotation loading
    annotations_ok = test_annotation_loading()
    
    # Test environment
    test_environment()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if annotations_ok:
        print("✅ Annotations are loading correctly from database")
        print("🔍 The issue is likely in the training process itself")
        print("💡 Possible causes:")
        print("   - RLE parsing failures due to odd-length data")
        print("   - Memory issues during training")
        print("   - GPU/CUDA problems")
        print("   - Training subprocess dying immediately")
    else:
        print("❌ Annotation loading is failing")
        print("💡 This explains the silent failure")

if __name__ == "__main__":
    main()

