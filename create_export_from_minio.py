#!/usr/bin/env python3
"""
Script to create a Label Studio export file from MinIO annotations
This can be used as a workaround if the direct MinIO integration doesn't work
"""

import os
import json
import sys
from datetime import datetime

def create_export_from_minio():
    """Create a Label Studio export file from MinIO annotations"""
    
    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError:
        print("❌ MinIO client not available. Please install: pip install minio")
        return False
    
    # Initialize MinIO client
    minio_client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False
    )
    
    bucket_name = "segmentation-platform"
    
    try:
        # List annotation files in the bucket
        annotation_objects = []
        for obj in minio_client.list_objects(bucket_name, prefix="annotations/", recursive=True):
            if obj.object_name.endswith('.json') or not obj.object_name.endswith('/'):
                annotation_objects.append(obj.object_name)
        
        if not annotation_objects:
            print(f"❌ No annotation files found in MinIO bucket {bucket_name}")
            return False
        
        print(f"✅ Found {len(annotation_objects)} annotation files in MinIO")
        
        # Create export directory if it doesn't exist
        export_dir = "label-studio-data/export/"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate export filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"project-1-at-{timestamp}-minio-export.json"
        filepath = os.path.join(export_dir, filename)
        
        # Collect all annotations
        all_annotations = []
        for obj_name in annotation_objects:
            try:
                # Download annotation data
                response = minio_client.get_object(bucket_name, obj_name)
                annotation_data = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                all_annotations.append(annotation_data)
                print(f"✅ Processed annotation: {obj_name}")
                
            except Exception as e:
                print(f"⚠️ Error processing annotation {obj_name}: {e}")
                continue
        
        if not all_annotations:
            print("❌ No valid annotations found")
            return False
        
        # Save as Label Studio export format
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Export file created: {filepath}")
        print(f"📊 Total annotations: {len(all_annotations)}")
        return filepath
        
    except S3Error as e:
        print(f"❌ MinIO error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Creating Label Studio export file from MinIO annotations...")
    result = create_export_from_minio()
    
    if result:
        print(f"✅ Success! Export file created: {result}")
        print("You can now run batch evaluation in Streamlit.")
    else:
        print("❌ Failed to create export file.")
        sys.exit(1)

