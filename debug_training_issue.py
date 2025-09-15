#!/usr/bin/env python3
"""
Debug script to identify why training fails silently after changing annotations
"""

import os
import sys
import json
import boto3
from botocore.exceptions import ClientError

def check_minio_connection():
    """Check MinIO connection and bucket access"""
    print("🔍 Checking MinIO connection...")
    
    try:
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
        
        # Test connection
        s3_client.list_buckets()
        print("✅ MinIO connection successful")
        
        return s3_client
    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")
        return None

def check_annotations(s3_client, bucket_name="segmentation-platform", annotation_prefix="annotations"):
    """Check annotation files in MinIO"""
    print(f"🔍 Checking annotations in bucket '{bucket_name}' with prefix '{annotation_prefix}'...")
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=annotation_prefix
        )
        
        if 'Contents' not in response:
            print(f"❌ No annotations found with prefix '{annotation_prefix}'")
            return False
        
        print(f"✅ Found {len(response['Contents'])} annotation files")
        
        # Check first few annotation files
        for i, obj in enumerate(response['Contents'][:5]):
            print(f"  📄 {obj['Key']} (size: {obj['Size']} bytes)")
            
            if obj['Size'] == 0:
                print(f"    ⚠️ Empty file!")
                continue
                
            # Try to read and parse the annotation
            try:
                response_obj = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                annotation_data = json.loads(response_obj['Body'].read().decode('utf-8'))
                
                # Check annotation structure
                if 'result' in annotation_data:
                    result = annotation_data['result']
                    print(f"    ✅ Valid annotation with {len(result)} items")
                    
                    # Check annotation types
                    types = [item.get('type') for item in result]
                    print(f"    📝 Annotation types: {types}")
                    
                    # Check for brush vs polygon
                    has_brush = 'brushlabels' in types
                    has_polygon = 'polygonlabels' in types
                    print(f"    🎨 Has brush: {has_brush}, Has polygon: {has_polygon}")
                    
                else:
                    print(f"    ❌ Invalid annotation structure - no 'result' key")
                    
            except Exception as e:
                print(f"    ❌ Error reading annotation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking annotations: {e}")
        return False

def check_class_configuration():
    """Check class configuration files"""
    print("🔍 Checking class configuration...")
    
    config_files = [
        "/app/class_config.json",
        "class_config.json",
        "models/class_config.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Found class config: {config_file}")
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"  📝 Classes: {config.get('class_names', [])}")
                return True
            except Exception as e:
                print(f"  ❌ Error reading config: {e}")
        else:
            print(f"❌ Not found: {config_file}")
    
    return False

def check_annotation_type_detection():
    """Test annotation type detection"""
    print("🔍 Testing annotation type detection...")
    
    try:
        sys.path.append('models/utils')
        from annotation_type_detector import AnnotationTypeDetector
        
        detector = AnnotationTypeDetector("segmentation-platform", "annotations")
        detection = detector.detect_annotation_type()
        
        print(f"✅ Detection result: {detection}")
        
        if detection['sample_annotations'] == 0:
            print("❌ No annotations were successfully processed!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error in annotation type detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔍 TRAINING SILENT FAILURE DIAGNOSTIC")
    print("=" * 60)
    
    # Check MinIO connection
    s3_client = check_minio_connection()
    if not s3_client:
        print("❌ Cannot proceed without MinIO connection")
        return
    
    # Check annotations
    annotations_ok = check_annotations(s3_client)
    
    # Check class configuration
    config_ok = check_class_configuration()
    
    # Check annotation type detection
    detection_ok = check_annotation_type_detection()
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"MinIO Connection: {'✅ OK' if s3_client else '❌ FAILED'}")
    print(f"Annotations: {'✅ OK' if annotations_ok else '❌ FAILED'}")
    print(f"Class Config: {'✅ OK' if config_ok else '❌ FAILED'}")
    print(f"Type Detection: {'✅ OK' if detection_ok else '❌ FAILED'}")
    
    if not annotations_ok:
        print("\n🚨 LIKELY ISSUE: Annotation files are missing or empty")
        print("   → Check if Label Studio properly exported annotations to MinIO")
        print("   → Verify the annotation prefix in Label Studio settings")
    elif not detection_ok:
        print("\n🚨 LIKELY ISSUE: Annotation type detection is failing")
        print("   → Check if annotation files have valid JSON structure")
        print("   → Verify annotation format matches expected Label Studio format")
    elif not config_ok:
        print("\n🚨 LIKELY ISSUE: Class configuration is missing")
        print("   → Run class detection in Streamlit first")
        print("   → Check if class names match between annotations and config")

if __name__ == "__main__":
    main()

