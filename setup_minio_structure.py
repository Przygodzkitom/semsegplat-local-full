#!/usr/bin/env python3
"""
Setup MinIO Storage Structure for Semantic Segmentation Platform

This script creates the recommended folder structure in MinIO:
- images/     (for source images)
- annotations/ (for Label Studio exports)
- models/     (for trained models)
"""

import boto3
import os
from botocore.exceptions import ClientError

def setup_minio_structure():
    """Setup the recommended MinIO folder structure"""
    
    # MinIO configuration
    endpoint_url = "http://localhost:9000"
    access_key = "minioadmin"
    secret_key = "minioadmin123"
    bucket_name = "segmentation-platform"
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        
        print("🔧 Setting up MinIO storage structure...")
        
        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"✅ Created bucket '{bucket_name}'")
            else:
                raise
        
        # Create folder structure by uploading empty objects
        folders = [
            "images/",
            "annotations/",
            "models/",
            "models/checkpoints/",
            "models/final/"
        ]
        
        for folder in folders:
            try:
                # Create folder by uploading an empty object
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=folder,
                    Body=""
                )
                print(f"✅ Created folder: {folder}")
            except Exception as e:
                print(f"⚠️ Folder {folder} might already exist: {e}")
        
        # List the structure
        print("\n📁 Current MinIO structure:")
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('/'):
                    print(f"  📂 {obj['Key']}")
        
        print("\n🎯 Recommended Label Studio Configuration:")
        print("\n📸 Source Storage (for Images):")
        print("  Storage Title: MinIO Images")
        print("  Storage Type: Amazon S3")
        print(f"  Bucket Name: {bucket_name}")
        print("  Prefix: images/")
        print("  Regex Filter: .*\.(jpg|jpeg|png|tif|tiff)$")
        print("  Use pre-signed URLs: ❌ OFF")
        print("  Proxy through Label Studio: ✅ ON")
        
        print("\n📝 Target Storage (for Annotations):")
        print("  Storage Title: MinIO Annotations")
        print("  Storage Type: Amazon S3")
        print(f"  Bucket Name: {bucket_name}")
        print("  Prefix: annotations/")
        print("  Use pre-signed URLs: ❌ OFF")
        print("  Proxy through Label Studio: ✅ ON")
        
        print("\n✅ MinIO structure setup complete!")
        print(f"🌐 Access MinIO Console: http://localhost:9001")
        print(f"   Login: minioadmin / minioadmin123")
        
    except Exception as e:
        print(f"❌ Error setting up MinIO structure: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_minio_structure()

