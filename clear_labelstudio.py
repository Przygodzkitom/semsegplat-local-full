#!/usr/bin/env python3
"""
Label Studio Database Cleanup Script
Clears the Label Studio SQLite database and related data
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

def clear_labelstudio_database():
    """Clear Label Studio database and related data"""
    print("🧹 Label Studio Database Cleanup")
    print("=" * 50)
    
    # Stop Label Studio container
    print("🔄 Stopping Label Studio container...")
    try:
        result = subprocess.run(['docker', 'compose', 'stop', 'label-studio'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Label Studio container stopped")
        else:
            print(f"⚠️ Warning stopping container: {result.stderr}")
    except Exception as e:
        print(f"❌ Error stopping container: {e}")
        return False
    
    # Clear the SQLite database
    db_path = "label-studio-data/label_studio.sqlite3"
    if os.path.exists(db_path):
        # Backup the database first
        backup_path = f"{db_path}.backup_{int(time.time())}"
        shutil.copy2(db_path, backup_path)
        print(f"📋 Database backed up to: {backup_path}")
        
        # Remove the database
        os.remove(db_path)
        print("✅ Label Studio database cleared!")
    else:
        print("ℹ️ No database file found to clear")
    
    # Clear other Label Studio data
    data_dirs = [
        "label-studio-data/export",
        "label-studio-data/media"
    ]
    
    cleared_dirs = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            os.makedirs(data_dir, exist_ok=True)
            cleared_dirs.append(data_dir)
    
    if cleared_dirs:
        print(f"✅ Cleared Label Studio data directories: {', '.join(cleared_dirs)}")
    
    # Restart Label Studio container
    print("🔄 Restarting Label Studio container...")
    try:
        result = subprocess.run(['docker', 'compose', 'start', 'label-studio'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Label Studio container restarted")
            print("⏳ Please wait a moment for Label Studio to initialize...")
        else:
            print(f"❌ Error restarting container: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error restarting container: {e}")
        return False
    
    print("🎉 Label Studio completely reset! You can now create a fresh project.")
    return True

def clear_minio_annotations():
    """Clear MinIO annotation files"""
    print("\n🗄️ MinIO Storage Cleanup")
    print("=" * 50)
    
    try:
        import boto3
        s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin123',
            region_name='us-east-1'
        )
        
        # List and delete annotation files
        response = s3_client.list_objects_v2(Bucket='segmentation-platform', Prefix='annotations/')
        deleted_count = 0
        
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(Bucket='segmentation-platform', Key=obj['Key'])
                deleted_count += 1
        
        print(f"✅ Cleared {deleted_count} annotation files from MinIO!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing MinIO: {str(e)}")
        return False

def clear_config_files():
    """Clear configuration files"""
    print("\n📁 Config Files Cleanup")
    print("=" * 50)
    
    config_files = [
        "config/label_studio_project.json",
        "config/class_config.json",
        "/app/class_config.json"
    ]
    
    cleared_files = []
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                os.remove(config_file)
                cleared_files.append(config_file)
            except Exception as e:
                print(f"⚠️ Could not remove {config_file}: {e}")
    
    if cleared_files:
        print(f"✅ Cleared config files: {', '.join(cleared_files)}")
    else:
        print("ℹ️ No config files found to clear")
    
    return True

def main():
    """Main cleanup function"""
    print("🧹 Complete System Cleanup")
    print("=" * 50)
    print("This will clear:")
    print("- Label Studio database and data")
    print("- MinIO annotation files")
    print("- Configuration files")
    print()
    
    response = input("Are you sure you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled")
        return
    
    success = True
    
    # Clear Label Studio database
    if not clear_labelstudio_database():
        success = False
    
    # Clear MinIO annotations
    if not clear_minio_annotations():
        success = False
    
    # Clear config files
    if not clear_config_files():
        success = False
    
    if success:
        print("\n🎉 Complete system cleanup successful!")
        print("You can now start fresh with your annotation testing.")
    else:
        print("\n⚠️ Some cleanup operations had issues, but the system should be mostly clean.")

if __name__ == "__main__":
    main()

