#!/usr/bin/env python3
"""
Comprehensive System Cleanup Script
Clears ALL project data for a true clean slate experience
"""

import os
import shutil
import subprocess
import time
import json
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

class ComprehensiveCleanup:
    def __init__(self):
        self.minio_endpoint = "http://localhost:9000"
        self.minio_access_key = "minioadmin"
        self.minio_secret_key = "minioadmin123"
        self.bucket_name = "segmentation-platform"
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def stop_containers(self):
        """Stop all relevant containers"""
        self.log("🔄 Stopping containers...")
        
        # First, get list of running containers
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.log("⚠️ Could not list running containers", "WARN")
                return
            
            running_containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            self.log(f"📋 Found {len(running_containers)} running containers")
            
        except Exception as e:
            self.log(f"⚠️ Error listing containers: {e}", "WARN")
            return
        
        # Define containers we want to stop
        target_containers = [
            "semsegplat-full_local_version-label-studio-1",
            "semsegplat-full_local_version-semseg-app-1", 
            "semsegplat-full_local_version-minio-1"
        ]
        
        for container in target_containers:
            if container in running_containers:
                try:
                    result = subprocess.run(
                        ['docker', 'stop', container], 
                        capture_output=True, 
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        self.log(f"✅ Stopped {container}")
                    else:
                        self.log(f"⚠️ Warning stopping {container}: {result.stderr}", "WARN")
                except subprocess.TimeoutExpired:
                    self.log(f"⚠️ Timeout stopping {container}", "WARN")
                except Exception as e:
                    self.log(f"❌ Error stopping {container}: {e}", "ERROR")
            else:
                self.log(f"ℹ️ Container {container} is not running")
    
    def clear_labelstudio_database(self):
        """Clear Label Studio database and related data"""
        self.log("🧹 Clearing Label Studio database...")
        
        # Clear the SQLite database
        db_path = "label-studio-data/label_studio.sqlite3"
        if os.path.exists(db_path):
            try:
                # Backup the database first
                backup_path = f"{db_path}.backup_{int(time.time())}"
                shutil.copy2(db_path, backup_path)
                self.log(f"📋 Database backed up to: {backup_path}")
                
                # Try to remove the database with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        os.remove(db_path)
                        self.log("✅ Label Studio database cleared!")
                        break
                    except PermissionError as e:
                        if attempt < max_retries - 1:
                            self.log(f"⚠️ Permission denied, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(2)
                        else:
                            self.log(f"❌ Could not remove database after {max_retries} attempts: {e}", "ERROR")
                    except Exception as e:
                        self.log(f"❌ Error removing database: {e}", "ERROR")
                        break
            except Exception as e:
                self.log(f"❌ Error backing up database: {e}", "ERROR")
        else:
            self.log("ℹ️ No database file found to clear")
        
        # Clear other Label Studio data directories
        data_dirs = [
            "label-studio-data/export",
            "label-studio-data/media"
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                try:
                    shutil.rmtree(data_dir)
                    os.makedirs(data_dir, exist_ok=True)
                    self.log(f"✅ Cleared directory: {data_dir}")
                except Exception as e:
                    self.log(f"⚠️ Error clearing directory {data_dir}: {e}", "WARN")
            else:
                os.makedirs(data_dir, exist_ok=True)
                self.log(f"✅ Created directory: {data_dir}")
        
        # Also clear any backup files older than 1 hour to prevent accumulation
        self.log("🧹 Cleaning up old backup files...")
        try:
            backup_files = [f for f in os.listdir("label-studio-data") if f.startswith("label_studio.sqlite3.backup_")]
            current_time = time.time()
            cleaned_count = 0
            
            for backup_file in backup_files:
                backup_path = os.path.join("label-studio-data", backup_file)
                try:
                    # Keep only the most recent backup, remove others older than 1 hour
                    file_time = os.path.getmtime(backup_path)
                    if current_time - file_time > 3600:  # 1 hour
                        os.remove(backup_path)
                        cleaned_count += 1
                except Exception as e:
                    self.log(f"⚠️ Could not remove old backup {backup_file}: {e}", "WARN")
            
            if cleaned_count > 0:
                self.log(f"✅ Cleaned up {cleaned_count} old backup files")
            else:
                self.log("ℹ️ No old backup files to clean up")
                
        except Exception as e:
            self.log(f"⚠️ Error cleaning up backup files: {e}", "WARN")
    
    def clear_minio_filesystem_data(self):
        """Clear MinIO data directly from filesystem"""
        self.log("🗑️ Clearing MinIO filesystem data...")
        
        minio_data_path = "minio-data/segmentation-platform"
        
        if not os.path.exists(minio_data_path):
            self.log("ℹ️ No MinIO data directory found")
            return True
        
        try:
            import shutil
            
            # Clear images directory
            images_path = os.path.join(minio_data_path, "images")
            if os.path.exists(images_path):
                shutil.rmtree(images_path)
                self.log("✅ Cleared MinIO images directory")
            else:
                self.log("ℹ️ No images directory found")
            
            # Clear annotations directory
            annotations_path = os.path.join(minio_data_path, "annotations")
            if os.path.exists(annotations_path):
                shutil.rmtree(annotations_path)
                self.log("✅ Cleared MinIO annotations directory")
            else:
                self.log("ℹ️ No annotations directory found")
            
            # Clear models directory if it exists
            models_path = os.path.join(minio_data_path, "models")
            if os.path.exists(models_path):
                shutil.rmtree(models_path)
                self.log("✅ Cleared MinIO models directory")
            else:
                self.log("ℹ️ No models directory found")
                
            return True
            
        except Exception as e:
            self.log(f"⚠️ Error clearing MinIO filesystem data: {e}", "WARN")
            return False

    def clear_minio_data(self):
        """Clear all MinIO data (images and annotations)"""
        self.log("🧹 Clearing MinIO data...")
        
        # First, clear filesystem data (most reliable)
        filesystem_success = self.clear_minio_filesystem_data()
        
        # Then try S3 API as backup
        s3_success = False
        try:
            # Initialize S3 client with proper MinIO configuration
            from botocore.config import Config
            
            config = Config(
                signature_version='s3v4',
                s3={
                    'addressing_style': 'path'
                }
            )
            
            s3_client = boto3.client(
                's3',
                endpoint_url=self.minio_endpoint,
                aws_access_key_id=self.minio_access_key,
                aws_secret_access_key=self.minio_secret_key,
                region_name='us-east-1',
                config=config
            )
            
            # Test connection first
            try:
                s3_client.head_bucket(Bucket=self.bucket_name)
                self.log("✅ MinIO connection successful")
            except Exception as e:
                self.log(f"⚠️ Cannot connect to MinIO: {e}", "WARN")
                return filesystem_success
            
            # Clear images
            self.log("🗑️ Clearing MinIO images...")
            try:
                response = s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix="images/"
                )
                
                if 'Contents' in response:
                    image_objects = [obj['Key'] for obj in response['Contents']]
                    if image_objects:
                        # Delete all image objects
                        for obj_key in image_objects:
                            s3_client.delete_object(Bucket=self.bucket_name, Key=obj_key)
                        self.log(f"✅ Cleared {len(image_objects)} image objects")
                    else:
                        self.log("ℹ️ No images to clear")
                else:
                    self.log("ℹ️ No images folder found")
                s3_success = True
            except Exception as e:
                self.log(f"⚠️ Error clearing images: {e}", "WARN")
            
            # Clear annotations
            self.log("🗑️ Clearing MinIO annotations...")
            try:
                response = s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix="annotations/"
                )
                
                if 'Contents' in response:
                    annotation_objects = [obj['Key'] for obj in response['Contents']]
                    if annotation_objects:
                        # Delete all annotation objects
                        for obj_key in annotation_objects:
                            s3_client.delete_object(Bucket=self.bucket_name, Key=obj_key)
                        self.log(f"✅ Cleared {len(annotation_objects)} annotation objects")
                    else:
                        self.log("ℹ️ No annotations to clear")
                else:
                    self.log("ℹ️ No annotations folder found")
                s3_success = True
            except Exception as e:
                self.log(f"⚠️ Error clearing annotations: {e}", "WARN")
                
        except Exception as e:
            self.log(f"⚠️ Error with S3 API cleanup: {e}", "WARN")
        
        # Return success if either filesystem or S3 cleanup worked
        if filesystem_success or s3_success:
            self.log("✅ MinIO data cleared successfully!")
            return True
        else:
            self.log("⚠️ MinIO data cleanup completed with some issues")
            return False
    
    def clear_config_files(self):
        """Clear configuration files (but keep model files)"""
        self.log("🧹 Clearing configuration files...")
        
        cleared_files = []
        
        # Clear config files from config/ directory
        config_dir = "config"
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith(".json"):
                    file_path = os.path.join(config_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleared_files.append(file_path)
                        self.log(f"✅ Cleared config file: {file_path}")
        
        # Define config files to clear from root directory (but keep model files)
        config_patterns = [
            "class_config.json",
            "project_config.json",
            "label_studio_config.json"
        ]
        
        for pattern in config_patterns:
            if os.path.exists(pattern):
                os.remove(pattern)
                cleared_files.append(pattern)
                self.log(f"✅ Cleared config file: {pattern}")
        
        # Clear any other config files in the root directory
        for file in os.listdir("."):
            if file.endswith("_config.json") and "model" not in file.lower():
                file_path = os.path.join(".", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    cleared_files.append(file)
                    self.log(f"✅ Cleared config file: {file}")
        
        if not cleared_files:
            self.log("ℹ️ No config files found to clear")
        else:
            self.log(f"✅ Cleared {len(cleared_files)} config files")
        
        return cleared_files
    
    def clear_debug_files(self):
        """Clear debug and temporary files"""
        self.log("🧹 Clearing debug files...")
        
        debug_patterns = [
            "debug_*.png",
            "debug_*.jpg",
            "debug_*.jpeg",
            "test_*.py",
            "temp_*",
            "*.tmp"
        ]
        
        cleared_files = []
        for pattern in debug_patterns:
            for file in Path(".").glob(pattern):
                if file.is_file():
                    file.unlink()
                    cleared_files.append(str(file))
                    self.log(f"✅ Cleared debug file: {file}")
        
        if not cleared_files:
            self.log("ℹ️ No debug files found to clear")
        else:
            self.log(f"✅ Cleared {len(cleared_files)} debug files")
        
        return cleared_files
    
    def restart_containers(self):
        """Restart all containers"""
        self.log("🔄 Restarting containers...")
        
        try:
            # First, check if docker-compose.yml exists
            if not os.path.exists('docker-compose.yml'):
                self.log("⚠️ docker-compose.yml not found, skipping container restart", "WARN")
                return False
            
            # Start containers in order
            result = subprocess.run(
                ['docker', 'compose', 'up', '-d'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.log("✅ Containers restarted successfully")
                
                # Wait for services to be ready
                self.log("⏳ Waiting for services to be ready...")
                time.sleep(10)
                
                # Check container status
                result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.log("📊 Container status:")
                    for line in result.stdout.split('\n')[1:]:  # Skip header
                        if line.strip():
                            self.log(f"  {line}")
                else:
                    self.log("⚠️ Could not check container status", "WARN")
                
                return True
            else:
                self.log(f"❌ Error restarting containers: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("⚠️ Timeout restarting containers", "WARN")
            return False
        except Exception as e:
            self.log(f"❌ Error restarting containers: {e}", "ERROR")
            return False
    
    def verify_cleanup(self):
        """Verify that cleanup was successful"""
        self.log("🔍 Verifying cleanup...")
        
        issues = []
        
        # Check Label Studio database
        db_path = "label-studio-data/label_studio.sqlite3"
        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            if size > 1024:  # Consider database significant if larger than 1KB
                issues.append(f"Label Studio database still exists ({size:,} bytes)")
            else:
                self.log("✅ Label Studio database cleared (empty file)")
        else:
            self.log("✅ Label Studio database removed")
        
        # Check export files
        export_dir = "label-studio-data/export/"
        if os.path.exists(export_dir):
            export_files = [f for f in os.listdir(export_dir) if f.endswith('.json')]
            if export_files:
                issues.append(f"Export files still exist: {export_files}")
            else:
                self.log("✅ Export directory cleared")
        else:
            self.log("✅ Export directory removed")
        
        # Check MinIO data
        try:
            from botocore.config import Config
            
            config = Config(
                signature_version='s3v4',
                s3={
                    'addressing_style': 'path'
                }
            )
            
            s3_client = boto3.client(
                's3',
                endpoint_url=self.minio_endpoint,
                aws_access_key_id=self.minio_access_key,
                aws_secret_access_key=self.minio_secret_key,
                region_name='us-east-1',
                config=config
            )
            
            # Test connection first
            try:
                s3_client.head_bucket(Bucket=self.bucket_name)
            except Exception as e:
                self.log(f"⚠️ Cannot connect to MinIO for verification: {e}", "WARN")
                # Check filesystem instead
                minio_data_path = "minio-data/segmentation-platform"
                if os.path.exists(minio_data_path):
                    images_path = os.path.join(minio_data_path, "images")
                    annotations_path = os.path.join(minio_data_path, "annotations")
                    
                    if os.path.exists(images_path) and os.listdir(images_path):
                        issues.append("MinIO images still exist in filesystem")
                    else:
                        self.log("✅ MinIO images cleared (filesystem check)")
                    
                    if os.path.exists(annotations_path) and os.listdir(annotations_path):
                        issues.append("MinIO annotations still exist in filesystem")
                    else:
                        self.log("✅ MinIO annotations cleared (filesystem check)")
                return
            
            # Check images
            response = s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="images/"
            )
            if 'Contents' in response and len(response['Contents']) > 0:
                issues.append(f"MinIO images still exist: {len(response['Contents'])} objects")
            else:
                self.log("✅ MinIO images cleared")
            
            # Check annotations
            response = s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="annotations/"
            )
            if 'Contents' in response and len(response['Contents']) > 0:
                issues.append(f"MinIO annotations still exist: {len(response['Contents'])} objects")
            else:
                self.log("✅ MinIO annotations cleared")
                
        except Exception as e:
            self.log(f"⚠️ Could not verify MinIO cleanup: {e}", "WARN")
        
        if issues:
            self.log("❌ Cleanup verification failed:", "ERROR")
            for issue in issues:
                self.log(f"  - {issue}", "ERROR")
            return False
        else:
            self.log("✅ Cleanup verification successful!")
            return True
    
    def run_comprehensive_cleanup(self):
        """Run the complete cleanup process"""
        self.log("🧹 Starting Comprehensive System Cleanup")
        self.log("=" * 60)
        self.log("This will clear:")
        self.log("- Label Studio database and all data")
        self.log("- MinIO images and annotations")
        self.log("- Export files")
        self.log("- Configuration files")
        self.log("- Debug and temporary files")
        self.log("=" * 60)
        
        success = True
        
        # Step 1: Clear MinIO data (while containers are running)
        if not self.clear_minio_data():
            success = False
        
        # Step 2: Stop containers
        self.stop_containers()
        time.sleep(5)  # Wait for containers to stop
        
        # Step 3: Clear Label Studio database
        self.clear_labelstudio_database()
        
        # Step 4: Clear config files
        self.clear_config_files()
        
        # Step 5: Clear debug files
        self.clear_debug_files()
        
        # Step 6: Restart containers
        if not self.restart_containers():
            success = False
        
        # Step 7: Verify cleanup
        if not self.verify_cleanup():
            success = False
        
        # Final result
        if success:
            self.log("🎉 Comprehensive cleanup completed successfully!")
            self.log("✅ You now have a true clean slate for new projects")
        else:
            self.log("⚠️ Cleanup completed with some issues", "WARN")
            self.log("Please check the logs above for any problems")
        
        return success

def main():
    """Main function"""
    print("🧹 Comprehensive System Cleanup Tool")
    print("=" * 50)
    print("This will completely reset your system for a clean slate.")
    print("All project data, images, annotations, and configurations will be cleared.")
    print()
    
    response = input("Are you sure you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled")
        return
    
    cleanup = ComprehensiveCleanup()
    success = cleanup.run_comprehensive_cleanup()
    
    if success:
        print("\n🎉 Cleanup successful! You can now start fresh.")
    else:
        print("\n⚠️ Cleanup completed with issues. Please review the logs.")

if __name__ == "__main__":
    main()
