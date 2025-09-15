#!/usr/bin/env python3
"""
IMPROVED Comprehensive System Cleanup Script
Clears ALL project data for a true clean slate experience
Addresses volume-specific cleanup and ensures complete reset
"""

import os
import shutil
import subprocess
import time
import json
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

class ImprovedComprehensiveCleanup:
    def __init__(self):
        self.minio_endpoint = "http://localhost:9000"
        self.minio_access_key = "minioadmin"
        self.minio_secret_key = "minioadmin123"
        self.bucket_name = "segmentation-platform"
        self.project_name = "semsegplat-full_local_version"
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def stop_containers(self):
        """Stop all relevant containers"""
        self.log("🔄 Stopping containers...")
        
        try:
            result = subprocess.run(
                ['docker', 'compose', 'down'],
                capture_output=True, 
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.log("✅ Containers stopped successfully")
            else:
                self.log(f"⚠️ Warning stopping containers: {result.stderr}", "WARN")
        except Exception as e:
            self.log(f"❌ Error stopping containers: {e}", "ERROR")
    
    def clear_docker_volumes(self):
        """Clear Docker volumes completely"""
        self.log("🗑️ Clearing Docker volumes...")
        
        try:
            # Remove the specific volumes for this project
            volume_names = [
                f"{self.project_name}_label-studio-data",
                f"{self.project_name}_minio-data"
            ]
            
            for volume_name in volume_names:
                try:
                    result = subprocess.run(
                        ['docker', 'volume', 'rm', volume_name],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        self.log(f"✅ Removed volume: {volume_name}")
                    else:
                        self.log(f"⚠️ Could not remove volume {volume_name}: {result.stderr}", "WARN")
                except Exception as e:
                    self.log(f"⚠️ Error removing volume {volume_name}: {e}", "WARN")
            
            # Recreate the volumes (they'll be recreated when containers start)
            self.log("✅ Docker volumes cleared")
            return True
            
        except Exception as e:
            self.log(f"❌ Error clearing Docker volumes: {e}", "ERROR")
            return False
    
    def clear_labelstudio_data(self):
        """Clear Label Studio data completely"""
        self.log("🧹 Clearing Label Studio data...")
        
        if os.path.exists("label-studio-data"):
            try:
                # Remove the entire directory
                shutil.rmtree("label-studio-data")
                self.log("✅ Removed label-studio-data directory")
                
                # Recreate the directory structure
                os.makedirs("label-studio-data/export", exist_ok=True)
                os.makedirs("label-studio-data/media", exist_ok=True)
                os.makedirs("label-studio-data/backup", exist_ok=True)
                self.log("✅ Recreated clean label-studio-data structure")
                
            except Exception as e:
                self.log(f"❌ Error clearing Label Studio data: {e}", "ERROR")
                return False
        else:
            # Create the directory structure if it doesn't exist
            os.makedirs("label-studio-data/export", exist_ok=True)
            os.makedirs("label-studio-data/media", exist_ok=True)
            os.makedirs("label-studio-data/backup", exist_ok=True)
            self.log("✅ Created clean label-studio-data structure")
        
        return True
    
    def clear_minio_data(self):
        """Clear MinIO data completely"""
        self.log("🗑️ Clearing MinIO data...")
        
        if os.path.exists("minio-data"):
            try:
                # Remove the entire directory
                shutil.rmtree("minio-data")
                self.log("✅ Removed minio-data directory")
                
                # Recreate the directory structure
                os.makedirs("minio-data/segmentation-platform/images", exist_ok=True)
                os.makedirs("minio-data/segmentation-platform/annotations", exist_ok=True)
                self.log("✅ Recreated clean minio-data structure")
                
            except Exception as e:
                self.log(f"❌ Error clearing MinIO data: {e}", "ERROR")
                return False
        else:
            # Create the directory structure if it doesn't exist
            os.makedirs("minio-data/segmentation-platform/images", exist_ok=True)
            os.makedirs("minio-data/segmentation-platform/annotations", exist_ok=True)
            self.log("✅ Created clean minio-data structure")
        
        return True
    
    def clear_model_data(self, clear_checkpoints=False):
        """Clear model data (optional)"""
        self.log("🤖 Clearing model data...")
        
        if clear_checkpoints:
            # Clear model checkpoints
            checkpoints_dir = "models/checkpoints"
            if os.path.exists(checkpoints_dir):
                try:
                    shutil.rmtree(checkpoints_dir)
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    self.log("✅ Cleared model checkpoints")
                except Exception as e:
                    self.log(f"⚠️ Error clearing checkpoints: {e}", "WARN")
            
            # Clear saved models
            saved_models_dir = "models/saved_models"
            if os.path.exists(saved_models_dir):
                try:
                    shutil.rmtree(saved_models_dir)
                    os.makedirs(saved_models_dir, exist_ok=True)
                    self.log("✅ Cleared saved models")
                except Exception as e:
                    self.log(f"⚠️ Error clearing saved models: {e}", "WARN")
        else:
            self.log("ℹ️ Preserving model checkpoints (use --clear-models to remove them)")
    
    def clear_config_files(self):
        """Clear configuration files"""
        self.log("🧹 Clearing configuration files...")
        
        cleared_files = []
        
        # Clear local config files
        config_patterns = [
            "*_config.json",
            "class_config.json",
            "project_config.json",
            "label_studio_config.json"
        ]
        
        for pattern in config_patterns:
            for file in Path(".").glob(pattern):
                if file.is_file() and "model" not in str(file).lower():
                    try:
                        file.unlink()
                        cleared_files.append(str(file))
                        self.log(f"✅ Cleared config file: {file}")
                    except Exception as e:
                        self.log(f"⚠️ Could not clear {file}: {e}", "WARN")
        
        # Clear test files
        test_files = ["test_annotation.json"]
        for test_file in test_files:
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                    cleared_files.append(test_file)
                    self.log(f"✅ Cleared test file: {test_file}")
                except Exception as e:
                    self.log(f"⚠️ Could not clear {test_file}: {e}", "WARN")
        
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
            "temp_*",
            "*.tmp",
            ".cleanup_completed"
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
            if not os.path.exists('docker-compose.yml'):
                self.log("⚠️ docker-compose.yml not found, skipping container restart", "WARN")
                return False
            
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
                time.sleep(15)
                
                return True
            else:
                self.log(f"❌ Error restarting containers: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Error restarting containers: {e}", "ERROR")
            return False
    
    def verify_cleanup(self):
        """Verify that cleanup was successful"""
        self.log("🔍 Verifying cleanup...")
        
        issues = []
        
        # Check Label Studio data - verify database is empty, not just that file exists
        if os.path.exists("label-studio-data/label_studio.sqlite3"):
            try:
                import sqlite3
                conn = sqlite3.connect("label-studio-data/label_studio.sqlite3")
                cursor = conn.cursor()
                
                # Check if there are any tasks or completions
                cursor.execute("SELECT COUNT(*) FROM task")
                task_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM task_completion")
                completion_count = cursor.fetchone()[0]
                
                conn.close()
                
                if task_count > 0 or completion_count > 0:
                    issues.append(f"Label Studio database still contains data: {task_count} tasks, {completion_count} completions")
                else:
                    self.log("✅ Label Studio database is empty (as expected after container restart)")
            except Exception as e:
                self.log(f"⚠️ Could not verify Label Studio database: {e}", "WARN")
        else:
            self.log("✅ Label Studio database file not found (will be created by container)")
        
        # Check MinIO data
        minio_images = "minio-data/segmentation-platform/images"
        minio_annotations = "minio-data/segmentation-platform/annotations"
        
        if os.path.exists(minio_images) and os.listdir(minio_images):
            issues.append("MinIO images still exist")
        else:
            self.log("✅ MinIO images cleared")
        
        if os.path.exists(minio_annotations) and os.listdir(minio_annotations):
            issues.append("MinIO annotations still exist")
        else:
            self.log("✅ MinIO annotations cleared")
        
        # Check for any remaining data files
        data_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if (file.endswith(('.png', '.jpg', '.jpeg')) and 
                    'minio-data' not in root and 'app' not in root):
                    data_files.append(os.path.join(root, file))
        
        if data_files:
            issues.append(f"Found {len(data_files)} data files outside expected locations")
        else:
            self.log("✅ No unexpected data files found")
        
        if issues:
            self.log("❌ Cleanup verification failed:", "ERROR")
            for issue in issues:
                self.log(f"  - {issue}", "ERROR")
            return False
        else:
            self.log("✅ Cleanup verification successful!")
            return True
    
    def run_comprehensive_cleanup(self, clear_models=False):
        """Run the complete cleanup process"""
        self.log("🧹 Starting IMPROVED Comprehensive System Cleanup")
        self.log("=" * 60)
        self.log("This will clear:")
        self.log("- Docker volumes completely")
        self.log("- Label Studio database and all data")
        self.log("- MinIO images and annotations")
        self.log("- All configuration files")
        self.log("- Debug and temporary files")
        if clear_models:
            self.log("- Model checkpoints and saved models")
        else:
            self.log("- Model checkpoints (preserved)")
        self.log("=" * 60)
        
        success = True
        
        # Step 1: Stop containers
        self.stop_containers()
        time.sleep(5)
        
        # Step 2: Clear Docker volumes
        if not self.clear_docker_volumes():
            success = False
        
        # Step 3: Clear Label Studio data
        if not self.clear_labelstudio_data():
            success = False
        
        # Step 4: Clear MinIO data
        if not self.clear_minio_data():
            success = False
        
        # Step 5: Clear model data (optional)
        self.clear_model_data(clear_models)
        
        # Step 6: Clear config files
        self.clear_config_files()
        
        # Step 7: Clear debug files
        self.clear_debug_files()
        
        # Step 8: Restart containers
        if not self.restart_containers():
            success = False
        
        # Step 9: Verify cleanup
        if not self.verify_cleanup():
            success = False
        
        # Final result
        if success:
            self.log("🎉 IMPROVED cleanup completed successfully!")
            self.log("✅ You now have a TRUE clean slate for new projects")
            self.log("🔄 All Docker volumes, data, and configurations have been reset")
        else:
            self.log("⚠️ Cleanup completed with some issues", "WARN")
            self.log("Please check the logs above for any problems")
        
        return success

def main():
    """Main function"""
    print("🧹 IMPROVED Comprehensive System Cleanup Tool")
    print("=" * 60)
    print("This will completely reset your system for a clean slate.")
    print("All project data, images, annotations, configurations, and Docker volumes will be cleared.")
    print()
    
    # Ask about model clearing
    clear_models = input("Do you want to clear model checkpoints too? (y/N): ").lower() == 'y'
    
    response = input("Are you sure you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled")
        return
    
    cleanup = ImprovedComprehensiveCleanup()
    success = cleanup.run_comprehensive_cleanup(clear_models)
    
    if success:
        print("\n🎉 Cleanup successful! You can now start completely fresh.")
    else:
        print("\n⚠️ Cleanup completed with issues. Please review the logs.")

if __name__ == "__main__":
    main()
