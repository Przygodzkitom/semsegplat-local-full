import streamlit as st
import psutil
import torch
import os
import gc
import numpy as np
import cv2
import json
import time
from pathlib import Path
from models.config import ModelConfig
from models.inferencer import Inferencer
from models.utils.gpu_detector import detect_gpu, print_device_info
from app.storage_manager import get_storage_manager
from app.label_studio.config import create_label_studio_project, sync_images_to_label_studio, get_project_images
from app.image_utils import process_uploaded_images, get_supported_formats, format_file_size, estimate_conversion_time

# Configure resource limits and memory management
def configure_resource_limits():
    try:
        # Monitor system memory for CPU execution
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            st.warning("System memory usage is very high (>90%). Performance may be affected.")
        elif memory.percent > 80:
            st.warning("System memory usage is high (>80%). Consider closing other applications.")
        
        st.sidebar.info(f"System Memory: {memory.percent}% used ({memory.available / 1024**3:.1f}GB available)")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        st.sidebar.info(f"CPU Cores: {cpu_count}")
        
        # Detect and display GPU information
        gpu_config = detect_gpu()
        
        if gpu_config['available']:
            st.sidebar.success(f"🚀 GPU Available: {gpu_config['name']}")
            st.sidebar.info(f"GPU Memory: {gpu_config['memory_gb']:.1f} GB")
            st.sidebar.info(f"CUDA Version: {gpu_config['cuda_version']}")
        else:
            st.sidebar.info("ℹ️ Running on CPU - Training and inference will be slower but functional")
        
    except Exception as e:
        st.error(f"Error configuring resource limits: {str(e)}")
        return False
    return True

def save_project_config(project_id, project_name, project_description=""):
    """Save project configuration to persistent storage"""
    try:
        # Try to save to the project directory (mounted volume)
        config_dir = Path("/app/project/config")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "label_studio_project.json"
        config_data = {
            "project_id": project_id,
            "project_name": project_name,
            "project_description": project_description,
            "created_at": str(Path().cwd()),
            "last_updated": str(Path().cwd())
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        st.success(f"✅ Project configuration saved")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save project configuration: {str(e)}")
        return False

def load_project_config():
    """Load project configuration from persistent storage"""
    try:
        # Try to load from the project directory (mounted volume)
        config_file = Path("/app/project/config") / "label_studio_project.json"
        
        # Fallback to local config directory if mounted volume doesn't exist
        if not config_file.exists():
            config_file = Path("config") / "label_studio_project.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Restore session state
            st.session_state.label_studio_project_id = config_data.get("project_id")
            st.session_state.label_studio_project_name = config_data.get("project_name")
            st.session_state.label_studio_project_description = config_data.get("project_description")
            
            st.info(f"✅ Loaded existing project: {config_data.get('project_name')} (ID: {config_data.get('project_id')})")
            return config_data
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ Could not load project configuration: {str(e)}")
        return None

# Initialize application with resource management
if not configure_resource_limits():
    st.error("Failed to configure resource limits. Please restart the application.")
    st.stop()

# Import basic libraries first
import cv2

# Try importing ML libraries
try:
    import torch
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    st.error(f"Error importing ML libraries: {str(e)}")

# Try importing local modules
try:
    from models.config import ModelConfig
    from models.inferencer import Inferencer
except Exception as e:
    st.error(f"Error importing local modules: {str(e)}")

st.title("Semantic Segmentation Platform")

def ensure_minio_folders():
    """Ensure required MinIO folders exist at startup"""
    st.write("🔍 Starting MinIO folder check...")
    
    # Check if we've already done this in this session
    if st.session_state.get('minio_folders_created', False):
        st.info("✅ MinIO folders already verified in this session")
        return True
        
    try:
        st.write("📦 Importing boto3...")
        import boto3
        from botocore.exceptions import ClientError
        st.write("✅ boto3 imported successfully")
        
        # Initialize MinIO client
        st.write("🔌 Initializing MinIO client...")
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin123',
            region_name='us-east-1'
        )
        st.write("✅ MinIO client initialized")
        
        # Ensure bucket exists
        bucket_created = False
        st.write("🪣 Checking bucket existence...")
        try:
            s3_client.head_bucket(Bucket="segmentation-platform")
            st.write("✅ Bucket exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                st.write("📦 Creating bucket...")
                s3_client.create_bucket(Bucket="segmentation-platform")
                bucket_created = True
                st.write("✅ Bucket created")
            else:
                st.error(f"❌ Bucket error: {str(e)}")
                return False
        
        # Ensure images folder exists
        st.write("📁 Creating images/ folder...")
        try:
            s3_client.put_object(
                Bucket="segmentation-platform",
                Key="images/"
            )
            st.write("✅ images/ folder created")
        except Exception as e:
            st.error(f"❌ Could not create images/ folder: {str(e)}")
            return False
        
                    # Ensure annotations folder exists
            st.write("📁 Creating annotations/ folder...")
            try:
                s3_client.put_object(
                    Bucket="segmentation-platform",
                    Key="annotations/"
                )
                st.write("✅ annotations/ folder created")
            except Exception as e:
                st.error(f"❌ Could not create annotations/ folder: {str(e)}")
                return False
        
        # Mark as completed
        st.session_state.minio_folders_created = True
        st.write("✅ Session state updated")
        
        if bucket_created:
            st.success("✅ MinIO bucket and folders created successfully!")
        else:
            st.success("✅ MinIO folders verified successfully!")
            
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to ensure MinIO folders: {str(e)}")
        st.write(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return False

def _clean_minio_storage():
    """Clear all data from MinIO storage"""
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin123',
            region_name='us-east-1'
        )
        
        bucket_name = 'segmentation-platform'
        
        # List all objects in the bucket
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            if 'Contents' in response:
                objects = [obj['Key'] for obj in response['Contents']]
                
                # Delete all objects
                for obj_key in objects:
                    s3_client.delete_object(Bucket=bucket_name, Key=obj_key)
                
                st.success(f"✅ Deleted {len(objects)} objects from MinIO")
            else:
                st.info("ℹ️ MinIO bucket is already empty")
                
        except Exception as e:
            st.warning(f"⚠️ Could not clear MinIO: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ Error clearing MinIO storage: {str(e)}")

def _clean_label_studio_database():
    """Clear Label Studio SQLite database and restart container"""
    try:
        st.info("🔄 Stopping Label Studio container...")
        result = subprocess.run(
            ["docker", "compose", "stop", "label-studio"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Label Studio container stopped")
        else:
            st.warning(f"⚠️ Could not stop container: {result.stderr}")
        
        # Delete the database file
        db_path = "label-studio-data/label_studio.sqlite3"
        if os.path.exists(db_path):
            os.remove(db_path)
            st.success("✅ Label Studio database cleared")
        else:
            st.info("ℹ️ Label Studio database file not found")
        
        # Restart Label Studio container
        st.info("🔄 Restarting Label Studio container...")
        result = subprocess.run(
            ["docker", "compose", "start", "label-studio"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Label Studio container restarted")
        else:
            st.warning(f"⚠️ Could not restart container: {result.stderr}")
            
    except Exception as e:
        st.error(f"❌ Error clearing Label Studio database: {str(e)}")

def _clean_project_config():
    """Clear saved project configuration"""
    try:
        config_paths = [
            "config/label_studio_project.json",
            "/app/project/config/label_studio_project.json"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                os.remove(config_path)
                st.success(f"✅ Project configuration cleared: {config_path}")
                
    except Exception as e:
        st.error(f"❌ Error clearing project configuration: {str(e)}")

def _clear_session_state():
    """Clear all Streamlit session state"""
    try:
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success("✅ Session state cleared")
        
    except Exception as e:
        st.error(f"❌ Error clearing session state: {str(e)}")

def _restart_all_containers():
    """Restart all containers to ensure complete clean state"""
    try:
        st.info("🔄 Restarting all containers for complete clean state...")
        
        # Stop all containers
        result = subprocess.run(
            ["docker", "compose", "stop"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            st.success("✅ All containers stopped")
        else:
            st.warning(f"⚠️ Could not stop containers: {result.stderr}")
        
        # Start all containers
        result = subprocess.run(
            ["docker", "compose", "up", "-d"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            st.success("✅ All containers started")
        else:
            st.warning(f"⚠️ Could not start containers: {result.stderr}")
            
    except Exception as e:
        st.error(f"❌ Error restarting containers: {str(e)}")

def load_existing_images(bucket_name=None):
    """Load existing images from storage on startup"""
    storage_manager = get_storage_manager(bucket_name=bucket_name)
    if storage_manager:
        try:
            # Only look in the images/ folder, not the entire bucket
            images = storage_manager.list_images(prefix="images/")
            if images:
                st.session_state.existing_images = images
                st.session_state.total_images = len(images)
                return images
        except Exception as e:
            st.warning(f"Could not load existing images: {str(e)}")
    return []

def main():
    
    # Load existing project configuration on startup
    if 'label_studio_project_id' not in st.session_state:
        load_project_config()
    
    # Initialize MinIO manager and load existing images
    if 'existing_images' not in st.session_state:
        st.session_state.existing_images = load_existing_images("segmentation-platform")
    
        # ALWAYS SHOW PROJECT STATUS - AFTER PROJECT IS LOADED
    st.markdown("### 📊 Project Status (Always Visible)")
    st.write("🔍 DEBUG: Project Status section is being rendered!")
    st.write("🔍 DEBUG: This should definitely be visible!")
    
    # Show session state info
    if 'label_studio_project_id' in st.session_state:
        project_id = st.session_state.label_studio_project_id
        st.info(f"🔍 Debug: Project ID found: {project_id}")
        st.success(f"✅ Active Project: {st.session_state.get('label_studio_project_name', 'Unknown')} (ID: {project_id})")
        
        # Annotation Type Configuration
        st.info("📋 Configure your annotation type directly in Label Studio:")
        st.info("1. Go to Project Settings → Labeling Interface")
        st.info("2. Choose your preferred annotation type (Brush, Polygon, Rectangle, Circle)")
        st.info("3. The training system will automatically handle your choice")
    else:
        st.info("🔍 Debug: No project ID in session state")
    
    # Annotation type is configured directly in Label Studio
    # No need for complex detection or display logic
    
    # Sidebar - always show
    with st.sidebar:
        # MinIO Status
        st.subheader("🗄️ MinIO Status")
        if st.button("🔄 Check MinIO Status", use_container_width=True):
            ensure_minio_folders()
        
        # Show current status
        if st.session_state.get('minio_folders_created', False):
            st.success("✅ MinIO folders ready")
        else:
            st.warning("⚠️ MinIO folders not verified")
        
        # Maintenance
        st.subheader("🧹 Maintenance")
        
        if st.button("🧹 Clean Temp", use_container_width=True):
            with st.spinner("Cleaning temporary files..."):
                try:
                    import shutil
                    from pathlib import Path
                    
                    # Clean temp directory
                    if Path("temp").exists():
                        shutil.rmtree("temp")
                        Path("temp").mkdir()
                    
                    # Clean logs (keep last 7 days)
                    if Path("logs").exists():
                        current_time = time.time()
                        for log_file in Path("logs").glob("*.log"):
                            if current_time - log_file.stat().st_mtime > 7 * 24 * 3600:  # 7 days
                                log_file.unlink()
                    
                    st.success("✅ Temporary files cleaned!")
                    
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {str(e)}")
        
        # Quick Reset for Testing
        st.subheader("🧪 Quick Reset (Testing)")
        st.info("Quick reset for testing - removes project config and session state:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Quick Reset - Project Only", type="secondary", use_container_width=True):
                try:
                    _clean_project_config()
                    _clear_session_state()
                    st.success("✅ Quick reset completed!")
                    st.info("🔄 Refreshing page...")
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Quick reset failed: {str(e)}")
        
        # Container management
        st.subheader("🐳 Container Management")
        st.info("Manage Docker containers for testing:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏹️ Stop All", type="secondary", use_container_width=True):
                try:
                    result = subprocess.run(["docker", "compose", "stop"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ All containers stopped")
                    else:
                        st.error(f"❌ Error stopping containers: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("▶️ Start All", type="secondary", use_container_width=True):
                try:
                    result = subprocess.run(["docker", "compose", "up", "-d"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ All containers started")
                    else:
                        st.error(f"❌ Error starting containers: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button("🔄 Restart All", type="secondary", use_container_width=True):
                try:
                    _restart_all_containers()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Clean Slate - Reset Everything
        st.subheader("🔄 Clean Slate (Reset Everything)")
        st.warning("⚠️ **DANGER ZONE**: This will completely reset your entire system!")
        st.info("Use this to start fresh for testing. This will delete:")
        st.info("• All images in MinIO")
        st.info("• All annotations in MinIO") 
        st.info("• Label Studio database (all projects)")
        st.info("• Saved project configuration")
        st.info("• Current session state")
        
        # Use session state to track confirmation
        if 'reset_confirmed' not in st.session_state:
            st.session_state.reset_confirmed = False
        
        if not st.session_state.reset_confirmed:
            if st.button("🗑️ RESET EVERYTHING - Clean Slate", type="secondary", use_container_width=True):
                st.session_state.reset_confirmed = True
                st.experimental_rerun()
        else:
            st.warning("⚠️ **FINAL WARNING**: This will delete ALL your data!")
            st.info("Click the button below to confirm and proceed with the reset.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ YES, RESET EVERYTHING", type="primary", use_container_width=True):
                    with st.spinner("🧹 Performing complete system reset..."):
                        try:
                            # Step 1: Clear MinIO storage
                            st.info("🗑️ Clearing MinIO storage...")
                            _clean_minio_storage()
                            
                            # Step 2: Clear Label Studio database and restart container
                            st.info("🗑️ Clearing Label Studio database...")
                            _clean_label_studio_database()
                            
                            # Step 3: Clear project configuration
                            st.info("🗑️ Clearing project configuration...")
                            _clean_project_config()
                            
                            # Step 4: Clear session state
                            st.info("🗑️ Clearing session state...")
                            _clear_session_state()
                            
                            # Step 5: Restart all containers for complete clean state
                            st.info("🔄 Restarting all containers...")
                            _restart_all_containers()
                            
                            st.success("✅ Complete system reset successful!")
                            st.info("🔄 Refreshing page in 5 seconds...")
                            time.sleep(5)
                            st.experimental_rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Reset failed: {str(e)}")
                            st.info("🔍 Check the error details above")
            
            with col2:
                if st.button("❌ CANCEL RESET", type="secondary", use_container_width=True):
                    st.session_state.reset_confirmed = False
                    st.experimental_rerun()
        
        st.divider()
        
        # Navigation
        if "current_step" not in st.session_state:
            st.session_state.current_step = "upload"

        steps = {
            "upload": "1. Upload Images",
            "annotate": "2. Annotate Images",
            "train": "3. Train Model",
            "inference": "4. Run Inference"
        }

        for step, label in steps.items():
            if st.button(label):
                st.session_state.current_step = step

    # Main content area
    if st.session_state.current_step == "upload":
        st.header("Upload Images")
        
        # Simple single-project approach - always upload to images/ folder
        selected_upload_project = "main"
        image_prefix = "images/"
        st.info(f"📁 Uploading to main project folder: {image_prefix}")
        
        # Load existing images from main images folder
        def load_project_images():
            try:
                import boto3
                s3_client = boto3.client(
                    's3',
                    endpoint_url='http://minio:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin123',
                    region_name='us-east-1'
                )
                
                # Always look in the main images/ folder
                response = s3_client.list_objects_v2(Bucket="segmentation-platform", Prefix="images/")
                images = []
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if not obj['Key'].endswith('/') and obj['Size'] > 0:
                            images.append(obj['Key'])
                
                return images
            except Exception as e:
                st.error(f"Error loading images: {str(e)}")
                return []
        
        # Load and show existing images
        existing_project_images = load_project_images()
        if existing_project_images:
            st.info(f"📁 Found {len(existing_project_images)} existing images")
        else:
            st.info(f"📁 No existing images found")
        
        # Enhanced file uploader with format detection and conversion
        st.subheader("📁 Image Upload & Format Conversion")
        
        # Show supported formats
        supported_formats = get_supported_formats()
        st.info(f"🖼️ Supported formats: {', '.join(supported_formats).upper()}")
        st.info("🔄 Non-PNG images will be automatically converted to PNG format for Label Studio compatibility")
        
        # File uploader with expanded format support
        uploaded_files = st.file_uploader(
            "Choose images", 
            accept_multiple_files=True, 
            type=supported_formats,
            help="Upload images in any supported format. They will be converted to PNG if needed."
        )
        
        # Add warning for large batches
        if uploaded_files and len(uploaded_files) > 50:
            st.warning(f"⚠️ **Large batch detected:** {len(uploaded_files)} files. Processing may take several minutes. Consider processing in smaller batches for better performance.")
        
        # Check if we already processed these files to prevent infinite loop
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            st.info(f"✅ Previously uploaded: {len(st.session_state.uploaded_files)} files")
            if st.button("Clear upload history and upload new files"):
                st.session_state.uploaded_files = []
                st.rerun()
        
        elif uploaded_files:
            # Process images: detect format and convert to PNG if needed
            st.subheader("🔄 Processing Images")
            
            # Show file information
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📊 **Total files:** {len(uploaded_files)}")
                total_size = sum(file.size for file in uploaded_files)
                st.write(f"📦 **Total size:** {format_file_size(total_size)}")
                
                # Estimate conversion time
                total_size_mb = total_size / (1024 * 1024)  # Convert to MB
                estimated_time = estimate_conversion_time(len(uploaded_files), total_size_mb)
                st.write(f"⏱️ **Est. conversion time:** {estimated_time}")
            
            with col2:
                st.write(f"🖼️ **Format summary:**")
                # Count formats to show summary instead of listing all files
                format_counts = {}
                
                # Show progress for format detection
                if len(uploaded_files) > 10:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    if len(uploaded_files) > 10:
                        progress_text.text(f"Analyzing format of {file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    file.seek(0)
                    try:
                        from PIL import Image
                        with Image.open(file) as img:
                            format_name = img.format.upper() if img.format else 'UNKNOWN'
                            format_counts[format_name] = format_counts.get(format_name, 0) + 1
                    except:
                        format_counts['UNKNOWN'] = format_counts.get('UNKNOWN', 0) + 1
                    finally:
                        file.seek(0)
                
                # Clear progress indicators
                if len(uploaded_files) > 10:
                    progress_text.empty()
                    progress_bar.empty()
                
                # Show format summary
                for format_name, count in format_counts.items():
                    st.write(f"• {format_name}: {count} files")
            
            # Process images for format conversion
            if st.button("🔄 Process & Convert Images", type="primary"):
                try:
                    # Process uploaded images (detect format and convert if needed)
                    processed_files = process_uploaded_images(uploaded_files)
                    
                    # Show conversion summary
                    converted_count = sum(1 for _, _, _, was_converted in processed_files if was_converted)
                    st.success(f"✅ Processing complete! {converted_count} images converted to PNG")
                    
                    # Show detailed conversion summary
                    if converted_count > 0:
                        st.subheader("📊 Conversion Summary")
                        conversion_details = []
                        for _, filename, original_format, was_converted in processed_files:
                            if was_converted:
                                conversion_details.append(f"• {filename}: {original_format} → PNG")
                        
                        for detail in conversion_details:
                            st.write(detail)
                    
                    # Upload to project-specific location
                    import boto3
                    s3_client = boto3.client(
                        's3',
                        endpoint_url='http://minio:9000',
                        aws_access_key_id='minioadmin',
                        aws_secret_access_key='minioadmin123',
                        region_name='us-east-1'
                    )
                    
                    # Upload progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    uploaded_to_storage = []
                    for i, (file_data, filename, original_format, was_converted) in enumerate(processed_files):
                        status_text.text(f"Uploading {filename}...")
                        
                        try:
                            # Reset file pointer
                            file_data.seek(0)
                            
                            # Create unique filename with timestamp
                            timestamp = int(time.time() * 1000)
                            final_filename = f"{timestamp}_{filename}"
                            
                            # Show upload info
                            if was_converted:
                                st.info(f"🔄 Uploading converted image: {filename} (was {original_format})")
                            else:
                                st.info(f"📤 Uploading original image: {filename} ({original_format})")
                            
                            # Upload to root bucket
                            s3_client.upload_fileobj(
                                file_data,
                                "segmentation-platform",
                                f"{image_prefix}{final_filename}"
                            )
                            
                            # Debug: Verify upload succeeded
                            st.success(f"✅ Upload completed for: {final_filename}")
                            
                            uploaded_to_storage.append(f"{image_prefix}{final_filename}")
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(processed_files))
                            
                        except Exception as e:
                            st.error(f"Failed to upload {filename}: {str(e)}")
                    
                    status_text.text("Upload complete!")
                    st.success(f"✅ Successfully uploaded {len(uploaded_to_storage)} images to project '{selected_upload_project}'")
                    
                    # Debug: Verify files actually exist in MinIO
                    st.info("🔍 Verifying uploads in MinIO...")
                    try:
                        response = s3_client.list_objects_v2(Bucket="segmentation-platform", Prefix="images/")
                        if 'Contents' in response:
                            actual_files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
                            st.info(f"📁 Found {len(actual_files)} files in MinIO: {actual_files}")
                        else:
                            st.warning("⚠️ No files found in MinIO after upload!")
                    except Exception as e:
                        st.error(f"❌ Error verifying uploads: {str(e)}")
                    
                    # Clear the uploaded files to prevent re-upload
                    st.session_state.uploaded_files = uploaded_to_storage
                    
                except Exception as e:
                    st.error(f"❌ Processing/Upload failed: {str(e)}")

    elif st.session_state.current_step == "annotate":
        st.header("Annotate Images")
        
        # Check if project is already configured
        if 'label_studio_project_id' in st.session_state and st.session_state.label_studio_project_id:
            project_id = st.session_state.label_studio_project_id
            project_name = st.session_state.get('label_studio_project_name', 'Unknown')
            
            st.success(f"✅ Project Already Configured: {project_name} (ID: {project_id})")
            
            # Show project actions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f'<a href="http://localhost:8080/projects/{project_id}/data" target="_blank" style="font-size:1.2em;font-weight:bold;">'
                    '🌐 Open Project in Label Studio</a>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<a href="http://localhost:8080/projects/{project_id}/settings" target="_blank" style="font-size:1.1em;">'
                    '⚙️ Project Settings</a>',
                    unsafe_allow_html=True
                )
            
            st.divider()
            
            # Reset project configuration option
            st.subheader("🔄 Reset Project Configuration")
            st.warning("⚠️ This will clear the current project configuration and allow you to set up a new project.")
            
            if st.button("🗑️ Reset and Start Fresh", type="secondary", use_container_width=True):
                try:
                    # Remove config file using full pathlib import
                    import pathlib
                    config_file = pathlib.Path("/app/project/config") / "label_studio_project.json"
                    if config_file.exists():
                        config_file.unlink()
                        st.success("✅ Configuration file removed")
                    
                    # Clear session state
                    if 'label_studio_project_id' in st.session_state:
                        del st.session_state.label_studio_project_id
                    if 'label_studio_project_name' in st.session_state:
                        del st.session_state.label_studio_project_name
                    if 'label_studio_project_description' in st.session_state:
                        del st.session_state.label_studio_project_description
                    
                    st.success("✅ Project configuration reset! You can now set up a new project.")
                    # Use the correct method for older Streamlit versions
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for older Streamlit versions
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"❌ Failed to reset configuration: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}: {str(e)}")
            
            st.divider()
            st.info("🎯 Your project is ready for annotation! Use the links above to access Label Studio.")
            return
        
        # Label Studio Setup Section (only shown if no project is configured)
        st.subheader("🏷️ Label Studio Setup")
        
        # Automatic Configuration Section
        st.markdown("### 🚀 Automatic Configuration")
        st.info("Use the automatic setup to create and configure Label Studio projects with MinIO storage!")
        
        # Project configuration - fixed values for simplicity
        project_name = "semantic-segmentation"
        project_description = "Automated semantic segmentation project with MinIO storage"
        

        
        # Token input section
        st.markdown("### 🔑 Authentication")
        st.info("To automatically configure Label Studio, you need to provide your personal access token.")
        
        # Instructions
        with st.expander("📋 How to get your token", expanded=False):
            st.markdown("""
            **1️⃣ Open Label Studio:**
            - 🌐 Go to: http://localhost:8080
            - 👤 Login: admin@example.com / admin
            
            **2️⃣ Generate Token:**
            - ⚙️ Click your username in top right → Account Settings
            - 🔑 Look for "Access Tokens" or "API Tokens"
            - ➕ Create new token with read/write permissions
            - 📋 Copy the generated token
            
            **3️⃣ Paste Token Below:**
            - 🔑 Paste your token in the field below
            - 🚀 Click Auto-Setup to configure everything automatically
            """)
        
        # Token input field
        personal_access_token = st.text_input(
            "Personal Access Token:",
            type="password",
            placeholder="Paste your Label Studio personal access token here",
            help="This token will be used to automatically configure your Label Studio project"
        )
        
        # Auto-setup button (only enabled if token is provided)
        if personal_access_token:
            if st.button("Auto-Setup Label Studio Project", type="primary", use_container_width=True):
                try:
                    from app.label_studio.auto_config import LabelStudioAutoConfig
                    
                    # Initialize auto-config with the provided token
                    auto_config = LabelStudioAutoConfig(base_url="http://label-studio:8080")
                    auto_config.personal_access_token = personal_access_token
                    
                    # Run automatic setup
                    with st.spinner("Setting up Label Studio project automatically..."):
                        project_id = auto_config.auto_setup_project(project_name, project_description)
                        
                        if project_id:
                            st.session_state.label_studio_project_id = project_id
                            st.session_state.label_studio_project_name = project_name
                            
                            # Save project configuration to persistent storage
                            save_project_config(project_id, project_name, project_description)
                            
                            st.success(f"Project setup complete! Project ID: {project_id}")
                            
                            # Show next steps
                            st.markdown("### Next Steps")
                            st.markdown("""
                            **1. Access Your Project:**
                            - Open: http://localhost:8080
                            - Login: admin@example.com
                            - Password: admin
                            - Your project should be visible in the projects list
                            
                            **2. Customize Classes (Optional):**
                            - Go to: Project Settings → Labeling Interface
                            - Modify the label configuration to add your specific classes
                            - Customize colors and labels as needed
                            - Save changes
                            
                            **3. Start Annotating:**
                            - Click on your project to open it
                            - Use the brush tool to paint over objects
                            - Assign labels to segmented regions
                            - Annotations are automatically saved to MinIO
                            
                            **4. Export Annotations:**
                            - Go to: Export → Export Annotations
                            - Choose format: JSON or COCO
                            - Annotations saved to: MinIO/annotations/
                            """)
                            
                            # Direct link to project
                            st.markdown(
                                f'<a href="http://localhost:8080/projects/{project_id}/data" target="_blank" style="font-size:1.2em;font-weight:bold;">'
                                'Open Project in Label Studio</a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("Automatic setup failed. Check the logs above for details.")
                            
                except Exception as e:
                    st.error(f"Error during automatic setup: {str(e)}")
                    st.info("Make sure Label Studio is running and accessible at http://localhost:8080")
        else:
            st.warning("Please provide your personal access token to enable automatic setup.")
            st.info("Follow the instructions above to get your token from Label Studio.")
        
        # Project Status Section
        st.markdown("### 📊 Project Status")
        st.write("🔍 DEBUG: Project Status section is being rendered!")
        st.write("🔍 DEBUG: This should definitely be visible!")
        st.error("🔴 ERROR: This error message should definitely show!")
        st.success("🟢 SUCCESS: This success message should definitely show!")
        
        # Debug: Always show session state info
        st.info(f"🔍 Debug: Current session state keys: {list(st.session_state.keys())}")
        if 'label_studio_project_id' in st.session_state:
            st.info(f"🔍 Debug: Project ID found: {st.session_state.label_studio_project_id}")
        else:
            st.info("🔍 Debug: No project ID in session state")
        
        # Check if we have a project ID in session state
        if 'label_studio_project_id' in st.session_state:
            project_id = st.session_state.label_studio_project_id
            project_name = st.session_state.get('label_studio_project_name', 'Unknown')
            
            st.success(f"✅ Active Project: {project_name} (ID: {project_id})")
            
            # Debug: Show session state
            st.info(f"🔍 Debug: Session state contains project_id: {project_id}")
            st.info(f"🔍 Debug: Session state keys: {list(st.session_state.keys())}")
            
            # Show project actions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f'<a href="http://localhost:8080/projects/{project_id}/data" target="_blank" style="font-size:1.2em;font-weight:bold;">'
                    '🌐 Open Project in Label Studio</a>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<a href="http://localhost:8080/projects/{project_id}/settings" target="_blank" style="font-size:1.1em;">'
                    '⚙️ Project Settings</a>',
                    unsafe_allow_html=True
                )
            
            # Check project status
            if st.button("🔄 Check Project Status", use_container_width=True):
                try:
                    from app.label_studio.auto_config import LabelStudioAutoConfig
                    
                    auto_config = LabelStudioAutoConfig(base_url="http://label-studio:8080")
                    project_info = auto_config.get_project_info(project_id)
                    
                    if project_info:
                        st.info(f"**Project Details:**")
                        st.write(f"• Tasks: {project_info.get('task_number', 0)}")
                        st.write(f"• Annotations: {project_info.get('total_annotations_number', 0)}")
                        st.write(f"• Created: {project_info.get('created_at', 'Unknown')}")
                        st.write(f"• Status: {'Active' if project_info.get('is_published', False) else 'Draft'}")
                    else:
                        st.warning("Could not fetch project information")
                        
                except Exception as e:
                    st.error(f"❌ Error checking project status: {str(e)}")
            
            # Annotation Type Configuration
            st.markdown("### 🎨 Annotation Type")
            st.info("""
            **Current Setup**: Your project is configured with brush annotations by default.
            
            **To Change Annotation Type**:
            1. 🌐 Open Label Studio: http://localhost:8080
            2. ⚙️ Go to Project Settings → Labeling Interface
            3. 🔄 Change the label configuration:
               - **Brush**: `<BrushLabels>` - Paint over regions
               - **Polygon**: `<PolygonLabels>` - Draw precise boundaries
               - **Rectangle**: `<RectangleLabels>` - Draw bounding boxes
               - **Circle**: `<CircleLabels>` - Draw circular regions
            4. 💾 Save changes
            
            **Training**: The system will automatically handle your annotation type choice.
            """)
        
        # No message needed when no active project
        
        # Manual Setup Section (collapsed by default)
        with st.expander("📋 Manual Setup Instructions (Alternative)", expanded=False):
            st.markdown("""
            **1️⃣ Access Label Studio:**
            - 🌐 Open: http://localhost:8080
            - 👤 Login: admin@example.com
            - 🔑 Password: admin
            
            **2️⃣ Create New Project:**
            - 📝 Project Name: """ + project_name + """
            - 📄 Description: Semantic segmentation project for automated annotation
            
            **3️⃣ Configure Labeling Interface:**
            - 🎨 Use the Label Studio GUI to define your classes
            - 🏷️ Add labels that match your project requirements
            - 📝 Background class is included by default
            - 🔧 You can add multiple object classes as needed
            - 🎯 **Choose Annotation Type**:
              - **Brush**: `<BrushLabels>` - Paint over regions (default, good for semantic segmentation)
              - **Polygon**: `<PolygonLabels>` - Draw precise boundaries (better for instance segmentation)
              - **Rectangle**: `<RectangleLabels>` - Draw bounding boxes
              - **Circle**: `<CircleLabels>` - Draw circular regions
            
            **4️⃣ Configure Source Storage (Images):**
            - 🔧 Go to: Settings → Cloud Storage → Add Source Storage
            - 📋 Use these settings:
              - Storage Type: Amazon S3
              - Storage Title: Images Storage
              - Bucket Name: segmentation-platform
              - Bucket Prefix: images/
              - Region Name: us-east-1
              - S3 Endpoint: http://minio:9000
              - Access Key ID: minioadmin
              - Secret Access Key: minioadmin123
              - Treat every bucket object as a source file: ✅ ON
              - Recursive scan: ✅ ON
              - Use pre-signed URLs: ❌ OFF
            
            **5️⃣ Configure Target Storage (Annotations):**
            - 🔧 Go to: Settings → Cloud Storage → Add Target Storage
            - 📋 Use these settings:
              - Storage Type: Amazon S3
              - Storage Title: Annotations Storage
              - Bucket Name: segmentation-platform
              - Bucket Prefix: annotations/
              - Region Name: us-east-1
              - S3 Endpoint: http://minio:9000
              - Access Key ID: minioadmin
              - Secret Access Key: minioadmin123
              - Can delete objects from storage: ❌ OFF (recommended)
            
            **6️⃣ Sync Storage:**
            - 🔄 Click 'Sync' button on both source and target storage
            - 📥 This will import images from MinIO into your project
            
            **7️⃣ Start Annotating:**
            - 🎯 Go to: Labeling Interface
            - 🖱️ Use your chosen annotation tool (brush, polygon, rectangle, or circle)
            - 🏷️ Assign labels based on your defined classes
            - 💾 The system will automatically handle your annotation type choice
            - 💡 **Tip**: The annotation type you chose will determine which training script is used
            
            **8️⃣ Export Annotations:**
            - 📤 Go to: Export → Export Annotations
            - 💾 Choose format: JSON or COCO
            - 📁 Annotations will be saved to MinIO in annotations/ folder
            """)
        
        # Troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **❓ Images not loading?**
            - ✅ Check MinIO bucket exists: segmentation-platform
            - ✅ Check images are in: segmentation-platform/images/
            - ✅ Check storage configuration in Label Studio
            - ✅ Try refreshing the page
            
            **❓ Can't connect to Label Studio?**
            - ✅ Check if Label Studio is running: http://localhost:8080
            - ✅ Check Docker containers: `docker compose ps`
            - ✅ Restart services: `docker compose restart`
            
            **❓ Can't connect to MinIO?**
            - ✅ Check MinIO is running: http://minio:9000
            - ✅ Check credentials: minioadmin
            - ✅ Check bucket exists: segmentation-platform
            
            **❓ Annotations not saving?**
            - ✅ Check target storage configuration
            - ✅ Check MinIO permissions
            - ✅ Try manual export
            
            **❓ Auto-setup failed?**
            - ✅ Check Label Studio is accessible at http://localhost:8080
            - ✅ Check MinIO is running and accessible
            - ✅ Check Docker containers are healthy
            - ✅ Try manual setup as alternative
            """)

    elif st.session_state.current_step == "train":
        st.header("Train Model")
        
        st.info("🎯 **Streamlit-Friendly Training Interface**")
        st.write("This will train a U-Net model with ResNet101 backbone on your LabelStudio annotations.")
        st.write("Training runs in the background and shows real-time progress.")
        
        # Define bucket name and use root bucket
        bucket_name = "segmentation-platform"
        selected_project = "root"
        annotation_prefix = "annotations/"
        
        st.info(f"🎯 Using root bucket")
        
        # Import class detector
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'utils'))
            from class_detector import ClassDetector
            class_detector = ClassDetector(bucket_name, annotation_prefix)
        except Exception as e:
            st.error(f"Error importing class detector: {str(e)}")
            st.stop()
        
        # Class Configuration Section
        st.subheader("📋 Class Configuration")
        
        # Check if there are any annotations to detect
        try:
            # Quick check for existing annotations
            import boto3
            s3_client = boto3.client(
                's3',
                endpoint_url='http://minio:9000',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin123',
                region_name='us-east-1'
            )
            
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=annotation_prefix)
            has_annotations = 'Contents' in response and len([obj for obj in response['Contents'] if obj['Size'] > 0]) > 0
            
            if not has_annotations:
                st.warning("⚠️ No annotations found in MinIO storage.")
                st.info("📋 To use the training section:")
                st.info("1. First create a Label Studio project in the Annotate section")
                st.info("2. Upload some images to MinIO")
                st.info("3. Create annotations in Label Studio")
                st.info("4. Come back here to detect classes and train")
                st.stop()
                
        except Exception as e:
            st.warning(f"⚠️ Could not check for annotations: {str(e)}")
        
        # Detect classes from Label Studio
        if st.button("🔍 Detect Classes from Label Studio"):
            with st.spinner("Detecting classes from annotations..."):
                try:
                    stats = class_detector.get_class_statistics()
                    st.session_state.detected_classes = stats
                    st.success(f"Detected {stats['num_classes']} classes from {stats['total_annotations']} annotations!")
                except Exception as e:
                    st.error(f"Error detecting classes: {str(e)}")
        
        # Show detected classes
        if 'detected_classes' in st.session_state:
            stats = st.session_state.detected_classes
            
            st.write("**Detected Classes:**")
            for i, (class_name, count) in enumerate(stats['classes'].items()):
                st.write(f"  • **{class_name}**: {count} annotations")
            
            # Class configuration
            st.write("**Configure Classes for Training:**")
            st.write("Classes will be used in this order (Background is always first):")
            
            # Create class list with Background first
            class_list = ["Background"] + stats['class_names']
            
            # Allow user to edit class order
            edited_classes = []
            for i, class_name in enumerate(class_list):
                if i == 0:  # Background is always first
                    edited_classes.append(class_name)
                    st.write(f"  {i+1}. **{class_name}** (Background - cannot be changed)")
                else:
                    # Allow reordering of object classes
                    new_name = st.text_input(f"  {i+1}. Class name:", value=class_name, key=f"class_{i}")
                    if new_name.strip():
                        edited_classes.append(new_name.strip())
                    else:
                        edited_classes.append(class_name)
            
            # Save class configuration
            if st.button("💾 Save Class Configuration"):
                try:
                    config = {
                        'class_names': edited_classes,
                        'detected_classes': stats['classes'],
                        'total_annotations': stats['total_annotations']
                    }
                    
                    with open("/app/class_config.json", 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    st.session_state.class_config = config
                    st.success("✅ Class configuration saved!")
                except Exception as e:
                    st.error(f"Error saving class configuration: {str(e)}")
        
        # Show training info
        with st.expander("📋 Training Configuration", expanded=False):
            if 'class_config' in st.session_state:
                config = st.session_state.class_config
                st.write("**Model Architecture:** U-Net with ResNet101 backbone")
                st.write(f"**Classes:** {' + '.join(config['class_names'])} ({len(config['class_names'])} classes)")
                st.write("**Data Source:** MinIO bucket with LabelStudio annotations")
                st.write("**Training:** 100 epochs with validation (full training)")
                st.write("**Disk Space:** Only saves final model (efficient)")
                st.write("**Progress:** Real-time updates in Streamlit")
                st.write("**Background:** Training runs in isolated process (no VM freeze)")
            else:
                st.write("**Model Architecture:** U-Net with ResNet101 backbone")
                st.write("**Classes:** Not configured yet - detect classes first")
                st.write("**Data Source:** MinIO bucket with LabelStudio annotations")
                st.write("**Training:** 100 epochs with validation (full training)")
                st.write("**Disk Space:** Only saves final model (efficient)")
                st.write("**Progress:** Real-time updates in Streamlit")
                st.write("**Background:** Training runs in isolated process (no VM freeze)")
        
        # Import training service
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
            from training_service import TrainingService
            training_service = TrainingService(
                bucket_name=bucket_name,
                annotation_prefix=annotation_prefix
            )
        except Exception as e:
            st.error(f"Error importing training service: {str(e)}")
            st.stop()
        
        # Check if training is already running
        if 'training_running' not in st.session_state:
            st.session_state.training_running = False
            st.session_state.training_log = []
            st.session_state.current_epoch = 0
        
        # Detect if there's already a training process running
        if not st.session_state.training_running:
            if training_service.detect_running_training():
                st.session_state.training_running = True
                st.info("🔄 Detected running training process - reconnected!")
        
        # Training control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.training_running:
                # Check if class configuration exists
                if 'class_config' not in st.session_state:
                    st.warning("⚠️ Please configure classes before starting training")
                    if st.button("🚀 Start Training", type="primary", disabled=True):
                        pass
                else:
                    if st.button("🚀 Start Training", type="primary"):
                        st.info("🔍 Debug: Starting training process...")
                        st.info(f"🔍 Debug: Using bucket_name='{bucket_name}', annotation_prefix='annotations/'")
                        success, message = training_service.start_training()
                        st.info(f"🔍 Debug: start_training() returned: success={success}, message='{message}'")
                        if success:
                            st.session_state.training_running = True
                            st.session_state.training_log = []
                            st.session_state.current_epoch = 0
                            st.success(message)
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to start training: {message}")
        
        with col2:
            if st.session_state.training_running:
                if st.button("⏹️ Stop Training"):
                    success, message = training_service.stop_training()
                    if success:
                        st.session_state.training_running = False
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to stop training: {message}")
        
        # Show training status
        if st.session_state.training_running:
            st.info("🔄 Training is running in isolated process... (This may take 15-30 minutes)")
            
            # Manual refresh button
            if st.button("🔄 Refresh Status"):
                st.experimental_rerun()
            
            # Get current status from training service
            status = training_service.get_status()
            
            # Update session state with current status
            st.session_state.current_epoch = status.get('current_epoch', 0)
            st.session_state.training_log = status.get('log', [])
            
            # Progress bar
            progress = status.get('progress', 0)
            progress_bar = st.progress(progress / 100)
            status_text = st.empty()
            status_text.text(f"Training epoch {st.session_state.current_epoch}/100 ({progress:.1f}%)")
            
            # Show memory and resource usage
            memory_usage = status.get('memory_usage', 0)
            gpu_memory_usage = status.get('gpu_memory_usage', 0)
            disk_usage = status.get('disk_usage', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("System Memory", f"{memory_usage:.1f} MB")
            with col2:
                st.metric("GPU Memory", f"{gpu_memory_usage:.1f} MB")
            with col3:
                st.metric("Disk Usage", f"{disk_usage:.1f}%")
            
            # Training log
            with st.expander("📋 Training Log", expanded=True):
                log_placeholder = st.empty()
                if st.session_state.training_log:
                    log_text = "\n".join(st.session_state.training_log[-20:])  # Show last 20 lines
                    log_placeholder.text(log_text)
                else:
                    log_placeholder.text("Starting training...")
            
            # Debug information (collapsed by default)
            with st.expander("🔧 Debug Info", expanded=False):
                st.write("**Raw Status Data:**")
                st.json(status)
                st.write("**Training Service Info:**")
                st.write(f"Progress File: {training_service.progress_file}")
                st.write(f"Log File: {training_service.log_file}")
                st.write(f"Is Running: {training_service.is_running}")
                if training_service.training_process:
                    st.write(f"Process ID: {training_service.training_process.pid}")
                    st.write(f"Process Return Code: {training_service.training_process.poll()}")
            
            # Check if training completed
            if status.get('status') == 'completed':
                st.session_state.training_running = False
                st.success("✅ Training completed successfully!")
                st.info("Check models/checkpoints/ for the final model.")
                
                # Show final log
                with st.expander("📋 Final Training Log", expanded=False):
                    st.text("\n".join(st.session_state.training_log[-50:]))
                    
            elif status.get('status') == 'failed':
                st.session_state.training_running = False
                st.error("❌ Training failed!")
                
                # Show error log
                with st.expander("📋 Error Log", expanded=True):
                    st.text("\n".join(st.session_state.training_log[-50:]))
            
            # Auto-refresh every 5 seconds (increased from 3)
            time.sleep(5)
            st.experimental_rerun()
        
        else:
            # Show completion message or start button
            if st.session_state.training_log and any("✅ Training complete!" in log for log in st.session_state.training_log):
                st.success("✅ Training completed successfully!")
                st.info("Check models/checkpoints/ for the final model.")
                
                # Show final log
                with st.expander("📋 Final Training Log", expanded=False):
                    st.text("\n".join(st.session_state.training_log[-50:]))
            else:
                st.info("Click 'Start Training' to begin the training process.")

    elif st.session_state.current_step == "inference":
        st.header("Run Inference")
        
        # List available models first
        checkpoints_dir = "models/checkpoints"
        model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        
        if not model_files:
            st.error(f"No model files (.pth) found in {checkpoints_dir}")
            st.info("Please train a model first in the training section.")
            st.stop()
        
        # Let user select from available models
        selected_model = st.selectbox("Select model", model_files)
        model_path = os.path.join(checkpoints_dir, selected_model)

        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please train a model first or provide correct path.")
            st.stop()

        # Load model configuration FIRST
        config_path = model_path.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                class_names = model_config.get('class_names', ["Background"])
                num_classes = len(class_names)
                st.success(f"✅ Loaded model config: {num_classes} classes - {', '.join(class_names)}")
            except Exception as e:
                st.warning(f"⚠️ Could not load model config: {e}. Using default configuration.")
                class_names = ["Background"]
                num_classes = 1
        else:
            st.warning("⚠️ No model config found. Using default configuration.")
            class_names = ["Background"]
            num_classes = 1

        # Show class information
        st.info(f"**Model Configuration:** {num_classes} classes - {', '.join(class_names)}")
        
        # Create ModelConfig with the correct number of classes
        config = ModelConfig(num_classes=num_classes, class_names=class_names)
        config.model_type = "resnet101"  # Match the training setup

        # Initialize session state for threshold if not exists
        if 'threshold' not in st.session_state:
            st.session_state.threshold = 0.3
            
        # Add threshold slider with session state
        new_threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.05,
            key='threshold_slider',
            help="Adjust the threshold for segmentation. Higher values make the model more selective."
        )

        # Only update threshold if it changed
        if new_threshold != st.session_state.threshold:
            st.session_state.threshold = new_threshold

        # Cache the model loading with a unique key based on model and config
        @st.cache_resource
        def get_inferencer(model_path, num_classes, class_names, threshold):
            # Create config inside the cached function to ensure it's fresh
            _config = ModelConfig(num_classes=num_classes, class_names=class_names)
            _config.model_type = "resnet101"
            return Inferencer(model_path, _config, threshold=threshold)

        try:
            # Clear cache if model or config changes
            cache_key = f"{model_path}_{num_classes}_{','.join(class_names)}_{st.session_state.threshold}"
            st.cache_resource.clear()
            
            inferencer = get_inferencer(model_path, num_classes, class_names, st.session_state.threshold)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Make sure the model was trained with the same class configuration.")
            st.stop()

        # File uploaders
        uploaded_image = st.file_uploader("Image", type=["jpg", "png"], key="image_uploader")
        
        # GT mask options
        st.subheader("Ground Truth Options")
        gt_option = st.radio(
            "Choose ground truth source:",
            ["Upload GT mask", "Use Label Studio annotations", "No GT (inference only)"],
            help="Select how to provide ground truth for evaluation"
        )
        
        uploaded_mask = None
        if gt_option == "Upload GT mask":
            uploaded_mask = st.file_uploader("Ground Truth Mask", type=["png", "jpg"], key="mask_uploader")
        elif gt_option == "Use Label Studio annotations":
            st.info("Will use annotations from storage for evaluation")
            # Add batch evaluation option
            if st.button("🔍 Run Batch Evaluation on Label Studio Data"):
                with st.spinner("Running batch evaluation on Label Studio annotations..."):
                    try:
                        from models.inference import batch_evaluate_with_labelstudio
                        
                        # Run batch evaluation
                        results = batch_evaluate_with_labelstudio(
                            image_dir="images/",
                            annotation_dir="annotations/",  # Fixed annotation prefix
                            model_path=model_path,
                            bucket_name=bucket_name,
                            num_classes=num_classes,
                            threshold=st.session_state.threshold
                        )
                        
                        if results['status'] == 'success':
                            st.success("Batch evaluation completed!")
                            
                            # Display results in a nice format
                            st.subheader("📊 Batch Evaluation Results")
                            
                            # Create metrics display
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Images Evaluated", results['images_evaluated'])
                                st.metric("Classes", results['num_classes'])
                            
                            with col2:
                                st.metric("Mean IoU", f"{results['overall_mean_iou']:.3f}")
                                st.metric("Model", selected_model)
                            
                            # Show detailed results
                            with st.expander("📋 Detailed Metrics", expanded=True):
                                st.write("**Class-wise IoU scores:**")
                                for i, miou in enumerate(results['mean_ious']):
                                    class_name = results['class_names'][i] if i < len(results['class_names']) else f"Class {i}"
                                    st.write(f"  {class_name}: {miou:.4f}")
                                
                                # Show object-wise metrics if available
                                if results['avg_metrics']:
                                    st.write("**Object-wise metrics:**")
                                    for class_name, metrics in results['avg_metrics'].items():
                                        st.write(f"  **{class_name}:**")
                                        st.write(f"    Precision: {metrics['precision']:.4f}")
                                        st.write(f"    Recall: {metrics['recall']:.4f}")
                                        st.write(f"    F1-Score: {metrics['f1']:.4f}")
                        else:
                            st.warning("No data found for evaluation")
                            st.info("Make sure you have Label Studio annotations in your MinIO bucket.")
                            
                    except Exception as e:
                        st.error(f"Error during batch evaluation: {str(e)}")
                        st.info("Make sure you have Label Studio annotations in your MinIO bucket.")

        # Cache the image processing
        @st.cache_data
        def process_image(image_bytes):
            return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

        if uploaded_image is not None:
            # Create a container for the results
            results_container = st.container()
            
            with results_container:
                # Process image
                image_bytes = uploaded_image.read()
                image = process_image(image_bytes)

                if gt_option == "Upload GT mask" and uploaded_mask is not None:
                    gt_mask = cv2.imdecode(np.frombuffer(uploaded_mask.read(), np.uint8), 0)
                    pred_mask, ious, metrics = inferencer.predict_and_compare(image, gt_mask)
                    overlayed_pred, overlayed_gt = inferencer.create_visualization(image, pred_mask, gt_mask)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(overlayed_pred, caption="Predicted Mask", use_column_width=True)
                    with col2:
                        st.image(overlayed_gt, caption="Ground Truth Mask", use_column_width=True)
                    
                    # Display metrics
                    st.subheader("📊 Evaluation Metrics")
                    
                    # IoU scores
                    st.write("**IoU per class:**")
                    for i, iou in enumerate(ious):
                        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                        st.write(f"  {class_name}: {iou:.3f}" if not np.isnan(iou) else f"  {class_name}: N/A")
                    
                    # Object-wise metrics if available
                    if metrics:
                        st.write("**Object-wise metrics:**")
                        for class_name, class_metrics in metrics.items():
                            st.write(f"  **{class_name}:**")
                            if isinstance(class_metrics, dict):
                                st.write(f"    Precision: {class_metrics.get('precision', 'N/A'):.3f}")
                                st.write(f"    Recall: {class_metrics.get('recall', 'N/A'):.3f}")
                                st.write(f"    F1-Score: {class_metrics.get('f1', 'N/A'):.3f}")
                                st.write(f"    True Positives: {class_metrics.get('tp', 'N/A')}")
                                st.write(f"    False Positives: {class_metrics.get('fp', 'N/A')}")
                                st.write(f"    False Negatives: {class_metrics.get('fn', 'N/A')}")
                                st.write(f"    Predicted Objects: {class_metrics.get('n_pred', 'N/A')}")
                                st.write(f"    Ground Truth Objects: {class_metrics.get('n_gt', 'N/A')}")
                            else:
                                st.write(f"    Raw metrics: {class_metrics}")
                    else:
                        st.warning("No object-wise metrics available")
                    
                elif gt_option == "No GT (inference only)":
                    pred_masks = inferencer.predict(image)
                    overlayed_pred, _ = inferencer.create_visualization(image, pred_masks)
                    st.image(overlayed_pred, caption="Predicted Mask", use_column_width=True)
                    st.info("No ground truth provided - inference only mode")
                    
                else:
                    st.warning("Please provide ground truth for evaluation")

if __name__ == "__main__":
    print("Starting application...")  # Debug print to console
    main()

