import streamlit as st
import psutil
import torch
import os
import gc
import numpy as np
import cv2
from models.config import ModelConfig
from models.inferencer import Inferencer
from models.utils.gpu_detector import detect_gpu, print_device_info
from app.storage_manager import get_storage_manager
from app.label_studio.config import create_label_studio_project, sync_images_to_label_studio, get_project_images
import json

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

# Initialize application with resource management
if not configure_resource_limits():
    st.error("Failed to configure resource limits. Please restart the application.")
    st.stop()

# Debug message at the start
st.write("Starting app initialization...")

# Import basic libraries first
import cv2
st.write("Basic libraries imported successfully")

# Try importing ML libraries
try:
    import torch
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    st.write("ML libraries imported successfully")
except Exception as e:
    st.error(f"Error importing ML libraries: {str(e)}")

# Try importing local modules
try:
    from models.config import ModelConfig
    from models.inferencer import Inferencer
    st.write("Local modules imported successfully")
except Exception as e:
    st.error(f"Error importing local modules: {str(e)}")

st.title("Semantic Segmentation Platform")

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
    st.write("Main function started")
    
    # Initialize MinIO manager and load existing images
    if 'existing_images' not in st.session_state:
        st.session_state.existing_images = load_existing_images("segmentation-platform")
    
    with st.sidebar:
        st.header("Project Configuration")
        st.write("Sidebar initialized")
        
        # Project Initialization Section
        st.subheader("🚀 Project Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Initialize Project Structure"):
                with st.spinner("Setting up project structure..."):
                    try:
                        # Create required directories
                        import subprocess
                        from pathlib import Path
                        
                        directories = [
                            "label-studio-data",
                            "models/checkpoints",
                            "models/saved_models",
                            "models/utils",
                            "logs",
                            "backups",
                            "temp"
                        ]
                        
                        for directory in directories:
                            Path(directory).mkdir(parents=True, exist_ok=True)
                        
                        st.success("✅ Directory structure created!")
                        
                        # Setup MinIO structure
                        try:
                            import boto3
                            from botocore.exceptions import ClientError
                            
                            st.info("🔗 Connecting to MinIO...")
                            
                            s3_client = boto3.client(
                                's3',
                                endpoint_url='http://minio:9000',
                                aws_access_key_id='minioadmin',
                                aws_secret_access_key='minioadmin123',
                                region_name='us-east-1'
                            )
                            
                            st.info("✅ MinIO connection established")
                            
                            # Create bucket if it doesn't exist
                            try:
                                s3_client.head_bucket(Bucket="segmentation-platform")
                                st.info("✅ Bucket 'segmentation-platform' exists")
                            except ClientError as e:
                                if e.response['Error']['Code'] == '404':
                                    st.info("📦 Creating bucket 'segmentation-platform'...")
                                    s3_client.create_bucket(Bucket="segmentation-platform")
                                    st.info("✅ Bucket created")
                                else:
                                    raise e
                            
                            # Create folder structure
                            st.info("📁 Creating folder structure...")
                            folders = ["images/", "annotations/", "models/", "models/checkpoints/", "models/final/"]
                            for folder in folders:
                                try:
                                    s3_client.put_object(Bucket="segmentation-platform", Key=folder, Body="")
                                    st.info(f"✅ Created folder: {folder}")
                                except Exception as e:
                                    st.warning(f"⚠️ Could not create folder {folder}: {str(e)}")
                            
                            st.success("✅ MinIO storage structure created!")
                            
                        except ImportError:
                            st.warning("⚠️ boto3 not available, skipping MinIO structure")
                            st.info("Install with: pip install boto3")
                        except Exception as e:
                            st.error(f"❌ MinIO setup failed: {str(e)}")
                            st.info("Debug info:")
                            st.code(f"Endpoint: http://minio:9000\nBucket: segmentation-platform\nError: {type(e).__name__}: {str(e)}")
                        
                        # Test services (from within container, use internal hostnames)
                        import requests
                        services = [
                            ("MinIO API", "http://minio:9000/minio/health/live"),
                            ("MinIO Console", "http://minio:9001"),
                            ("Label Studio", "http://label-studio:8080"),
                        ]
                        
                        st.info("🔗 Testing service connectivity...")
                        for service_name, url in services:
                            try:
                                response = requests.get(url, timeout=5)
                                if response.status_code in [200, 302]:
                                    st.success(f"✅ {service_name}: {url}")
                                else:
                                    st.warning(f"⚠️ {service_name}: {url} (Status: {response.status_code})")
                            except requests.exceptions.RequestException:
                                st.error(f"❌ {service_name}: {url}")
                        
                        # Also show external URLs for user access
                        st.info("🌐 External Access URLs:")
                        st.success("✅ MinIO API: http://localhost:9000")
                        st.success("✅ MinIO Console: http://localhost:9001")
                        st.success("✅ Label Studio: http://localhost:8080")
                        st.success("✅ Streamlit App: http://localhost:8501")
                        
                    except Exception as e:
                        st.error(f"❌ Error during initialization: {str(e)}")
        
        with col2:
            st.warning("⚠️ **Danger Zone** - Use with caution!")
            
            # Show current storage info
            try:
                import boto3
                s3_client = boto3.client(
                    's3',
                    endpoint_url='http://minio:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin123',
                    region_name='us-east-1'
                )
                
                response = s3_client.list_objects_v2(Bucket="segmentation-platform")
                file_count = len(response.get('Contents', []))
                
                if file_count > 0:
                    st.info(f"📊 Current storage: {file_count} files")
                    
                    # Backup option
                    if st.button("💾 Create Backup", type="primary"):
                        with st.spinner("Creating backup..."):
                            try:
                                import os
                                from datetime import datetime
                                
                                # Create backup directory
                                backup_dir = f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                os.makedirs(backup_dir, exist_ok=True)
                                
                                # Download all files to backup
                                response = s3_client.list_objects_v2(Bucket="segmentation-platform")
                                if 'Contents' in response:
                                    for obj in response['Contents']:
                                        # Create local path
                                        local_path = os.path.join(backup_dir, obj['Key'])
                                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                                        
                                        # Download file
                                        s3_client.download_file("segmentation-platform", obj['Key'], local_path)
                                    
                                    st.success(f"✅ Backup created: {backup_dir}")
                                    st.info(f"📁 Backup location: {os.path.abspath(backup_dir)}")
                                else:
                                    st.warning("⚠️ No files to backup")
                                    
                            except Exception as e:
                                st.error(f"❌ Backup failed: {str(e)}")
                    
                    # Safe reset with confirmation
                    if st.button("🗑️ Clear All Data", type="secondary"):
                        st.error("🚨 **WARNING: This will permanently delete ALL files!**")
                        st.write("This action will remove:")
                        st.write("- All uploaded images")
                        st.write("- All annotations")
                        st.write("- All model files")
                        st.write("- All other data")
                        
                        # Require explicit confirmation
                        confirmation = st.text_input("Type 'DELETE' to confirm:", key="reset_confirmation")
                        
                        if confirmation == "DELETE":
                            with st.spinner("Clearing storage..."):
                                try:
                                    # List and delete all objects
                                    response = s3_client.list_objects_v2(Bucket="segmentation-platform")
                                    if 'Contents' in response:
                                        for obj in response['Contents']:
                                            s3_client.delete_object(Bucket="segmentation-platform", Key=obj['Key'])
                                    
                                    # Clear session state
                                    if 'existing_images' in st.session_state:
                                        del st.session_state.existing_images
                                    if 'total_images' in st.session_state:
                                        del st.session_state.total_images
                                    
                                    st.success("✅ Storage cleared! All files removed.")
                                    st.rerun()  # Refresh the page
                                    
                                except Exception as e:
                                    st.error(f"❌ Error clearing storage: {str(e)}")
                        elif confirmation:
                            st.error("❌ Confirmation text doesn't match. Please type 'DELETE' exactly.")
                else:
                    st.success("✅ Storage is empty")
                    
            except Exception as e:
                st.error(f"❌ Could not check storage status: {str(e)}")
        
        # Service Management
        st.subheader("⚙️ Service Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Services"):
                with st.spinner("Starting services..."):
                    try:
                        result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                                              capture_output=True, text=True, check=True)
                        st.success("✅ Services started!")
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Error starting services: {e.stderr}")
        
        with col2:
            if st.button("⏹️ Stop Services"):
                with st.spinner("Stopping services..."):
                    try:
                        result = subprocess.run(['docker', 'compose', 'down'], 
                                              capture_output=True, text=True, check=True)
                        st.success("✅ Services stopped!")
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Error stopping services: {e.stderr}")
        
        # Service Status
        if st.button("📊 Check Service Status"):
            with st.spinner("Checking service status..."):
                try:
                    result = subprocess.run(['docker', 'compose', 'ps'], 
                                          capture_output=True, text=True, check=True)
                    st.code(result.stdout)
                except subprocess.CalledProcessError as e:
                    st.error(f"❌ Error checking status: {e.stderr}")
        
        # Quick Access Links
        st.subheader("🔗 Quick Access")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏷️ Label Studio"):
                st.markdown("[Open Label Studio](http://localhost:8080)")
                st.info("Username: admin@example.com | Password: admin")
        
        with col2:
            if st.button("☁️ MinIO Console"):
                st.markdown("[Open MinIO Console](http://localhost:9001)")
                st.info("Username: minioadmin | Password: minioadmin123")
        
        # Label Studio Configuration Helper
        st.subheader("🏷️ Label Studio Setup")
        
        if st.button("📋 Show Label Studio Config"):
            st.info("**Label Studio Configuration Steps:**")
            st.markdown("""
            1. **Go to**: http://localhost:8080
            2. **Login**: admin@example.com / admin
            3. **Create Project** or open existing
            4. **Go to**: Settings → Cloud Storage
            5. **Add Source Storage**:
               - Storage Type: Amazon S3
               - Bucket: segmentation-platform
               - Prefix: images/
               - Use pre-signed URLs: ❌ OFF
               - Proxy through Label Studio: ✅ ON
            6. **Add Target Storage** (optional):
               - Storage Type: Amazon S3
               - Bucket: segmentation-platform
               - Prefix: annotations/
               - Use pre-signed URLs: ❌ OFF
               - Proxy through Label Studio: ✅ ON
            """)
        
        # Backup and Maintenance
        st.subheader("💾 Backup & Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Backup Data"):
                with st.spinner("Creating backup..."):
                    try:
                        import subprocess
                        from datetime import datetime
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_dir = f"backups/backup_{timestamp}"
                        
                        # Create backup directory
                        Path(backup_dir).mkdir(parents=True, exist_ok=True)
                        
                        # Backup Label Studio data
                        if Path("label-studio-data").exists():
                            subprocess.run(['cp', '-r', 'label-studio-data', f"{backup_dir}/"])
                        
                        # Backup MinIO data (if possible)
                        try:
                            subprocess.run([
                                'docker', 'run', '--rm',
                                '-v', 'semsegplat-full_local_version_minio-data:/data',
                                '-v', f"{os.path.abspath(backup_dir)}:/backup",
                                'alpine', 'sh', '-c', 'cp -r /data/* /backup/minio-data/ 2>/dev/null || echo "No MinIO data to backup"'
                            ])
                        except:
                            st.warning("⚠️ Could not backup MinIO data")
                        
                        st.success(f"✅ Backup created: {backup_dir}")
                        
                    except Exception as e:
                        st.error(f"❌ Backup failed: {str(e)}")
        
        with col2:
            if st.button("🧹 Clean Temp"):
                with st.spinner("Cleaning temporary files..."):
                    try:
                        import shutil
                        
                        # Clean temp directory
                        if Path("temp").exists():
                            shutil.rmtree("temp")
                            Path("temp").mkdir()
                        
                        # Clean logs (keep last 7 days)
                        if Path("logs").exists():
                            import time
                            current_time = time.time()
                            for log_file in Path("logs").glob("*.log"):
                                if current_time - log_file.stat().st_mtime > 7 * 24 * 3600:  # 7 days
                                    log_file.unlink()
                        
                        st.success("✅ Temporary files cleaned!")
                        
                    except Exception as e:
                        st.error(f"❌ Cleanup failed: {str(e)}")
        
        # System Information
        st.subheader("📊 System Status")
        
        # Check if services are running
        try:
            import requests
            
            services_status = {}
            
            # Check MinIO
            try:
                response = requests.get("http://minio:9000/minio/health/live", timeout=3)
                services_status["MinIO"] = "🟢 Running" if response.status_code == 200 else "🟡 Issues"
            except:
                services_status["MinIO"] = "🔴 Stopped"
            
            # Check Label Studio
            try:
                response = requests.get("http://label-studio:8080", timeout=3)
                services_status["Label Studio"] = "🟢 Running" if response.status_code in [200, 302] else "🟡 Issues"
            except:
                services_status["Label Studio"] = "🔴 Stopped"
            
            # Display status
            for service, status in services_status.items():
                st.write(f"{service}: {status}")
            
            # Check directory structure
            st.write("**Directory Structure:**")
            from pathlib import Path
            directories = ["label-studio-data", "models/checkpoints", "models/saved_models", "logs", "backups"]
            for directory in directories:
                if Path(directory).exists():
                    st.write(f"✅ {directory}/")
                else:
                    st.write(f"❌ {directory}/ (missing)")
            
        except Exception as e:
            st.warning(f"Could not check system status: {str(e)}")
        
        # Bucket Configuration
        st.subheader("📦 MinIO Bucket Configuration")
        
        # Fixed bucket name
        bucket_name = "segmentation-platform"
        st.info(f"Bucket: **{bucket_name}** (fixed)")
        
        # Test bucket connection
        if st.button("🔍 Test Bucket Connection"):
            try:
                storage_manager = get_storage_manager(bucket_name=bucket_name)
                if storage_manager:
                    images = storage_manager.list_images(prefix="images/")
                    st.success(f"✅ Connected to bucket '{bucket_name}' - Found {len(images)} images in images/ folder")
                else:
                    st.error(f"❌ Could not connect to bucket '{bucket_name}'")
            except Exception as e:
                st.error(f"❌ Error connecting to bucket: {str(e)}")
        
        # List available buckets
        if st.button("📋 List Available Buckets"):
            try:
                storage_manager = get_storage_manager()
                if storage_manager:
                    if storage_manager.provider_type == 'minio':
                        st.info("Using local MinIO storage. Bucket: segmentation-platform")
                    else:
                        st.info(f"Using {storage_manager.provider_type} storage.")
                else:
                    st.error("❌ Storage manager not available")
            except Exception as e:
                st.error(f"❌ Error listing buckets: {str(e)}")
        
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
        
        # Show existing images count
        if st.session_state.existing_images:
            st.info(f"📁 Found {len(st.session_state.existing_images)} existing images in storage")
        
        # File uploader
        uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        
        if uploaded_files:
            storage_manager = get_storage_manager(bucket_name="segmentation-platform")
            if storage_manager:
                # Upload progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                uploaded_to_storage = []
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {file.name}...")
                    
                    try:
                        # Reset file pointer
                        file.seek(0)
                        
                        # Upload to storage
                        result = storage_manager.upload_image(file)
                        uploaded_to_storage.append(result)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"Failed to upload {file.name}: {str(e)}")
                
                status_text.text("Upload complete!")
                st.success(f"✅ Successfully uploaded {len(uploaded_to_storage)} images to storage")
                
                # Refresh existing images list
                st.session_state.existing_images = load_existing_images("segmentation-platform")
                
                # Store uploaded files info in session state
                st.session_state.uploaded_files = uploaded_to_storage
            else:
                st.error("❌ Storage manager not available. Please check your configuration.")

    elif st.session_state.current_step == "annotate":
        st.header("Annotate Images")
        
        # Always show Label Studio link
        LABEL_STUDIO_URL = "http://localhost:8080"  # Use localhost for local deployment
        st.markdown(
            f'<a href="{LABEL_STUDIO_URL}" target="_blank" style="font-size:1.2em;font-weight:bold;">'
            '👉 Open Label Studio for Annotation</a>',
            unsafe_allow_html=True
        )
        
        # Check if we have images (either uploaded in this session or existing in storage)
        has_images = (
            "uploaded_files" in st.session_state and st.session_state.uploaded_files or
            st.session_state.existing_images
        )
        
        if has_images:
            total_images = len(st.session_state.existing_images) if st.session_state.existing_images else 0
            if "uploaded_files" in st.session_state:
                total_images += len(st.session_state.uploaded_files)
            
            st.write(f"Ready to annotate {total_images} images")
        else:
            st.warning("No images found. Please upload some images first.")
            st.info("Go to the 'Upload Images' step to add images to your project.")

    elif st.session_state.current_step == "train":
        st.header("Train Model")
        
        st.info("🎯 **Streamlit-Friendly Training Interface**")
        st.write("This will train a U-Net model with ResNet101 backbone on your LabelStudio annotations.")
        st.write("Training runs in the background and shows real-time progress.")
        
        # Import class detector
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'utils'))
            from class_detector import ClassDetector
            class_detector = ClassDetector(bucket_name, "annotations/")  # Fixed annotation prefix
        except Exception as e:
            st.error(f"Error importing class detector: {str(e)}")
            st.stop()
        
        # Class Configuration Section
        st.subheader("📋 Class Configuration")
        
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
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
            from training_service import TrainingService
            training_service = TrainingService(
                bucket_name=bucket_name,
                annotation_prefix="annotations/"  # Fixed annotation prefix
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
            import time
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

        # Import os for file operations
        import os
        
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
