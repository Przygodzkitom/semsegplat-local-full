import streamlit as st
import psutil
import torch
import os
import gc
import numpy as np
import cv2
import json
import time
import boto3
from pathlib import Path

print("=" * 80)
print("🚀 MAIN.PY LOADED - DEBUG VERSION 2024")
print("=" * 80)
from models.config import ModelConfig
from models.inference_single import Inferencer
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
        # Save to local config directory (consistent location)
        config_dir = Path("config")
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

def discover_all_config_files():
    """Discover config files in the single designated location"""
    config_locations = []
    
    # Single config location (works in both local and Docker)
    config_paths = [
        Path("config") / "label_studio_project.json",
        Path("config") / "class_config.json"
    ]
    
    for path in config_paths:
        if path.exists():
            try:
                size = path.stat().st_size
                config_locations.append({
                    "path": str(path),
                    "size": size,
                    "exists": True
                })
            except:
                pass
    
    return config_locations

def clear_all_config_files():
    """Clear all discovered config files"""
    cleared_files = []
    config_locations = discover_all_config_files()
    
    for config in config_locations:
        try:
            path = Path(config["path"])
            if path.exists():
                path.unlink()
                cleared_files.append(str(path))
        except Exception as e:
            st.error(f"Failed to delete {config['path']}: {str(e)}")
    
    return cleared_files

def load_project_config():
    """Load project configuration from persistent storage"""
    try:
        # Load from local config directory (consistent location)
        config_file = Path("config") / "label_studio_project.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Restore session state
            st.session_state.label_studio_project_id = config_data.get("project_id")
            st.session_state.label_studio_project_name = config_data.get("project_name")
            st.session_state.label_studio_project_description = config_data.get("project_description")
            
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
    from models.inference_single import Inferencer
except Exception as e:
    st.error(f"Error importing local modules: {str(e)}")

# ATI_Box Logo
from pathlib import Path

# Try to load logo image, fallback to text if not found
logo_path = Path("app/logo.jpg")
logo_path_png = Path("app/logo.png")

if logo_path.exists():
    st.image(str(logo_path), width=300)
elif logo_path_png.exists():
    st.image(str(logo_path_png), width=300)
else:
    # Fallback to text-based header
    st.markdown("## 🧬 ATI_Box")
    st.markdown("*ANNOTATION • TRAINING • INFERENCE*")
    st.markdown("**IN ONE PLACE**")

st.divider()











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
    
    # Check for clean slate URL parameter (using st.experimental_get_query_params for older Streamlit)
    try:
        query_params = st.experimental_get_query_params()
        if query_params.get("clean_slate") == ["true"]:
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Clean slate activated! Session state cleared.")
            st.experimental_rerun()
    except:
        # If query params not available, skip this check
        pass
    
    # Clean Slate Operations Section
    with st.expander("🧹 Clean Slate Operations", expanded=False):
        st.markdown("### 🎯 Complete System Reset")
        st.info("For a true clean slate (removing all data, images, annotations, and project history), use the external cleanup script.")
        
        st.markdown("#### 📋 Clean Slate Instructions")
        st.code("""
# Step 1: Stop all containers
docker-compose down

# Step 2: Run the comprehensive cleanup script
python comprehensive_cleanup.py

# Step 3: Restart containers
docker-compose up -d
        """, language="bash")
        
        st.markdown("#### 🔧 Alternative Manual Cleanup")
        st.code("""
# Stop containers
docker-compose down

# Remove all volumes (destructive - removes all data)
docker volume rm $(docker volume ls -q)

# Remove local data directories
rm -rf minio-data/ label-studio-data/

# Restart containers
docker-compose up -d
        """, language="bash")
        
        st.warning("⚠️ **Warning**: Clean slate operations will remove ALL data including images, annotations, and trained models. This cannot be undone!")
        
        st.markdown("#### 🧹 Local Cleanup (Limited Scope)")
        st.caption("These options only clear local files and session state - they do NOT clear Docker volumes or persistent data.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Session State", key="clear_session"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Session state cleared! Refreshing...")
                st.experimental_rerun()
        
        with col2:
            if st.button("🧹 Clear Debug Files", key="clear_debug"):
                with st.spinner("Clearing debug files..."):
                    try:
                        debug_patterns = [
                            "debug_*.png",
                            "debug_*.jpg", 
                            "debug_*.jpeg",
                            "temp_*",
                            "*.tmp"
                        ]
                        
                        cleared_files = []
                        for pattern in debug_patterns:
                            for file in os.listdir("."):
                                if file.startswith(pattern.split("*")[0]) and file.endswith(pattern.split("*")[1]):
                                    if os.path.isfile(file):
                                        os.remove(file)
                                        cleared_files.append(file)
                        
                        if cleared_files:
                            st.success(f"✅ Cleared debug files: {', '.join(cleared_files)}")
                        else:
                            st.info("ℹ️ No debug files to clear")
                        
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error clearing debug files: {e}")
        
    
    # Load existing project configuration on startup
    if 'label_studio_project_id' not in st.session_state:
        load_project_config()
    
    # Debug: Show session state info
    if 'label_studio_project_id' in st.session_state:
        st.info(f"🔍 DEBUG: Session state has project ID: {st.session_state.label_studio_project_id}")
        st.info(f"🔍 DEBUG: Session state has project name: {st.session_state.get('label_studio_project_name', 'None')}")
    

    
    # Initialize MinIO manager and load existing images
    if 'existing_images' not in st.session_state:
        st.session_state.existing_images = load_existing_images("segmentation-platform")
    
        # Project Status
    st.markdown("### 📊 Project Status")
    
    if 'label_studio_project_id' in st.session_state:
        project_id = st.session_state.label_studio_project_id
        st.success(f"✅ Active Project: {st.session_state.get('label_studio_project_name', 'Unknown')} (ID: {project_id})")
        


    else:
        st.info("ℹ️ No active project. Create a new project to get started.")
    
    # Annotation type is configured directly in Label Studio
    # No need for complex detection or display logic
    
    # Sidebar - always show
    with st.sidebar:

        

        

        

        

        

        
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
                st.experimental_rerun()
        
        elif uploaded_files:
            # Automatically process images: detect format and convert to PNG if needed
            st.subheader("🔄 Processing Images Automatically")
            
            try:
                # Process uploaded images (detect format and convert if needed)
                processed_files = process_uploaded_images(uploaded_files)
                
                # Show conversion summary
                converted_count = sum(1 for _, _, _, was_converted in processed_files if was_converted)
                total_files = len(processed_files)
                
                if converted_count > 0:
                    st.success(f"✅ Processing complete! {converted_count} out of {total_files} images converted to PNG")
                    
                    # Show detailed conversion summary
                    st.subheader("📊 Conversion Summary")
                    conversion_details = []
                    for _, filename, original_format, was_converted in processed_files:
                        if was_converted:
                            conversion_details.append(f"• {filename}: {original_format} → PNG")
                        else:
                            conversion_details.append(f"• {filename}: {original_format} (already PNG)")
                    
                    for detail in conversion_details:
                        st.write(detail)
                else:
                    st.success(f"✅ All {total_files} images are already in PNG format - no conversion needed!")
                
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
                        
                        st.success(f"✅ Upload completed for: {final_filename}")
                        
                        uploaded_to_storage.append(f"{image_prefix}{final_filename}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(processed_files))
                        
                    except Exception as e:
                        st.error(f"Failed to upload {filename}: {str(e)}")
                
                status_text.text("Upload complete!")
                st.success(f"✅ Successfully uploaded {len(uploaded_to_storage)} images to project '{selected_upload_project}'")
                
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
            
            st.info("🎯 Your project is ready for annotation! Use the links above to access Label Studio.")
            

            
            return
        
        # Label Studio Setup Section (only shown if no project is configured)
        st.subheader("🏷️ Label Studio Setup")
        
        # Manual Project Linking Section
        st.markdown("### 🔗 Link to Existing Project")
        st.info("If you already have a project in Label Studio, you can link to it here:")
        
        existing_project_id = st.text_input(
            "Existing Project ID:",
            placeholder="Enter the ID of your existing Label Studio project",
            help="You can find this in the Label Studio URL: http://localhost:8080/projects/PROJECT_ID/"
        )
        
        if existing_project_id and existing_project_id.strip():
            if st.button("🔗 Link to Existing Project", type="primary", use_container_width=True):
                try:
                    # Verify the project exists
                    from app.label_studio.auto_config import LabelStudioAutoConfig
                    
                    auto_config = LabelStudioAutoConfig(base_url="http://label-studio:8080")
                    project_info = auto_config.get_project_info(int(existing_project_id))
                    
                    if project_info:
                        # Link to existing project
                        st.session_state.label_studio_project_id = int(existing_project_id)
                        st.session_state.label_studio_project_name = project_info.get('title', 'Unknown')
                        st.session_state.label_studio_project_description = project_info.get('description', '')
                        
                        # Save project configuration
                        save_project_config(int(existing_project_id), st.session_state.label_studio_project_name, st.session_state.label_studio_project_description)
                        
                        st.success(f"✅ Successfully linked to existing project: {st.session_state.label_studio_project_name}")
                        st.info("🔄 Refreshing page...")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error(f"❌ Project with ID {existing_project_id} not found in Label Studio")
                        
                except Exception as e:
                    st.error(f"❌ Error linking to project: {str(e)}")
        
        st.divider()
        
        # Automatic Configuration Section
        st.markdown("### 🚀 Create New Project")
        st.info("Or create a new project automatically:")
        
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
        
        # Check if we have a project ID in session state
        if 'label_studio_project_id' in st.session_state:
            project_id = st.session_state.label_studio_project_id
            project_name = st.session_state.get('label_studio_project_name', 'Unknown')
            
            st.success(f"✅ Active Project: {project_name} (ID: {project_id})")
            

            
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
        
        # Annotation Type Detection Section
        st.subheader("🎨 Annotation Type Detection")
        
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
        
        # Detect annotation type and classes
        if st.button("🔍 Detect Annotation Type & Classes"):
            with st.spinner("Detecting annotation type and classes..."):
                try:
                    # Import annotation type detector
                    import sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'utils'))
                    from annotation_type_detector import AnnotationTypeDetector
                    
                    # Detect annotation type
                    detector = AnnotationTypeDetector(bucket_name, annotation_prefix)
                    detection = detector.detect_annotation_type()
                    recommendation = detector.get_recommended_class_config()
                    
                    st.session_state.annotation_detection = detection
                    st.session_state.class_recommendation = recommendation
                    
                    # Display results
                    st.success(f"✅ Annotation type detected!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Annotation Type", detection['type'].title())
                        st.metric("Sample Annotations", detection['sample_annotations'])
                    
                    with col2:
                        st.metric("Background Handling", detection['background_handling'].title())
                        st.metric("Classes Found", len(detection['class_names']))
                    
                    # Show detailed information
                    with st.expander("📋 Detailed Detection Results", expanded=True):
                        st.write(f"**Annotation Type:** {detection['type']}")
                        st.write(f"**Background Handling:** {detection['background_handling']}")
                        st.write(f"**Has Explicit Background:** {detection['has_explicit_background']}")
                        st.write(f"**Classes Found:** {detection['class_names']}")
                        st.write(f"**Recommendation:** {recommendation['recommendation']}")
                        
                        if detection['type'] == 'polygon':
                            st.info("🎯 **Polygon Annotations Detected**")
                            st.write("- Background is automatically class 0 (unlabeled areas)")
                            st.write("- Training script will use polygon-specific handling")
                        elif detection['type'] == 'brush':
                            st.info("🎨 **Brush Annotations Detected**")
                            if detection['has_explicit_background']:
                                st.write("- Background is explicitly defined in annotations")
                                st.write("- Training script will use brush-specific handling with explicit background")
                            else:
                                st.write("- No explicit background class found")
                                st.write("- Training script will use brush-specific handling without background")
                        else:
                            st.warning("⚠️ **Mixed/Unknown Annotation Types**")
                            st.write("- Multiple annotation types detected")
                            st.write("- Training script will default to brush handling")
                    
                except Exception as e:
                    st.error(f"Error detecting annotation type: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
        
        # Class Configuration Section
        st.subheader("📋 Class Configuration")
        
        # Show detected classes if available
        if 'class_recommendation' in st.session_state:
            recommendation = st.session_state.class_recommendation
            
            st.write("**Recommended Class Configuration:**")
            st.write(f"Classes: {recommendation['class_names']}")
            st.write(f"Annotation Type: {recommendation['annotation_type']}")
            st.write(f"Background Handling: {recommendation['background_handling']}")
            
            # Allow user to edit class order
            st.write("**Configure Classes for Training:**")
            edited_classes = []
            for i, class_name in enumerate(recommendation['class_names']):
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
                        'annotation_type': recommendation['annotation_type'],
                        'background_handling': recommendation['background_handling'],
                        'detection_result': st.session_state.annotation_detection
                    }
                    
                    with open("/app/class_config.json", 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    st.session_state.class_config = config
                    st.success("✅ Class configuration saved!")
                except Exception as e:
                    st.error(f"Error saving class configuration: {str(e)}")
        
        # Legacy class detection (fallback)
        elif st.button("🔍 Detect Classes from Label Studio (Legacy)"):
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
                annotation_type = config.get('annotation_type', 'unknown')
                background_handling = config.get('background_handling', 'unknown')
                
                st.write("**Model Architecture:** U-Net with ResNet101 backbone")
                st.write(f"**Classes:** {' + '.join(config['class_names'])} ({len(config['class_names'])} classes)")
                st.write(f"**Annotation Type:** {annotation_type.title()}")
                st.write(f"**Background Handling:** {background_handling.title()}")
                
                # Show which training script will be used
                if annotation_type == 'polygon':
                    st.write("**Training Script:** Polygon-specific (background = class 0)")
                elif annotation_type == 'brush':
                    if background_handling == 'explicit':
                        st.write("**Training Script:** Brush-specific (explicit background)")
                    else:
                        st.write("**Training Script:** Brush-specific (no background)")
                else:
                    st.write("**Training Script:** Brush-specific (default)")
                
                st.write("**Data Source:** MinIO bucket with LabelStudio annotations")
                st.write("**Training:** 100 epochs with validation (full training)")
                st.write("**Disk Space:** Only saves final model (efficient)")
                st.write("**Progress:** Real-time updates in Streamlit")
                st.write("**Background:** Training runs in isolated process (no VM freeze)")
            else:
                st.write("**Model Architecture:** U-Net with ResNet101 backbone")
                st.write("**Classes:** Not configured yet - detect classes first")
                st.write("**Annotation Type:** Will be auto-detected")
                st.write("**Training Script:** Will be auto-selected based on annotation type")
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
                        success, message = training_service.start_training()
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
        
        # Define bucket name for MinIO access
        bucket_name = "segmentation-platform"
        
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
        
        # Try to read encoder from model's config file
        try:
            config_file = model_path.replace('.pth', '_config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    model_config = json.load(f)
                if 'encoder_name' in model_config:
                    config.encoder_name = model_config['encoder_name']
                    print(f"Using encoder from model config: {config.encoder_name}")
                else:
                    # No encoder_name in config, use optimal encoder for current device
                    from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
                    gpu_config = detect_gpu()
                    optimal_config = get_optimal_model_config(gpu_config)
                    config.encoder_name = optimal_config['encoder']
                    print(f"No encoder_name in model config, using optimal encoder: {config.encoder_name}")
            else:
                # No config file, use optimal encoder for current device
                from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
                gpu_config = detect_gpu()
                optimal_config = get_optimal_model_config(gpu_config)
                config.encoder_name = optimal_config['encoder']
                print(f"No model config file found, using optimal encoder: {config.encoder_name}")
        except Exception as e:
            # Error reading config, use optimal encoder for current device
            from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
            gpu_config = detect_gpu()
            optimal_config = get_optimal_model_config(gpu_config)
            config.encoder_name = optimal_config['encoder']
            print(f"Error reading model config: {e}, using optimal encoder: {config.encoder_name}")

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
            
            # Try to read encoder from model's config file
            try:
                config_file = model_path.replace('.pth', '_config.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        model_config = json.load(f)
                    if 'encoder_name' in model_config:
                        _config.encoder_name = model_config['encoder_name']
                        print(f"Using encoder from model config: {_config.encoder_name}")
                    else:
                        # No encoder_name in config, use optimal encoder for current device
                        from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
                        gpu_config = detect_gpu()
                        optimal_config = get_optimal_model_config(gpu_config)
                        _config.encoder_name = optimal_config['encoder']
                        print(f"No encoder_name in model config, using optimal encoder: {_config.encoder_name}")
                else:
                    # No config file, use optimal encoder for current device
                    from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
                    gpu_config = detect_gpu()
                    optimal_config = get_optimal_model_config(gpu_config)
                    _config.encoder_name = optimal_config['encoder']
                    print(f"No model config file found, using optimal encoder: {_config.encoder_name}")
            except Exception as e:
                # Error reading config, use optimal encoder for current device
                from models.utils.gpu_detector import detect_gpu, get_optimal_model_config
                gpu_config = detect_gpu()
                optimal_config = get_optimal_model_config(gpu_config)
                _config.encoder_name = optimal_config['encoder']
                print(f"Error reading model config: {e}, using optimal encoder: {_config.encoder_name}")
            
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
                        from models.inference_batch import batch_evaluate_with_minio_annotations
                        
                        # Run batch evaluation directly from MinIO annotations
                        print("=" * 60)
                        print("🚀 RUNNING BATCH EVALUATION FROM MINIO ANNOTATIONS")
                        print("=" * 60)
                        results = batch_evaluate_with_minio_annotations(
                            bucket_name=bucket_name,
                            model_path=model_path,
                            num_classes=num_classes,
                            threshold=st.session_state.threshold
                        )
                        
                        if results['status'] == 'success':
                            st.success("Batch evaluation completed!")
                            
                            # Display debug info if available
                            if 'debug_info' in results and results['debug_info']:
                                with st.expander("🔍 Debug Information", expanded=True):
                                    for debug_line in results['debug_info']:
                                        st.text(debug_line)
                            
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

    main()

