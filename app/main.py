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
print(" MAIN.PY LOADED - DEBUG VERSION 2024")
print("=" * 80)
from models.config import ModelConfig
from models.inference_single import Inferencer
from models.utils.gpu_detector import detect_gpu, print_device_info
from app.storage_manager import get_storage_manager
from app.label_studio.config import create_label_studio_project, sync_images_to_label_studio, get_project_images
from app.image_utils import process_uploaded_images, get_supported_formats, format_file_size, estimate_conversion_time, convert_to_png, detect_image_format
from models.batch_analysis import analyze_single_image, create_analysis_overlay, aggregate_batch_results

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
            st.sidebar.success(f" GPU Available: {gpu_config['name']}")
            st.sidebar.info(f"GPU Memory: {gpu_config['memory_gb']:.1f} GB")
            st.sidebar.info(f"CUDA Version: {gpu_config['cuda_version']}")
        else:
            st.sidebar.info(" Running on CPU - Training and inference will be slower but functional")
        
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
        
        st.success(f" Project configuration saved")
        return True
    except Exception as e:
        st.error(f" Failed to save project configuration: {str(e)}")
        return False

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
        st.warning(f" Could not load project configuration: {str(e)}")
        return None



# Initialize application with resource management
if not configure_resource_limits():
    st.error("Failed to configure resource limits. Please restart the application.")
    st.stop()

# ATI_Box Logo

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
    
    # Load existing project configuration on startup
    if 'label_studio_project_id' not in st.session_state:
        load_project_config()

    # Initialize MinIO manager and load existing images
    if 'existing_images' not in st.session_state:
        st.session_state.existing_images = load_existing_images("segmentation-platform")
    
    # Annotation type is configured directly in Label Studio
    # No need for complex detection or display logic
    
    # Sidebar - always show
    with st.sidebar:
        # Navigation
        if "current_step" not in st.session_state:
            st.session_state.current_step = "upload"

        st.markdown("""
        <style>
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: #1565C0;
            color: white;
            font-size: 1.15rem;
            font-weight: bold;
            width: 100%;
            height: 3.2rem;
            border-radius: 8px;
            border: none;
            margin-bottom: 6px;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #1976D2;
            color: white;
        }
        section[data-testid="stSidebar"] div.stButton > button:active,
        section[data-testid="stSidebar"] div.stButton > button:focus {
            background-color: #0D47A1;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        steps = {
            "upload": "1. Upload Images",
            "annotate": "2. Annotate Images",
            "train": "3. Train Model",
            "inference": "4. Evaluate Model",
            "batch_analysis": "5. Batch Analysis"
        }

        for step, label in steps.items():
            if st.button(label, use_container_width=True):
                st.session_state.current_step = step

    # Main content area
    if st.session_state.current_step == "upload":
        st.markdown("<h1 style='text-align: center;'>UPLOAD IMAGES</h1>", unsafe_allow_html=True)

        # Simple single-project approach - always upload to images/ folder
        selected_upload_project = "main"
        image_prefix = "images/"
        # Load and show existing images
        existing_project_images = st.session_state.get('existing_images', [])
        if existing_project_images:
            st.info(f"Found {len(existing_project_images)} existing images")
        else:
            st.info(f"No existing images found")
        
        # Enhanced file uploader with format detection and conversion
        st.subheader("Image Upload & Format Conversion")
        
        # Show supported formats
        supported_formats = get_supported_formats()
        st.info(f"Supported formats: {', '.join(supported_formats).upper()}")
        st.info("Non-PNG images will be automatically converted to PNG format for Label Studio compatibility")
        st.warning(
            "**Note on automatic conversion:** While the app can convert most formats to PNG, "
            "some images may not convert correctly — in particular **dim or low-contrast TIFFs** "
            "(e.g. microscopy, scientific, or HDR images) where the automatic brightness normalization "
            "may not match your expectations. "
            "For best results, **convert your images to PNG manually before uploading** "
            "(e.g. using ImageJ, FIJI, or any image editor) so you can verify they look correct."
        )
        
        # File uploader with expanded format support
        uploaded_files = st.file_uploader(
            "Choose images", 
            accept_multiple_files=True, 
            type=supported_formats,
            help="Upload images in any supported format. They will be converted to PNG if needed."
        )
        
        # Add warning for large batches
        if uploaded_files and len(uploaded_files) > 50:
            st.warning(f" **Large batch detected:** {len(uploaded_files)} files. Processing may take several minutes. Consider processing in smaller batches for better performance.")
        
        if uploaded_files:
            # Automatically process images: detect format and convert to PNG if needed
            st.subheader(" Processing Images Automatically")
            
            try:
                # Process uploaded images (detect format and convert if needed)
                processed_files = process_uploaded_images(uploaded_files)
                
                # Show conversion summary
                converted_count = sum(1 for _, _, _, was_converted in processed_files if was_converted)
                total_files = len(processed_files)
                
                if converted_count > 0:
                    st.success(f" Processing complete! {converted_count} out of {total_files} images converted to PNG")
                    
                    # Show detailed conversion summary
                    st.subheader(" Conversion Summary")
                    conversion_details = []
                    for _, filename, original_format, was_converted in processed_files:
                        if was_converted:
                            conversion_details.append(f"• {filename}: {original_format} → PNG")
                        else:
                            conversion_details.append(f"• {filename}: {original_format} (already PNG)")
                    
                    for detail in conversion_details:
                        st.write(detail)
                else:
                    st.success(f" All {total_files} images are already in PNG format - no conversion needed!")
                
                # Upload to project-specific location
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
                batch_timestamp = int(time.time() * 1000)
                for i, (file_data, filename, original_format, was_converted) in enumerate(processed_files):
                    status_text.text(f"Uploading {filename}...")

                    try:
                        # Reset file pointer
                        file_data.seek(0)

                        # Create unique filename: batch timestamp + index prevents collisions
                        final_filename = f"{batch_timestamp}_{i}_{filename}"
                        
                        # Show upload info
                        if was_converted:
                            st.info(f" Uploading converted image: {filename} (was {original_format})")
                        else:
                            st.info(f" Uploading original image: {filename} ({original_format})")
                        
                        # Upload to root bucket
                        s3_client.upload_fileobj(
                            file_data,
                            "segmentation-platform",
                            f"{image_prefix}{final_filename}"
                        )
                        
                        st.success(f" Upload completed for: {final_filename}")
                        
                        uploaded_to_storage.append(f"{image_prefix}{final_filename}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(processed_files))
                        
                    except Exception as e:
                        st.error(f"Failed to upload {filename}: {str(e)}")
                
                status_text.text("Upload complete!")
                st.success(f" Successfully uploaded {len(uploaded_to_storage)} images to project '{selected_upload_project}'")
                
                st.info(" Verifying uploads in MinIO...")
                try:
                    response = s3_client.list_objects_v2(Bucket="segmentation-platform", Prefix="images/")
                    if 'Contents' in response:
                        actual_files = [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
                        st.info(f" Found {len(actual_files)} files in MinIO: {actual_files}")
                    else:
                        st.warning(" No files found in MinIO after upload!")
                except Exception as e:
                    st.error(f" Error verifying uploads: {str(e)}")
                
                # Clear the uploaded files to prevent re-upload
                st.session_state.uploaded_files = uploaded_to_storage
                
            except Exception as e:
                st.error(f" Processing/Upload failed: {str(e)}")

    elif st.session_state.current_step == "annotate":
        st.markdown("<h1 style='text-align: center;'>ANNOTATE IMAGES</h1>", unsafe_allow_html=True)
        
        # Check if project is already configured (simple check - no validation needed)
        if 'label_studio_project_id' in st.session_state and st.session_state.label_studio_project_id:
            project_id = st.session_state.label_studio_project_id
            project_name = st.session_state.get('label_studio_project_name', 'Unknown')
            
            # Project is configured - show the interface
            st.success(f" Project Already Configured: {project_name} (ID: {project_id})")
            
            # Show project actions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f'<a href="http://localhost:8080/projects/{project_id}/data" target="_blank" style="font-size:1.2em;font-weight:bold;">'
                    ' Open Project in Label Studio</a>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<a href="http://localhost:8080/projects/{project_id}/settings" target="_blank" style="font-size:1.1em;">'
                    ' Project Settings</a>',
                    unsafe_allow_html=True
                )
            
            st.info(" Your project is ready for annotation! Use the links above to access Label Studio.")
            return
        
        # Label Studio Setup Section (only shown if no project is configured)
        st.subheader(" Create new Label Studio project")
        st.info("You only do it once when creating a new project.")

        # Project configuration - fixed values for simplicity
        project_name = "semantic-segmentation"
        project_description = "Automated semantic segmentation project with MinIO storage"

        # Token input section
        st.markdown("<p style='font-size:1.2em;font-weight:bold;'>1. Open Label Studio:</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Go to: <a href='http://localhost:8080' target='_blank'>http://localhost:8080</a></p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Login: admin@example.com / admin</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2em;font-weight:bold;'>2. Generate Token:</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Click your username in top right → Account Settings</p>", unsafe_allow_html=True)
        st.image("/app/app/LS_screen_1.jpg")
        st.image("/app/app/LS_screen_2.jpg")
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Look for \"Access Tokens\" or \"API Tokens\"</p>", unsafe_allow_html=True)
        st.image("/app/app/LS_screen_3.jpg")
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Create new token with read/write permissions</p>", unsafe_allow_html=True)
        st.image("/app/app/LS_screen_4.jpg")
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Copy the generated token</p>", unsafe_allow_html=True)
        st.image("/app/app/LS_screen_5.jpg")
        st.markdown("<p style='font-size:1.2em;font-weight:bold;'>3. Paste Token Below:</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1em;font-weight:bold;'>- Paste your token in the field below and press ENTER</p>", unsafe_allow_html=True)
        
        # Token input field
        personal_access_token = st.text_input(
            "",
            type="password",
            placeholder="Paste your Label Studio personal access token here",
            help="This token will be used to automatically configure your Label Studio project"
        )
        
        # Auto-setup button (only enabled if token is provided)
        if personal_access_token:
            if st.button("PRESS THIS BUTTON to Auto-Setup Label Studio Project", type="primary", use_container_width=True):
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

    elif st.session_state.current_step == "train":
        st.markdown("<h1 style='text-align: center;'>TRAIN MODEL</h1>", unsafe_allow_html=True)
        
        st.info(" **Streamlit-Friendly Training Interface**")
        st.write("This will train a U-Net model with ResNet101 backbone on your LabelStudio annotations.")
        st.write("Training runs in the background and shows real-time progress.")
        
        # Define bucket name and use root bucket
        bucket_name = "segmentation-platform"
        selected_project = "root"
        annotation_prefix = "annotations/"
        
        st.info(f" Using root bucket")
        
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
        st.subheader(" Annotation Type Detection")
        
        # Check if there are any annotations to detect
        try:
            # Quick check for existing annotations
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
                st.warning(" No annotations found in MinIO storage.")
                st.info(" To use the training section:")
                st.info("1. First create a Label Studio project in the Annotate section")
                st.info("2. Upload some images to MinIO")
                st.info("3. Create annotations in Label Studio")
                st.info("4. Come back here to detect classes and train")
                st.stop()
                
        except Exception as e:
            st.warning(f" Could not check for annotations: {str(e)}")
        
        # Detect annotation type and classes
        if st.button(" Detect Annotation Type & Classes"):
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
                    st.success(f" Annotation type detected!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Annotation Type", detection['type'].title())
                        st.metric("Sample Annotations", detection['sample_annotations'])
                    
                    with col2:
                        st.metric("Background Handling", detection['background_handling'].title())
                        st.metric("Classes Found", len(detection['class_names']))
                    
                    # Show detailed information
                    with st.expander(" Detailed Detection Results", expanded=True):
                        st.write(f"**Annotation Type:** {detection['type']}")
                        st.write(f"**Background Handling:** {detection['background_handling']}")
                        st.write(f"**Has Explicit Background:** {detection['has_explicit_background']}")
                        st.write(f"**Classes Found:** {detection['class_names']}")
                        st.write(f"**Recommendation:** {recommendation['recommendation']}")
                        
                        if detection['type'] == 'polygon':
                            st.info(" **Polygon Annotations Detected**")
                            st.write("- Background is automatically class 0 (unlabeled areas)")
                            st.write("- Training script will use polygon-specific handling")
                        elif detection['type'] == 'brush':
                            st.info(" **Brush Annotations Detected**")
                            if detection['has_explicit_background']:
                                st.write("- Background is explicitly defined in annotations")
                                st.write("- Training script will use brush-specific handling with explicit background")
                            else:
                                st.write("- No explicit background class found")
                                st.write("- Training script will use brush-specific handling without background")
                        else:
                            st.warning(" **Mixed/Unknown Annotation Types**")
                            st.write("- Multiple annotation types detected")
                            st.write("- Training script will default to brush handling")
                    
                except Exception as e:
                    st.error(f"Error detecting annotation type: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
        
        # Class Configuration Section
        st.subheader(" Class Configuration")
        
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
                new_name = st.text_input(f" {i+1}. Class name:", value=class_name, key=f"class_{i}")
                if new_name.strip():
                    edited_classes.append(new_name.strip())
                else:
                    edited_classes.append(class_name)
            
            # Save class configuration
            if st.button(" Save Class Configuration", key="save_class_config"):
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
                    st.success(" Class configuration saved!")
                except Exception as e:
                    st.error(f"Error saving class configuration: {str(e)}")
        
        # Legacy class detection (fallback)
        elif st.button(" Detect Classes from Label Studio (Legacy)"):
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
                st.write(f" • **{class_name}**: {count} annotations")
            
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
                    st.write(f" {i+1}. **{class_name}** (Background - cannot be changed)")
                else:
                    # Allow reordering of object classes
                    new_name = st.text_input(f" {i+1}. Class name:", value=class_name, key=f"class_{i}")
                    if new_name.strip():
                        edited_classes.append(new_name.strip())
                    else:
                        edited_classes.append(class_name)
            
            # Save class configuration
            if st.button(" Save Class Configuration", key="save_legacy_class_config"):
                try:
                    config = {
                        'class_names': edited_classes,
                        'detected_classes': stats['classes'],
                        'total_annotations': stats['total_annotations']
                    }
                    
                    with open("/app/class_config.json", 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    st.session_state.class_config = config
                    st.success(" Class configuration saved!")
                except Exception as e:
                    st.error(f"Error saving class configuration: {str(e)}")
        
        # Show training info
        with st.expander(" Training Configuration", expanded=False):
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
                st.write("**Training:** 100 epochs with validation")
                st.write("**Disk Space:** Only saves final model (efficient)")
                st.write("**Progress:** Real-time updates in Streamlit")
                st.write("**Background:** Training runs in isolated process (no VM freeze)")
            else:
                st.write("**Model Architecture:** U-Net with ResNet101 backbone")
                st.write("**Classes:** Not configured yet - detect classes first")
                st.write("**Annotation Type:** Will be auto-detected")
                st.write("**Training Script:** Will be auto-selected based on annotation type")
                st.write("**Data Source:** MinIO bucket with LabelStudio annotations")
                st.write("**Training:** 100 epochs with validation")
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
                st.info(" Detected running training process - reconnected!")
        
        # Training control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.training_running:
                # Check if class configuration exists
                if 'class_config' not in st.session_state:
                    st.warning(" Please configure classes before starting training")
                    if st.button(" Start Training", type="primary", disabled=True):
                        pass
                else:
                    if st.button(" Start Training", type="primary"):
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
                if st.button("⏹ Stop Training"):
                    success, message = training_service.stop_training()
                    if success:
                        st.session_state.training_running = False
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to stop training: {message}")
        
        # Show training status
        if st.session_state.training_running:
            st.info(" Training is running in isolated process... (This may take 15-30 minutes)")
            
            # Manual refresh button
            if st.button(" Refresh Status"):
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
            total_epochs = status.get('total_epochs', 100)
            status_text.text(f"Training epoch {st.session_state.current_epoch}/{total_epochs} ({progress:.1f}%)")
            
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
            with st.expander(" Training Log", expanded=True):
                log_placeholder = st.empty()
                if st.session_state.training_log:
                    log_text = "\n".join(st.session_state.training_log[-20:])  # Show last 20 lines
                    log_placeholder.text(log_text)
                else:
                    log_placeholder.text("Starting training...")
            

            
            # Check if training completed
            if status.get('status') == 'completed':
                st.session_state.training_running = False
                st.success(" Training completed successfully!")
                st.info("Check models/checkpoints/ for the final model.")
                
                # Show final log
                with st.expander(" Final Training Log", expanded=False):
                    st.text("\n".join(st.session_state.training_log[-50:]))
                    
            elif status.get('status') == 'failed':
                st.session_state.training_running = False
                st.error(" Training failed!")
                
                # Show error log
                with st.expander(" Error Log", expanded=True):
                    st.text("\n".join(st.session_state.training_log[-50:]))
            
            # Auto-refresh every 5 seconds (increased from 3)
            time.sleep(5)
            st.experimental_rerun()
        
        else:
            # Show completion message or start button
            if st.session_state.training_log and any(" Training complete!" in log for log in st.session_state.training_log):
                st.success(" Training completed successfully!")
                st.info("Check models/checkpoints/ for the final model.")
                
                # Show final log
                with st.expander(" Final Training Log", expanded=False):
                    st.text("\n".join(st.session_state.training_log[-50:]))
            else:
                st.info("Click 'Start Training' to begin the training process.")

    elif st.session_state.current_step == "inference":
        st.markdown("<h1 style='text-align: center;'>Evaluate model</h1>", unsafe_allow_html=True)
        
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
                st.success(f" Loaded model config: {num_classes} classes - {', '.join(class_names)}")
            except Exception as e:
                st.warning(f" Could not load model config: {e}. Using default configuration.")
                class_names = ["Background"]
                num_classes = 1
        else:
            st.warning(" No model config found. Using default configuration.")
            class_names = ["Background"]
            num_classes = 1

        # Show class information
        st.info(f"**Model Configuration:** {num_classes} classes - {', '.join(class_names)}")
        
        # Create ModelConfig with the correct number of classes
        config = ModelConfig(num_classes=num_classes, class_names=class_names)
        
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
        st.info(f"Supported formats: {', '.join(get_supported_formats()).upper()}. Non-PNG images will be automatically converted.")
        uploaded_image = st.file_uploader("Image", type=get_supported_formats(), key="image_uploader",
                                          help="Upload an image in any supported format. Non-PNG images will be automatically converted to PNG.")
        
        # GT mask options
        st.subheader("Evaluation Mode")
        gt_option = st.radio(
            "Choose evaluation mode:",
            ["Use Label Studio annotations", "No GT (inference only)"],
            help="Evaluate with Label Studio annotations or run inference without ground truth"
        )

        if gt_option == "Use Label Studio annotations":
            st.info("Will use annotations from storage for evaluation")
            
            # Add batch evaluation option
            if st.button(" Run Batch Evaluation on Label Studio Data"):
                with st.spinner("Running batch evaluation on Label Studio annotations..."):
                    try:
                        from models.inference_batch import batch_evaluate_with_minio_annotations
                        
                        # Run batch evaluation directly from MinIO annotations
                        print("=" * 60)
                        print(" RUNNING BATCH EVALUATION FROM MINIO ANNOTATIONS")
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
                                with st.expander(" Debug Information", expanded=True):
                                    for debug_line in results['debug_info']:
                                        st.text(debug_line)
                            
                            # Display results in a nice format
                            st.subheader(" Batch Evaluation Results")
                            
                            # Create metrics display
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Images Evaluated", results['images_evaluated'])
                                st.metric("Classes", results['num_classes'])
                            
                            with col2:
                                st.metric("Mean IoU", f"{results['overall_mean_iou']:.3f}")
                                st.metric("Model", selected_model)
                            
                            # Show detailed results
                            with st.expander(" Detailed Metrics", expanded=True):
                                st.write("**Class-wise IoU scores:**")
                                for i, miou in enumerate(results['mean_ious']):
                                    class_name = results['class_names'][i] if i < len(results['class_names']) else f"Class {i}"
                                    st.write(f" {class_name}: {miou:.4f}")
                                
                                # Show object-wise metrics if available
                                if results['avg_metrics']:
                                    st.write("**Object-wise metrics:**")
                                    for class_name, metrics in results['avg_metrics'].items():
                                        st.write(f" **{class_name}:**")
                                        st.write(f" Precision: {metrics['precision']:.4f}")
                                        st.write(f" Recall: {metrics['recall']:.4f}")
                                        st.write(f" F1-Score: {metrics['f1']:.4f}")
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
                # Convert to PNG if needed
                detected_format, _ = detect_image_format(uploaded_image)
                if detected_format not in ['PNG', 'UNKNOWN']:
                    st.info(f"Converting {uploaded_image.name} from {detected_format} to PNG...")
                    converted_data, _ = convert_to_png(uploaded_image)
                    image_bytes = converted_data.read()
                    st.success(f"Converted to PNG successfully")
                else:
                    image_bytes = uploaded_image.read()
                image = process_image(image_bytes)

                if gt_option == "No GT (inference only)":
                    pred_masks = inferencer.predict(image)
                    overlayed_pred, _ = inferencer.create_visualization(image, pred_masks)
                    st.image(overlayed_pred, caption="Predicted Mask", use_column_width=True)
                    st.info("No ground truth provided - inference only mode")
                else:
                    # Use Label Studio annotations mode
                    st.info(" Use the 'Run Batch Evaluation on Label Studio Data' button above to evaluate with annotated images.")

    elif st.session_state.current_step == "batch_analysis":
        st.markdown("<h1 style='text-align: center;'>BATCH ANALYSIS</h1>", unsafe_allow_html=True)
        st.info("Upload images for quantitative segmentation analysis. "
                "This counts objects and measures their area per class. "
                "No ground truth comparison is performed.")

        # Model selection
        checkpoints_dir = "models/checkpoints"
        model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]

        if not model_files:
            st.error(f"No model files (.pth) found in {checkpoints_dir}")
            st.info("Please train a model first in the training section.")
            st.stop()

        selected_model = st.selectbox("Select model", model_files,
                                       key="ba_model_select")
        model_path = os.path.join(checkpoints_dir, selected_model)

        # Load model config
        config_path = model_path.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                class_names = model_config.get('class_names', ["Background"])
                num_classes = len(class_names)
                st.success(f"Model config: {num_classes} classes - {', '.join(class_names)}")
            except Exception as e:
                st.warning(f"Could not load model config: {e}")
                class_names = ["Background"]
                num_classes = 1
        else:
            st.warning("No model config found. Using default configuration.")
            class_names = ["Background"]
            num_classes = 1

        # Threshold slider
        if 'ba_threshold' not in st.session_state:
            st.session_state.ba_threshold = 0.3

        ba_threshold = st.slider(
            "Segmentation Threshold", 0.0, 1.0,
            value=st.session_state.ba_threshold, step=0.05,
            key='ba_threshold_slider',
            help="Higher values = more selective segmentation."
        )
        st.session_state.ba_threshold = ba_threshold

        # Minimum object area filter
        min_area = st.number_input(
            "Minimum object area (pixels)",
            min_value=0, max_value=10000, value=100, step=50,
            key='ba_min_area',
            help="Objects smaller than this pixel count are ignored."
        )

        # File upload
        st.subheader("Upload Images for Analysis")
        st.info(f"Supported formats: {', '.join(get_supported_formats()).upper()}. Non-PNG images will be automatically converted.")
        uploaded_files = st.file_uploader(
            "Choose images",
            accept_multiple_files=True,
            type=get_supported_formats(),
            key="ba_file_uploader",
            help="Upload images in any supported format. Non-PNG images will be automatically converted to PNG before analysis."
        )

        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} image(s) for analysis.")

            if st.button("Run Batch Analysis", key="ba_run"):
                # Load model (cached)
                @st.cache_resource
                def get_ba_inferencer(_model_path, _num_classes,
                                      _class_names, _threshold):
                    _config = ModelConfig(
                        num_classes=_num_classes,
                        class_names=list(_class_names)
                    )
                    cfg_file = _model_path.replace('.pth', '_config.json')
                    if os.path.exists(cfg_file):
                        try:
                            with open(cfg_file, 'r') as f:
                                mc = json.load(f)
                            if 'encoder_name' in mc:
                                _config.encoder_name = mc['encoder_name']
                        except Exception:
                            pass
                    return Inferencer(_model_path, _config,
                                     threshold=_threshold)

                try:
                    inferencer = get_ba_inferencer(
                        model_path, num_classes,
                        tuple(class_names), ba_threshold
                    )
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.stop()

                # Process images
                all_results = []
                all_overlays = []
                filenames = []

                progress = st.progress(0)
                status = st.empty()
                converted_count = 0

                for idx, uploaded_file in enumerate(uploaded_files):
                    status.text(
                        f"Analyzing {uploaded_file.name} "
                        f"({idx+1}/{len(uploaded_files)})..."
                    )

                    try:
                        detected_format, _ = detect_image_format(uploaded_file)
                        if detected_format not in ['PNG', 'UNKNOWN']:
                            converted_data, _ = convert_to_png(uploaded_file)
                            file_bytes = converted_data.read()
                            converted_count += 1
                        else:
                            file_bytes = uploaded_file.read()
                        image = cv2.imdecode(
                            np.frombuffer(file_bytes, np.uint8), 1
                        )
                        if image is None:
                            st.warning(f"Could not decode {uploaded_file.name}")
                            continue

                        # predict() expects BGR input
                        pred_masks = inferencer.predict(image)

                        # Convert to RGB for display/overlay
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Analyze
                        analysis = analyze_single_image(
                            pred_masks, class_names,
                            min_object_area=min_area
                        )

                        # Create overlay
                        overlay = create_analysis_overlay(
                            image_rgb, pred_masks,
                            class_names, analysis,
                            min_object_area=min_area
                        )

                        all_results.append(analysis)
                        all_overlays.append(overlay)
                        filenames.append(uploaded_file.name)

                    except Exception as e:
                        st.warning(
                            f"Error processing {uploaded_file.name}: {e}"
                        )

                    progress.progress((idx + 1) / len(uploaded_files))

                status.text("Analysis complete!")
                progress.progress(1.0)
                if converted_count > 0:
                    st.info(f"Converted {converted_count} image(s) to PNG before analysis.")

                # Store results in session state
                st.session_state.ba_results = all_results
                st.session_state.ba_overlays = all_overlays
                st.session_state.ba_filenames = filenames

        # Display results (persists across reruns)
        if 'ba_results' in st.session_state and st.session_state.ba_results:
            all_results = st.session_state.ba_results
            all_overlays = st.session_state.ba_overlays
            filenames = st.session_state.ba_filenames

            st.subheader("Summary Table")

            import pandas as pd
            summary_df = aggregate_batch_results(all_results, filenames)
            st.dataframe(summary_df, use_container_width=True)

            # CSV Export
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name="batch_analysis_results.csv",
                mime="text/csv",
                key="ba_csv_download"
            )

            # Per-image detail view
            st.subheader("Per-Image Details")

            selected_image = st.selectbox(
                "Select image to inspect",
                filenames,
                key="ba_image_select"
            )

            if selected_image:
                img_idx = filenames.index(selected_image)
                result = all_results[img_idx]
                overlay = all_overlays[img_idx]

                st.image(overlay, caption=f"Analysis: {selected_image}",
                         use_column_width=True)

                for class_name, class_data in result['classes'].items():
                    with st.expander(
                        f"{class_name}: {class_data['object_count']} objects, "
                        f"{class_data['total_area_percent']:.2f}% area",
                        expanded=True
                    ):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Objects", class_data['object_count'])
                        col2.metric("Area (px)",
                                    f"{class_data['total_area_pixels']:,}")
                        col3.metric("Area (%)",
                                    f"{class_data['total_area_percent']:.2f}%")

                        if class_data['objects']:
                            obj_df = pd.DataFrame([
                                {
                                    'Object ID': o['id'],
                                    'Area (px)': o['area_pixels'],
                                    'Area (%)': f"{o['area_percent']:.3f}",
                                    'Bounding Box': str(o['bbox'])
                                }
                                for o in class_data['objects']
                            ])
                            st.dataframe(obj_df, use_container_width=True)

if __name__ == "__main__":

    main()

