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
import threading
from pathlib import Path

print("=" * 80)
print(" MAIN.PY LOADED - DEBUG VERSION 2024")
print("=" * 80)
from models.config import ModelConfig
from models.inference_single import Inferencer
from models.utils.gpu_detector import detect_gpu, print_device_info
from app.storage_manager import get_storage_manager
from app.label_studio.config import create_label_studio_project, sync_images_to_label_studio, get_project_images, push_images_to_label_studio
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

col_l, col_m, col_r = st.columns([1, 1, 1])
with col_m:
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











def _upload_worker(raw_files, image_prefix, progress):
    """Background thread: convert images to PNG and upload to MinIO.
    Updates the shared progress dict — no Streamlit calls allowed here."""
    import io as _io
    from PIL import Image as _Image

    try:
        s3 = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin123',
            region_name='us-east-1',
        )
    except Exception as e:
        progress['errors'].append(f"Storage connection failed: {e}")
        progress['in_progress'] = False
        progress['complete'] = False
        return

    batch_ts = int(time.time() * 1000)

    for i, (file_bytes, original_name) in enumerate(raw_files):
        try:
            buf = _io.BytesIO(file_bytes)
            with _Image.open(buf) as img:
                fmt = (img.format or 'PNG').upper()
                if fmt != 'PNG':
                    if img.mode in ('I', 'F'):
                        arr = np.array(img, dtype=np.float32)
                        mn, mx = arr.min(), arr.max()
                        arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros(arr.shape, dtype=np.uint8)
                        img = _Image.fromarray(arr, mode='L')
                    elif img.mode == 'RGBA':
                        bg = _Image.new('RGB', img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1])
                        img = bg
                    elif img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                    out = _io.BytesIO()
                    img.save(out, format='PNG', optimize=True)
                    out.seek(0)
                    final_name = os.path.splitext(original_name)[0] + '.png'
                else:
                    out = _io.BytesIO(file_bytes)
                    final_name = original_name

            unique = f"{batch_ts}_{i}_{final_name}"
            s3.upload_fileobj(out, "segmentation-platform", f"{image_prefix}{unique}")
            progress['uploaded'].append(f"{image_prefix}{unique}")
        except Exception as e:
            progress['errors'].append(f"{original_name}: {str(e)}")

        progress['done'] = i + 1

    # Push to Label Studio if project already configured
    try:
        ls_id = progress.get('ls_project_id')
        if progress['uploaded'] and ls_id:
            push_images_to_label_studio(ls_id, progress['uploaded'])
    except Exception:
        pass

    progress['in_progress'] = False
    progress['complete'] = len(progress['uploaded']) == len(raw_files)


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

def _render_loss_chart(train_losses, val_losses):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color='#f97316', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, label='Val Loss', color='#3b82f6', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


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

        image_prefix = "images/"

        # Initialise upload progress tracker
        if 'upload_progress' not in st.session_state:
            st.session_state.upload_progress = {}
        progress = st.session_state.upload_progress

        if progress.get('in_progress'):
            # Upload running in background — show live progress and auto-refresh
            done = progress.get('done', 0)
            total = progress.get('total', 0)
            st.info(f"Uploading images in background… {done} / {total} done")
            st.progress(done / total if total > 0 else 0)
            time.sleep(1)
            st.experimental_rerun()

        elif progress.get('complete') is True:
            st.success(f"All {progress.get('total', 0)} images uploaded successfully.")
            if progress.get('errors'):
                for err in progress['errors']:
                    st.warning(err)
            if st.button("Upload more images"):
                st.session_state.upload_progress = {}
                st.experimental_rerun()

        elif progress.get('complete') is False:
            st.error(f"Upload finished with {len(progress.get('errors', []))} error(s).")
            for err in progress.get('errors', []):
                st.write(f"• {err}")
            if st.button("Retry / upload new images"):
                st.session_state.upload_progress = {}
                st.experimental_rerun()

        else:
            # Idle — show file uploader
            existing_project_images = st.session_state.get('existing_images', [])
            if existing_project_images:
                st.info(f"Found {len(existing_project_images)} existing images")
            else:
                st.info("No existing images found")

            st.subheader("Image Upload & Format Conversion")
            supported_formats = get_supported_formats()
            st.info(f"Supported formats: {', '.join(supported_formats).upper()}")
            st.warning(
                "**Note on automatic conversion:** While the app can convert most formats to PNG, "
                "some images may not convert correctly — in particular **dim or low-contrast TIFFs** "
                "(e.g. microscopy, scientific, or HDR images) where the automatic brightness normalization "
                "may not match your expectations. "
                "For best results, **convert your images to PNG manually before uploading** "
                "(e.g. using ImageJ, FIJI, or any image editor) so you can verify they look correct."
            )

            uploaded_files = st.file_uploader(
                "Choose images",
                accept_multiple_files=True,
                type=supported_formats,
                help="Upload images in any supported format. They will be converted to PNG if needed."
            )

            if uploaded_files and len(uploaded_files) > 50:
                st.warning(f"**Large batch detected:** {len(uploaded_files)} files. Processing may take several minutes.")

            if uploaded_files:
                # Read file bytes immediately — the uploader widget disappears on navigation
                raw_files = [(f.read(), f.name) for f in uploaded_files]

                new_progress = {
                    'in_progress': True,
                    'done': 0,
                    'total': len(raw_files),
                    'errors': [],
                    'uploaded': [],
                    'complete': None,
                    'ls_project_id': st.session_state.get('label_studio_project_id'),
                }
                st.session_state.upload_progress = new_progress

                t = threading.Thread(
                    target=_upload_worker,
                    args=(raw_files, image_prefix, new_progress),
                    daemon=True,
                )
                t.start()
                st.experimental_rerun()

    elif st.session_state.current_step == "annotate":
        st.markdown("<h1 style='text-align: center;'>ANNOTATE IMAGES</h1>", unsafe_allow_html=True)
        
        # Check if project is already configured (simple check - no validation needed)
        if 'label_studio_project_id' in st.session_state and st.session_state.label_studio_project_id:
            project_id = st.session_state.label_studio_project_id
            project_name = st.session_state.get('label_studio_project_name', 'Unknown')
            
            st.markdown(
                f'<a href="http://localhost:8080/projects/{project_id}/data" target="_blank">'
                '<button style="background-color:#FF4B4B;color:white;border:none;padding:0.5em 1em;'
                'border-radius:0.4em;font-size:1em;font-weight:bold;cursor:pointer;width:100%;">'
                'Open Project in Label Studio</button></a>',
                unsafe_allow_html=True
            )
            return
        
        # Label Studio Setup Section (only shown if no project is configured)
        st.subheader("Create new Label Studio project")

        project_name = "semantic-segmentation"
        project_description = "Automated semantic segmentation project with MinIO storage"

        # --- Annotation mode ---
        st.markdown("#### Annotation tool")
        annotation_type = st.radio(
            "Choose how annotators will draw regions:",
            options=["brush", "polygon"],
            format_func=lambda x: "Brush (paint pixels)" if x == "brush" else "Polygon (draw outlines)",
            horizontal=True,
        )

        # --- Background handling note ---
        if annotation_type == "brush":
            st.info(
                "**Brush mode:** any pixel you leave unpainted is automatically treated as background "
                "by the training pipeline — you do not need to paint background areas. "
                "Adding a 'Background' class here is redundant."
            )
        else:
            st.info(
                "**Polygon mode:** background is implicit — any area not covered by a polygon is "
                "treated as background automatically. Do not add a Background class."
            )

        # --- Class editor ---
        st.markdown("#### Classes")
        st.caption("Define the foreground label classes for this project. At least one class is required.")

        # Initialise class list in session state — default depends on annotation type
        if "ls_classes" not in st.session_state or st.session_state.get("ls_classes_annotation_type") != annotation_type:
            if annotation_type == "brush":
                st.session_state.ls_classes = [
                    {"name": "Object", "color": "#ff0000"},
                ]
            else:
                st.session_state.ls_classes = [
                    {"name": "Object", "color": "#ff0000"},
                ]
            st.session_state.ls_classes_annotation_type = annotation_type

        def _add_class():
            st.session_state.ls_classes.append(
                {"name": f"Class {len(st.session_state.ls_classes) + 1}", "color": "#00aa00"}
            )

        def _delete_class(idx):
            st.session_state.ls_classes.pop(idx)

        # Render editable rows
        for i, cls in enumerate(st.session_state.ls_classes):
            col_name, col_color, col_del = st.columns([4, 2, 1])
            with col_name:
                new_name = st.text_input(
                    f"Class name {i+1}", value=cls["name"], key=f"cls_name_{i}", label_visibility="collapsed"
                )
                st.session_state.ls_classes[i]["name"] = new_name
            with col_color:
                new_color = st.color_picker(
                    f"Color {i+1}", value=cls["color"], key=f"cls_color_{i}", label_visibility="collapsed"
                )
                st.session_state.ls_classes[i]["color"] = new_color
            with col_del:
                if len(st.session_state.ls_classes) > 1:
                    st.button("✕", key=f"cls_del_{i}", on_click=_delete_class, args=(i,))

        st.button("+ Add class", on_click=_add_class)

        # --- Setup button ---
        st.markdown("---")
        _up = st.session_state.get('upload_progress', {})
        _uploading = _up.get('in_progress', False)
        _upload_failed = _up.get('complete') is False
        uploads_blocked = _uploading or _upload_failed
        if _uploading:
            st.info("Images are still uploading in the background. Please wait until the upload completes before setting up Label Studio.")
        elif _upload_failed:
            st.warning("Some images failed to upload. Please go back to Upload Images and resolve the errors before setting up Label Studio.")
        if st.button("Setup Label Studio Project", type="primary", use_container_width=True, disabled=uploads_blocked):
            class_names = [c["name"] for c in st.session_state.ls_classes if c["name"].strip()]
            class_colors = [c["color"] for c in st.session_state.ls_classes if c["name"].strip()]

            # Training dataloaders expect Background at index 0 — prepend it here.
            # The LS label config only lists the user-defined foreground classes
            # (background is implicit in both modes); the saved class_config.json
            # has Background at index 0 so the training scripts get correct indices.
            training_class_names = ["Background"] + class_names

            if not class_names:
                st.error("Please add at least one class before setting up.")
            else:
                try:
                    from app.label_studio.auto_config import LabelStudioAutoConfig

                    auto_config = LabelStudioAutoConfig(base_url="http://label-studio:8080")

                    with st.spinner("Setting up Label Studio project automatically..."):
                        project_id = auto_config.auto_setup_project(
                            project_name,
                            project_description,
                            annotation_type=annotation_type,
                            class_names=class_names,
                            class_colors=class_colors,
                        )

                    if project_id:
                        st.session_state.label_studio_project_id = project_id
                        st.session_state.label_studio_project_name = project_name
                        save_project_config(project_id, project_name, project_description)

                        # Save class config for training — Background at index 0, then user classes
                        class_config = {
                            "class_names": training_class_names,
                            "annotation_type": annotation_type,
                            "background_handling": "implicit",
                        }
                        try:
                            with open("/app/class_config.json", "w") as f:
                                json.dump(class_config, f, indent=2)
                            st.session_state.class_config = class_config
                        except Exception as e:
                            st.warning(f"Could not save class config: {e}")

                        st.success(f"Project setup complete! Project ID: {project_id}")
                        st.markdown(
                            f'<a href="http://localhost:8080/projects/{project_id}/data" target="_blank">'
                            '<button style="background-color:#FF4B4B;color:white;border:none;padding:0.5em 1em;'
                            'border-radius:0.4em;font-size:1em;font-weight:bold;cursor:pointer;width:100%;">'
                            'Open Project in Label Studio</button></a>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Setup failed. Check the messages above for details.")

                except Exception as e:
                    st.error(f"Error during setup: {str(e)}")
                    st.info("Make sure Label Studio is running and accessible at http://localhost:8080")

    elif st.session_state.current_step == "train":
        st.markdown("<h1 style='text-align: center;'>TRAIN MODEL</h1>", unsafe_allow_html=True)
        
        # Define bucket name and use root bucket
        bucket_name = "segmentation-platform"
        selected_project = "root"
        annotation_prefix = "annotations/"
        
        # Import class detector
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'utils'))
            from class_detector import ClassDetector
            class_detector = ClassDetector(bucket_name, annotation_prefix)
        except Exception as e:
            st.error(f"Error importing class detector: {str(e)}")
            st.stop()
        

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
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>Before training, click the button below to detect annotation type and classes from Label Studio.</p>", unsafe_allow_html=True)
        st.markdown("""
        <style>
        div.element-container:has(span#detect-btn-anchor) + div.element-container div.stButton > button,
        div.element-container:has(span#detect-btn-anchor) + div.element-container div.stButton > button * {
            background-color: #E65100 !important;
            color: white !important;
            font-size: 1.4rem !important;
            font-weight: 800 !important;
            width: 100% !important;
            height: 3.2rem !important;
            line-height: 3.2rem !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 8px !important;
            border: none !important;
        }
        div.element-container:has(span#detect-btn-anchor) + div.element-container div.stButton > button:hover,
        div.element-container:has(span#detect-btn-anchor) + div.element-container div.stButton > button:hover * {
            background-color: #FF6D00 !important;
            color: white !important;
        }
        </style>
        <span id="detect-btn-anchor"></span>
        """, unsafe_allow_html=True)
        if st.button(" Detect Annotation Type & Classes", use_container_width=True):
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

                    # Save class configuration automatically
                    config = {
                        'class_names': recommendation['class_names'],
                        'annotation_type': recommendation['annotation_type'],
                        'background_handling': recommendation['background_handling'],
                        'detection_result': detection
                    }
                    with open("/app/class_config.json", 'w') as f:
                        json.dump(config, f, indent=2)
                    st.session_state.class_config = config

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Annotation Type", detection['type'].title())
                    with col2:
                        st.metric("Classes Found", len(detection['class_names']))

                    with st.expander(" Detection Results", expanded=True):
                        st.write(f"**Annotation Type:** {detection['type'].title()}")
                        st.write(f"**Classes Found:** {detection['class_names']}")
                        st.info("Background class is always derived from unannotated pixels — you do not need to annotate background regardless of annotation type.")

                except Exception as e:
                    st.error(f"Error detecting annotation type: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
        
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
        if not st.session_state.training_running:
            st.markdown("""
            <style>
            div.element-container:has(span#start-training-btn-anchor) + div.element-container div.stButton > button,
            div.element-container:has(span#start-training-btn-anchor) + div.element-container div.stButton > button * {
                background-color: #E65100 !important;
                color: white !important;
                font-size: 1.4rem !important;
                font-weight: 800 !important;
                width: 100% !important;
                height: 3.2rem !important;
                line-height: 3.2rem !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                border-radius: 8px !important;
                border: none !important;
            }
            div.element-container:has(span#start-training-btn-anchor) + div.element-container div.stButton > button:hover,
            div.element-container:has(span#start-training-btn-anchor) + div.element-container div.stButton > button:hover * {
                background-color: #FF6D00 !important;
                color: white !important;
            }
            div.element-container:has(span#start-training-btn-anchor) + div.element-container div.stButton > button:disabled,
            div.element-container:has(span#start-training-btn-anchor) + div.element-container div.stButton > button:disabled * {
                background-color: #BF360C !important;
                opacity: 0.5 !important;
            }
            </style>
            <span id="start-training-btn-anchor"></span>
            """, unsafe_allow_html=True)
            if 'class_config' not in st.session_state:
                if st.button(" Start Training", disabled=True, use_container_width=True):
                    pass
            else:
                if st.button(" Start Training", use_container_width=True):
                    success, message = training_service.start_training()
                    if success:
                        st.session_state.training_running = True
                        st.session_state.training_log = []
                        st.session_state.current_epoch = 0
                        st.session_state.pop('loss_history', None)
                        try:
                            import os as _os
                            _os.remove("/app/loss_history.json")
                        except FileNotFoundError:
                            pass
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to start training: {message}")

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

            # Loss chart — read from dedicated file (written once per completed epoch)
            try:
                with open("/app/loss_history.json") as _lf:
                    _lh = json.load(_lf)
                train_losses = _lh.get('train_losses', [])
                val_losses = _lh.get('val_losses', [])
                if train_losses:
                    st.session_state.loss_history = {'train_losses': train_losses, 'val_losses': val_losses}
            except (FileNotFoundError, ValueError):
                train_losses = []
                val_losses = []
            display = st.session_state.get('loss_history', {})
            if display.get('train_losses'):
                _render_loss_chart(display['train_losses'], display.get('val_losses', []))

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

                saved_history = st.session_state.get('loss_history', {})
                saved_train = saved_history.get('train_losses', [])
                saved_val = saved_history.get('val_losses', [])
                if saved_train:
                    _render_loss_chart(saved_train, saved_val)

                # Show final log
                with st.expander(" Final Training Log", expanded=False):
                    st.text("\n".join(st.session_state.training_log[-50:]))

    elif st.session_state.current_step == "inference":
        st.markdown("<h1 style='text-align: center;'>EVALUATE MODEL</h1>", unsafe_allow_html=True)
        
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
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>Select model from a dropdown list</p>", unsafe_allow_html=True)
        selected_model = st.selectbox("", model_files)
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
        st.markdown("""
        <style>
        /* Slider track container */
        div[data-baseweb="slider"] div.st-d7 {
            height: 12px !important;
        }
        /* Slider track rail */
        div[data-baseweb="slider"] div.st-au {
            height: 12px !important;
        }
        /* Thumb */
        div[data-baseweb="slider"] div.css-1vzeuhh {
            width: 24px !important;
            height: 24px !important;
        }
        /* Value above thumb */
        div[data-baseweb="slider"] div.StyledThumbValue {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
        }
        /* Min/max tick labels */
        div[data-testid="stTickBarMin"],
        div[data-testid="stTickBarMax"] {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>Segmentation Threshold</p>", unsafe_allow_html=True)
        new_threshold = st.slider(
            "",
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

        # Section 1: inference on unseen image
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>If you want to check how the model performs on previously unseen images, drag and drop one here.</p>", unsafe_allow_html=True)
        st.info(f"Supported formats: {', '.join(get_supported_formats()).upper()}. Non-PNG images will be automatically converted.")
        uploaded_image = st.file_uploader("", type=get_supported_formats(), key="image_uploader",
                                          help="Upload an image in any supported format. Non-PNG images will be automatically converted to PNG.")

        # Section 2: batch evaluation on held-out test images
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>Evaluate the model on the held-out test set — images reserved before training and never used by the model.</p>", unsafe_allow_html=True)
        st.markdown("""
        <style>
        div.element-container:has(span#batch-eval-btn-anchor) + div.element-container div.stButton > button,
        div.element-container:has(span#batch-eval-btn-anchor) + div.element-container div.stButton > button * {
            background-color: #E65100 !important;
            color: white !important;
            font-size: 1.4rem !important;
            font-weight: 800 !important;
            width: 100% !important;
            height: 3.2rem !important;
            line-height: 3.2rem !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 8px !important;
            border: none !important;
        }
        div.element-container:has(span#batch-eval-btn-anchor) + div.element-container div.stButton > button:hover,
        div.element-container:has(span#batch-eval-btn-anchor) + div.element-container div.stButton > button:hover * {
            background-color: #FF6D00 !important;
            color: white !important;
        }
        </style>
        <span id="batch-eval-btn-anchor"></span>
        """, unsafe_allow_html=True)
        if st.button(" Run Batch Evaluation on Label Studio Data", use_container_width=True):
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
                            st.markdown("""
                            <style>
                            div.css-q8sbsg > p {
                                font-size: 1.8rem !important;
                                font-weight: 800 !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            with st.expander(" Detailed Metrics", expanded=True):
                                st.markdown("<p style='font-size:1.2rem; font-weight:800; margin-bottom:4px;'>Class-wise IoU scores</p>", unsafe_allow_html=True)
                                for i, miou in enumerate(results['mean_ious']):
                                    class_name = results['class_names'][i] if i < len(results['class_names']) else f"Class {i}"
                                    st.markdown(f"<p style='font-size:1.1rem; font-weight:600; margin:2px 0;'>&nbsp;&nbsp;{class_name}: <b>{miou:.4f}</b></p>", unsafe_allow_html=True)

                                # Show object-wise metrics if available
                                if results['avg_metrics']:
                                    st.markdown("<p style='font-size:1.2rem; font-weight:800; margin-top:12px; margin-bottom:4px;'>Object-wise metrics</p>", unsafe_allow_html=True)
                                    for class_name, metrics in results['avg_metrics'].items():
                                        st.markdown(f"<p style='font-size:1.15rem; font-weight:800; margin-top:8px;'>{class_name}</p>", unsafe_allow_html=True)
                                        cols = st.columns(3)
                                        cols[0].metric("Precision", f"{metrics['precision']:.4f}")
                                        cols[1].metric("Recall", f"{metrics['recall']:.4f}")
                                        cols[2].metric("F1-Score", f"{metrics['f1']:.4f}")
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

                pred_masks = inferencer.predict(image)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                overlayed_pred, _ = inferencer.create_visualization(image_rgb, pred_masks)
                st.image(overlayed_pred, caption="Predicted Mask", use_column_width=True)

    elif st.session_state.current_step == "batch_analysis":
        st.markdown("<h1 style='text-align: center;'>BATCH ANALYSIS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>Upload images for quantitative segmentation analysis. This counts objects and measures their area per class.</p>", unsafe_allow_html=True)

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

        st.markdown("""
        <style>
        div[data-baseweb="slider"] div.st-d7 {
            height: 12px !important;
        }
        div[data-baseweb="slider"] div.st-au {
            height: 12px !important;
        }
        div[data-baseweb="slider"] div.css-1vzeuhh {
            width: 24px !important;
            height: 24px !important;
        }
        div[data-baseweb="slider"] div.StyledThumbValue {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stTickBarMin"],
        div[data-testid="stTickBarMax"] {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2rem; font-weight:800;'>Segmentation Threshold</p>", unsafe_allow_html=True)
        ba_threshold = st.slider(
            "", 0.0, 1.0,
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
                            class_names,
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

