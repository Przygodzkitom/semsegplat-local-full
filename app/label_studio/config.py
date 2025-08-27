import os
from label_studio_sdk import Client
from google.cloud import storage
import streamlit as st
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def verify_label_studio_connection():
    """Verify Label Studio connection and credentials"""
    try:
        # Check if Label Studio is responding
        try:
            health = requests.get("http://labelstudio:8080/api/health", timeout=5)
            if health.status_code != 200:
                return False
        except:
            return False
            
        # Check for API Key
        if not os.getenv("LABEL_STUDIO_API_KEY"):
            st.error("LABEL_STUDIO_API_KEY not found in .env file.")
            return False

        try:
            # Verify API key is valid by fetching projects
            client = Client(
                url="http://labelstudio:8080",
                api_key=os.getenv("LABEL_STUDIO_API_KEY")
            )
            client.get_projects()
            return True
        except Exception as e:
            st.error(f"Failed to connect with API Key: {e}")
            return False
    except Exception as e:
        st.error(f"Connection verification error: {str(e)}")
        return False

def setup_minio_storage(bucket_name):
    """Setup MinIO S3-compatible storage connection for Label Studio"""
    if not bucket_name:
        st.warning("No MinIO bucket name provided")
        return None
        
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        return {
            "bucket": bucket_name,
            "prefix": "images",  # Changed from "upload" to "images" to match our structure
            "regex": ".*\\.(jpg|jpeg|png|gif|bmp)$"  # More specific regex for images
        }
    except Exception as e:
        st.error(f"GCS Error: {str(e)}")
        return None

def create_label_studio_project(project_name, class_names, bucket_name=None):
    """Create a Label Studio project with optional GCS storage"""
    try:
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        if not api_key:
            raise ValueError("LABEL_STUDIO_API_KEY not set in environment.")
            
        # Initialize client with token
        client = Client(
            url="http://labelstudio:8080",
            api_key=api_key
        )
        
        # Generate label config for semantic segmentation
        label_config = f"""
        <View>
          <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
          <BrushLabels name="label" toName="image">
            {''.join([f'<Label value="{c}" background="#{hash(c) % 0xFFFFFF:06x}"/>' for c in class_names])}
          </BrushLabels>
          <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA">
            <Header value="Instructions"/>
            <Text name="instructions" value="Paint over the regions in the image. Use the mouse wheel to zoom in/out."/>
          </View>
        </View>
        """
        
        # Create project with additional settings
        project = client.create_project(
            title=project_name,
            label_config=label_config,
            params={
                'show_instruction': True,
                'show_skip_button': True,
                'enable_empty_annotation': True,
                'show_annotation_history': True,
                'auto_save_annotation': True
            }
        )
        
        # Setup storage if GCS bucket is provided
        if bucket_name:
            storage_settings = setup_gcs_storage(bucket_name)
            if storage_settings:
                # Configure source storage (where images are stored)
                source_storage = {
                    "type": "gcs",
                    "bucket": storage_settings["bucket"],
                    "prefix": storage_settings["prefix"],
                    "regex": storage_settings["regex"],
                    "use_blob_urls": True,
                    "recursive_scan": True
                }
                
                # Configure target storage (where annotations are stored)
                target_storage = {
                    "type": "gcs",
                    "bucket": storage_settings["bucket"],
                    "prefix": "annotations",
                    "create_local_copy": False
                }
                
                # Update project with storage settings
                project.params.update({
                    "source_storage": source_storage,
                    "target_storage": target_storage
                })
                
                st.success(f"✅ GCS storage configured for project '{project_name}'")
                st.info(f"Source: gs://{bucket_name}/{storage_settings['prefix']}")
                st.info(f"Target: gs://{bucket_name}/annotations")
        
        return project.id
        
    except Exception as e:
        st.error(f"Error creating project: {str(e)}")
        if hasattr(e, 'response'):
            st.error(f"Response from server: {e.response.text if e.response else 'No response'}")
        raise

def upload_to_gcs(file_data, bucket_name, destination_blob_name):
    """Upload a file to GCS bucket"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_file(file_data)
        return blob.public_url
    except Exception as e:
        st.error(f"Error uploading to GCS: {str(e)}")
        raise

def sync_images_to_label_studio(project_id, bucket_name):
    """Sync images from GCS to Label Studio project"""
    try:
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        if not api_key:
            raise ValueError("LABEL_STUDIO_API_KEY not set in environment.")
        
        client = Client(
            url="http://labelstudio:8080",
            api_key=api_key
        )
        
        project = client.get_project(project_id)
        
        # Get storage configuration
        storage_config = project.params.get("source_storage", {})
        if storage_config.get("type") != "gcs":
            st.warning("Project is not configured with GCS storage")
            return False
        
        # Sync storage
        sync_result = project.sync_storage()
        st.success(f"✅ Synced {sync_result.get('count', 0)} images from GCS to Label Studio")
        return True
        
    except Exception as e:
        st.error(f"Error syncing images: {str(e)}")
        return False

def get_project_images(project_id):
    """Get list of images in a Label Studio project"""
    try:
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        if not api_key:
            raise ValueError("LABEL_STUDIO_API_KEY not set in environment.")
        
        client = Client(
            url="http://labelstudio:8080",
            api_key=api_key
        )
        
        project = client.get_project(project_id)
        tasks = project.get_tasks()
        
        images = []
        for task in tasks:
            if 'image' in task['data']:
                images.append({
                    'id': task['id'],
                    'url': task['data']['image'],
                    'annotations': len(task.get('annotations', []))
                })
        
        return images
        
    except Exception as e:
        st.error(f"Error getting project images: {str(e)}")
        return []
