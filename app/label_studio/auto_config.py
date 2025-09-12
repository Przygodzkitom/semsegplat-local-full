import os
import requests
import json
import time
import streamlit as st
from typing import Dict, List, Optional, Tuple
import boto3
from botocore.exceptions import ClientError

class LabelStudioAutoConfig:
    """Automatic configuration for Label Studio projects with MinIO storage"""
    
    def __init__(self, base_url: str = None):
        # Use container name when running in Docker, localhost when running locally
        if base_url is None:
            # Check for LABEL_STUDIO_URL environment variable first
            env_url = os.getenv("LABEL_STUDIO_URL")
            if env_url:
                self.base_url = env_url
            elif os.getenv("DOCKER_ENV"):
                self.base_url = "http://label-studio:8080"
            else:
                self.base_url = "http://localhost:8080"
        else:
            self.base_url = base_url
            
        self.access_token = None
        self.refresh_token = None
        self.session = requests.Session()
        
        # MinIO configuration
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
        self.minio_bucket = os.getenv("MINIO_BUCKET_NAME", "segmentation-platform")
        
        # Debug: Show MinIO configuration
        st.info(f"🔧 MinIO Configuration:")
        st.info(f"   Endpoint: {self.minio_endpoint}")
        st.info(f"   Access Key: {self.minio_access_key}")
        st.info(f"   Secret Key: {self.minio_secret_key[:10]}...")
        st.info(f"   Bucket: {self.minio_bucket}")
        
        # Label Studio personal access token (refresh token)
        self.personal_access_token = os.getenv("LABEL_STUDIO_PERSONAL_ACCESS_TOKEN", "admin")
        
        # Default credentials (these should match your docker-compose setup)
        self.username = os.getenv("LABEL_STUDIO_USERNAME", "admin@example.com")
        self.password = os.getenv("LABEL_STUDIO_PASSWORD", "admin")
    
    def authenticate(self) -> bool:
        """Authenticate with Label Studio using personal access token"""
        try:
            # Debug: Show connection details
            st.info(f"🔗 Connecting to Label Studio at: {self.base_url}")
            st.info(f"🔑 Using personal access token")
            
            # Use the personal access token as the refresh token
            self.refresh_token = self.personal_access_token
            
            # Get access token using refresh token
            return self._refresh_access_token()
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False
    
    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        try:
            refresh_url = f"{self.base_url}/api/token/refresh/"
            refresh_data = {"refresh": self.refresh_token}
            
            response = self.session.post(refresh_url, json=refresh_data)
            if response.status_code != 200:
                st.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return False
            
            token_data = response.json()
            self.access_token = token_data.get("access")
            
            if not self.access_token:
                st.error("No access token received from refresh")
                return False
            
            # Set authorization header for future requests
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            })
            
            return True
            
        except Exception as e:
            st.error(f"Token refresh error: {str(e)}")
            return False
    
    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refresh if needed"""
        if not self.access_token:
            return self.authenticate()
        
        # Check if token is expired by making a test request
        try:
            response = self.session.get(f"{self.base_url}/api/projects/")
            if response.status_code == 401:
                # Token expired, refresh it
                return self._refresh_access_token()
            return True
        except Exception:
            return self._refresh_access_token()
    
    def authenticate_with_sdk(self) -> bool:
        """Alternative authentication using Label Studio SDK"""
        try:
            from label_studio_sdk import Client
            
            st.info(f"🔗 Connecting to Label Studio using SDK at: {self.base_url}")
            st.info(f"🔑 Using token: {self.personal_access_token[:10]}...")
            
            # Check if we have a valid token
            if not self.personal_access_token or self.personal_access_token == "admin":
                st.error("❌ No valid personal access token provided")
                return False
            
            # Try to connect using the SDK (which handles authentication differently)
            client = Client(
                url=self.base_url,
                api_key=self.personal_access_token
            )
            
            # Test connection by getting projects
            projects = client.get_projects()
            st.success("✅ SDK authentication successful!")
            return True
            
        except Exception as e:
            st.error(f"SDK authentication error: {str(e)}")
            return False
    
    def create_project(self, project_name: str, description: str = "") -> Optional[int]:
        """Create a new Label Studio project"""
        if not self._ensure_valid_token():
            return None
        
        try:
            # Generate label config for semantic segmentation
            label_config = self._generate_label_config()
            
            project_data = {
                "title": project_name,
                "description": description,
                "label_config": label_config,
                "enable_empty_annotation": True,
                "show_instruction": True,
                "show_skip_button": True,
                "show_annotation_history": True,
                "show_collab_predictions": True,
                "sampling": "Sequential sampling",
                "overlap_cohort_percentage": 100,
                "maximum_annotations": 1,
                "min_annotations_to_start_training": 1
            }
            
            response = self.session.post(
                f"{self.base_url}/api/projects/",
                json=project_data
            )
            
            if response.status_code == 201:
                project = response.json()
                project_id = project["id"]
                st.success(f"✅ Created Label Studio project: {project_name} (ID: {project_id})")
                return project_id
            else:
                st.error(f"Failed to create project: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error creating project: {str(e)}")
            return None
    
    def _generate_label_config(self) -> str:
        """Generate XML label configuration for semantic segmentation"""
        return """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <BrushLabels name="label" toName="image">
    <Label value="Background" background="#000000"/>
    <Label value="Object" background="#FF0000"/>
  </BrushLabels>
  <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA">
    <Header value="Instructions"/>
    <Text name="instructions" value="Paint over the regions in the image. Use the mouse wheel to zoom in/out. Background class is automatically included. You can customize classes in Label Studio settings."/>
  </View>
</View>"""
    
    def configure_minio_storage(self, project_id: int) -> bool:
        """Configure MinIO storage for the project"""
        if not self._ensure_valid_token():
            return False
        
        try:
            # First, ensure MinIO bucket exists
            if not self._ensure_minio_bucket():
                return False
            
            # Configure source storage (images)
            source_storage_id = self._add_source_storage(project_id)
            if not source_storage_id:
                return False
            
            # Configure target storage (annotations)
            target_storage_id = self._add_target_storage(project_id)
            if not target_storage_id:
                return False
            
            # Store the export storage ID for later use
            st.session_state.export_storage_id = target_storage_id
            
            # Sync storage to import images
            if self._sync_storage(project_id, source_storage_id):
                st.success("✅ MinIO storage configured and synced successfully!")
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error configuring MinIO storage: {str(e)}")
            return False

    # Annotation type detection removed - user configures directly in Label Studio
    # No need for complex API calls or token authentication
    
    def _ensure_minio_bucket(self) -> bool:
        """Ensure MinIO bucket exists, create if it doesn't"""
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=self.minio_endpoint,
                aws_access_key_id=self.minio_access_key,
                aws_secret_access_key=self.minio_secret_key,
                region_name='us-east-1'
            )
            
            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=self.minio_bucket)
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    # Bucket doesn't exist, create it
                    s3_client.create_bucket(Bucket=self.minio_bucket)
                    st.info(f"📦 Created MinIO bucket: {self.minio_bucket}")
                    return True
                else:
                    st.error(f"MinIO error: {str(e)}")
                    return False
                    
        except Exception as e:
            st.error(f"Error ensuring MinIO bucket: {str(e)}")
            return False
    
    def _add_source_storage(self, project_id: int) -> Optional[int]:
        """Add source storage for images"""
        try:
            storage_data = {
                "project": project_id,
                "storage_type": "s3",
                "title": "Images Storage",
                "bucket": self.minio_bucket,
                "prefix": "images/",
                "region_name": "us-east-1",
                "s3_endpoint": self.minio_endpoint,
                "aws_access_key_id": self.minio_access_key,
                "aws_secret_access_key": self.minio_secret_key,
                "use_blob_urls": True,
                "recursive_scan": True,
                "treat_every_bucket_object_as_source_file": True,
                "presign": False
            }
            
            # Debug: Show what we're sending
            st.info(f"📤 Source storage request data: {json.dumps(storage_data, indent=2)}")
            st.info(f"🔍 Endpoint value being sent: '{storage_data['s3_endpoint']}'")
            
            response = self.session.post(
                f"{self.base_url}/api/storages/s3/",
                json=storage_data
            )
            
            if response.status_code == 201:
                storage = response.json()
                storage_id = storage["id"]
                st.info(f"📁 Added source storage for images (ID: {storage_id})")
                return storage_id
            else:
                st.error(f"Failed to add source storage: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error adding source storage: {str(e)}")
            return None
    
    def _add_target_storage(self, project_id: int) -> Optional[int]:
        """Add target storage for annotations"""
        try:
            # Export storage configuration based on Label Studio API docs
            storage_data = {
                "project": project_id,
                "storage_type": "s3",
                "title": "Annotations Export Storage",
                "description": "Export annotations to MinIO storage",
                "bucket": self.minio_bucket,
                "prefix": "annotations",  # NO trailing slash - this was the fix!
                "aws_access_key_id": self.minio_access_key,
                "aws_secret_access_key": self.minio_secret_key,
                "aws_session_token": None,
                "aws_sse_kms_key_id": None,
                "region_name": "us-east-1",
                "s3_endpoint": "http://minio:9000",
                "can_delete_objects": False,
                "use_blob_urls": False,
                "recursive_scan": False,
                "presign": False,
                "presign_ttl": 1,
                "force_path_style": True
            }
            
            # Use the correct export storage endpoint from the API docs
            endpoint = f"{self.base_url}/api/storages/export/s3"
            
            st.info(f"🔄 Creating export storage using endpoint: {endpoint}")
            st.info(f"📤 Request data: {json.dumps(storage_data, indent=2)}")
            st.info(f"🔍 s3_endpoint value: '{storage_data['s3_endpoint']}' (with http:// and force_path_style)")
            st.info(f"🔍 s3_endpoint type: {type(storage_data['s3_endpoint'])}")
            st.info(f"🔍 force_path_style: {storage_data.get('force_path_style', 'Not set')}")
            
            response = self.session.post(endpoint, json=storage_data)
            
            if response.status_code == 201:
                storage = response.json()
                storage_id = storage["id"]
                st.success(f"📁 Added export storage for annotations (ID: {storage_id})")
                return storage_id
            else:
                st.error(f"❌ Failed to create export storage: {response.status_code}")
                st.error(f"📥 Response: {response.text}")
                
                # Try fallback to regular S3 storage endpoint
                st.info(f"🔄 Trying fallback to regular S3 storage endpoint...")
                fallback_endpoint = f"{self.base_url}/api/storages/s3/"
                fallback_response = self.session.post(fallback_endpoint, json=storage_data)
                
                if fallback_response.status_code == 201:
                    storage = fallback_response.json()
                    storage_id = storage["id"]
                    st.success(f"📁 Added storage using fallback endpoint (ID: {storage_id})")
                    return storage_id
                else:
                    st.error(f"❌ Fallback also failed: {fallback_response.status_code}")
                    st.error(f"📥 Fallback response: {fallback_response.text}")
                    return None
                
        except Exception as e:
            st.error(f"Error adding target storage: {str(e)}")
            return None
    
    def _sync_storage(self, project_id: int, storage_id: int) -> bool:
        """Sync storage to import images"""
        try:
            # Try multiple sync endpoint variations
            sync_endpoints = [
                f"{self.base_url}/api/storages/s3/{storage_id}/sync/",
                f"{self.base_url}/api/storages/{storage_id}/sync/",
                f"{self.base_url}/api/storages/{storage_id}/sync",
                f"{self.base_url}/api/storages/s3/{storage_id}/sync"
            ]
            
            for sync_url in sync_endpoints:
                try:
                    response = self.session.post(sync_url)
                    
                    if response.status_code == 200:
                        sync_result = response.json()
                        count = sync_result.get("count", 0)
                        st.success(f"✅ Synced {count} images from MinIO")
                        return True
                    elif response.status_code == 404:
                        continue
                    else:
                        continue
                        
                except Exception as e:
                    continue
            
            st.error("❌ All sync endpoints failed")
            return False
                
        except Exception as e:
            st.error(f"Error syncing storage: {str(e)}")
            return False
    
    def get_project_info(self, project_id: int) -> Optional[Dict]:
        """Get project information"""
        if not self._ensure_valid_token():
            return None
        
        try:
            response = self.session.get(f"{self.base_url}/api/projects/{project_id}/")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get project info: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error getting project info: {str(e)}")
            return None
    
    def list_projects(self) -> List[Dict]:
        """List all projects"""
        # Ensure we're authenticated before listing projects
        if not self.authenticate():
            return []
        
        try:
            response = self.session.get(f"{self.base_url}/api/projects/")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to list projects: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error listing projects: {str(e)}")
            return []
    
    def auto_setup_project(self, project_name: str, description: str = "") -> Optional[int]:
        """Complete automatic setup: create project and configure storage"""
        try:
            st.info(f"🚀 Starting automatic setup for project: {project_name}")
            
            # Step 1: Try SDK authentication first, fallback to HTTP API
            if not self.authenticate_with_sdk():
                st.info("🔄 SDK authentication failed, trying HTTP API...")
                if not self.authenticate():
                    st.error("❌ All authentication methods failed")
                    return None
            
            # Step 2: Check if project already exists
            existing_project = self._find_existing_project(project_name)
            if existing_project:
                project_id = existing_project["id"]
                st.info(f"📁 Project '{project_name}' already exists (ID: {project_id})")
                
                # Check if storage is configured
                if self._check_storage_configuration(project_id):
                    st.success(f"✅ Project '{project_name}' is ready to use!")
                    st.info(f"🌐 Access Label Studio at: {self.base_url}")
                    st.info(f"👤 Login: {self.username}")
                    st.info(f"🔑 Password: {self.password}")
                    return project_id
                else:
                    st.info("🔄 Configuring storage for existing project...")
                    if self.configure_minio_storage(project_id):
                        st.success(f"✅ Project '{project_name}' setup complete!")
                        return project_id
                    else:
                        st.error("❌ Storage configuration failed")
                        return None
            
            # Step 3: Skip cleanup to preserve uploaded images
            st.info("📁 Preserving existing images for new project...")
            # self._cleanup_old_project_data()  # Disabled to preserve uploaded images
            
            # Step 4: Create new project
            project_id = self.create_project(project_name, description)
            if not project_id:
                st.error("❌ Project creation failed")
                return None
            
            # Step 5: Configure storage
            if not self.configure_minio_storage(project_id):
                st.error("❌ Storage configuration failed")
                return None
            
            # Step 6: Configure project to use export storage
            if not self._configure_project_export(project_id):
                st.warning("⚠️ Export storage configured but project export settings may need manual configuration")
            
            st.success(f"🎉 Project '{project_name}' setup complete!")
            st.info(f"🌐 Access Label Studio at: {self.base_url}")
            st.info(f"👤 Login: {self.username}")
            st.info(f"🔑 Password: {self.password}")
            
            return project_id
            
        except Exception as e:
            st.error(f"❌ Auto setup failed: {str(e)}")
            return None
    
    def _find_existing_project(self, project_name: str) -> Optional[Dict]:
        """Find existing project by name"""
        try:
            if not self._ensure_valid_token():
                return None
            
            projects = self.list_projects()
            for project in projects:
                if project.get("title") == project_name:
                    return project
            return None
        except Exception:
            return None
    
    def _check_storage_configuration(self, project_id: int) -> bool:
        """Check if project has storage configured"""
        try:
            if not self._ensure_valid_token():
                return False
            
            project_info = self.get_project_info(project_id)
            if not project_info:
                return False
            
            # Check if project has storage configured by looking at storage endpoints
            storage_response = self.session.get(f"{self.base_url}/api/storages/s3/")
            if storage_response.status_code == 200:
                storages = storage_response.json()
                project_storages = [s for s in storages if s.get("project") == project_id]
                
                if len(project_storages) >= 2:  # Should have both source and target storage
                    st.info(f"✅ Project has {len(project_storages)} storage configurations")
                    return True
                else:
                    st.info(f"⚠️ Project has only {len(project_storages)} storage configurations (need 2)")
                    return False
            else:
                st.warning("Could not check storage configuration")
                return False
                
        except Exception as e:
            st.warning(f"Storage check error: {str(e)}")
            return False
    
    def _cleanup_old_project_data(self):
        """Clean up old project data from MinIO storage"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            st.info("🧹 Cleaning up old annotations and images...")
            
            # Initialize MinIO client
            s3_client = boto3.client(
                's3',
                endpoint_url=self.minio_endpoint,
                aws_access_key_id=self.minio_access_key,
                aws_secret_access_key=self.minio_secret_key,
                region_name='us-east-1'
            )
            
            # Clean up old annotations
            try:
                response = s3_client.list_objects_v2(
                    Bucket=self.minio_bucket,
                    Prefix="annotations"
                )
                
                if 'Contents' in response:
                    old_annotations = [obj['Key'] for obj in response['Contents'] if obj['Size'] > 0]
                    if old_annotations:
                        for annotation_key in old_annotations:
                            s3_client.delete_object(Bucket=self.minio_bucket, Key=annotation_key)
                        st.info(f"🧹 Cleaned up {len(old_annotations)} old annotation files")
                    else:
                        st.info("🧹 No old annotations to clean up")
                else:
                    st.info("🧹 No annotations folder found")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not clean up annotations: {str(e)}")
            
            # Clean up old images (optional - you might want to keep these)
            try:
                response = s3_client.list_objects_v2(
                    Bucket=self.minio_bucket,
                    Prefix="images/"
                )
                
                if 'Contents' in response:
                    old_images = [obj['Key'] for obj in response['Contents'] if obj['Size'] > 0]
                    if old_images:
                        for image_key in old_images:
                            s3_client.delete_object(Bucket=self.minio_bucket, Key=image_key)
                        st.info(f"🧹 Cleaned up {len(old_images)} old image files")
                    else:
                        st.info("🧹 No old images to clean up")
                else:
                    st.info("🧹 No images folder found")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not clean up images: {str(e)}")
            
            st.success("✅ Cleanup completed!")
            
        except Exception as e:
            st.warning(f"⚠️ Cleanup failed: {str(e)}")
    
    def _configure_project_export(self, project_id: int) -> bool:
        """Configure project to use export storage for annotations"""
        try:
            st.info("⚙️ Configuring project export settings...")
            
            # Get project info to check current settings
            project_info = self.get_project_info(project_id)
            if not project_info:
                st.warning("⚠️ Could not get project info for export configuration")
                return False
            
            # Check if we have the export storage ID from session state
            export_storage_id = st.session_state.get('export_storage_id')
            
            if export_storage_id:
                st.success(f"✅ Using export storage ID from session: {export_storage_id}")
            else:
                # Fallback: Try to find export storage via API calls
                st.info("🔄 Export storage ID not in session, trying API discovery...")
                
                # Method 1: Try the export storage endpoint
                try:
                    export_storage_response = self.session.get(f"{self.base_url}/api/storages/export/s3/")
                    if export_storage_response.status_code == 200:
                        export_storages = export_storage_response.json()
                        project_export_storages = [s for s in export_storages if s.get("project") == project_id]
                        if project_export_storages:
                            export_storage_id = project_export_storages[0]["id"]
                            st.success(f"✅ Found export storage via API (ID: {export_storage_id})")
                except Exception as e:
                    st.info(f"Method 1 failed: {str(e)}")
                
                # Method 2: Try the regular S3 storage endpoint and filter by prefix
                if not export_storage_id:
                    try:
                        s3_storage_response = self.session.get(f"{self.base_url}/api/storages/s3/")
                        if s3_storage_response.status_code == 200:
                            s3_storages = s3_storage_response.json()
                            # Look for storage with annotations prefix (no trailing slash)
                            project_export_storages = [s for s in s3_storages if s.get("project") == project_id and s.get("prefix") == "annotations"]
                            if project_export_storages:
                                export_storage_id = project_export_storages[0]["id"]
                                st.success(f"✅ Found export storage via S3 endpoint (ID: {export_storage_id})")
                    except Exception as e:
                        st.info(f"Method 2 failed: {str(e)}")
            
            if export_storage_id:
                # Try to configure the project to use export storage for annotations
                try:
                    # Update project settings to enable export storage
                    project_update_data = {
                        "enable_annotation_export": True,
                        "export_storage": export_storage_id
                    }
                    
                    st.info(f"🔄 Updating project settings with: {json.dumps(project_update_data, indent=2)}")
                    
                    update_response = self.session.patch(
                        f"{self.base_url}/api/projects/{project_id}/",
                        json=project_update_data
                    )
                    
                    if update_response.status_code == 200:
                        st.success("✅ Project export settings configured successfully!")
                        return True
                    else:
                        st.warning(f"⚠️ Could not update project export settings: {update_response.status_code}")
                        st.info(f"Response: {update_response.text}")
                        return False
                        
                except Exception as e:
                    st.warning(f"⚠️ Project export configuration failed: {str(e)}")
                    return False
            else:
                st.warning("⚠️ No export storage found for project")
                return False
                
        except Exception as e:
            st.warning(f"⚠️ Export configuration failed: {str(e)}")
            return False
    
    def export_project_data(self, project_id: int, export_format: str = "JSON") -> Optional[str]:
        """Export project data and save to local export directory"""
        try:
            if not self.authenticate():
                st.error("❌ Failed to authenticate with Label Studio")
                return None
            
            # Create export directory if it doesn't exist
            export_dir = "label-studio-data/export/"
            os.makedirs(export_dir, exist_ok=True)
            
            # Trigger export via API
            export_url = f"{self.base_url}/api/projects/{project_id}/export"
            export_params = {
                "exportType": export_format,
                "download_all_tasks": True
            }
            
            st.info(f"🔄 Triggering export for project {project_id}...")
            st.info(f"📤 Export URL: {export_url}")
            st.info(f"📤 Export params: {export_params}")
            
            # Make the export request
            response = self.session.get(export_url, params=export_params)
            
            if response.status_code == 200:
                # Generate filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
                filename = f"project-{project_id}-at-{timestamp}-{project_id:08x}.json"
                filepath = os.path.join(export_dir, filename)
                
                # Save the export data
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                st.success(f"✅ Export completed successfully!")
                st.info(f"📁 Export saved to: {filepath}")
                return filepath
                
            else:
                st.error(f"❌ Export failed with status {response.status_code}")
                st.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            return None
    
    def get_project_info(self, project_id: int) -> Optional[Dict]:
        """Get project information including annotation count"""
        try:
            if not self.authenticate():
                st.error("❌ Failed to authenticate with Label Studio")
                return None
            
            # Get project details
            project_url = f"{self.base_url}/api/projects/{project_id}/"
            response = self.session.get(project_url)
            
            if response.status_code == 200:
                project_data = response.json()
                
                # Get task count
                tasks_url = f"{self.base_url}/api/projects/{project_id}/tasks/"
                tasks_response = self.session.get(tasks_url)
                
                task_count = 0
                annotated_count = 0
                
                if tasks_response.status_code == 200:
                    tasks_data = tasks_response.json()
                    task_count = tasks_data.get('count', 0)
                    
                    # Count annotated tasks
                    for task in tasks_data.get('results', []):
                        if task.get('annotations') and len(task['annotations']) > 0:
                            annotated_count += 1
                
                return {
                    'id': project_data.get('id'),
                    'title': project_data.get('title'),
                    'task_count': task_count,
                    'annotated_count': annotated_count,
                    'created_at': project_data.get('created_at'),
                    'updated_at': project_data.get('updated_at')
                }
            else:
                st.error(f"❌ Failed to get project info: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"❌ Error getting project info: {str(e)}")
            return None

