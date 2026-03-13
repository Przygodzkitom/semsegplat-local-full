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
            env_url = os.getenv("LABEL_STUDIO_URL")
            if env_url:
                self.base_url = env_url
            elif os.getenv("DOCKER_ENV"):
                self.base_url = "http://label-studio:8080"
            else:
                self.base_url = "http://localhost:8080"
        else:
            self.base_url = base_url

        self.session = requests.Session()

        # MinIO configuration
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
        self.minio_bucket = os.getenv("MINIO_BUCKET_NAME", "segmentation-platform")

        # Label Studio credentials (match docker-compose defaults)
        self.username = os.getenv("LABEL_STUDIO_USERNAME", "admin@example.com")
        self.password = os.getenv("LABEL_STUDIO_PASSWORD", "admin")

    def _wait_for_label_studio(self, timeout: int = 60) -> bool:
        """Wait until Label Studio's health endpoint responds. Returns True when ready."""
        health_url = f"{self.base_url}/api/health"
        deadline = time.time() + timeout
        placeholder = st.empty()
        while time.time() < deadline:
            try:
                r = requests.get(health_url, timeout=3)
                if r.status_code == 200:
                    placeholder.empty()
                    return True
            except Exception:
                pass
            remaining = int(deadline - time.time())
            placeholder.info(f"⏳ Waiting for Label Studio to start… ({remaining}s remaining)")
            time.sleep(3)
        placeholder.empty()
        return False

    def authenticate(self) -> bool:
        """Authenticate with Label Studio via form login (session cookie + CSRF)."""
        try:
            # Wait for Label Studio to be reachable before attempting login
            if not self._wait_for_label_studio(timeout=60):
                st.error("❌ Label Studio did not become available within 60 seconds. Is the container running?")
                return False

            # Step 1: GET login page to obtain CSRF token
            self.session.get(f"{self.base_url}/user/login/", timeout=10)
            csrf = self.session.cookies.get("csrftoken", "")

            # Step 2: POST login form
            login_resp = self.session.post(
                f"{self.base_url}/user/login/",
                data={
                    "email": self.username,
                    "password": self.password,
                    "csrfmiddlewaretoken": csrf,
                },
                headers={"Referer": f"{self.base_url}/user/login/"},
                allow_redirects=True,
                timeout=10,
            )

            if "sessionid" not in self.session.cookies:
                st.error(f"❌ Login failed — no session cookie (status {login_resp.status_code})")
                return False

            # Step 3: keep CSRF header for mutating API requests
            csrf_after = self.session.cookies.get("csrftoken", csrf)
            self.session.headers.update({
                "Content-Type": "application/json",
                "X-CSRFToken": csrf_after,
            })
            return True

        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False

    def _ensure_valid_token(self) -> bool:
        """Ensure session is still authenticated, re-login if needed."""
        if "sessionid" not in self.session.cookies:
            return self.authenticate()
        try:
            r = self.session.get(f"{self.base_url}/api/projects/", timeout=5)
            if r.status_code == 401:
                return self.authenticate()
            return True
        except Exception:
            return self.authenticate()
    
    def create_project(
        self,
        project_name: str,
        description: str = "",
        annotation_type: str = "brush",
        class_names: List[str] = None,
        class_colors: List[str] = None,
    ) -> Optional[int]:
        """Create a new Label Studio project"""
        if not self._ensure_valid_token():
            return None

        try:
            label_config = self._generate_label_config(
                annotation_type=annotation_type,
                class_names=class_names,
                class_colors=class_colors,
            )

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
    
    def _generate_label_config(
        self,
        annotation_type: str = "brush",
        class_names: List[str] = None,
        class_colors: List[str] = None,
    ) -> str:
        """Generate XML label configuration for semantic segmentation."""
        if not class_names:
            class_names = ["Background", "Object"]
        if not class_colors:
            # auto-generate distinct colours if none provided
            class_colors = [f"#{abs(hash(n)) % 0xFFFFFF:06x}" for n in class_names]

        label_tag = "BrushLabels" if annotation_type == "brush" else "PolygonLabels"
        instruction = (
            "Paint over the regions in the image using the brush tool. "
            "Use the mouse wheel to zoom in/out."
            if annotation_type == "brush"
            else "Click to place polygon vertices. Close the polygon by clicking the first point. "
            "Use the mouse wheel to zoom in/out."
        )

        labels_xml = "\n    ".join(
            f'<Label value="{name}" background="{color}"/>'
            for name, color in zip(class_names, class_colors)
        )

        return f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <{label_tag} name="label" toName="image">
    {labels_xml}
  </{label_tag}>
  <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA">
    <Header value="Instructions"/>
    <Text name="instructions" value="{instruction}"/>
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

            # Cache both IDs so later syncs don't need to re-list storages
            st.session_state.source_storage_id = source_storage_id
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
                "presign": False,
            }
            response = self.session.post(f"{self.base_url}/api/storages/s3/", json=storage_data)
            if response.status_code == 201:
                return response.json()["id"]
            else:
                st.error(f"Failed to add source storage: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error adding source storage: {str(e)}")
            return None
    
    def _add_target_storage(self, project_id: int) -> Optional[int]:
        """Add export storage for annotations"""
        try:
            storage_data = {
                "project": project_id,
                "storage_type": "s3",
                "title": "Annotations Export Storage",
                "bucket": self.minio_bucket,
                "prefix": "annotations",
                "aws_access_key_id": self.minio_access_key,
                "aws_secret_access_key": self.minio_secret_key,
                "region_name": "us-east-1",
                "s3_endpoint": self.minio_endpoint,
                "can_delete_objects": False,
                "use_blob_urls": False,
                "recursive_scan": False,
                "presign": False,
                "presign_ttl": 1,
                "force_path_style": True,
            }
            response = self.session.post(f"{self.base_url}/api/storages/export/s3", json=storage_data)
            if response.status_code == 201:
                return response.json()["id"]
            # Fallback to generic S3 endpoint
            response = self.session.post(f"{self.base_url}/api/storages/s3/", json=storage_data)
            if response.status_code == 201:
                return response.json()["id"]
            st.error(f"Failed to add export storage: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            st.error(f"Error adding target storage: {str(e)}")
            return None
    
    def push_tasks(self, project_id: int, storage_keys: list) -> bool:
        """Push newly uploaded images as tasks directly to Label Studio via the API.

        storage_keys: list of MinIO object keys, e.g. ['images/foo.png', 'images/bar.png']
        """
        if not self._ensure_valid_token():
            return False

        tasks = [
            {"data": {"image": f"s3://{self.minio_bucket}/{key}"}, "project": project_id}
            for key in storage_keys
        ]

        try:
            resp = self.session.post(f"{self.base_url}/api/tasks/bulk/", json=tasks)
            if resp.status_code in (200, 201):
                st.success(f"✅ {len(tasks)} image(s) added to Label Studio")
                return True

            # Bulk endpoint not available — fall back to one task at a time
            if resp.status_code == 404:
                failed = 0
                for task in tasks:
                    r = self.session.post(f"{self.base_url}/api/tasks/", json=task)
                    if r.status_code not in (200, 201):
                        failed += 1
                if failed == 0:
                    st.success(f"✅ {len(tasks)} image(s) added to Label Studio")
                    return True
                st.warning(f"⚠️ {failed}/{len(tasks)} tasks failed to add")
                return False

            st.warning(f"⚠️ Could not add tasks: {resp.status_code} — {resp.text[:200]}")
            return False

        except Exception as e:
            st.warning(f"Error pushing tasks: {str(e)}")
            return False

    def sync_project_storage(self, project_id: int) -> bool:
        """Authenticate and sync the source storage for a project (for use after new uploads)."""
        if not self._ensure_valid_token():
            return False

        try:
            # Use cached storage ID from setup if available
            storage_id = st.session_state.get("source_storage_id")

            if not storage_id:
                # Fall back to listing — try project-scoped endpoint first
                for url in [
                    f"{self.base_url}/api/storages/s3/?project={project_id}",
                    f"{self.base_url}/api/storages/s3/",
                ]:
                    resp = self.session.get(url)
                    if resp.status_code == 200:
                        storages = resp.json()
                        match = next(
                            (s for s in storages
                             if s.get("project") == project_id and s.get("prefix", "").startswith("images")),
                            None
                        )
                        if match:
                            storage_id = match["id"]
                            st.session_state.source_storage_id = storage_id
                            break

            if not storage_id:
                st.warning("No source storage ID found — cannot sync. Re-open the app to restore the project session.")
                return False

            return self._sync_storage(project_id, storage_id)

        except Exception as e:
            st.warning(f"Storage sync error: {str(e)}")
            return False

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
    
    def auto_setup_project(
        self,
        project_name: str,
        description: str = "",
        annotation_type: str = "brush",
        class_names: List[str] = None,
        class_colors: List[str] = None,
    ) -> Optional[int]:
        """Complete automatic setup: authenticate, create project and configure storage."""
        try:
            st.info(f"🚀 Starting automatic setup for project: {project_name}")

            # Step 1: Authenticate (auto-fetches token via login — no user action needed)
            if not self.authenticate():
                st.error("❌ Authentication failed")
                return None

            # Step 2: Check if project already exists
            existing_project = self._find_existing_project(project_name)
            if existing_project:
                project_id = existing_project["id"]
                st.info(f"📁 Project '{project_name}' already exists (ID: {project_id})")

                if self._check_storage_configuration(project_id):
                    st.success(f"✅ Project '{project_name}' is ready to use!")
                    return project_id
                else:
                    st.info("🔄 Configuring storage for existing project...")
                    if self.configure_minio_storage(project_id):
                        st.success(f"✅ Project '{project_name}' setup complete!")
                        return project_id
                    else:
                        st.error("❌ Storage configuration failed")
                        return None

            # Step 3: Create new project with chosen annotation type and classes
            project_id = self.create_project(
                project_name,
                description,
                annotation_type=annotation_type,
                class_names=class_names,
                class_colors=class_colors,
            )
            if not project_id:
                st.error("❌ Project creation failed")
                return None

            # Step 4: Configure MinIO storage
            if not self.configure_minio_storage(project_id):
                st.error("❌ Storage configuration failed")
                return None

            # Step 5: Configure project export storage
            if not self._configure_project_export(project_id):
                st.warning("⚠️ Export storage configured but project export settings may need manual configuration")

            st.success(f"🎉 Project '{project_name}' setup complete!")
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
        """Link export storage to the project."""
        try:
            # Find the export storage that was just created for this project
            export_storage_id = st.session_state.get("export_storage_id")

            if not export_storage_id:
                for endpoint in (
                    f"{self.base_url}/api/storages/export/s3/",
                    f"{self.base_url}/api/storages/s3/",
                ):
                    try:
                        r = self.session.get(endpoint)
                        if r.status_code == 200:
                            storages = r.json()
                            match = next(
                                (s for s in storages
                                 if s.get("project") == project_id and s.get("prefix") == "annotations"),
                                None,
                            )
                            if match:
                                export_storage_id = match["id"]
                                break
                    except Exception:
                        continue

            if not export_storage_id:
                return False

            r = self.session.patch(
                f"{self.base_url}/api/projects/{project_id}/",
                json={"enable_annotation_export": True, "export_storage": export_storage_id},
            )
            return r.status_code == 200

        except Exception:
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
                
                st.success(f"✅ Export saved to: {filepath}")
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
                    # API may return a plain list or a paginated dict
                    if isinstance(tasks_data, list):
                        task_list = tasks_data
                        task_count = len(task_list)
                    else:
                        task_count = tasks_data.get('count', 0)
                        task_list = tasks_data.get('results', [])
                    annotated_count = sum(
                        1 for t in task_list if t.get('annotations')
                    )
                
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

