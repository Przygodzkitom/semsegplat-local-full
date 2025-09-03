import os
import io
from datetime import datetime
import uuid
from abc import ABC, abstractmethod

# Try to import streamlit, but don't fail if it's not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock st object for non-streamlit environments
    class MockStreamlit:
        def success(self, msg): print(f"✅ {msg}")
        def error(self, msg): print(f"❌ {msg}")
        def warning(self, msg): print(f"⚠️ {msg}")
        def info(self, msg): print(f"ℹ️ {msg}")
    st = MockStreamlit()

class StorageProvider(ABC):
    """Abstract base class for storage providers"""
    
    @abstractmethod
    def upload_image(self, file_data, filename=None):
        """Upload an image file to storage"""
        pass
    
    @abstractmethod
    def list_images(self, prefix="images/"):
        """List all images in storage"""
        pass
    
    @abstractmethod
    def download_image(self, blob_name):
        """Download an image from storage"""
        pass
    
    @abstractmethod
    def delete_image(self, blob_name):
        """Delete an image from storage"""
        pass
    
    @abstractmethod
    def get_image_url(self, blob_name):
        """Get the URL of an image"""
        pass

class MinIOProvider(StorageProvider):
    """MinIO S3-compatible storage provider"""
    
    def __init__(self, bucket_name=None):
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        self.bucket_name = bucket_name or os.getenv('MINIO_BUCKET_NAME', 'segmentation-platform')
        self.endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        
        self.s3_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize MinIO S3 client and bucket"""
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name='us-east-1'  # MinIO default region
            )
            
            # Test connection and create bucket if it doesn't exist
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # Bucket doesn't exist, create it
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    if STREAMLIT_AVAILABLE:
                        st.success(f"✅ Created MinIO bucket: {self.bucket_name}")
                    else:
                        print(f"✅ Created MinIO bucket: {self.bucket_name}")
                else:
                    raise
            

                
        except Exception as e:
            error_msg = f"❌ Failed to connect to MinIO bucket '{self.bucket_name}': {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            raise
    
    def upload_image(self, file_data, filename=None):
        """Upload an image file to MinIO"""
        if not self.s3_client:
            raise Exception("MinIO client not initialized")
        
        try:
            # Generate unique filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"images/{timestamp}_{unique_id}_{file_data.name}"
            else:
                filename = f"images/{filename}"
            
            # Reset file pointer to beginning
            file_data.seek(0)
            
            # Upload to MinIO
            self.s3_client.upload_fileobj(file_data, self.bucket_name, filename)
            
            success_msg = f"✅ Uploaded {file_data.name} to MinIO"
            if STREAMLIT_AVAILABLE:
                st.success(success_msg)
            else:
                print(success_msg)
            
            return {
                'filename': filename,
                'url': f"s3://{self.bucket_name}/{filename}",
                'size': 0,  # Not essential, set to 0
                'uploaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"❌ Failed to upload {file_data.name}: {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            raise
    
    def list_images(self, prefix="images/"):
        """List all images in the MinIO bucket"""
        if not self.s3_client:
            raise Exception("MinIO client not initialized")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            images = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                        images.append({
                            'name': obj['Key'],
                            'url': f"s3://{self.bucket_name}/{obj['Key']}",
                            'size': obj['Size'],
                            'updated': obj['LastModified'].isoformat(),
                            'filename': os.path.basename(obj['Key'])
                        })
            
            return images
            
        except Exception as e:
            error_msg = f"❌ Failed to list images: {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            return []
    
    def list_annotations(self, prefix="annotations/"):
        """List all annotation files in the MinIO bucket"""
        if not self.s3_client:
            raise Exception("MinIO client not initialized")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            annotations = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Look for JSON files, text files, or numbered files (Label Studio format)
                    if (obj['Key'].lower().endswith(('.json', '.txt')) or 
                        obj['Key'].split('/')[-1].isdigit()):
                        annotations.append({
                            'name': obj['Key'],
                            'url': f"s3://{self.bucket_name}/{obj['Key']}",
                            'size': obj['Size'],
                            'updated': obj['LastModified'].isoformat(),
                            'filename': os.path.basename(obj['Key'])
                        })
            
            return annotations
            
        except Exception as e:
            error_msg = f"❌ Failed to list annotations: {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            return []
    
    def download_image(self, blob_name):
        """Download an image from MinIO"""
        if not self.s3_client:
            raise Exception("MinIO client not initialized")
        
        try:
            file_data = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, blob_name, file_data)
            file_data.seek(0)
            return file_data
            
        except Exception as e:
            error_msg = f"❌ Failed to download {blob_name}: {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            raise
    
    def delete_image(self, blob_name):
        """Delete an image from MinIO"""
        if not self.s3_client:
            raise Exception("MinIO client not initialized")
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=blob_name)
            success_msg = f"✅ Deleted {blob_name} from MinIO"
            if STREAMLIT_AVAILABLE:
                st.success(success_msg)
            else:
                print(success_msg)
            
        except Exception as e:
            error_msg = f"❌ Failed to delete {blob_name}: {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            raise
    
    def get_image_url(self, blob_name):
        """Get the S3 URL of an image"""
        if not self.s3_client:
            raise Exception("MinIO client not initialized")
        
        try:
            s3_url = f"s3://{self.bucket_name}/{blob_name}"
            return s3_url
        except Exception as e:
            error_msg = f"❌ Failed to get URL for {blob_name}: {str(e)}"
            if STREAMLIT_AVAILABLE:
                st.error(error_msg)
            else:
                print(error_msg)
            return None

class StorageManager:
    """Unified storage manager that works with MinIO"""
    
    def __init__(self, bucket_name=None):
        """
        Initialize storage manager with MinIO provider
        
        Args:
            bucket_name: bucket name for MinIO storage
        """
        self.bucket_name = bucket_name
        self.provider = MinIOProvider(self.bucket_name)
    
    def upload_image(self, file_data, filename=None):
        """Upload an image file to MinIO storage"""
        return self.provider.upload_image(file_data, filename)
    
    def list_images(self, prefix="images/"):
        """List all images in MinIO storage"""
        return self.provider.list_images(prefix)
    
    def list_annotations(self, prefix="annotations/"):
        """List all annotation files in MinIO storage"""
        return self.provider.list_annotations(prefix)
    
    def download_image(self, blob_name):
        """Download an image from MinIO storage"""
        return self.provider.download_image(blob_name)
    
    def delete_image(self, blob_name):
        """Delete an image from MinIO storage"""
        return self.provider.delete_image(blob_name)
    
    def get_image_url(self, blob_name):
        """Get the URL of an image in MinIO storage"""
        return self.provider.get_image_url(blob_name)

def get_storage_manager(bucket_name=None):
    """Get or create storage manager instance (for Streamlit environments)"""
    # Always create a new instance to avoid session state issues
    try:
        return StorageManager(bucket_name)
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.error(f"Failed to create storage manager: {str(e)}")
        else:
            print(f"Failed to create storage manager: {str(e)}")
        return None
