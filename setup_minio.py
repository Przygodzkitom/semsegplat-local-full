#!/usr/bin/env python3
"""
MinIO Setup Script for Semantic Segmentation Platform

This script helps you set up and configure MinIO for local S3-compatible storage.
MinIO provides the same API as Amazon S3, making it perfect for local development.
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

# MinIO configuration
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_CONSOLE = "http://localhost:9001"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
DEFAULT_BUCKET = "segmentation-platform"

def check_dependencies():
    """Check if required tools are available"""
    print("🔍 Checking dependencies...")
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ Docker found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found or not running")
        print("Please install Docker and ensure it's running")
        return False
    
    return True

def check_minio_running():
    """Check if MinIO is already running"""
    try:
        response = requests.get(f"{MINIO_ENDPOINT}/minio/health/live", timeout=5)
        if response.status_code == 200:
            print("✅ MinIO is already running")
            return True
    except requests.exceptions.RequestException:
        pass
    
    try:
        response = requests.get(f"{MINIO_CONSOLE}/api/v1/login", timeout=5)
        if response.status_code == 200:
            print("✅ MinIO console is accessible")
            return True
    except requests.exceptions.RequestException:
        pass
    
    return False

def start_minio():
    """Start MinIO using Docker Compose"""
    print("🚀 Starting MinIO...")
    
    try:
        # Check if docker-compose.yml exists
        if not os.path.exists('docker-compose.yml'):
            print("❌ docker-compose.yml not found")
            return False
        
        # Start MinIO service
        result = subprocess.run([
            'docker-compose', 'up', '-d', 'minio'
        ], capture_output=True, text=True, check=True)
        
        print("✅ MinIO started successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start MinIO: {e}")
        print(f"Error output: {e.stderr}")
        return False

def wait_for_minio():
    """Wait for MinIO to be ready"""
    print("⏳ Waiting for MinIO to be ready...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{MINIO_ENDPOINT}/minio/health/live", timeout=5)
            if response.status_code == 200:
                print("✅ MinIO is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
    
    print("❌ MinIO failed to start within expected time")
    return False

def test_minio_connection():
    """Test MinIO connection using boto3"""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        print("🧪 Testing MinIO connection...")
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name='us-east-1'
        )
        
        # Test bucket operations
        try:
            # Check if bucket exists
            s3_client.head_bucket(Bucket=DEFAULT_BUCKET)
            print(f"✅ Bucket '{DEFAULT_BUCKET}' exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Create bucket
                s3_client.create_bucket(Bucket=DEFAULT_BUCKET)
                print(f"✅ Created bucket '{DEFAULT_BUCKET}'")
            else:
                raise
        
        # Test upload
        test_content = b"Hello MinIO!"
        s3_client.put_object(
            Bucket=DEFAULT_BUCKET,
            Key='test.txt',
            Body=test_content
        )
        print("✅ Upload test successful")
        
        # Test download
        response = s3_client.get_object(Bucket=DEFAULT_BUCKET, Key='test.txt')
        downloaded_content = response['Body'].read()
        if downloaded_content == test_content:
            print("✅ Download test successful")
        else:
            print("❌ Download test failed")
            return False
        
        # Clean up test file
        s3_client.delete_object(Bucket=DEFAULT_BUCKET, Key='test.txt')
        print("✅ Cleanup test successful")
        
        return True
        
    except ImportError:
        print("❌ boto3 not installed. Please install it with: pip install boto3")
        return False
    except Exception as e:
        print(f"❌ MinIO connection test failed: {str(e)}")
        return False

def update_env_file():
    """Update .env file with MinIO configuration"""
    env_file = Path('.env')
    
    # Read existing content
    env_content = ""
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
    
    # Add or update MinIO configuration
    minio_config = f"""
# MinIO Configuration
MINIO_ENDPOINT={MINIO_ENDPOINT}
MINIO_ACCESS_KEY={MINIO_ACCESS_KEY}
MINIO_SECRET_KEY={MINIO_SECRET_KEY}
MINIO_BUCKET_NAME={DEFAULT_BUCKET}
"""
    
    # Check if MinIO config already exists
    if 'MINIO_ENDPOINT' not in env_content:
        with open(env_file, 'a') as f:
            f.write(minio_config)
        print("✅ Added MinIO configuration to .env file")
    else:
        print("✅ MinIO configuration already exists in .env file")

def create_label_studio_minio_config():
    """Create Label Studio configuration for MinIO"""
    print("📝 Creating Label Studio MinIO configuration...")
    
    config_content = f"""# Label Studio MinIO Configuration

## MinIO Storage Setup for Label Studio

### 1. Access Label Studio
Open your browser and go to: **http://localhost:8080**

Login with:
- **Username**: `admin@example.com`
- **Password**: `admin`

### 2. Create a New Project (or use existing)

1. Click **"Create Project"**
2. Enter a project name (e.g., "Semantic Segmentation")
3. Click **"Create"**

### 3. Configure MinIO Storage

1. In your project, go to **Settings** → **Cloud Storage**
2. Click **"Add Source Storage"**
3. Select **"Amazon S3"** (MinIO is S3-compatible)
4. Fill in the details:
   - **Bucket Name**: `{DEFAULT_BUCKET}`
   - **Prefix**: `images/`
   - **Regex Filter**: `.*\\.(jpg|jpeg|png|tif|tiff)$`
   - **Use Blob URLs**: ✅ Checked
   - **Recursive Scan**: ✅ Checked
   - **AWS Access Key ID**: `{MINIO_ACCESS_KEY}`
   - **AWS Secret Access Key**: `{MINIO_SECRET_KEY}`
   - **S3 Endpoint**: `{MINIO_ENDPOINT}`
   - **Region**: `us-east-1`

### 4. Test the Connection

1. Click **"Test Connection"**
2. You should see: ✅ **"Connection successful"**
3. Click **"Add Storage"**

### 5. Sync Images

1. In the Cloud Storage section, click **"Sync Storage"**
2. You should see your images being imported
3. Check the **"Tasks"** tab to see the imported images

## 🔧 Troubleshooting

### If connection fails:
1. **Check MinIO is running**: `docker-compose ps minio`
2. **Check credentials**: Use the default MinIO credentials
3. **Check endpoint**: Make sure it's `{MINIO_ENDPOINT}`

### If images don't appear:
1. **Check regex**: Make sure it matches your file extensions
2. **Check prefix**: Should be `images/` (with trailing slash)
3. **Try manual sync**: Click "Sync Storage" again

### If you see permission errors:
1. **Restart MinIO**: `docker-compose restart minio`
2. **Check bucket exists**: The setup script should create it automatically
"""
    
    with open('LABEL_STUDIO_MINIO_SETUP.md', 'w') as f:
        f.write(config_content)
    
    print("✅ Created Label Studio MinIO setup guide: LABEL_STUDIO_MINIO_SETUP.md")

def main():
    print("🚀 MinIO Setup for Semantic Segmentation Platform")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check if MinIO is already running
    if check_minio_running():
        print("✅ MinIO is already running")
    else:
        # Start MinIO
        if not start_minio():
            return False
        
        # Wait for MinIO to be ready
        if not wait_for_minio():
            return False
    
    # Test connection
    if not test_minio_connection():
        return False
    
    # Update environment file
    update_env_file()
    
    # Create Label Studio configuration
    create_label_studio_minio_config()
    
    print("\n🎉 MinIO setup completed successfully!")
    print(f"✅ MinIO API: {MINIO_ENDPOINT}")
    print(f"✅ MinIO Console: {MINIO_CONSOLE}")
    print(f"✅ Bucket: {DEFAULT_BUCKET}")
    print(f"✅ Access Key: {MINIO_ACCESS_KEY}")
    print(f"✅ Secret Key: {MINIO_SECRET_KEY}")
    print("\n📖 Next steps:")
    print("1. Upload images through your application")
    print("2. Configure Label Studio using the guide in LABEL_STUDIO_MINIO_SETUP.md")
    print("3. Access MinIO console at http://localhost:9001 (login: minioadmin/minioadmin123)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
