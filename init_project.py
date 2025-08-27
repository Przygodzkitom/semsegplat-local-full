#!/usr/bin/env python3
"""
Project Initialization Script for Semantic Segmentation Platform

This script sets up the complete project structure on any new machine:
- Creates all required directories
- Sets up MinIO storage structure
- Validates Docker environment
- Creates configuration files
- Tests connectivity
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

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
    
    # Check Docker Compose
    try:
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose not found")
        print("Please install Docker Compose")
        return False
    
    return True

def create_directory_structure():
    """Create all required directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "label-studio-data",
        "models/checkpoints",
        "models/saved_models",
        "models/utils",
        "app",
        "docker",
        "logs",
        "backups",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")

def setup_minio_structure():
    """Setup MinIO storage structure"""
    print("\n☁️ Setting up MinIO storage structure...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("⚠️ boto3 not available, skipping MinIO structure setup")
        print("   Install with: pip install boto3")
        return True
    
    # MinIO configuration
    endpoint_url = "http://localhost:9000"
    access_key = "minioadmin"
    secret_key = "minioadmin123"
    bucket_name = "segmentation-platform"
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        
        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3_client.create_bucket(Bucket=bucket_name)
                print(f"✅ Created bucket '{bucket_name}'")
            else:
                raise
        
        # Create folder structure
        folders = [
            "images/",
            "annotations/",
            "models/",
            "models/checkpoints/",
            "models/final/"
        ]
        
        for folder in folders:
            try:
                s3_client.put_object(Bucket=bucket_name, Key=folder, Body="")
                print(f"✅ Created folder: {folder}")
            except Exception as e:
                print(f"⚠️ Folder {folder} might already exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up MinIO structure: {e}")
        print("   Make sure MinIO is running: docker compose up -d minio")
        return False

def start_services():
    """Start Docker services"""
    print("\n🚀 Starting Docker services...")
    
    try:
        # Start MinIO first
        result = subprocess.run(['docker', 'compose', 'up', '-d', 'minio'], 
                              capture_output=True, text=True, check=True)
        print("✅ MinIO started")
        
        # Wait for MinIO to be ready
        print("⏳ Waiting for MinIO to be ready...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
                if response.status_code == 200:
                    print("✅ MinIO is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
            print(f"   Attempt {attempt + 1}/{max_attempts}...")
        else:
            print("⚠️ MinIO might not be ready yet, continuing...")
        
        # Start all services
        result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                              capture_output=True, text=True, check=True)
        print("✅ All services started")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting services: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_config_files():
    """Create configuration files if they don't exist"""
    print("\n⚙️ Creating configuration files...")
    
    # Create .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# Label Studio Configuration
LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=admin
LABEL_STUDIO_HOST=http://localhost
LABEL_STUDIO_PORT=8080
LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
LABEL_STUDIO_ALLOW_ORGANIZATION_WEBHOOKS=false
LABEL_STUDIO_USE_REDIS=false
LABEL_STUDIO_FORCE_HTTPS=false
DJANGO_DB_ENGINE=django.db.backends.sqlite3
DJANGO_SETTINGS_MODULE=core.settings.label_studio
SECRET_KEY=zgwknz^7laj%n#a&cd8xcmelk+l8z_v8zn7sqfnbkehqlsdt#5

# MinIO Configuration
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=segmentation-platform
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def test_connectivity():
    """Test connectivity to all services"""
    print("\n🔗 Testing connectivity...")
    
    services = [
        ("MinIO API", "http://localhost:9000/minio/health/live"),
        ("MinIO Console", "http://localhost:9001"),
        ("Label Studio", "http://localhost:8080"),
        ("Streamlit App", "http://localhost:8501")
    ]
    
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 302]:  # 302 is redirect for login
                print(f"✅ {service_name}: {url}")
            else:
                print(f"⚠️ {service_name}: {url} (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"❌ {service_name}: {url} - {e}")

def create_readme():
    """Create a project README with setup instructions"""
    print("\n📝 Creating project documentation...")
    
    readme_content = """# Semantic Segmentation Platform

## 🚀 Quick Start

This project is now initialized and ready to use!

### Access Points

- **Label Studio**: http://localhost:8080
  - Username: admin@example.com
  - Password: admin

- **MinIO Console**: http://localhost:9001
  - Username: minioadmin
  - Password: minioadmin123

- **Streamlit App**: http://localhost:8501

### Storage Structure

```
s3://segmentation-platform/
├── images/          # Source images for annotation
├── annotations/     # Label Studio exports
└── models/          # Trained models
```

### Useful Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f

# Restart specific service
docker compose restart label-studio
```

### Label Studio Configuration

1. Go to http://localhost:8080
2. Login with admin@example.com / admin
3. Create a new project
4. Go to Settings → Cloud Storage
5. Add Source Storage:
   - Storage Type: Amazon S3
   - Bucket: segmentation-platform
   - Prefix: images/
   - Use pre-signed URLs: OFF
   - Proxy through Label Studio: ON

### Backup Data

```bash
# Backup Label Studio data
cp -r label-studio-data label-studio-data-backup-$(date +%Y%m%d)

# Backup MinIO data
docker run --rm -v semsegplat-full_local_version_minio-data:/data -v $(pwd)/minio-backup:/backup alpine sh -c "cp -r /data/* /backup/"
```

## 📁 Project Structure

```
.
├── app/                    # Streamlit application
├── models/                 # ML models and training
├── label-studio-data/      # Persistent Label Studio data
├── docker/                 # Docker configuration
├── logs/                   # Application logs
├── backups/                # Backup directory
└── temp/                   # Temporary files
```

## 🆘 Troubleshooting

- **Services not starting**: Check Docker is running
- **Data persistence issues**: Check label-studio-data/ directory
- **MinIO connection**: Verify MinIO is running on port 9000
- **Label Studio issues**: Check logs with `docker compose logs label-studio`

## 📚 Documentation

- `LABEL_STUDIO_MINIO_SETTINGS.md` - Label Studio configuration
- `LABEL_STUDIO_PERSISTENCE.md` - Data persistence guide
- `MINIO_SETUP.md` - MinIO setup and usage
"""
    
    readme_file = Path('PROJECT_README.md')
    readme_file.write_text(readme_content)
    print("✅ Created PROJECT_README.md")

def main():
    """Main initialization function"""
    print("🚀 Semantic Segmentation Platform - Project Initialization")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install required tools.")
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Start services
    if not start_services():
        print("\n⚠️ Service startup had issues, but continuing...")
    
    # Setup MinIO structure
    setup_minio_structure()
    
    # Create config files
    create_config_files()
    
    # Test connectivity
    test_connectivity()
    
    # Create documentation
    create_readme()
    
    print("\n" + "=" * 60)
    print("✅ Project initialization complete!")
    print("\n🎯 Next Steps:")
    print("1. Access Label Studio: http://localhost:8080")
    print("2. Configure MinIO storage in Label Studio")
    print("3. Upload images via Streamlit: http://localhost:8501")
    print("4. Start annotating!")
    print("\n📚 Read PROJECT_README.md for detailed instructions")

if __name__ == "__main__":
    main()

