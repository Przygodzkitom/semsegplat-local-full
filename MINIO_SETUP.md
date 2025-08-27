# MinIO Local Storage Setup Guide

This guide explains how to set up and use MinIO as a local S3-compatible storage solution for your semantic segmentation platform, replacing Google Cloud Storage.

## 🎯 Overview

MinIO is an open-source, S3-compatible object storage server that provides:
- ✅ **Local storage** - No cloud dependencies
- ✅ **S3-compatible API** - Works with existing S3 clients
- ✅ **Easy setup** - Simple Docker deployment
- ✅ **Web console** - Visual management interface
- ✅ **High performance** - Optimized for local development

## 🚀 Quick Setup

### 1. Prerequisites

- Docker and Docker Compose installed
- Python environment with required packages
- At least 1GB of free disk space

### 2. Automatic Setup

Run the setup script to configure everything automatically:

```bash
python setup_minio.py
```

This script will:
- ✅ Check Docker installation
- ✅ Start MinIO container
- ✅ Create the default bucket
- ✅ Test upload/download functionality
- ✅ Update your `.env` file
- ✅ Create Label Studio configuration guide

### 3. Manual Setup (Alternative)

If you prefer manual setup:

1. **Start MinIO:**
   ```bash
   docker-compose up -d minio
   ```

2. **Update your `.env` file:**
   ```env
   # MinIO Configuration
   MINIO_ENDPOINT=http://localhost:9000
   MINIO_ACCESS_KEY=minioadmin
   MINIO_SECRET_KEY=minioadmin123
   MINIO_BUCKET_NAME=segmentation-platform
   ```

3. **Test the connection:**
   ```bash
   python -c "
   import boto3
   s3 = boto3.client('s3', endpoint_url='http://localhost:9000',
                     aws_access_key_id='minioadmin',
                     aws_secret_access_key='minioadmin123')
   s3.create_bucket(Bucket='segmentation-platform')
   print('✅ MinIO connection successful')
   "
   ```

## 📁 Storage Structure

Your MinIO bucket will be organized as follows:

```
s3://segmentation-platform/
├── images/                    # Uploaded images
│   ├── 20241201_143022_abc123_image1.jpg
│   ├── 20241201_143023_def456_image2.png
│   └── ...
├── annotations/              # Label Studio annotations
│   ├── project_1/
│   ├── project_2/
│   └── ...
└── models/                   # Trained models (optional)
    ├── checkpoints/
    └── final/
```

## 🔧 Configuration

### Environment Variables

Add these to your `.env` file:

```env
# MinIO Configuration
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=segmentation-platform

# Label Studio (existing)
LABEL_STUDIO_API_KEY=your-api-key
LABEL_STUDIO_URL=http://labelstudio:8080
```

### Docker Compose Configuration

The `docker-compose.yml` file includes MinIO service:

```yaml
minio:
  image: minio/minio:latest
  ports:
    - "9000:9000"  # API port
    - "9001:9001"  # Console port
  environment:
    - MINIO_ROOT_USER=minioadmin
    - MINIO_ROOT_PASSWORD=minioadmin123
  volumes:
    - minio-data:/data
  command: server /data --console-address ":9001"
```

## 🖥️ Usage

### 1. Upload Images

1. Navigate to the "Upload Images" section
2. Select images to upload
3. Images are automatically uploaded to MinIO
4. Progress is shown during upload
5. Images persist across app restarts

### 2. View Existing Images

- The app automatically loads existing images from MinIO on startup
- Images are displayed in a grid with file information
- You can delete individual images if needed
- Refresh button to reload the image list

### 3. Label Studio Integration

1. Create a Label Studio project with S3 integration
2. Configure MinIO as S3-compatible storage
3. Images are automatically synced from MinIO
4. Annotations are stored back to MinIO
5. No manual import/export needed

### 4. Training and Inference

- Training can use images directly from MinIO URLs
- Models can be saved to MinIO for persistence
- Inference can work with MinIO-stored images

## 🌐 Web Console

Access the MinIO web console at: **http://localhost:9001**

Login credentials:
- **Username**: `minioadmin`
- **Password**: `minioadmin123`

The console provides:
- 📁 **Bucket management** - Create, delete, and configure buckets
- 📤 **File upload** - Drag-and-drop file upload
- 📋 **File browser** - Navigate and manage files
- 🔧 **Settings** - Configure bucket policies and access

## 🔍 Troubleshooting

### Common Issues

1. **MinIO Not Starting:**
   ```
   ❌ Failed to start MinIO
   ```
   **Solution:** Check Docker is running and ports 9000/9001 are available

2. **Connection Refused:**
   ```
   ❌ Connection to MinIO failed
   ```
   **Solution:** Wait for MinIO to fully start (can take 30-60 seconds)

3. **Bucket Not Found:**
   ```
   ❌ Bucket 'segmentation-platform' not found
   ```
   **Solution:** Run the setup script to create the bucket automatically

4. **Label Studio Sync Issues:**
   ```
   ⚠️ Could not sync images automatically
   ```
   **Solution:** Check Label Studio S3 configuration matches MinIO settings

### Debug Commands

```bash
# Check MinIO status
docker-compose ps minio

# View MinIO logs
docker-compose logs minio

# Test MinIO connection
curl http://localhost:9000/minio/health/live

# Access MinIO console
open http://localhost:9001

# Test S3 operations
python -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin123')
print('Buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])
"
```

## 🔒 Security Considerations

### Default Credentials

The default MinIO credentials are:
- **Access Key**: `minioadmin`
- **Secret Key**: `minioadmin123`

**⚠️ Important:** These are development credentials. For production:

1. **Change default credentials:**
   ```bash
   docker-compose down
   export MINIO_ROOT_USER=your-secure-username
   export MINIO_ROOT_PASSWORD=your-secure-password
   docker-compose up -d minio
   ```

2. **Use environment variables:**
   ```env
   MINIO_ROOT_USER=your-secure-username
   MINIO_ROOT_PASSWORD=your-secure-password
   ```

3. **Restrict network access:**
   - Only expose MinIO on localhost
   - Use VPN for remote access
   - Configure firewall rules

### Data Persistence

MinIO data is stored in a Docker volume:
- **Volume name**: `minio-data`
- **Location**: Docker managed volume
- **Backup**: Consider backing up the volume for data safety

## 📈 Performance Optimization

### Storage Configuration

1. **Use SSD storage** for better performance
2. **Allocate sufficient memory** to Docker (4GB+ recommended)
3. **Monitor disk space** - MinIO stores all data locally

### Network Configuration

1. **Local access only** - MinIO runs on localhost
2. **No external dependencies** - Works offline
3. **Fast access** - No network latency



## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify Docker is running and accessible
3. Check MinIO logs: `docker-compose logs minio`
4. Test basic connectivity: `curl http://localhost:9000/minio/health/live`
5. Verify bucket exists and is accessible

## 📚 Additional Resources

- [MinIO Documentation](https://docs.min.io/)
- [S3 API Reference](https://docs.aws.amazon.com/AmazonS3/latest/API/)
- [Label Studio S3 Integration](https://labelstud.io/guide/storage.html#Amazon-S3)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🎉 Benefits of MinIO

### Key Advantages:

1. **No cloud costs** - Free local storage
2. **No internet dependency** - Works offline
3. **Faster access** - No network latency
4. **Full control** - Your data, your infrastructure
5. **Easy backup** - Simple file system backup
6. **Development friendly** - Perfect for local development
7. **S3-compatible** - Works with existing S3 tools and libraries
