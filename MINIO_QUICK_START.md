# MinIO Quick Start Guide

Get your semantic segmentation platform running with local MinIO storage in 5 minutes!

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install boto3 botocore

# Ensure Docker is running
docker --version
```

### 2. Start MinIO

```bash
# Start MinIO using the updated docker-compose
docker-compose up -d minio

# Wait 30 seconds for MinIO to be ready
sleep 30
```

### 3. Run Setup Script

```bash
# Configure everything automatically
python setup_minio.py
```

### 4. Start Your Application

```bash
# Start the full application stack
docker-compose up -d
```

### 5. Access Your Platform

- **Main App**: http://localhost:8501
- **Label Studio**: http://localhost:8080
- **MinIO Console**: http://localhost:9001 (login: minioadmin/minioadmin123)

## 🔧 Configuration

The setup script automatically creates this configuration in your `.env` file:

```env
# MinIO Configuration
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=segmentation-platform
```

## 📁 Storage Structure

Your data will be organized as:

```
s3://segmentation-platform/
├── images/          # Uploaded images
├── annotations/     # Label Studio annotations
└── models/         # Trained models
```

## 🎯 What You Get

✅ **Local storage** - No cloud costs or dependencies  
✅ **S3-compatible API** - Works with existing tools  
✅ **Web console** - Visual file management  
✅ **Persistent data** - Survives container restarts  
✅ **Fast access** - No network latency  



## 🆘 Troubleshooting

### MinIO won't start
```bash
# Check Docker is running
docker ps

# Check ports are available
netstat -tulpn | grep :9000
```

### Can't connect to MinIO
```bash
# Check MinIO status
docker-compose ps minio

# View logs
docker-compose logs minio

# Test connection
curl http://localhost:9000/minio/health/live
```

### Application can't find storage
```bash
# Verify environment variables
cat .env | grep MINIO

# Test storage connection
python -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:9000',
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin123')
print('Buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])
"
```

## 📚 Next Steps

1. **Upload images** through the web interface
2. **Configure Label Studio** using the guide in `LABEL_STUDIO_MINIO_SETUP.md`
3. **Start annotating** your images
4. **Train models** with your annotations
5. **Run inference** on new images

## 💡 Tips

- **Backup your data**: MinIO data is stored in Docker volumes
- **Monitor disk space**: All data is stored locally
- **Use the console**: Access http://localhost:9001 for file management
- **Check logs**: `docker-compose logs minio` for troubleshooting

## 🎉 You're Ready!

Your semantic segmentation platform is now running with local MinIO storage. No more cloud dependencies, no more costs, and no more network issues!
