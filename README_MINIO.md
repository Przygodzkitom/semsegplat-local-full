# Semantic Segmentation Platform with Local MinIO Storage

This project now supports local S3-compatible storage using MinIO, eliminating the need for cloud storage dependencies.

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites
- Docker and Docker Compose
- Python 3.8+

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup MinIO
```bash
# Run the automated setup
python setup_minio.py
```

### 4. Start the Application
```bash
# Start all services
docker-compose up -d
```

### 5. Access Your Platform
- **Main App**: http://localhost:8501
- **Label Studio**: http://localhost:8080  
- **MinIO Console**: http://localhost:9001 (login: minioadmin/minioadmin123)

## 📁 What You Get

✅ **Local Storage** - No cloud costs or dependencies  
✅ **S3-Compatible API** - Works with existing tools  
✅ **Web Console** - Visual file management  
✅ **Persistent Data** - Survives container restarts  
✅ **Fast Access** - No network latency  

## 🔧 Configuration

The setup script automatically configures:
- MinIO server on localhost:9000
- Default bucket: `segmentation-platform`
- Access credentials: `minioadmin`/`minioadmin123`
- Environment variables in `.env` file

## 📚 Documentation

- **Quick Start**: `MINIO_QUICK_START.md`
- **Full Setup Guide**: `MINIO_SETUP.md`
- **Label Studio Integration**: `LABEL_STUDIO_MINIO_SETUP.md`

## 🆘 Troubleshooting

### MinIO won't start
```bash
# Check Docker
docker ps

# Check ports
netstat -tulpn | grep :9000
```

### Can't connect to MinIO
```bash
# Check status
docker-compose ps minio

# View logs
docker-compose logs minio
```

## 🎯 Usage

1. **Upload Images** - Use the web interface to upload images
2. **Configure Label Studio** - Follow the setup guide for annotation
3. **Train Models** - Use your annotations to train segmentation models
4. **Run Inference** - Apply trained models to new images

## 💡 Tips

- **Backup**: MinIO data is stored in Docker volumes
- **Monitor Space**: All data is stored locally
- **Console Access**: Use http://localhost:9001 for file management
- **Logs**: `docker-compose logs minio` for troubleshooting

## 🎉 Benefits

- **No Cloud Costs** - Everything runs locally
- **No Internet Dependency** - Works offline
- **Full Control** - Your data, your infrastructure
- **Development Friendly** - Perfect for local development



