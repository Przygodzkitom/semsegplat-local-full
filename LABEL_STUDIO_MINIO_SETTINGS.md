# Label Studio MinIO Storage Configuration Guide

## 🎯 Overview

This guide provides the correct settings for configuring Label Studio to work with MinIO storage in this semantic segmentation platform.

## 🚀 Quick Setup

### 1. Access Label Studio
- **URL**: `http://localhost:8080`
- **Username**: `admin@example.com`
- **Password**: `admin`

### 2. Navigate to Storage Settings
1. Open your project in Label Studio
2. Go to **Settings** → **Cloud Storage**
3. Click **"Add Source Storage"**

## 📋 Correct Settings Configuration

### Source Storage (for Images)

```
Storage Title: MinIO Images
Storage Type: Amazon S3
Bucket Name: segmentation-platform
Prefix: images/
Regex Filter: .*\.(jpg|jpeg|png|tif|tiff)$
Use Blob URLs: ✅ Checked
Recursive Scan: ✅ Checked
Access Key ID: minioadmin
Secret Access Key: minioadmin123
Endpoint URL: http://minio:9000
Region: us-east-1
Use pre-signed URLs: ❌ OFF (Unchecked) ⚠️ CRITICAL
Proxy through Label Studio: ✅ ON (Checked) ⚠️ CRITICAL
```

### Target Storage (for Annotations) - Optional

```
Storage Title: MinIO Annotations
Storage Type: Amazon S3
Bucket Name: segmentation-platform
Prefix: annotations/
Access Key ID: minioadmin
Secret Access Key: minioadmin123
Endpoint URL: http://minio:9000
Region: us-east-1
Use pre-signed URLs: ❌ OFF (Unchecked)
Proxy through Label Studio: ✅ ON (Checked)
```

## ⚠️ Critical Settings Explained

### **Use pre-signed URLs: OFF**
- **Why**: Pre-signed URLs don't work with MinIO's internal Docker network
- **What happens if ON**: Images won't load in the browser
- **What happens if OFF**: Label Studio proxies images through its server

### **Proxy through Label Studio: ON**
- **Why**: Browsers can't directly access `http://minio:9000`
- **What happens**: Label Studio fetches images from MinIO and serves them
- **Security**: Keeps MinIO credentials secure within containers

### **Endpoint URL: http://minio:9000**
- **Why**: Uses Docker internal network hostname
- **Don't use**: `http://localhost:9000` (won't work from Label Studio container)

## 🔧 Verification Steps

### 1. Test Connection
1. Click **"Test Connection"**
2. Should see: ✅ **"Connection successful"**
3. If failed, check:
   - MinIO container is running: `docker ps | grep minio`
   - Network connectivity: `docker exec label-studio curl http://minio:9000/minio/health/live`

### 2. Sync Images
1. Click **"Add Storage"**
2. Click **"Sync Storage"**
3. Check **"Tasks"** tab for imported images

### 3. Test Image Loading
1. Open an image in the annotation interface
2. Image should load without errors
3. If images don't load, ensure "Use pre-signed URLs" is OFF

## 🐛 Troubleshooting

### Images Don't Load
- ✅ Ensure "Use pre-signed URLs" is **OFF**
- ✅ Ensure "Proxy through Label Studio" is **ON**
- ✅ Check MinIO container is running
- ✅ Verify bucket contains images in `images/` prefix

### Connection Test Fails
- ✅ Check MinIO is running: `docker ps | grep minio`
- ✅ Verify credentials: `minioadmin` / `minioadmin123`
- ✅ Check endpoint URL: `http://minio:9000`
- ✅ Ensure bucket exists: `segmentation-platform`

### Permission Errors
- ✅ Restart Label Studio: `docker-compose restart label-studio`
- ✅ Check MinIO logs: `docker-compose logs minio`
- ✅ Verify bucket permissions in MinIO console

## 📁 Expected Storage Structure

```
s3://segmentation-platform/
├── images/                    # Source images for annotation
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── annotations/              # Label Studio exports (optional)
│   ├── project_1/
│   ├── project_2/
│   └── ...
└── models/                   # Trained models
    ├── checkpoints/
    └── final/
```

## 🔄 Environment Variables

If you need to configure via environment variables, add to `label-studio.env`:

```env
# MinIO S3-compatible storage settings
LABEL_STUDIO_STORAGE_BACKEND=s3
LABEL_STUDIO_S3_ENDPOINT=http://minio:9000
LABEL_STUDIO_S3_ACCESS_KEY_ID=minioadmin
LABEL_STUDIO_S3_SECRET_ACCESS_KEY=minioadmin123
LABEL_STUDIO_S3_BUCKET_NAME=segmentation-platform
LABEL_STUDIO_S3_REGION=us-east-1
LABEL_STUDIO_S3_USE_SSL=false
LABEL_STUDIO_S3_VERIFY_SSL=false
```

## 🎯 Summary

The key to successful MinIO integration with Label Studio is:
1. **Use pre-signed URLs: OFF** ⚠️
2. **Proxy through Label Studio: ON** ⚠️
3. **Endpoint URL: http://minio:9000** (Docker internal hostname)
4. **Test connection before syncing**

This configuration ensures Label Studio can properly access and serve images from your local MinIO storage.

