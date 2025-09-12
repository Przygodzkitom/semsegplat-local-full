# 🐳 Docker Volume System Documentation

## Overview

This document provides a comprehensive analysis of the Docker volume mounting system used in the Semantic Segmentation Platform. The system uses a hybrid approach combining named volumes with bind mounts to ensure data persistence and optimal performance.

## 📋 Table of Contents

1. [Volume Architecture](#volume-architecture)
2. [Configuration Files](#configuration-files)
3. [Volume Definitions](#volume-definitions)
4. [Service-Specific Mounts](#service-specific-mounts)
5. [Data Flow and Persistence](#data-flow-and-persistence)
6. [Scripts and Automation](#scripts-and-automation)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## 🏗️ Volume Architecture

### High-Level Architecture

```
Host System
├── /home/tomasz/Dokumenty/semsegplat-full_local_version /
│   ├── label-studio-data/          # Label Studio data (bind mounted)
│   ├── minio-data/                 # MinIO data (bind mounted)
│   ├── models/                     # Model files (bind mounted)
│   └── app/                        # Application code (bind mounted)
│
Docker Volumes (Named)
├── semsegplat-full_local_version_label-studio-data
│   └── Bind to: ./label-studio-data
└── semsegplat-full_local_version_minio-data
    └── Bind to: ./minio-data
```

### Volume Types Used

1. **Named Volumes with Bind Mounts**: Primary data persistence
2. **Direct Bind Mounts**: Code and configuration files
3. **Hybrid Approach**: Combines benefits of both approaches

## 📁 Configuration Files

### 1. `docker-compose.yml` (CPU Version)

```yaml
version: '3.8'

services:
  minio:
    volumes:
      - minio-data:/data

  semseg-app:
    volumes:
      - ./models:/app/models
      - ./app:/app/app
      - ./label-studio-data:/app/label-studio-data  # ← CRITICAL FIX

  label-studio:
    volumes:
      - label-studio-data:/label-studio/data
      - ./label-studio-data:/label-studio/data/backup

volumes:
  label-studio-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./label-studio-data
  minio-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./minio-data
```

### 2. `docker-compose.gpu.yml` (GPU Version)

```yaml
version: '3.8'

services:
  minio:
    volumes:
      - minio-data:/data

  semseg-app:
    volumes:
      - ./models/checkpoints:/app/models/checkpoints  # ← DIFFERENT from CPU version
      - ./app:/app/app
      # ← MISSING: ./label-studio-data:/app/label-studio-data

  label-studio:
    volumes:
      - label-studio-data:/label-studio/data
      - ./label-studio-data:/label-studio/data/backup

volumes:
  # Same volume definitions as CPU version
```

## 🔧 Volume Definitions

### Named Volumes with Bind Mounts

#### 1. `label-studio-data` Volume

```yaml
label-studio-data:
  driver: local
  driver_opts:
    type: none
    o: bind
    device: ./label-studio-data
```

**Purpose**: Stores Label Studio database and configuration
**Host Path**: `/home/tomasz/Dokumenty/semsegplat-full_local_version /label-studio-data`
**Container Path**: `/label-studio/data`
**Data Persistence**: ✅ Full persistence across container restarts

#### 2. `minio-data` Volume

```yaml
minio-data:
  driver: local
  driver_opts:
    type: none
    o: bind
    device: ./minio-data
```

**Purpose**: Stores MinIO object storage data
**Host Path**: `/home/tomasz/Dokumenty/semsegplat-full_local_version /minio-data`
**Container Path**: `/data`
**Data Persistence**: ✅ Full persistence across container restarts

### Direct Bind Mounts

#### 1. Models Directory

**CPU Version**:
```yaml
- ./models:/app/models
```

**GPU Version**:
```yaml
- ./models/checkpoints:/app/models/checkpoints
```

**Purpose**: Provides access to trained models and checkpoints
**Data Persistence**: ✅ Full persistence (read-only for containers)

#### 2. Application Code

```yaml
- ./app:/app/app
```

**Purpose**: Mounts application source code for development
**Data Persistence**: ✅ Full persistence with live updates

#### 3. Label Studio Data Access (CRITICAL FIX)

**CPU Version**:
```yaml
- ./label-studio-data:/app/label-studio-data
```

**GPU Version**:
```yaml
# MISSING - This is the bug that was fixed!
```

**Purpose**: Allows semseg-app to access Label Studio database
**Data Persistence**: ✅ Full persistence
**Status**: ✅ Fixed in CPU version, ❌ Missing in GPU version

## 🚀 Service-Specific Mounts

### MinIO Service

```yaml
volumes:
  - minio-data:/data
```

**Container Path**: `/data`
**Purpose**: Object storage for images and annotations
**Data Structure**:
```
/data/
└── segmentation-platform/
    ├── images/
    │   ├── 1757667859849_Snap-5539.png/
    │   ├── 1757667860025_Snap-5710.png/
    │   └── 1757667860182_Snap-5537.png/
    └── annotations/
```

### Label Studio Service

```yaml
volumes:
  - label-studio-data:/label-studio/data
  - ./label-studio-data:/label-studio/data/backup
```

**Primary Container Path**: `/label-studio/data`
**Backup Container Path**: `/label-studio/data/backup`
**Purpose**: Database, media files, and configuration
**Data Structure**:
```
/label-studio/data/
├── label_studio.sqlite3          # Main database
├── media/                        # Uploaded files
├── export/                       # Export files
└── backup/                       # Backup directory
```

### Semantic Segmentation App Service

#### CPU Version (Fixed)
```yaml
volumes:
  - ./models:/app/models
  - ./app:/app/app
  - ./label-studio-data:/app/label-studio-data  # ← CRITICAL FIX
```

#### GPU Version (Has Bug)
```yaml
volumes:
  - ./models/checkpoints:/app/models/checkpoints
  - ./app:/app/app
  # ← MISSING: ./label-studio-data:/app/label-studio-data
```

**Purpose**: Access to models, code, and Label Studio database
**Critical Issue**: GPU version missing Label Studio data access

## 📊 Data Flow and Persistence

### Data Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Host System   │    │  Docker Volumes  │    │   Containers    │
│                 │    │                  │    │                 │
│ label-studio-   │◄──►│ label-studio-    │◄──►│ Label Studio    │
│ data/           │    │ data (bind)      │    │ /label-studio/  │
│                 │    │                  │    │ data            │
│ minio-data/     │◄──►│ minio-data       │◄──►│ MinIO /data     │
│                 │    │ (bind)           │    │                 │
│ models/         │────┼──────────────────┼───►│ semseg-app      │
│ app/            │    │                  │    │ /app/models     │
│                 │    │                  │    │ /app/app        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Persistence Strategy

1. **Critical Data**: Uses named volumes with bind mounts
   - Label Studio database
   - MinIO object storage
   - Ensures data survives container recreation

2. **Code and Models**: Uses direct bind mounts
   - Live code updates
   - Model file access
   - Development flexibility

3. **Backup Strategy**: Dual mounting
   - Primary: Named volume for Label Studio
   - Backup: Direct bind mount for redundancy

## 🤖 Scripts and Automation

### 1. `start.sh` - Auto-Detection Script

```bash
#!/bin/bash
# Detects GPU availability and chooses appropriate configuration

if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA Docker runtime detected!"
    $DOCKER_COMPOSE -f docker-compose.gpu.yml up  # ← Uses GPU config (has bug)
else
    echo "ℹ️  NVIDIA Docker runtime not available"
    $DOCKER_COMPOSE up  # ← Uses CPU config (fixed)
fi
```

**Issue**: GPU detection uses buggy configuration
**Impact**: GPU users get broken volume mounts

### 2. `deploy.sh` - Deployment Script

```bash
#!/bin/bash
# Sets up complete project on new machine

# Creates required directories
mkdir -p label-studio-data
mkdir -p models/checkpoints
mkdir -p models/saved_models
mkdir -p models/utils
mkdir -p app
mkdir -p docker
mkdir -p logs
mkdir -p backups
mkdir -p temp

# Starts services
docker compose up -d
```

**Purpose**: Initial setup and directory creation
**Volume Dependencies**: Creates directories that volumes bind to

### 3. Environment Configuration

#### `.env` File
```bash
LABEL_STUDIO_URL='http://labelstudio:8080'
LABEL_STUDIO_API_KEY='4fda6ab9c9c32429cf723c2219f10b77b52ebd34'
LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=admin
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=segmentation-platform
```

**Purpose**: Environment variables for container configuration
**Volume Impact**: Affects how services access mounted data

## 🔍 Current Volume State Analysis

### Running Container Mounts

#### semseg-app Container
```json
[
  {
    "Type": "bind",
    "Source": "/home/tomasz/Dokumenty/semsegplat-full_local_version /models",
    "Destination": "/app/models",
    "Mode": "rw"
  },
  {
    "Type": "bind", 
    "Source": "/home/tomasz/Dokumenty/semsegplat-full_local_version /app",
    "Destination": "/app/app",
    "Mode": "rw"
  },
  {
    "Type": "bind",
    "Source": "/home/tomasz/Dokumenty/semsegplat-full_local_version /label-studio-data",
    "Destination": "/app/label-studio-data",
    "Mode": "rw"
  }
]
```

#### label-studio Container
```json
[
  {
    "Type": "volume",
    "Name": "semsegplat-full_local_version_label-studio-data",
    "Source": "/var/lib/docker/volumes/semsegplat-full_local_version_label-studio-data/_data",
    "Destination": "/label-studio/data",
    "Driver": "local"
  },
  {
    "Type": "bind",
    "Source": "/host_mnt/home/tomasz/Dokumenty/semsegplat-full_local_version /label-studio-data",
    "Destination": "/label-studio/data/backup",
    "Mode": "rw"
  }
]
```

#### minio Container
```json
[
  {
    "Type": "volume",
    "Name": "semsegplat-full_local_version_minio-data", 
    "Source": "/var/lib/docker/volumes/semsegplat-full_local_version_minio-data/_data",
    "Destination": "/data",
    "Driver": "local"
  }
]
```

## 🐛 Known Issues and Fixes

### Issue 1: Missing Label Studio Data Access in GPU Version

**Problem**: `docker-compose.gpu.yml` missing critical volume mount
```yaml
# MISSING in GPU version:
- ./label-studio-data:/app/label-studio-data
```

**Impact**: GPU users cannot access Label Studio database
**Status**: ❌ Not fixed in GPU version
**Solution**: Add the missing volume mount to GPU configuration

### Issue 2: Inconsistent Model Mounting

**CPU Version**:
```yaml
- ./models:/app/models  # Mounts entire models directory
```

**GPU Version**:
```yaml
- ./models/checkpoints:/app/models/checkpoints  # Only checkpoints subdirectory
```

**Impact**: Different model access patterns between versions
**Status**: ⚠️ Intentional but inconsistent

### Issue 3: Space in Directory Path

**Problem**: Directory path contains space: `semsegplat-full_local_version /`
**Impact**: Potential issues with volume mounting on some systems
**Status**: ⚠️ Works but not ideal

## 🛠️ Troubleshooting

### Problem: "Label Studio database not found"

**Symptoms**:
- Batch evaluation fails
- Database path not accessible
- "No annotations found" error

**Root Cause**: Missing volume mount in container
**Solution**: Ensure `./label-studio-data:/app/label-studio-data` is mounted

**Verification**:
```bash
docker exec -it <container> ls -la /app/label-studio-data/
```

### Problem: "Image not found in MinIO"

**Symptoms**:
- Images exist in MinIO but not accessible
- Database has different image names than MinIO

**Root Cause**: Database/container synchronization issue
**Solution**: Restart containers to sync data

**Verification**:
```bash
docker exec -it <container> ls -la /data/segmentation-platform/images/
```

### Problem: Volume Mount Failures

**Symptoms**:
- Container fails to start
- "bind mount" errors
- Permission denied

**Root Cause**: Host directory doesn't exist or wrong permissions
**Solution**: Create directories and fix permissions

**Fix**:
```bash
mkdir -p label-studio-data minio-data models app
chmod 755 label-studio-data minio-data models app
```

## ✅ Best Practices

### 1. Volume Mount Strategy

- **Use named volumes with bind mounts** for critical data
- **Use direct bind mounts** for code and configuration
- **Avoid mounting entire project directory** (performance impact)

### 2. Data Persistence

- **Always use named volumes** for databases and object storage
- **Test data persistence** after container recreation
- **Regular backups** of critical data directories

### 3. Development Workflow

- **Use direct bind mounts** for live code updates
- **Separate data and code** volumes for clarity
- **Version control** volume configurations

### 4. Production Deployment

- **Use consistent volume configurations** across environments
- **Document volume dependencies** clearly
- **Test volume mounting** in target environment

## 🔧 Recommended Fixes

### 1. Fix GPU Configuration

Update `docker-compose.gpu.yml`:
```yaml
semseg-app:
  volumes:
    - ./models/checkpoints:/app/models/checkpoints
    - ./app:/app/app
    - ./label-studio-data:/app/label-studio-data  # ← ADD THIS LINE
```

### 2. Standardize Model Mounting

**Option A**: Use consistent mounting in both versions
```yaml
# Both CPU and GPU versions:
- ./models:/app/models
```

**Option B**: Document the difference clearly
```yaml
# CPU version - full models directory
- ./models:/app/models

# GPU version - only checkpoints (performance optimization)
- ./models/checkpoints:/app/models/checkpoints
```

### 3. Fix Directory Naming

Rename project directory to remove space:
```bash
mv "semsegplat-full_local_version " "semsegplat-full_local_version"
```

## 📈 Performance Considerations

### Volume Mount Performance

| Mount Type | Performance | Use Case |
|------------|-------------|----------|
| Named Volume (bind) | High | Critical data |
| Direct Bind Mount | Medium | Code/config |
| Named Volume (internal) | Low | Temporary data |

### Optimization Tips

1. **Minimize bind mounts** - Only mount necessary directories
2. **Use named volumes** for frequently accessed data
3. **Avoid mounting large directories** - Use specific subdirectories
4. **Consider volume drivers** for production (NFS, etc.)

## 🎯 Summary

The Docker volume system uses a hybrid approach that combines the benefits of named volumes (for data persistence) and direct bind mounts (for development flexibility). The main issue was a missing volume mount in the GPU configuration that prevented the semantic segmentation app from accessing the Label Studio database.

**Key Points**:
- ✅ CPU version is properly configured
- ❌ GPU version missing critical volume mount
- ✅ Data persistence works correctly
- ⚠️ Some inconsistencies between CPU/GPU versions
- ✅ Overall architecture is sound

The system successfully provides data persistence, development flexibility, and proper separation of concerns between different types of data and code.

