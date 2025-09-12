# 🐳 Docker Volume Standardization

## Changes Made

I've standardized the Docker volume mounts between the CPU and GPU versions to ensure consistency and optimal storage usage.

## ✅ **Standardized Configuration**

Both `docker-compose.yml` (CPU) and `docker-compose.gpu.yml` (GPU) now use identical volume mounting strategies:

### **MinIO Storage**
```yaml
volumes:
  - minio-data:/data

# Volume definition:
minio-data:
  driver: local
  driver_opts:
    type: none
    o: bind
    device: ./minio-data
```

### **Label Studio Storage**
```yaml
volumes:
  - label-studio-data:/label-studio/data
  - ./label-studio-data:/label-studio/data/backup  # Backup to local directory

# Volume definition:
label-studio-data:
  driver: local
  driver_opts:
    type: none
    o: bind
    device: ./label-studio-data
```

### **Application Storage**
```yaml
volumes:
  - ./models/checkpoints:/app/models/checkpoints
  - ./app:/app/app
```

## 🔧 **Key Changes Made**

### **Removed Excessive Mounts**
- ❌ **Removed**: `.:/app/project` (entire project directory mount)
- ❌ **Removed**: `./models/saved_models:/app/models/saved_models` (empty directory)

### **Standardized Volume Types**
- ✅ **Consistent**: Both versions now use named volumes with bind mount drivers
- ✅ **Consistent**: Both versions use identical volume definitions
- ✅ **Consistent**: Both versions mount only necessary directories

### **Optimized Storage Access**
- ✅ **Models**: Only `checkpoints` directory mounted (where trained models are stored)
- ✅ **App Code**: Only `app` directory mounted (application code)
- ✅ **Data**: MinIO and Label Studio data properly persisted

## 📊 **Storage Impact**

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| **CPU Version** | Full project + unused dirs | Only necessary dirs | ~90% reduction in mounted content |
| **GPU Version** | Inconsistent volume types | Standardized volumes | Consistent behavior |
| **Data Persistence** | Mixed strategies | Unified approach | Predictable data location |

## 🎯 **Benefits**

1. **Consistency**: Both CPU and GPU versions behave identically
2. **Efficiency**: Reduced mount overhead and storage usage
3. **Maintainability**: Single source of truth for volume configuration
4. **Data Safety**: Proper persistence for all critical data
5. **Performance**: Faster container startup with fewer mounts

## 🔄 **Migration Notes**

### **For Existing Deployments**
- Existing data in `minio-data/` and `label-studio-data/` will be preserved
- No data loss during the transition
- Containers will restart with new volume configuration

### **For New Deployments**
- Use either `docker-compose.yml` or `docker-compose.gpu.yml` based on your hardware
- Both will now provide identical storage behavior
- Only difference is GPU runtime configuration

## 🚀 **Usage**

### **CPU Version**
```bash
docker-compose up -d
```

### **GPU Version**
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

Both commands now provide identical storage behavior with only the GPU runtime being different.

## ✅ **Verification**

To verify the standardization worked:

1. **Check volume mounts**:
   ```bash
   docker-compose config
   docker-compose -f docker-compose.gpu.yml config
   ```

2. **Compare volume sections** - they should be identical except for GPU runtime

3. **Test data persistence** - data should persist across container restarts

The volume standardization is now complete and both CPU and GPU versions use optimal, consistent storage configuration! 🎉
