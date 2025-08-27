# Label Studio Data Persistence Guide

## Overview
This guide ensures that Label Studio projects, annotations, and user data persist across Docker restarts and rebuilds.

## Current Configuration

### Docker Compose Setup
The Label Studio service is configured with proper data persistence:

```yaml
label-studio:
  image: heartexlabs/label-studio:latest
  environment:
    - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data
  command: label-studio start --username admin@example.com --password admin
  volumes:
    - label-studio-data:/label-studio/data
    - ./label-studio-data:/label-studio/data/backup  # Backup to local directory
```

### Data Storage Locations
- **Primary Storage**: `./label-studio-data/` (bind mounted to `/label-studio/data` in container)
- **Backup Location**: `./label-studio-data/backup/` (additional backup)
- **Database**: `./label-studio-data/label_studio.sqlite3`
- **Media Files**: `./label-studio-data/media/`
- **Export Files**: `./label-studio-data/export/`
- **Test Data**: `./label-studio-data/test_data/`

## Quick Commands

### Start Services (Preserves Data)
```bash
docker compose up -d
```

### Stop Services (Preserves Data)
```bash
docker compose down
```

### Restart Services (Preserves Data)
```bash
docker compose restart
```

### Rebuild and Restart (Preserves Data)
```bash
docker compose build semseg-app && docker compose up -d
```

### Complete Reset (⚠️ Destroys Data)
```bash
docker compose down -v  # Removes volumes
rm -rf label-studio-data/  # Removes local data
docker compose up -d
```

## Troubleshooting

### Check Data Persistence
```bash
# Check if data directory exists
ls -la label-studio-data/

# Check container data
docker exec semsegplat-full_local_version-label-studio-1 ls -la /data

# Check database file
docker exec semsegplat-full_local_version-label-studio-1 ls -la /data/label_studio.sqlite3
```

### Reset Label Studio Only (Preserves MinIO Data)
```bash
# Stop only Label Studio
docker compose stop label-studio

# Remove Label Studio container (keeps volume)
docker compose rm -f label-studio

# Restart Label Studio
docker compose up -d label-studio
```

### Backup Data
```bash
# Create backup
cp -r label-studio-data/ label-studio-data-backup-$(date +%Y%m%d_%H%M%S)/

# Restore from backup
cp -r label-studio-data-backup-YYYYMMDD_HHMMSS/* label-studio-data/
```

## Expected Behavior

### After Docker Restart
- ✅ Projects should remain
- ✅ Annotations should remain
- ✅ User accounts should remain
- ✅ Project settings should remain

### After Docker Rebuild
- ✅ All data should persist
- ✅ No need to recreate projects
- ✅ Annotations should be available for training

### After Complete Reset
- ❌ All data will be lost
- ❌ Need to recreate projects
- ❌ Need to re-annotate images

## Verification Steps

1. **Create a test project** in Label Studio
2. **Add some annotations**
3. **Stop and restart** Docker services
4. **Verify** the project and annotations are still there
5. **Test training** with the persisted annotations

### Quick Verification Test

```bash
# 1. Check current data
ls -la label-studio-data/

# 2. Stop services
docker compose down

# 3. Check data still exists
ls -la label-studio-data/

# 4. Restart services
docker compose up -d

# 5. Check container data
docker exec semsegplat-full_local_version-label-studio-1 ls -la /label-studio/data

# 6. Verify database file size (should be > 0)
ls -lh label-studio-data/label_studio.sqlite3
```

## Important Notes

- **Never use `docker compose down -v`** unless you want to lose all data
- **Always use `docker compose down`** (without `-v`) to preserve volumes
- **The `label-studio-data/` directory** contains all persistent data
- **MinIO data** is stored separately and persists independently
- **Model checkpoints** are stored in `./models/checkpoints/` and persist separately

## Migration Between Machines

To move your Label Studio data to another machine:

1. **Copy the data directory**:
   ```bash
   scp -r label-studio-data/ user@new-machine:/path/to/project/
   ```

2. **Copy MinIO data** (if needed):
   ```bash
   # Use MinIO client or download from MinIO console
   ```

3. **Copy model checkpoints** (if needed):
   ```bash
   scp -r models/checkpoints/ user@new-machine:/path/to/project/models/
   ```

4. **Start services** on the new machine:
   ```bash
   docker compose up -d
   ```

