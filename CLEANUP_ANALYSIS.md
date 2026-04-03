# 🧹 Comprehensive Cleanup Script Documentation

## Overview

`comprehensive_cleanup.py` provides a complete system reset for fresh project starts. It removes all user data, annotations, images, and model checpoints.

## Usage

```bash
python3 comprehensive_cleanup.py
```

You will be prompted to confirm before any data is deleted.

## What It Does

### Cleanup Steps (in order)

1. **Stop containers** — runs `docker compose down` before touching any data
2. **Clear Docker volumes** — attempts to remove named Docker volumes (`semsegplat-full_local_version_label-studio-data`, `_minio-data`). This step is a no-op with the current bind-mount setup but is harmless.
3. **Clear Label Studio data** — completely removes and recreates `label-studio-data/` with empty `export/`, `media/`, `backup/` subdirectories
4. **Clear MinIO data** — completely removes and recreates `minio-data/` including `.minio.sys/`, with empty `segmentation-platform/images/` and `segmentation-platform/annotations/` subdirectories
5. **Clear model data** — clears `models/checkpoints/`, `models/saved_models/`, and `models/brush_cache/`
6. **Clear config files** — removes `*_config.json`, `class_config.json`, `project_config.json`, `label_studio_config.json`, and `test_annotation.json` from the project root
7. **Clear debug files** — removes `debug_*.png/jpg/jpeg`, `temp_*`, `*.tmp`, `.cleanup_completed`
8. **Restart containers** — runs `docker compose up -d` and waits 15 seconds for services to be ready
9. **Verify cleanup** — checks that Label Studio database is empty, MinIO image/annotation directories are empty, and no unexpected image files remain outside known locations

### What Is Always Preserved

- Application code (`app/`, `models/` source files)
- Docker and compose configuration files

## Notes

- **True clean slate**: because `label-studio-data/` and `minio-data/` are fully removed and recreated, this includes MinIO system files (`.minio.sys/`) — Label Studio and MinIO will reinitialise from scratch on next startup
- **Bind mounts**: data lives in `./label-studio-data` and `./minio-data` on the host; the Docker volume removal step (step 2) is retained for safety but has no effect with the current compose setup
