#!/usr/bin/env python3
"""
Update .env file to use MinIO instead of GCS
"""

import os
from pathlib import Path

def update_env_file():
    """Update .env file to use MinIO configuration"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Remove GCS configuration
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Skip GCS-related lines
        if line.startswith('GCS_BUCKET_NAME='):
            continue
        if line.startswith('# Google Cloud Storage'):
            continue
        if line.strip() == '':
            continue
        new_lines.append(line)
    
    # Add MinIO configuration
    minio_config = """
# MinIO Configuration
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=segmentation-platform
"""
    
    # Write updated content
    with open(env_file, 'w') as f:
        f.write('\n'.join(new_lines))
        f.write(minio_config)
    
    print("✅ Updated .env file to use MinIO")
    print("✅ Removed GCS configuration")
    print("✅ Added MinIO configuration")
    
    return True

if __name__ == "__main__":
    update_env_file()



