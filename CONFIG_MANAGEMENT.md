# Config File Management Guide

## Overview
This document describes the simplified configuration file system for the Semantic Segmentation Platform.

## Config File Locations (Simplified)

### Single Config Location
- `config/label_studio_project.json` - Project configuration
- `config/class_config.json` - Class configuration for annotations

**Note**: This single location works in both local and Docker environments through volume mounting.

## Config File Contents

### label_studio_project.json
```json
{
  "project_id": 1,
  "project_name": "semantic-segmentation",
  "project_description": "Automated semantic segmentation project with MinIO storage",
  "created_at": "/app",
  "last_updated": "/app"
}
```

### class_config.json
```json
{
  "class_names": [
    "Background",
    "Background", 
    "Cell"
  ],
  "detected_classes": {
    "Background": 11,
    "Cell": 11
  },
  "total_annotations": 22
}
```

## Cleanup Commands (Simplified)

### Manual Cleanup
```bash
# Remove config files (works for both local and Docker)
rm -f config/label_studio_project.json
rm -f config/class_config.json
```

### Complete Reset
```bash
# Remove all config files and restart
rm -f config/*.json
docker compose restart semseg-app
```

## UI Cleanup Options

The app now includes a "Config Management & Cleanup" section with:

1. **Config File Discovery** - Shows all found config files
2. **Clear Session State** - Clears Streamlit session state
3. **Clear Config Files** - Removes all discovered config files
4. **Complete Reset** - Clears both session state and config files

## Troubleshooting

### Problem: App shows project even after "clean slate"
**Solution**: Check for config files in the single location:
```bash
ls -la config/
```

### Problem: Session state persists across browser refreshes
**Solution**: Use incognito/private window or clear browser storage

### Problem: Config files reappear after deletion
**Solution**: Check if files are being recreated by the app and investigate the source

## Best Practices

1. **Only use the single config location** (`config/` directory)
2. **Use the UI cleanup tools** for convenience
3. **Keep config files simple** - avoid creating multiple locations
4. **Test cleanup procedures** after making changes to the app
5. **Maintain this simplified approach** - resist adding new config locations
