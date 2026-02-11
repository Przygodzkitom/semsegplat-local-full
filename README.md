# Semantic Segmentation Platform

A comprehensive platform for semantic segmentation using Label Studio for annotation, MinIO for storage, and PyTorch for training and inference.

## 🎯 Overview

This platform provides a complete workflow for semantic segmentation projects:

- **Image Annotation**: Label Studio with persistent projects
- **Data Storage**: MinIO S3-compatible object storage
- **Model Training**: U-Net with ResNet101 backbone
- **Inference**: Real-time segmentation with configurable thresholds
- **Web Interface**: Streamlit-based management interface

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM available
- 10GB+ free disk space
- NVIDIA GPU (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd semsegplat-full_local_version
```

### Starting the Platform

**Option 1: Quick Start (Recommended for daily use)**
```bash
# Linux/Mac - Auto-detects GPU and creates directories
chmod +x start.sh
./start.sh

# Windows - Auto-detects GPU and creates directories
start.bat
```

**Option 2: Full Deployment (Recommended for first-time setup)**
```bash
# Linux/Mac - Includes validation and health checks
chmod +x deploy.sh
./deploy.sh
```

**Option 3: Manual Start**
```bash
# Only if you prefer manual control
docker compose up -d
```

### Access the Platform

- **Streamlit App**: http://localhost:8501
- **Label Studio**: http://localhost:8080 (admin@example.com / admin)
- **MinIO Console**: http://localhost:9001 (minioadmin / minioadmin123)

### Script Comparison

| Script | Best For | Features |
|--------|----------|----------|
| **start.sh/bat** | Daily use, quick restart | Auto-creates directories, GPU detection, fast startup |
| **deploy.sh** | First-time setup, troubleshooting | Full validation, health checks, detailed output |

Both scripts automatically create required data directories, so no manual setup is needed!

## 📁 Project Structure

```
semsegplat-full_local_version/
├── app/                          # Streamlit application
│   ├── main.py                   # Main Streamlit interface
│   ├── storage_manager.py        # MinIO storage management
│   └── config_manager.py         # Configuration management
├── models/                       # ML models and training
│   ├── training.py               # Training script
│   ├── inference.py              # Inference and evaluation
│   ├── inferencer.py             # Model inference wrapper
│   ├── checkpoints/              # Trained model checkpoints
│   └── utils/                    # Model utilities
├── docker/                       # Docker configuration
│   └── Dockerfile                # Application Dockerfile
├── docker-compose.yml            # Main Docker Compose
├── docker-compose.gpu.yml        # GPU-enabled Docker Compose
├── requirements.txt              # Python dependencies
├── deploy.sh                     # Deployment script
├── init_project.py               # Python deployment script
└── README.md                     # This file
```

## 🔧 Configuration

## 🚨 Troubleshooting

### Common Issues

#### "S3 endpoint domain: ." Error
**Symptom**: Annotations cannot be saved in Label Studio  
**Cause**: Trailing slash in export storage prefix  
**Solution**: Change `annotations/` to `annotations` in export storage configuration  
**Details**: See [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md)

#### Annotations Not Saving
**Symptom**: Clicking "Submit" in Label Studio shows errors  
**Cause**: Export storage not properly configured  
**Solution**: Verify export storage prefix format and project export settings  
**Details**: See [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md)

#### Export Storage "Not Found"
**Symptom**: Project setup shows "No export storage found"  
**Cause**: Storage configuration or project export settings issue  
**Solution**: Check storage creation and project configuration  
**Details**: See [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md)

### Environment Variables

Create a `.env` file (optional):

```bash
# Label Studio
LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=admin

# MinIO
MINIO_BUCKET_NAME=segmentation-platform

# Optional: GPU settings
NVIDIA_VISIBLE_DEVICES=all
```

### Label Studio Setup

1. Access Label Studio at `http://localhost:8080`
2. Login with `admin@example.com` / `admin`
3. Create a new project
4. Configure storage settings (see `LABEL_STUDIO_MINIO_SETTINGS.md`)

#### 🚨 Critical Storage Configuration

**Export Storage Prefix**: Must be `annotations` (NO trailing slash)  
**Source Storage Prefix**: Can be `images/` (trailing slash OK)

**Why**: The trailing slash in export storage prefix causes "S3 endpoint domain: ." errors and prevents annotations from saving. See [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md) for full details.

### MinIO Storage Structure

```
segmentation-platform/
├── images/                       # Uploaded images
├── annotations/                  # Label Studio annotations
└── models/                       # Model artifacts
```

## 🎨 Usage

### 1. Upload Images

- Use the Streamlit interface at `http://localhost:8501`
- Upload images through the "Upload Images" section
- Images are stored in MinIO under `images/` prefix

### 2. Annotate Images

- Click "Annotate Images" in Streamlit to open Label Studio
- Create polygon or brush annotations
- Annotations are automatically saved to MinIO

### 3. Train Model

- In Streamlit, go to the "Training" section
- Configure classes (detected automatically from annotations)
- Start training (runs in background)
- Monitor progress in real-time

### 4. Run Inference

- Select a trained model from the dropdown
- Upload an image or use batch evaluation
- Adjust segmentation threshold as needed
- View results with ground truth comparison

## 🔄 Data Persistence

All data is stored on your host machine using Docker bind mounts, ensuring data survives container restarts and rebuilds.

### Label Studio Data

- **Location**: `./label-studio-data/` (on host)
- **Contains**: Database, project configs, user settings
- **Container Path**: `/label-studio/data`
- **Auto-created**: By start.sh/bat or deploy.sh

### MinIO Data

- **Location**: `./minio-data/` (on host)
- **Contains**: Images, annotations, model artifacts
- **Container Path**: `/data`
- **Access**: MinIO Console at `http://localhost:9001`
- **Auto-created**: By start.sh/bat or deploy.sh

### Model Checkpoints

- **Location**: `./models/checkpoints/` (on host)
- **Contains**: Trained models (.pth) and configs (_config.json)
- **Container Path**: `/app/models/checkpoints`
- **Auto-created**: By start.sh/bat or deploy.sh

### Backup Recommendations

```bash
# Backup all data
tar -czf backup-$(date +%Y%m%d).tar.gz \
  label-studio-data/ \
  minio-data/ \
  models/checkpoints/

# Restore from backup
tar -xzf backup-20240101.tar.gz
```

## 🛠️ Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit locally
streamlit run app/main.py

# Run training locally
python models/training.py
```

### Docker Development

```bash
# Build and run with GPU support
docker compose -f docker-compose.gpu.yml up -d

# View logs
docker compose logs -f semseg-app

# Access container shell
docker compose exec semseg-app bash
```

## 📊 Monitoring

### Service Status

Check service status in Streamlit sidebar:
- MinIO connectivity
- Label Studio connectivity
- Model availability
- Storage usage

### Logs

```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs label-studio
docker compose logs semseg-app
docker compose logs minio
```

## 🔧 Troubleshooting

### Common Issues

1. **Label Studio projects not persisting**
   - Check `LABEL_STUDIO_PERSISTENCE.md`
   - Verify `label-studio-data/` directory exists

2. **MinIO connection issues**
   - Ensure MinIO is running: `docker compose ps`
   - Check endpoint URL in Label Studio settings

3. **Model loading errors**
   - Verify model checkpoint exists
   - Check class configuration matches training

4. **Training failures**
   - Check GPU availability (if using GPU)
   - Verify annotation format
   - Check MinIO connectivity

### Reset Options

```bash
# Reset everything (⚠️ destroys all data)
docker compose down -v
rm -rf label-studio-data/
docker compose up -d

# Reset only Label Studio (preserves MinIO data)
docker compose stop label-studio
docker compose rm -f label-studio
docker compose up -d label-studio

# Reset only MinIO (preserves Label Studio data)
docker compose stop minio
docker compose rm -f minio
docker compose up -d minio
```

## 📚 Documentation

- [Label Studio MinIO Settings](LABEL_STUDIO_MINIO_SETTINGS.md)
- [Label Studio Persistence](LABEL_STUDIO_PERSISTENCE.md)
- [MinIO Setup](MINIO_SETUP.md)
- [GPU Detection Setup](GPU_DETECTION_SETUP.md)
- [Dynamic Class Configuration](DYNAMIC_CLASS_CONFIGURATION.md)
- [Inference Class Awareness](INFERENCE_CLASS_AWARENESS.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Label Studio](https://labelstud.io/) for annotation tools
- [MinIO](https://min.io/) for object storage
- [PyTorch](https://pytorch.org/) for deep learning
- [Streamlit](https://streamlit.io/) for web interface
- [U-Net](https://arxiv.org/abs/1505.04597) architecture

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation files
3. Open an issue on GitHub
4. Check service logs for detailed error messages

