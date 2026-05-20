# Semantic Segmentation Platform

A platform for semantic segmentation: annotate images in Label Studio, store data in MinIO, train U-Net models with PyTorch, and run inference — all through a Streamlit UI.

## Overview

| Service | Purpose | Port |
|---|---|---|
| Streamlit | UI, training, inference | 8501 |
| Label Studio | Image annotation | 8080 |
| MinIO | S3-compatible storage | 9000 / 9001 |

## Getting Started

See [INSTALL.md](INSTALL.md) for the full guide. You do **not** need to clone this repository — the platform runs from pre-built Docker images. The short version:

1. Install Docker ([Linux](https://docs.docker.com/engine/install/), [macOS](https://docs.docker.com/desktop/install/mac-install/), [Windows](https://docs.docker.com/desktop/install/windows-install/))
2. Download the startup files into a new folder:

   **Linux / macOS**
   ```bash
   mkdir semseg-platform && cd semseg-platform
   curl -O https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.yml
   curl -O https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.gpu.yml
   curl -O https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/start.sh
   chmod +x start.sh
   ```

   **Windows (PowerShell)**
   ```powershell
   mkdir semseg-platform; cd semseg-platform
   Invoke-WebRequest -Uri https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.yml -OutFile docker-compose.yml
   Invoke-WebRequest -Uri https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.gpu.yml -OutFile docker-compose.gpu.yml
   Invoke-WebRequest -Uri https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/start.bat -OutFile start.bat
   ```

3. Start the platform:

   > **Before running:** Make sure Docker is running. On Windows and macOS, open Docker Desktop and wait for the whale icon in the taskbar/menu bar to stop animating. On Linux, verify with `docker ps` — if it errors, run `sudo systemctl start docker`.

   ```bash
   ./start.sh      # Linux / macOS
   .\start.bat     # Windows (PowerShell — note the .\)
   ```

On the first run Docker pulls all images (~5–7 GB, one time only).

### Access the applications

| Application | URL | Default credentials |
|---|---|---|
| Streamlit | http://localhost:8501 | none |
| Label Studio | http://localhost:8080 | admin@example.com / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |

## Usage

### 1. Upload Images

Use the Streamlit UI at http://localhost:8501. Images are stored in MinIO under the `images/` prefix.

### 2. Annotate Images

Open Label Studio from the Streamlit sidebar. Create polygon or brush segmentation masks. Annotations are saved automatically to MinIO.

### 3. Train a Model

Go to the Training section in Streamlit. Classes are detected automatically from your annotations. Training runs in the background with real-time progress. GPU training takes 15–30 min; CPU takes 2–4 hours.

### 4. Run Inference

Select a trained model checkpoint, upload an image, and view the segmentation result. Adjust the threshold as needed.

## Data Persistence

All data lives on your machine via Docker bind mounts and survives container restarts and image updates.

| Data | Host path | Contents |
|---|---|---|
| Label Studio | `./label-studio-data/` | database, project config, user settings |
| MinIO | `./minio-data/` | images, annotations, model artifacts |
| Model checkpoints | `./models/checkpoints/` | trained `.pth` files and configs |

### Backup

```bash
tar -czf semseg-backup-$(date +%Y%m%d).tar.gz \
  minio-data/ \
  label-studio-data/ \
  models/checkpoints/
```

Restore by extracting the archive into your `semseg-platform/` folder before running `docker compose up`.

## Troubleshooting

### Label Studio — "S3 endpoint domain: ." error

**Cause**: Trailing slash in the export storage prefix.  
**Fix**: Change `annotations/` to `annotations` (no trailing slash) in the export storage configuration.  
See [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md) for full details.

### Label Studio Setup

1. Access Label Studio at http://localhost:8080
2. Log in with `admin@example.com` / `admin`
3. Create a new project and configure storage

**Export Storage Prefix**: must be `annotations` (no trailing slash)  
**Source Storage Prefix**: `images/` (trailing slash is fine here)

### Common issues

| Issue | Fix |
|---|---|
| Label Studio projects not persisting | Verify `label-studio-data/` exists and is writable |
| Label Studio blank page or 502 | Database still initialising — wait 60 s and refresh |
| MinIO connection error | Wait 30 s after startup; check `docker compose logs minio` |
| Model loading error | Verify checkpoint exists; check class config matches training |
| Training failure | Check GPU availability and annotation format |
| Port conflict | Edit the left-hand port in `docker-compose.yml` (e.g. `"8502:8501"`) |
| Out of memory | Close other apps; increase Docker memory limit in Docker Desktop Settings → Resources |

### Logs

```bash
docker compose ps                    # check service status
docker compose logs semseg-app      # Streamlit + training
docker compose logs label-studio    # annotation tool
docker compose logs minio           # storage
```

### Reset options

```bash
# Reset only Label Studio (preserves MinIO and model data)
docker compose stop label-studio
docker compose rm -f label-studio
docker compose up -d label-studio

# Reset only MinIO (preserves Label Studio and model data)
docker compose stop minio
docker compose rm -f minio
docker compose up -d minio
```

## Documentation

- [INSTALL.md](INSTALL.md) — full installation guide (start here)
- [CRITICAL_FIX_DOCUMENTATION.md](CRITICAL_FIX_DOCUMENTATION.md) — Label Studio storage fix
- [DOCKER_SETUP.md](DOCKER_SETUP.md) — Docker configuration details
- [MANAGING_MULTIPLE_PROJECTS.md](MANAGING_MULTIPLE_PROJECTS.md) — running multiple projects

## Development

This section is for contributors who have cloned the repository.

### Repository structure

```
semsegplat-full_local_version/
├── app/                     # Streamlit application
│   ├── main.py              # main interface
│   ├── storage_manager.py   # MinIO integration
│   └── config_manager.py    # configuration
├── models/                  # ML models
│   ├── training.py          # training script
│   ├── inference.py         # evaluation
│   ├── inferencer.py        # inference wrapper
│   └── utils/               # utilities
├── docker/
│   └── Dockerfile
├── docker-compose.yml       # CPU / auto-detect
├── docker-compose.gpu.yml   # GPU overlay
├── start.sh                 # Linux/macOS startup script
└── start.bat                # Windows startup script
```

### Local development

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

### Docker development

```bash
# CPU (or auto-detect GPU at runtime)
docker compose up -d

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Logs
docker compose logs -f semseg-app

# Shell access
docker compose exec semseg-app bash
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests if applicable
4. Submit a pull request

## Security considerations for hosting

This platform is designed for **local use only**. If you intend to host it on a server or expose it to a network beyond your own machine, be aware of the following before doing so:

### Default credentials are public

The default credentials for MinIO and Label Studio are hardcoded in `docker-compose.yml` and visible in this public repository:

| Service | Default username | Default password |
|---|---|---|
| MinIO | `minioadmin` | `minioadmin123` |
| Label Studio | `admin@example.com` | `admin` |

Anyone who has read this repository knows these values. Before exposing the platform to a network, change all credentials in `docker-compose.yml` to strong unique values.

### Ports to protect

If hosted on a server, the following ports must be firewalled or placed behind authentication:

| Port | Service |
|---|---|
| 8501 | Streamlit UI |
| 8080 | Label Studio |
| 9000 | MinIO S3 API |
| 9001 | MinIO admin console |

### Model checkpoints from untrusted sources

PyTorch model files (`.pth`) can execute arbitrary code when loaded. Only load checkpoints from sources you trust.

## License

MIT — see [LICENSE.md](LICENSE.md).

## Acknowledgments

- [Label Studio](https://labelstud.io/) — annotation
- [MinIO](https://min.io/) — object storage
- [PyTorch](https://pytorch.org/) — deep learning
- [Streamlit](https://streamlit.io/) — web interface
- [U-Net](https://arxiv.org/abs/1505.04597) — model architecture
