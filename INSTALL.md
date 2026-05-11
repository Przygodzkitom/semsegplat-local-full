# Installation Guide — Semantic Segmentation Platform

This guide covers a fresh installation using pre-built images from the registry. You do **not** need to clone the repository or build anything yourself.

---

## What you will be running

Three services start together and communicate over an internal Docker network:

| Service | What it does | Port |
|---|---|---|
| **semseg-app** | Streamlit UI, model training, inference | 8501 |
| **label-studio** | Image annotation tool | 8080 |
| **minio** | S3-compatible local file storage | 9000 / 9001 |

All data (images, annotations, trained models) is stored in folders on your machine, not inside containers. This means data survives container restarts, updates, and reinstalls.

---

## System requirements

- 8 GB RAM minimum (16 GB recommended for training)
- 15 GB free disk space (images + Docker layers)
- Internet connection for the initial image download (~5–7 GB total, one time only)
- NVIDIA GPU with CUDA support — optional, CPU-only works but training is slower

---

## Step 1 — Install Docker

Docker is the only software you need to install.

### Linux

Install Docker Engine and the Compose plugin from the official repository. The exact commands depend on your distribution; the official instructions are at [docs.docker.com/engine/install](https://docs.docker.com/engine/install). After installation, verify:

```bash
docker --version
docker compose version
```

**Required — add your user to the `docker` group:**

```bash
sudo usermod -aG docker $USER
```

Then log out and log back in (or run `newgrp docker` in the current terminal). This is mandatory — without it every `docker` command fails with a "permission denied" error on `/var/run/docker.sock`. Verify the fix worked before continuing:

```bash
docker ps
```

If that prints a table (even an empty one) without an error, you are ready to proceed.

### macOS

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/). It bundles Docker Engine, Docker Compose, and a lightweight Linux VM. After installation, open Docker Desktop and wait for the whale icon in the menu bar to stop animating before running any commands.

### Windows

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/). WSL 2 backend is required (Docker Desktop will prompt you to enable it). After installation, open Docker Desktop and wait for it to finish starting before running any commands.

---

## Step 2 — GPU support (optional, NVIDIA only)

Skip this step if you do not have an NVIDIA GPU, or if you plan to run on CPU only. The platform works on CPU — training just takes longer.

### Linux — NVIDIA Container Toolkit

```bash
# Add the NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Windows / macOS

GPU passthrough for Docker requires Linux. On Windows, GPU access works through WSL 2 — install the latest NVIDIA driver for Windows (CUDA support is included automatically). On macOS, Apple Silicon is not supported via NVIDIA; the app runs on CPU.

---

## Step 3 — Get the files

Create a folder anywhere on your machine to hold the application data and configuration, then download the startup files into it:

```bash
mkdir semseg-platform
cd semseg-platform
```

**Linux / macOS** — download with `curl`:

```bash
curl -O https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.gpu.yml
curl -O https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/start.sh
chmod +x start.sh
```

**Windows** — download with PowerShell:

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.yml -OutFile docker-compose.yml
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/docker-compose.gpu.yml -OutFile docker-compose.gpu.yml
Invoke-WebRequest -Uri https://raw.githubusercontent.com/Przygodzkitom/semsegplat-local-full/main/start.bat -OutFile start.bat
```

Alternatively, download all four files manually from the repository releases page and place them in the `semseg-platform` folder.

---

## Step 4 — Start the platform

On the first run Docker pulls all three images from the registry. This is a one-time download of roughly 5–7 GB and may take 10–20 minutes depending on your connection speed.

The startup scripts create the required data directories automatically, detect whether you have an NVIDIA GPU, and launch the correct compose configuration.

> **Before running:** Make sure Docker is running. On Windows and macOS, open Docker Desktop and wait for the whale icon in the taskbar/menu bar to stop animating. On Linux, verify with `docker ps` — if it errors, run `sudo systemctl start docker`.

**Linux / macOS:**

```bash
./start.sh
```

**Windows (PowerShell):**

```powershell
.\start.bat
```

> **Note:** In PowerShell you must prefix with `.\`. Running `start.bat` alone invokes the Windows `start` command instead of the script.

The script prints which configuration it selected (GPU or CPU) and starts the containers in the foreground so you can watch the log output directly. After the first run you can also use it for daily restarts — it is safe to run repeatedly.

**What the scripts create for you:**

```
semseg-platform/
├── docker-compose.yml
├── docker-compose.gpu.yml
├── start.sh  /  start.bat
├── minio-data/              ← MinIO object storage
├── label-studio-data/       ← Label Studio database and projects
└── models/
    └── checkpoints/         ← trained model checkpoints (persisted on host)
```

The application source code lives inside the image. Only the `checkpoints/` subfolder is mounted from your machine so trained models survive container updates.

### Manual start (without the scripts)

If you prefer not to use the scripts, create the data directories yourself and run compose directly:

```bash
# Linux / macOS
mkdir -p minio-data label-studio-data models/checkpoints

# Windows (PowerShell)
New-Item -ItemType Directory -Force minio-data, label-studio-data, models\checkpoints
```

Then start:

```bash
docker compose up -d                                                    # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU
```

Watch the startup progress:

```bash
docker compose logs -f
```

Wait until you see Label Studio print something like `Starting web server...` and semseg-app prints the Streamlit URL. This takes about 30–60 seconds after images finish downloading.

---

## Step 6 — Open the applications

| Application | URL | Default credentials |
|---|---|---|
| Streamlit (main UI) | http://localhost:8501 | none |
| Label Studio | http://localhost:8080 | admin@example.com / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |

On the very first start Label Studio initialises its database, which can take an extra 30–60 seconds. If http://localhost:8080 shows a loading screen, wait a moment and refresh.

---

## Typical first workflow

1. **Upload images** — open http://localhost:8501, go to the Upload tab, and upload your images. They are stored in MinIO.
2. **Annotate** — click "Open Label Studio" in the Streamlit sidebar, log in, open your project, and draw segmentation masks.
3. **Train** — return to Streamlit, go to the Training tab, pick your classes, and start training. GPU training takes 15–30 min; CPU takes 2–4 hours.
4. **Infer** — go to the Inference tab, select a trained checkpoint, upload a new image, and view the result.

---

## Stopping and restarting

```bash
# Stop all containers (data is preserved)
docker compose down

# Restart — using the script (recommended, auto-detects GPU)
./start.sh        # Linux / macOS
.\start.bat       # Windows (PowerShell)

# Restart — manually
docker compose up -d                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d  # GPU
```

Containers can also be stopped and started individually:

```bash
docker compose stop semseg-app
docker compose start semseg-app
```

---

## Updating to a newer version

Pull the latest images and recreate the containers:

```bash
docker compose pull
docker compose up -d
```

Your data in `minio-data/`, `label-studio-data/`, and `models/` is untouched.

---

## Troubleshooting

### Checking service status

```bash
docker compose ps
```

All three services should show `Up` or `running`.

### Reading logs

```bash
docker compose logs semseg-app      # Streamlit + training logs
docker compose logs label-studio    # Annotation tool logs
docker compose logs minio           # Storage logs
```

### Common issues

**"permission denied" on /var/run/docker.sock (Linux)**

Your user is not in the `docker` group. Run:

```bash
sudo usermod -aG docker $USER
newgrp docker   # applies immediately in the current terminal
```

Then retry. If `newgrp docker` is not enough (e.g. the script was launched from a file manager), log out and back in fully.

**Port already in use**

Something else on your machine is using port 8501, 8080, 9000, or 9001. Stop the conflicting service, or edit `docker-compose.yml` and change the left-hand side of the port mapping (e.g. `"8502:8501"`) then restart.

**Docker Desktop not running (macOS / Windows)**

Open Docker Desktop and wait for it to finish starting before running `docker compose` commands.

**GPU not detected inside the container**

Verify the NVIDIA Container Toolkit is installed and the test command works:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If that fails, revisit Step 2. If you used the GPU compose file but see `No GPU detected` in logs, the toolkit may need a Docker daemon restart (`sudo systemctl restart docker`).

**"no nvidia runtime" error**

You ran the GPU compose command on a machine without the NVIDIA Container Toolkit. Switch to the CPU command:

```bash
docker compose up -d
```

The app detects GPU availability at runtime and falls back to CPU automatically.

**Label Studio shows blank page or 502**

The database is still initialising. Wait 60 seconds and refresh.

**MinIO connection error in Streamlit**

MinIO takes a few seconds to start. Wait 30 seconds after `docker compose up` and refresh http://localhost:8501. If the error persists, check `docker compose logs minio`.

**Out of memory during training**

Training large datasets requires significant RAM and VRAM. Close other applications to free memory. On Docker Desktop you can increase the memory limit in Settings → Resources.

---

## Data backup

All persistent data lives in three directories. Back them up with:

```bash
tar -czf semseg-backup-$(date +%Y%m%d).tar.gz \
  minio-data/ \
  label-studio-data/ \
  models/checkpoints/
```

Restore by extracting the archive into the same `semseg-platform/` folder before running `docker compose up`.

**Important**: when restoring, copy the data directories into place before running `docker compose up`. Starting the containers first and copying data into a running MinIO or Label Studio instance can corrupt the database.
