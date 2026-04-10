# Managing Multiple Projects

This document explains how data is stored in this platform and how to run multiple independent projects side-by-side.

---

## How Data is Stored

The platform uses three services orchestrated by Docker Compose:

| Service | Purpose | Data location on your disk |
|---|---|---|
| `semseg-app` | Main Streamlit UI | `./models/`, `./app/` |
| `label-studio` | Annotation tool | `./label-studio-data/` |
| `minio` | Image storage | `./minio-data/` |

**Key point:** the data folders (`label-studio-data/` and `minio-data/`) live on your local disk as regular folders — not inside Docker. This means:

- Stopping or restarting containers does **not** erase your data.
- All images, annotations, trained models, and Label Studio project settings persist between sessions.
- The entire state of one "project" is captured by these two folders plus `models/checkpoints/`.

---

## What a "New Project" Means

Because data is stored in local folders, a new independent project = a **separate copy of the whole app folder** with fresh (empty) data directories and different network ports.

Docker Desktop alone is not sufficient for this — it can start and stop existing stacks but cannot launch a second independent copy of a `docker-compose.yml` from a different folder. A terminal is required for the steps below.

---

## Creating a New Project — Step by Step

### Step 1 — Copy the app folder

Use your file manager or the terminal. Give the copy a meaningful name.

```bash
cp -r semsegplat-full_local_version semsegplat-project-B
```

### Step 2 — Clear the data directories in the new copy

This gives you a clean slate. The old project folder is untouched.

```bash
cd semsegplat-project-B
rm -rf label-studio-data && mkdir label-studio-data
rm -rf minio-data && mkdir minio-data
```

> Note: using `rm -rf minio-data` (not `minio-data/*`) ensures the hidden `.minio.sys/` directory is also removed, so MinIO starts completely fresh.

### Step 3 — Change the ports in `docker-compose.yml` *(only if running instances simultaneously)*

**If you stop the old instance before starting the new one, skip this step** — the ports will be free.

If you want both instances running at the same time, they must use different host ports. Open `docker-compose.yml` in any text editor and change every port that conflicts:

| Service | Original | New project (example) |
|---|---|---|
| Streamlit app | `8501:8501` | `8502:8501` |
| Label Studio | `8080:8080` | `8081:8080` |
| MinIO API | `9000:9000` | `9002:9000` |
| MinIO Console | `9001:9001` | `9003:9001` |

Only the **left** number (host port) needs to change. The right number is internal to Docker and must stay the same.

### Step 4 — Give the new stack a unique project name

Docker Compose uses the folder name as a project name by default. If you renamed the folder this is automatic. You can also set it explicitly in the `.env` file or via the `-p` flag:

```bash
docker compose -p project-b up -d
```

### Step 5 — Start the new stack

```bash
cd semsegplat-project-B
docker compose up -d
```

The new instance is now accessible at the ports you set in Step 3, e.g. `http://localhost:8502`.

---

## Switching Between Projects

**Sequential use (simplest — no port changes needed):**

Stop the current project, then start the next one. Both use the default ports.

```bash
cd semsegplat-project-A
docker compose down

cd ../semsegplat-project-B
docker compose up -d
```

**Simultaneous use (requires different ports — see Step 3):**

Both instances run at the same time, accessible by their respective ports:

| Project | App URL | Label Studio URL |
|---|---|---|
| Project A (original) | http://localhost:8501 | http://localhost:8080 |
| Project B | http://localhost:8502 | http://localhost:8081 |

---

## Stopping a Project Instance

```bash
cd semsegplat-project-B
docker compose down
```

This stops and removes the containers but **leaves all data intact** in `label-studio-data/` and `minio-data/`. Run `docker compose up -d` again any time to resume.

To stop all instances at once from Docker Desktop: go to the **Containers** tab and stop each stack individually.

---

## Backing Up and Restoring a Project

### What to back up

Three folders contain everything needed to fully restore a project:

| Folder | Contains |
|---|---|
| `minio-data/` | All uploaded images and exported annotation masks |
| `label-studio-data/` | Label Studio database (tasks, annotations, project settings) |
| `models/checkpoints/` | Trained model weights (`.pth`) and their config files (`.json`) |

Everything else (app code, Dockerfiles, compose files) is in git and does not need to be backed up separately.

### Creating a backup

Copy the three folders while the stack is stopped to avoid partial writes:

```bash
cd semsegplat-project-A
docker compose down

cp -r minio-data       ../backup-project-A/minio-data
cp -r label-studio-data ../backup-project-A/label-studio-data
cp -r models/checkpoints ../backup-project-A/checkpoints
```

Or back up the entire project folder at once (simplest):

```bash
cp -r semsegplat-project-A semsegplat-project-A-backup-$(date +%Y-%m-%d)
```

Or as a compressed archive:

```bash
tar -czf project-A-backup-$(date +%Y-%m-%d).tar.gz semsegplat-project-A/
```

### Restoring from a backup

1. Clone or copy the app folder (code only, no data):

   ```bash
   cp -r semsegplat-full_local_version semsegplat-restored
   cd semsegplat-restored
   ```

2. Remove the empty data folders created by the copy, then replace them with your backups:

   ```bash
   rm -rf minio-data label-studio-data models/checkpoints
   cp -r ../backup-project-A/minio-data        ./minio-data
   cp -r ../backup-project-A/label-studio-data ./label-studio-data
   cp -r ../backup-project-A/checkpoints       ./models/checkpoints
   ```

3. Start the stack:

   ```bash
   docker compose up -d
   ```

> **Important — order matters:** always copy the data folders *before* running `docker compose up`. If Label Studio starts against an empty `label-studio-data/` folder it will initialise a fresh database, overwriting any file you copy in afterwards. If this happens: stop the stack (`docker compose down`), replace `label-studio-data/` with your backup again, then restart.

All images, annotations, and trained models will be accessible through the GUI immediately — no re-import needed.

---

## Summary

| Task | Tool needed |
|---|---|
| Start / stop an existing instance | Docker Desktop or terminal |
| Create a new project instance | Terminal (steps 1–5 above) |
| Switch between running projects | Browser (different ports) |
| Back up a project | File manager or terminal |
