@echo off
REM Semantic Segmentation Platform Startup Script
REM Universal script that works on any machine - GPU or CPU-only

echo 🚀 Semantic Segmentation Platform - Universal Startup
echo ===================================================

REM Create required directories if they don't exist
echo 📁 Ensuring directory structure...
if not exist "label-studio-data" mkdir label-studio-data
if not exist "minio-data" mkdir minio-data
if not exist "models\checkpoints" mkdir models\checkpoints
echo ✅ Directory structure ready
echo.

echo 🔍 Starting with GPU configuration...

REM Try legacy docker-compose first for GPU — it correctly honors "runtime: nvidia"
docker-compose -f docker-compose.gpu.yml up
if %errorlevel% neq 0 (
    echo ⚠️  Legacy docker-compose not available, trying modern docker compose...
    docker compose -f docker-compose.gpu.yml up
    if %errorlevel% neq 0 (
        echo ⚠️  GPU configuration failed, falling back to CPU
        echo 🔄 Starting with CPU configuration
        docker compose up
        if %errorlevel% neq 0 (
            echo ⚠️  Modern docker compose failed, trying legacy
            docker-compose up
        )
    )
)
