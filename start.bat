@echo off
REM Semantic Segmentation Platform Startup Script
REM Universal script that works on any machine - GPU or CPU-only

echo 🚀 Semantic Segmentation Platform - Universal Startup
echo ===================================================

echo 🔍 Starting with GPU configuration...

REM Try GPU configuration first
docker compose -f docker-compose.gpu.yml up
if %errorlevel% neq 0 (
    echo ⚠️  GPU configuration failed, trying legacy docker-compose
    docker-compose -f docker-compose.gpu.yml up
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
