FROM python:3.10-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Install PyTorch with CUDA support (will fall back to CPU if no GPU)
# This allows runtime detection of GPU availability
RUN pip3 install --no-cache-dir torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Then install packages from requirements.txt to ensure version compatibility
RUN pip3 install --no-cache-dir -r requirements.txt

# Clean up pip cache
RUN pip3 cache purge

COPY . .
RUN rm -rf .streamlit
EXPOSE 8501

# The app will automatically detect GPU at runtime using the gpu_detector module
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"] 