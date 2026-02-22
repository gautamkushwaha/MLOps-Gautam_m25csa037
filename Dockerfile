# Base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Prevent python buffering
ENV DEBIAN_FRONTEND=noninteractive

# Working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command (training)
CMD ["python", "train.py"]
