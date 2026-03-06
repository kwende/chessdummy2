# CUDA runtime base image with cuDNN (good for PyTorch GPU)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      ca-certificates \
      zstd \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA wheels (cu121). Unpinned so you can "just run";
# if you want reproducibility later, pin versions.
RUN python3 -m pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121

# Your script deps
RUN python3 -m pip install --no-cache-dir \
      python-chess zstandard numpy tqdm tensorboard

WORKDIR /workspace

# Default to a shell so you can run whatever command you want
CMD ["bash"]