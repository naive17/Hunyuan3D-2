FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3.11-venv \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install PyTorch with CUDA support
RUN python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Clone the repository
WORKDIR /app
RUN git clone https://github.com/Tencent/Hunyuan3D-2.git

# Install requirements
WORKDIR /app/Hunyuan3D-2
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Build custom rasterizer with explicit CUDA architecture settings
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py build_ext --inplace && python3 setup.py install

# Build differentiable renderer
WORKDIR /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py build_ext --inplace && python3 setup.py install

# Install Gradio
WORKDIR /app/Hunyuan3D-2
RUN python3 -m pip install --no-cache-dir gradio sentencepiece

# Expose port 8080
EXPOSE 8080

# Run Gradio app on custom port
CMD ["bash", "-c", "python3 gradio_app.py --enable_t23d --host 0.0.0.0 --port 8080 --low_vram_mode --enable_flashvdm"]