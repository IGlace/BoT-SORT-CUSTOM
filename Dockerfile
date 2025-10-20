FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

#RUN packages for opencv as https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6  -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 \
    build-essential \
    cmake \
    git \
    nano \
    wget \
    unzip \
    libhdf5-dev \
    liblapack-dev \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    graphviz \
    libgl1-mesa-glx \
    libcurl4 \
    g++ \
    fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt /tmp/requirements.txt

# Install the rest of your project dependencies
RUN python3.7 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install COCO API
RUN python3.7 -m pip install --no-cache-dir cython && \
    python3.7 -m pip install --no-cache-dir 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' && \
    python3.7 -m pip install --no-cache-dir cython_bbox

# Install FAISS with GPU support
RUN python3.7 -m pip install --no-cache-dir faiss-gpu