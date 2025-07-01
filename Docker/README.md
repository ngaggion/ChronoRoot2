# ChronoRoot 2.0 Docker

This directory contains the Dockerfile and documentation for the official ChronoRoot 2.0 Docker image, which provides the recommended way to use all components of the ChronoRoot 2.0 platform.

## Overview

The Docker image `ngaggion/chronoroot` contains a complete environment for running all ChronoRoot 2.0 components:

- nnUNet for segmentation (in the `base` conda environment)
- Standard Root Phenotyping Interface (in the `ChronoRootInterface` conda environment)
- Screening Interface (in the `ChronoRootInterface` conda environment)
- Functional PCA analysis tools (in the `FDA` conda environment)
- All necessary dependencies and libraries
- Pre-downloaded nnUNet models

The image is based on `pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime` and uses three separate conda environments to manage different components of the system.

## System Requirements

- Docker installed on your system
- NVIDIA GPU with appropriate drivers (for optimal performance)
- nvidia-docker2 package installed (for GPU acceleration)
- At least 30GB of free disk space
- 16GB RAM recommended (8GB minimum)

## Installation

### 1. Install Docker

Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your operating system.

For Ubuntu, you can use the following command:

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

Then, install Docker:
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 2. Install NVIDIA Container Toolkit (nvidia-docker2)

For GPU acceleration, you need to install nvidia-docker2. See the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for more details.

Here are the steps for Ubuntu/Debian:

```bash
# Add the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
sudo apt-get update

# Install nvidia-docker2
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Restart Docker to apply changes
sudo systemctl restart docker
```

### 3. Verify Installation

Verify that the NVIDIA Container Toolkit is working correctly:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

This command should display information about your GPU, confirming that Docker can access it.

## Downloading the ChronoRoot 2.0 Image

Pull the latest version of the ChronoRoot 2.0 Docker image:

```bash
docker pull ngaggion/chronoroot:latest
```

## Updating the ChronoRoot 2.0 Image

If you want to update the nnUNet models and github repo locally, you can build the image from the `Update` directory:

```bash
cd Docker/Update
docker build -t ngaggion/chronoroot:latest .
```

## Running ChronoRoot 2.0

### Basic Usage

For command-line only usage (e.g., running segmentation without GUI):

```bash
MOUNT="YOUR_LOCAL_DATA_PATH"
docker run -it --gpus all -v $MOUNT:/DATA/ ngaggion/chronoroot:latest
```

### With Graphical Interface

To run ChronoRoot 2.0 with the graphical interfaces, you need to enable X11 forwarding:

```bash
# Allow Docker to access X server
xhost +local:docker

# Run container with display forwarding
MOUNT="YOUR_LOCAL_DATA_PATH"
docker run -it --gpus all \
    -v $MOUNT:/DATA/ \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ngaggion/chronoroot:latest

# Restrict X server access when done
xhost -local:docker
```

## Usage Inside the Container

Once inside the container, you'll be in the ChronoRootInterface environment. You can:

### Run the Standard Interface

```bash
# Make sure you're in the ChronoRootInterface environment
conda activate ChronoRootInterface

# Start the Standard Interface
cd /app/ChronoRoot2/chronoRootApp
python run.py
```

### Run the Screening Interface

```bash
# Make sure you're in the ChronoRootInterface environment
conda activate ChronoRootInterface

# Start the Screening Interface
cd /app/ChronoRoot2/chronoRootScreeningApp
python run.py
```

### Run the Segmentation Pipeline

```bash
# Switch to the base environment where nnUNet is installed
conda activate base

# Go to segmentation directory
cd /app/ChronoRoot2/segmentationApp

# Edit test.sh to specify your input/output paths
nano test.sh

# Run segmentation
./test.sh
```

## Data Management

The Docker container mounts your local directory to `/DATA/` inside the container. This is the recommended location for storing your input videos and output results.

Example directory structure:

```
/DATA/
├── input_videos/          # Place your raw videos here <- Segmentations will be stored within them
└── analysis_results/      # Analysis results will be stored here
```

## Docker Image Structure

The image is built with three conda environments:

1. **Base environment**: Contains nnUNet for segmentation
2. **ChronoRootInterface**: Contains dependencies for both user interfaces
3. **FDA**: Contains libraries for Functional Principal Component Analysis

## Dockerfile Breakdown

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
```
- Base image with PyTorch and CUDA 11.8 support

```dockerfile
RUN apt-get update && \
    apt-get install -y wget git python3-pyqt5 nano gedit
```
- Installs system dependencies and useful utilities

```dockerfile
RUN git clone https://github.com/ngaggion/nnUNet /nnUNet
WORKDIR /nnUNet
RUN /opt/conda/bin/conda run pip install -e .
```
- Installs the modified nnUNet package in the base environment

```dockerfile
RUN /opt/conda/bin/conda create -y -n ChronoRootInterface python=3.8.2 \
    graph-tool=2.29 pyqt=5.9.2 numpy=1.18.1 scikit-image=0.16.2 pandas=1.0.3 seaborn=0.12.2 -c conda-forge/label/cf202003 && \
    /opt/conda/bin/conda install -y -n ChronoRootInterface -c conda-forge pyzbar pip
```
- Creates the ChronoRootInterface environment with precise dependency versions
- Note the use of specific conda-forge channel (cf202003) for compatibility

```dockerfile
RUN /opt/conda/bin/conda create -y -n FDA && \
    /opt/conda/bin/conda install -y -n FDA -c conda-forge scikit-fda scipy pandas matplotlib seaborn ipykernel
```
- Creates the FDA environment for functional data analysis

```dockerfile
RUN git clone https://github.com/ngaggion/ChronoRoot2.git /app
```
- Clones the ChronoRoot 2.0 repository into the /app directory

```dockerfile
WORKDIR /app/Segmentation
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz && \
    tar -xvf git-lfs-linux-amd64-v3.5.1.tar.gz && \
    [...] \
    git clone https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet
```
- Installs git-lfs and downloads pre-trained segmentation models

## Building the Image

If you want to build the Docker image locally:

```bash
cd Docker
docker build -t chronoroot:latest .
```

## Customization

If you need to adapt the image for different hardware requirements, you can modify the base image in the first line:

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
```

For example, to use CUDA 12.1:

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
```

## Working with the Pre-installed Models

The container comes with pre-downloaded nnUNet models from Hugging Face. These models are located in:

```
/app/Segmentation/ChronoRoot_nnUNet
```

## Getting Updates

To update to the latest version of ChronoRoot 2.0:

```bash
# Pull the latest image
docker pull ngaggion/chronoroot:latest

# Inside the container, update the code
git -C /app/ChronoRoot2 pull
```

## Available Tools

The Docker container includes several useful utilities:

- `nano` and `gedit` text editors for file editing
- `git` for version control
- `git-lfs` for handling large files (e.g., models)

## Troubleshooting

If you encounter issues with the graphical interface, make sure:
- X11 forwarding is properly configured
- Your NVIDIA drivers are compatible with CUDA 11.8
- The container has access to your GPU

For GPU memory errors, you can try:
- Reducing batch size in segmentation
- Processing videos in smaller segments
- Using a machine with more GPU memory