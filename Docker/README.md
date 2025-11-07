# ChronoRoot 2.0 Docker

This directory contains the Dockerfile and documentation for the official ChronoRoot 2.0 Docker image, which provides the recommended way to use all components of the ChronoRoot 2.0 platform.

## Overview

The Docker image `ngaggion/chronoroot` contains a complete environment for running all ChronoRoot 2.0 components:

- nnUNetv2 for AI-powered segmentation
- Standard Root Phenotyping Interface for detailed individual plant analysis
- Screening Interface for high-throughput batch processing
- Functional PCA analysis tools (scikit-fda)
- All necessary dependencies and libraries
- Pre-downloaded nnUNet models

The image is based on `pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime` and uses a single unified conda environment (`ChronoRoot`) that contains all components, simplifying usage and maintenance.

## System Requirements

- Docker installed on your system
- NVIDIA GPU with appropriate drivers (recommended for segmentation performance)
- nvidia-docker2 package installed (for GPU acceleration)
- At least 30GB of free disk space
- 16GB RAM recommended (8GB minimum)

## Installation

### 1. Install Docker

Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your operating system.

For Ubuntu, you can use the following commands:

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

# Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

For Windows, ChronoRoot has been tested under Windows Subsystem for Linux, version 2. Please refer to [this link](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers) for information on how to set up Docker with WSL2.

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

## Running ChronoRoot 2.0

### Linux Systems

For Linux systems, you need to enable X11 forwarding to use the graphical interfaces:

```bash
# Allow Docker to access X server
xhost +local:docker

# Run container with display forwarding
MOUNT="YOUR_LOCAL_DATA_PATH"
docker run -it --gpus all \
    -v $MOUNT:/DATA/ \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --shm-size=8gb \
    ngaggion/chronoroot:latest

# When finished, restrict X server access
xhost -local:docker
```

### Windows Systems (WSL2)

For Windows Subsystem for Linux 2, use the following command:

```bash
MOUNT="YOUR_LOCAL_DATA_PATH"

docker run -it --gpus all \
    -v $MOUNT:/DATA/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -e DISPLAY \
    -e WAYLAND_DISPLAY \
    -e XDG_RUNTIME_DIR \
    -e PULSE_SERVER \
    --shm-size=8gb \
    ngaggion/chronoroot:latest
```

### Without GPU

If you don't have an NVIDIA GPU or want to run without GPU acceleration, simply remove the `--gpus all` flag:

```bash
docker run -it \
    -v $MOUNT:/DATA/ \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --shm-size=8gb \
    ngaggion/chronoroot:latest
```

> **Note**: Segmentation will be significantly slower without GPU acceleration.

## Usage Inside the Container

Once inside the container, you'll automatically be in the `ChronoRoot` conda environment. The container includes convenient aliases for quick access to each component:

### Quick Access with Aliases

```bash
chronoroot     # Launch Standard Root Phenotyping Interface
screening      # Launch High-throughput Screening Interface
segmentation   # Launch Segmentation Interface
```

### Manual Activation

If you prefer to activate manually or the aliases don't work:

#### Standard Root Phenotyping Interface

```bash
conda activate ChronoRoot
cd /app/chronoRootApp
python run.py
```

#### Screening Interface

```bash
conda activate ChronoRoot
cd /app/chronoRootScreeningApp
python run.py
```

#### Segmentation Pipeline

```bash
conda activate ChronoRoot
cd /app/segmentationApp
python run.py
```

## Data Management

The Docker container mounts your local directory to `/DATA/` inside the container. The mounted directory should contain your videos, projects, and results.

Example directory structure:

```
/DATA/
├── Videos/                # Place your raw videos here
│   ├── experiment1/
│   └── experiment2/
├── Projects/              # Analysis projects will be created here
│   ├── project1/
│   └── project2/
└── Results/               # Final results and reports
```

**Important Notes:**
- Segmentation results are stored within the video directories
- Analysis projects can be created anywhere but `/DATA/Projects/` is recommended
- All data in `/DATA/` persists between container runs

## Docker Image Structure

The image uses a single unified conda environment called `ChronoRoot` that contains:

- Python 3.13.9
- PyQt 5.15.11 for graphical interfaces
- nnUNetv2 2.6.2 for segmentation
- scikit-fda 0.10.1 for functional data analysis
- All analysis dependencies (pandas, seaborn, scipy, etc.)

This unified environment simplifies usage and eliminates the need to switch between environments.

## Troubleshooting

### Display Issues

If the graphical interface doesn't appear:

1. **On Linux**: Verify X11 forwarding is enabled
   ```bash
   xhost +local:docker
   ```

2. **On WSL2**: Ensure WSLg is properly configured
   - Update to the latest WSL version: `wsl --update`
   - Verify display variables are set correctly

### GPU Issues

If GPU is not detected:

1. Verify nvidia-docker2 is installed: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`
2. Check NVIDIA drivers are compatible with CUDA 11.8
3. Ensure Docker daemon has been restarted after installing nvidia-docker2

### Memory Issues

If you encounter out-of-memory errors:

- Increase `--shm-size` parameter (default is 8gb)
  ```bash
  docker run -it --gpus all --shm-size=16gb [...]
  ```

### Permission Issues

If you encounter permission errors when accessing `/DATA/`:

```bash
# On the host system, ensure the mounted directory has correct permissions
sudo chmod -R 755 /your/data/path
```

## Getting Help

If you encounter issues not covered here:

1. Check the [main repository issues](https://github.com/ngaggion/ChronoRoot2/issues)
2. Review the [component-specific documentation](../README.md)
3. Open a new issue with detailed information about your problem.