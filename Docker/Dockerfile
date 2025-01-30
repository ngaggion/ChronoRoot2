FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /

RUN /opt/conda/bin/conda init bash

# Install necessary packages
RUN apt-get update && \
    apt-get install -y wget git python3-pyqt5 nano gedit

# nnUNet is installed in the base environment
RUN git clone https://github.com/ngaggion/nnUNet /nnUNet

# Set working directory
WORKDIR /nnUNet

# Install nnUNet
RUN /opt/conda/bin/conda run pip install -e . && \
    /opt/conda/bin/conda clean -a -y

WORKDIR /

# Create and activate Conda environment for GUI Interface 
RUN /opt/conda/bin/conda create -y -n ChronoRootInterface python=3.8.2 \
    graph-tool=2.29 pyqt=5.9.2 numpy=1.18.1 scikit-image=0.16.2 pandas=1.0.3 seaborn=0.12.2 -c conda-forge/label/cf202003 && \
    /opt/conda/bin/conda install -y -n ChronoRootInterface -c conda-forge pyzbar pip && \
    /opt/conda/envs/ChronoRootInterface/bin/pip install opencv-python && \
    echo "conda activate ChronoRootInterface" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -a -y

# Create and activate Conda environment for GUI Interface 
RUN /opt/conda/bin/conda create -y -n FDA && \
    /opt/conda/bin/conda install -y -n FDA -c conda-forge scikit-fda scipy pandas matplotlib seaborn ipykernel && \
    /opt/conda/bin/conda clean -a -y

# Install necessary packages to run SORT
RUN /opt/conda/bin/conda run -n ChronoRootInterface pip install filterpy==1.4.5 lapx lifelines && \
    /opt/conda/bin/conda clean -a -y
WORKDIR /

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
    
# Clone the repository
RUN git clone https://github.com/ngaggion/ChronoRoot2.git /app

WORKDIR /app/Segmentation

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz && \
    tar -xvf git-lfs-linux-amd64-v3.5.1.tar.gz && \
    rm git-lfs-linux-amd64-v3.5.1.tar.gz && \
    cd git-lfs-3.5.1 && \
    ./install.sh && \
    cd .. && \
    rm -rf git-lfs-3.5.1 && \
    git lfs install && \
    git clone https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet

# Set working directory
WORKDIR /app

RUN git pull
