FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /

RUN /opt/conda/bin/conda init bash

# Install necessary packages
RUN apt-get update && \
    apt-get install -y wget git python3-pyqt5 nano gedit libzbar0 libzbar-dev  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz && \
    tar -xvf git-lfs-linux-amd64-v3.5.1.tar.gz && \
    rm git-lfs-linux-amd64-v3.5.1.tar.gz && \
    cd git-lfs-3.5.1 && \
    ./install.sh && \
    cd .. && \
    rm -rf git-lfs-3.5.1 && \
    git lfs install

# Create and activate Conda environment for GUI Interface 
RUN /opt/conda/bin/conda create -y -n ChronoRoot python=3.13.9 \
    networkx=3.5 \
    pyqt=5.15.11 \
    numpy=2.2 \
    scikit-image=0.25.2 \
    pandas=2.3.3 \
    seaborn=0.13.2 \
    filterpy=1.4.5 \
    lifelines=0.30.0 \
    scikit-fda=0.10.1 \
    scipy=1.16.3 \
    pyzbar=0.1.9  \
    pip \
    -c conda-forge && \
    /opt/conda/envs/ChronoRoot/bin/pip install opencv-python lapx==0.9.2 nnunetv2==2.6.2 && \
    echo "conda activate ChronoRoot" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -a -y

RUN echo "alias segmentation='source activate ChronoRoot; cd /app/segmentationApp; python run.py'" >> ~/.bashrc && \
    echo "alias chronoroot='source activate ChronoRoot; cd /app/chronoRootApp; python run.py'" >> ~/.bashrc && \
    echo "alias screening='source activate ChronoRoot; cd /app/chronoRootScreeningApp; python run.py'" >> ~/.bashrc