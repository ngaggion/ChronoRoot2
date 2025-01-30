# ChronoRoot 2.0

ChronoRoot 2.0 is a comprehensive tool for plant root phenotyping, comprising two distinct stages: segmentation using the state-of-the-art method [nnUNet](https://github.com/MIC-DKFZ/nnUNet), and a set of two user interfaces for conducting the phenotyping procedure and generating reports. It offers flexibility for local installation or can be utilized through a Docker image.

## Local Installation of ChronoRootInterface

To install the ChronoRoot interface locally using Anaconda and pip packages, follow these steps:

```bash
conda create -y -n ChronoRootInterface python=3.8
conda activate ChronoRootInterface
conda install -c "conda-forge/label/cf202003" graph-tool=2.29 pyqt=5.9.2
conda install numpy scikit-image pandas seaborn
conda install -c conda-forge pyzbar
pip install opencv-python filterpy lifelines
```

To perform Functional PCA on the temporal series, a different environment is needed, which can be created as follows:

```bash
conda create -n FDA
conda activate FDA
conda install -c conda-forge scikit-fda scipy pandas matplotlib seaborn ipykernel
```

This environment won't be directly used by the user, as it will be automatically called inside a subprocess when generating the report if necessary.

### Usage of both interfaces

Activate the ChronoRootInterface environment:

```bash
conda activate ChronoRootInterface
```

Then, run the interface:

```bash
python run.py
```

## Local Installation of nnUNet

For local installation of nnUNet, minor modifications were made to allow soft dense segmentation outputs for temporal post-processing. Follow the instructions in the repository [github.com/ngaggion/nnUNet](https://github.com/ngaggion/nnUNet) to set it up.

### Usage

Before running the segmentation pipeline, download nnUNet models and data [available here](https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet) inside the "segmentationApp" folder. You'll need [git-lfs](https://git-lfs.com/) to clone the dataset from HuggingFace. Install git-lfs, then clone the dataset:

```bash
cd Segmentation
git lfs install
git clone https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet
```

Complete .test file with paths to your videos individually (per camera), then run it:

```bash
cd segmentationApp
conda activate nnUNet
./test.sh
```

## Combined ChronoRoot Docker Image

This Docker image provides an environment for running ChronoRoot completely, both for segmentation and with the user interface, along with GPU support and X11 forwarding. Please note that nnUNet is installed on the "base" environment, meanwhile the app is on "ChronoRootInterface" environment.

### Usage

Download the Docker image from the Docker Hub repository by running:

```bash
docker pull ngaggion/chronoroot:latest
```

If you are planning to use the user interface, execute the following commands to allow Docker to access the local X server:

```bash
xhost +local:docker
```

Then, run the Docker container with the following command:

```bash
MOUNT="YOUR_LOCAL_INFORMATION_PATH"

docker run -it --gpus all \
    -v $MOUNT:/DATA/ \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    chronoroot:latest
```

It's recommended to always pull from the repo when starting the docker.

```bash
git pull
```

After using the container, it's recommended to restrict access to the X server with the following command:

```bash
xhost -local:docker
```

### Docker Usage Notes

To enable GPU support within the Docker container, it's required to install the nvidia-docker2 package. **Please note that we are using CUDA 11.8, given your GPU restrictions you may want to build your own image.** In this case, you'll **only** need to modify the first line of the Dockerfile using any official pytorch >= 2.0.0 docker image that works with your hardware and build it from scratch.

For Ubuntu-based distributions please follow these steps:

1. **Add the GPG key:**

    ```bash
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    ```

2. **Add the repository:**

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/nvidia-docker/$distribution/$(arch)/" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```

3. **Update your package list:**

    ```bash
    sudo apt-get update
    ```

4. **Install NVIDIA Docker2:**

    ```bash
    sudo apt-get install -y nvidia-docker2
    ```

### Singularity

As sudo access may not be available for usage after installing ChronoRoot, Singularity may be used, as it provides a easy way to use Docker images:

```bash
    singularity build ChronoRoot.simg docker://ngaggion/chronoroot
```

For more information, refer to [Singularity's documentation](https://docs.sylabs.io/guides/2.6/user-guide/build_a_container.html).