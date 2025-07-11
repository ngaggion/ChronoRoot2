FROM ngaggion/chronoroot:latest

RUN echo "alias segmentation='source activate ChronoRootInterface; cd /app/segmentationApp; python run.py'" >> ~/.bashrc && \
    echo "alias chronoroot='source activate ChronoRootInterface; cd /app/chronoRootApp; python run.py'" >> ~/.bashrc && \
    echo "alias screening='source activate ChronoRootInterface; cd /app/chronoRootScreeningApp; python run.py'" >> ~/.bashrc
    
# Set working directory
WORKDIR /app/Segmentation

RUN git pull

# Set working directory
WORKDIR /app/Segmentation

RUN git pull

# Add a timestamp for last update 
RUN date > /app/Segmentation/ChronoRoot_nnUNet/last_update.txt

WORKDIR /app/