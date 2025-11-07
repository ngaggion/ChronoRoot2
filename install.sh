#!/bin/bash
set -e

echo "Installing ChronoRoot 2.0..."

# Detect OS and install system dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing system dependencies..."
    sudo apt-get update && sudo apt-get install -y libzbar0 libzbar-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected. Installing system dependencies..."
    brew install zbar
else
    echo "Unsupported OS. For Windows, please use WSL or Docker."
    exit 1
fi

conda env create -f environment.yml

echo "Installation complete!"
echo "Activate with: conda activate ChronoRoot"