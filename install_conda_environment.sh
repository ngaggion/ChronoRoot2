#!/bin/bash
set -e

echo "Installing ChronoRoot 2.0..."

conda env create -f environment.yml

echo "Installation complete!"
echo "Activate with: conda activate ChronoRoot"