# ChronoRoot 2.0 - Segmentation Module

This directory contains the AI-powered segmentation module for ChronoRoot 2.0, based on a modified version of the nnUNet architecture.

## Overview

The segmentation module identifies six distinct plant structures from infrared images:

1. Main root (primary root axis)
2. Lateral roots (secondary roots)
3. Seed (pre- and post-germination structures)
4. Hypocotyl (stem region between root-shoot junction and cotyledons)
5. Leaves (including both cotyledons and true leaves)
6. Petiole (leaf attachment structures)

## Directory Structure

```
segmentationApp/
├── ensemble_multiclass.py    # Implementation of nnUNet ensemble prediction with temporal consistency
├── name_handling.py          # Utilities for managing file paths and naming convention
├── test.sh                   # Script for running inference on new data
└── train.sh                  # Script for training new models (optional)
```

## Getting Started

### Prerequisites

Before running the segmentation pipeline, you need to download the pre-trained nnUNet models:

```bash
# Install git-lfs for handling large files
git lfs install

# Clone the dataset repository
git clone https://huggingface.co/datasets/ngaggion/ChronoRoot_nnUNet
```

This will provide the necessary nnUNet folder structure containing pre-trained models.

## Usage

### Running Inference

1. Edit the `test.sh` script to specify the path to your data:

```bash
#!/bin/bash
# Function to process input paths
process_input_path() {
  local input_path=$1
  # Kept fixed
  dataset="789"
  config="2d"
  save_prob="--save_probabilities"
  
  # Rename files to nnUNet format
  python name_handling.py "$input_path"
  
  # Folds to use, can be {0,1,2,3,4}. 1+ folds will be ensembled (by averaging predictions)
  for fold in {0..0}; do
    output_path="${input_path}/Segmentation/Fold_${fold}"
    mkdir -p "$output_path"
    nnUNetv2_predict_chrono -i "$input_path" -o "$output_path" -d "$dataset" -c "$config" -f "$fold" $save_prob
    python name_handling.py "$input_path" --revert_seg --segpath "$output_path"
  done
  
  # Revert file names and run temporal ensembling
  python name_handling.py "$input_path" --revert
  python ensemble_multiclass.py "$input_path" --alpha 0.9
}

# Add the paths to the nnUNet files
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Replace with your input data path
process_input_path /path/to/data
```

2. Run the segmentation:

```bash
chmod +x test.sh
./test.sh
```

### What the Segmentation Pipeline Does

The pipeline performs these steps:

1. **File Preparation**: Renames input files to match nnUNet conventions using `name_handling.py`
2. **Segmentation**: Runs the nnUNet prediction on each image
3. **Name Restoration**: Reverts files back to their original names
4. **Temporal Consistency**: Applies temporal consistency processing using `ensemble_multiclass.py`

### Output Format

The segmentation produces multi-class masks where each pixel value represents a specific plant structure:
- 0: Background
- 1: Main root
- 2: Lateral roots
- 3: Seed
- 4: Hypocotyl
- 5: Leaves
- 6: Petiole

Outputs include:
- Original grayscale segmentation masks
- Color-coded visualization of segmentation results

## Temporal Post-processing

The `ensemble_multiclass.py` script implements a weighted trailing average approach for temporal consistency. This enhances tracking robustness by incorporating historical structural information alongside new observations.

The key parameter is `alpha` (default 0.9), which controls the weight of previous frames in the accumulation. Higher values result in more temporal consistency but can reduce responsiveness to rapid changes.

## Training New Models (Advanced)

If you need to train custom models for different plant species, edit the `train.sh` script:

```bash
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Plan and preprocess the dataset
nnUNetv2_plan_and_preprocess -d 789 --verify_dataset_integrity

# Train each fold
nnUNetv2_train 789 2d 0 
nnUNetv2_train 789 2d 1 
nnUNetv2_train 789 2d 2 
nnUNetv2_train 789 2d 3 
nnUNetv2_train 789 2d 4 
```

This will:
1. Plan and preprocess your dataset (which should be prepared according to nnUNet guidelines)
2. Train 5 separate models (folds) for cross-validation
3. Save the trained models in the nnUNet results directory

Note that you can incorporate new images to the nnUNet_raw data files, modify the total number of files in the nnUNet dataset json and then run this script to retrain chronoRoot with extra new annotated images.

## Advanced Usage

### Modifying Inference Parameters

You can customize the segmentation process by editing parameters in `test.sh`:

- Change the folds used for prediction (0-4)
- Adjust the `alpha` parameter for temporal consistency
- Enable/disable probability map saving

### Customizing the Ensemble Process

The `ensemble_multiclass.py` script accepts these parameters:

```bash
python ensemble_multiclass.py [PATH] --alpha [WEIGHT] --num_classes [NUM_CLASSES]
```

- `PATH`: Directory containing input images and segmentation results
- `--alpha`: Weight factor for temporal averaging (default: 0.85)
- `--num_classes`: Number of segmentation classes (default: 7)