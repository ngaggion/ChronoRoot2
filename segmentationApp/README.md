# ChronoRoot 2.0 - Segmentation Module

This directory contains the AI-powered segmentation module for ChronoRoot 2.0, designed to automatically identify plant root structures from images.

-----

## Overview

The segmentation module uses a deep learning model to analyze infrared images and identify six distinct plant structures:

1.  **Main root** (primary root axis)
2.  **Lateral roots** (secondary roots)
3.  **Seed** (pre- and post-germination structures)
4.  **Hypocotyl** (stem region between root-shoot junction and cotyledons)
5.  **Leaves** (including both cotyledons and true leaves)
6.  **Petiole** (leaf attachment structures)

This application provides both a graphical user interface (GUI) and command-line interface (CLI) to manage and run segmentation and post-processing tasks on large datasets.

### Directory Structure

```
segmentationApp/
├── bash_usage.sh             # Example bash script for demo processing
├── cli.py                    # Command-line interface for quick processing
├── config.json               # Stores user settings (Conda env, alpha, etc.)
├── models/                   # Contains pre-trained nnUNet models
│   ├── Arabidopsis/
│   │   ├── dataset_fingerprint.json
│   │   ├── dataset.json
│   │   ├── fold_0/
│   │   │   └── checkpoint_final.pth
│   │   └── plans.json
│   └── Tomato/
│       ├── dataset_fingerprint.json
│       ├── dataset.json
│       ├── fold_0/
│       │   └── checkpoint_final.pth
│       └── plans.json
├── nnUNet_wrapper.py         # Internal script for nnUNet
├── postprocess.py            # Script for temporal post-processing
├── README.md                 # This file
├── run.py                    # The main GUI application
├── screenshots/
│   └── MainScreen.png        # Screenshot of the GUI interface
└── trainerOrganization/      # Tools for training custom models
    ├── CreateArabidopsisDataset.ipynb  # Dataset preparation for Arabidopsis
    ├── CreateTomatoDataset.ipynb        # Dataset preparation for Tomato
    ├── dataset_Arabidopsis.json        # Dataset configuration
    ├── dataset_Tomato.json              # Dataset configuration
    └── train.sh                         # Training script for nnUNet
```

-----

## How to Use (GUI Interface)

This module is designed to be run through the main graphical interface.

```bash
# Run the GUI interface using Docker
segmentation

# Or run directly with Python
python run.py
```

The GUI provides a complete workflow for processing your data:

  * **Multi-robot support**: Load and monitor multiple robot datasets.
  * **Queue management**: Add folders to a processing queue.
  * **Real-time progress**: Live progress bars for segmentation and post-processing.
  * **Parameter control**: Set the Conda environment, species, fast mode, and post-processing `alpha` value directly in the UI.
  * **Status monitoring**: Clear visual indicators for folder status (Not Started, Segmented, Complete, Error).

![Main Interface](screenshots/MainScreen.png)

### Workflow

1.  **Launch the App:** Run `python run.py`.
2.  **Set Parameters (Top Bar):**
      * **Conda:** Enter the name of your Conda environment.
      * **Alpha:** Set the alpha value for temporal post-processing (see below).
      * **Species:** Select "arabidopsis" or "tomato".
      * **Fast Mode:** Check this for faster processing (less augmentation).
3.  **Load Robot:** Click **"Load Robot"** and select the *root folder* containing your experiment data (e.g., `/path/to/Robot_1/`).
4.  **Process Data:**
      * The table will fill with all sub-folders.
      * Find folders with the status **"Not Started"** and click their **"Add to Queue"** button. This will schedule them for segmentation and post-processing.
      * If a folder is **"Segmented"** but you want to re-run post-processing (e.g., with a new alpha), click **"Postprocess"**.
      * If a folder is **"Complete"** but the stored alpha doesn't match the current setting, a **"Re-process"** button will appear.

-----

## Command-Line Interface (CLI)

For faster processing without the GUI overhead, use the CLI tool `cli.py`. This is ideal for batch processing or integration into automated pipelines.

### Basic Usage

```bash
# Basic segmentation (arabidopsis by default)
python cli.py /path/to/images

# Specify species (arabidopsis or tomato)
python cli.py /path/to/images --species tomato

# Fast mode (disable test-time augmentation for 2-3x speedup)
python cli.py /path/to/images --fast

# Segmentation + post-processing
python cli.py /path/to/images --postprocess

# Custom alpha value for post-processing
python cli.py /path/to/images --postprocess --alpha 0.9

# Post-processing only (if segmentation already done)
python cli.py /path/to/images --postprocess-only --alpha 0.85

# Verbose output
python cli.py /path/to/images --verbose
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Path to folder containing images | Required |
| `--species` | Model to use: `arabidopsis` or `tomato` | `arabidopsis` |
| `--device` | Computing device: `cuda`, `cpu`, or `mps` | `cuda` |
| `--fast` | Enable fast mode (no augmentation) | `False` |
| `--verbose` | Show detailed processing information | `False` |
| `--postprocess` | Run post-processing after segmentation | `False` |
| `--postprocess-only` | Skip segmentation, only run post-processing | `False` |
| `--alpha` | Temporal smoothing parameter (0.0-1.0) | `0.85` (arab.) / `0.99` (tomato) |
| `--num-classes` | Number of segmentation classes | `7` |

### Example Workflows

**Process a single experiment:**
```bash
python cli.py /data/Robot_1/Experiment_001 --species arabidopsis --postprocess
```

**Batch process multiple folders:**
```bash
for folder in /data/Robot_1/*; do
    python cli.py "$folder" --fast --postprocess --alpha 0.9
done
```

**Re-run post-processing with different alpha:**
```bash
python cli.py /data/Robot_1/Experiment_001 --postprocess-only --alpha 0.7
```

See `bash_usage.sh` for a complete demo workflow example.

-----

## Output Format

The segmentation produces multi-class masks where each pixel value represents a specific plant structure:

  * `0`: Background
  * `1`: Main root
  * `2`: Lateral roots
  * `3`: Seed
  * `4`: Hypocotyl
  * `5`: Leaves
  * `6`: Petiole

The final outputs are saved in the `Segmentation/Ensemble` folder within each data directory.

-----

## Temporal Post-processing (The "Alpha" Value)

Biological structures don't change drastically from one image to the next. The post-processing step uses this fact to recover from occasional mis-segmentations by applying temporal smoothing across the image sequence.

The **`alpha`** value controls this smoothing:

  * A **higher alpha** (e.g., 0.9) results in more temporal smoothing by memory accumulation, making the segmentation more stable over time. Makes sense for slow-growing roots (e.g., Arabidopsis).
  * A **lower alpha** (e.g., 0.5) allows for quicker adaptation to changes, useful for faster-growing plants.

Modifications on the post-processing script can allow the usage of different methods if needed. E.g. Tomato contains a different post-processing method that better suits its growth dynamics, as the plant moves faster and has more sudden changes in structure, where it does not stores the previous segmentations but the class presented to have class stability over time.

-----

## About the AI Model (nnUNet)

This tool uses **nnUNet** ("no-new-Net"), a powerful, self-configuring framework for biomedical image segmentation. As configured to use with Docker, the nnUNet environment should already be set as "base". If you are running the app in a different Conda environment, ensure that nnUNet is installed and properly configured and set up the environment name in the GUI.

  * **Official nnUNet Repository:** [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

-----

## Training Custom Models

### Dataset Access

The complete annotated ChronoRoot 2.0 dataset is publicly available on HuggingFace:

🤗 **Dataset:** [https://huggingface.co/datasets/ngaggion/ChronoRoot2](https://huggingface.co/datasets/ngaggion/ChronoRoot2)

The dataset is organized by robot and video folder, containing both raw images and their corresponding annotations.

### Dataset Preparation

The `trainerOrganization/` folder contains Jupyter notebooks that help convert the raw dataset into the nnUNet format:

1. **Download the dataset** from HuggingFace
2. **Run the appropriate notebook:**
   - `CreateArabidopsisDataset.ipynb` for Arabidopsis data
   - `CreateTomatoDataset.ipynb` for Tomato data

These notebooks will:
- Organize images into the nnUNet folder structure
- Generate proper train/test splits
- Ensure images from the same video stay in the same split (no data leakage)
- `CreateTomatoDataset.ipynb` will also include the creation of a complete training set using both the Tomato and Arabidopsis datasets for better generalization, but removing the "Petiole" class and updating both "Petiole" and "Leaves" to a single "Aerial" class.

### Training Process

Once your dataset is prepared, use the provided training script:

```bash
# Set nnUNet environment variables
export nnUNet_raw="/app/Segmentation/ChronoRoot_nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/app/Segmentation/ChronoRoot_nnUNet/nnUNet_preprocessed"
export nnUNet_results="/app/Segmentation/ChronoRoot_nnUNet/nnUNet_results"

# Plan and preprocess dataset (dataset ID 789 for ChronoRoot)
nnUNetv2_plan_and_preprocess -d 789 --verify_dataset_integrity

# Train on 5 folds (standard nnUNet cross-validation)
nnUNetv2_train 789 2d 0 
nnUNetv2_train 789 2d 1 
nnUNetv2_train 789 2d 2 
nnUNetv2_train 789 2d 3 
nnUNetv2_train 789 2d 4
```

After training, copy your model files to the `models/` directory following the structure described below.
-----

## Replacing or Adding Models

The `models/` directory contains the pre-trained nnUNet models. This folder structure is a direct copy of a standard nnUNet `nnUNet_results` directory.

  * The provided models (Arabidopsis and Tomato) are **nnUNet residual M models**.
  * The app uses the `plans.json` file to configure the AI and the `checkpoint_final.pth` file in the `fold_0/` directory as the trained model.

You can replace or add new models (e.g., for a different species) by:

1.  Training a new nnUNet model (see Training Custom Models section above).
2.  Creating a new folder inside `models/` (e.g., `models/Maize/`).
3.  Copying your `dataset.json`, `plans.json`, and the `fold_0/` directory (containing `checkpoint_final.pth`) from your nnUNet results into this new folder.

**Important:** The `plans.json` file is critical. If you ever change the model checkpoint, you **must** also use the matching `plans.json` file from that specific training run.