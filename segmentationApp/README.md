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

This application provides a graphical user interface (GUI) to manage and run segmentation and post-processing tasks on large datasets.

### Directory Structure

```
segmentationApp/
├── config.json               # Stores user settings (Conda env, alpha, etc.)
├── models/                   # Contains pre-trained nnUNet models
│   ├── Arabidopsis/
│   │   ├── dataset_fingerprint.json
│   │   ├── dataset.json
│   │   ├── fold_0/
│   │   │   └── checkpoint_final.pth
│   │   └── plans.json
│   └── Tomato/
│   │   ├── dataset_fingerprint.json
│   │   ├── dataset.json
│   │   ├── fold_0/
│   │   │   └── checkpoint_final.pth
│   │   └── plans.json
├── nnUNet_wrapper.py         # Internal script for nnUNet
├── postprocess.py            # Script for temporal post-processing
├── README.md                 # This file
├── run.py                    # The main GUI application
└── screenshots/
    └── MainScreen.png        # Screenshot of the GUI interface
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

Biological structures don't change drastically from one image to the next. The post-processing step uses this fact to "smooth" the segmentation results over time, making them more consistent.

The **`alpha`** value controls this smoothing:

  * A **higher alpha** (e.g., 0.9) results in more temporal smoothing. This is good for stable structures but less responsive to rapid changes.
  * A **lower alpha** (e.g., 0.5) is more responsive to the current frame but may appear "jerkier" over time.

-----

## About the AI Model (nnUNet)

This tool uses **nnUNet** ("no-new-Net"), a powerful, self-configuring framework for biomedical image segmentation. It is widely recognized for achieving state-of-the-art results.

  * **Official nnUNet Repository:** [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

### Replacing or Adding Models

The `models/` directory contains the pre-trained nnUNet models. This folder structure is a direct copy of a standard nnUNet `nnUNet_results` directory.

  * **Origin:** The provided models (Arabidopsis and Tomato) are **nnUNet residual M models**.
  * **Structure:** The app uses the `plans.json` file to configure the AI and the `checkpoint_final.pth` file in the `fold_0/` directory as the trained model.

You can replace or add new models (e.g., for a different species) by:

1.  Training a new nnUNet model.
2.  Creating a new folder inside `models/` (e.g., `models/Maize/`).
3.  Copying your `dataset.json`, `plans.json`, and the `fold_0/` directory (containing `checkpoint_final.pth`) from your nnUNet results into this new folder.

**Important:** The `plans.json` file is critical. If you ever change the model checkpoint, you **must** also use the matching `plans.json` file from that specific training run.