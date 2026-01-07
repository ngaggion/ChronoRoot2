# ChronoRoot Apptainer/Singularity Installation

A standalone installation system for ChronoRoot using Apptainer or Singularity containers.

> **System Compatibility:** This installation has been explicitly tested on **Ubuntu 24.04** and **Windows 10 (via WSL)**.

## Overview

This installer is designed to be **fully standalone**. It will automatically detect missing system dependencies (such as Apptainer/Singularity, Git, or Git LFS) and install them for you.

## Prerequisites

* **Internet Connection:** To download the repository and container images.
* **Sudo/Administrator Privileges:** Required if the installer needs to install missing dependencies.
* **Disk Space:** ~20-30 GB free space (depending on image choice).

### Windows Specifics

* **WSL 2 (Windows Subsystem for Linux) is MANDATORY.**
* You must have a Linux distribution (e.g., Ubuntu) installed and initialized.

---

## Installation

Please choose the installer corresponding to your operating system. Both installers will automatically download necessary files, install system requirements, build the environment, and **create icons in your system's Start Menu**.

### Option A: Linux Installation (`installer_linux.sh`)

Run the following in your terminal:

```bash
wget https://raw.githubusercontent.com/ngaggion/ChronoRoot2/master/apptainerInstaller/installer_linux.sh
bash installer_linux.sh

```

### Option B: Windows Installation (`installer_windows.sh`)

> **⚠️ IMPORTANT:** This script **must** be launched inside your **WSL Terminal** (e.g., Ubuntu on Windows), **NOT** in PowerShell or Command Prompt.

1. Open your WSL terminal (e.g., click "Ubuntu" in your Start Menu).
2. Run the installer:

```bash
wget https://raw.githubusercontent.com/ngaggion/ChronoRoot2/master/apptainerInstaller/installer_windows.sh
bash installer_windows.sh

```

### What the Installer Does

1. **Checks & Installs Dependencies:** Automatically installs Apptainer, Git, and Git LFS if they are missing (requires sudo).
2. **Clones Repository:** Downloads ChronoRoot2 to `~/.local/chronoroot/`.
3. **Builds Container:** Creates the Singularity image (~13.5 GB or ~23.5 GB).
4. **Integrates with OS:** Creates launcher scripts and adds "ChronoRoot" applications directly to your **Start Menu**.

---

## Launching Applications

After installation, you can launch the apps via the GUI on both platforms:

### 1. Via Start Menu (Recommended)

* **Linux:** Open your Application Menu and search for **"ChronoRoot"**.
* **Windows:** Open the Windows Start Menu and search for **"ChronoRoot"** (The shortcut launches the app via WSL transparently).

### 2. Via Command Line

If you prefer the terminal:

```bash
~/.local/chronoroot/ChronoRootApp.sh
~/.local/chronoroot/ChronoRootScreeningApp.sh
~/.local/chronoroot/ChronoRootSegmentationApp.sh

```

---

## Installation Structure

Location: `~/.local/chronoroot/`

```text
~/.local/chronoroot/
├── ChronoRoot2/                  # Source code (Git Repository)
├── Image_ChronoRoot.sif          # Singularity container image
├── ChronoRootApp.sh              # Launcher scripts
├── ChronoRootScreeningApp.sh
├── ChronoRootSegmentationApp.sh
└── uninstall.sh                  # Uninstallation script

```

## Singularity Image Options

During installation, you will be prompted to choose a version:

### Option 1: Build Standard Image (nodemo)

* Builds from `docker://ngaggion/chronorootbase:nodemo`.
* Recommended for standard production use.
* **Size:** ~13.5 GB.

### Option 2: Build Image with Demo Data (full)

* Builds from `docker://ngaggion/chronorootbase:full`.
* **Includes ~10 GB of demo imaging data** stored in `/Demo`.
* Recommended for tutorials and testing.
* **Size:** ~23.5 GB.

### Option 3: Use Existing .sif File

* If you already have the image locally (e.g., from a USB drive), provide the path to skip the download.

---

## Updating

To update the application code (bug fixes, new features):

```bash
cd ~/.local/chronoroot/ChronoRoot2
git pull

```

**Note:** This updates the Python code immediately. You generally do not need to rebuild the Singularity image unless specified in release notes.

## Uninstalling

```bash
bash ~/.local/chronoroot/uninstall.sh

```

This removes the repository, Singularity image, launchers, and cleans up Start Menu entries.

---

## Troubleshooting

### GUI doesn't appear

```bash
xhost +local:
echo $DISPLAY  # Should show :0 or similar

```

### Image build fails

* Check internet connection.
* Verify **~30 GB** free disk space.
* If the process is killed, check RAM usage (Tested on 16 GB RAM).

### Desktop entries/Icons don't appear

* **Linux:** Run `update-desktop-database ~/.local/share/applications`
* **Windows:** Check if shortcuts exist in: `C:\Users\<User>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\ChronoRoot`

## Support

* [GitHub Issues](https://github.com/ngaggion/ChronoRoot2/issues)