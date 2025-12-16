# ChronoRoot Apptainer/Singularity Installation

Simple installation system for Linux using Apptainer or Singularity containers.

## Prerequisites

- **Apptainer** or **Singularity** installed
- **Git** installed
- **Git LFS** installed (for large files in the repository)
- Ubuntu 20.04+ or compatible Linux distribution
- ~20 GB free disk space

Install prerequisites:
```bash
sudo apt update
sudo apt install -y git git-lfs software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
git lfs install
```

**Note:** Git LFS is required because the repository contains large model files. The installer will offer to install it automatically if not found.

## Quick Installation

Download and run the installer:

```bash
wget https://raw.githubusercontent.com/ngaggion/ChronoRoot2/master/apptainerInstaller/install.sh
bash install.sh
```

The installer will:
1. Clone the ChronoRoot2 repository to `~/.local/chronoroot/ChronoRoot2/`
2. Build the Singularity image (~13.5 GB download)
3. Create launcher scripts
4. Add applications to your system menu

## Installation Structure

After installation:

```
~/.local/chronoroot/
├── ChronoRoot2/                  # Full git repository (this is the source code)
│   ├── chronoRootApp/
│   ├── chronoRootScreeningApp/
│   ├── segmentationApp/
│   ├── logo.jpg
│   └── ...
├── Image_ChronoRoot.sif          # Singularity image (~13.5 GB)
├── ChronoRootApp.sh              # Launcher scripts
├── ChronoRootScreeningApp.sh
├── ChronoRootSegmentationApp.sh
└── uninstall.sh                  # Removal script
```

## Launching Applications

After installation, launch from:

1. **Application Menu**: Search for "ChronoRoot"
2. **Command Line**:
   ```bash
   ~/.local/chronoroot/ChronoRootApp.sh
   ~/.local/chronoroot/ChronoRootScreeningApp.sh
   ~/.local/chronoroot/ChronoRootSegmentationApp.sh
   ```

## Updating

To update the application code (bug fixes, new features):

```bash
cd ~/.local/chronoroot/ChronoRoot2
git pull
```

That's it! The launchers already point to the repository, so changes take effect immediately.

**Note:** This only updates the Python code. To rebuild the Singularity image (rare), you would need to reinstall.

## Uninstalling

```bash
bash ~/.local/chronoroot/uninstall.sh
```

This removes everything: the repository, Singularity image, launchers, and desktop entries.

## Singularity Image Options

During installation, you can choose:

### Option 1: Build from Docker Hub (Recommended)

Automatically builds from `docker://ngaggion/chronorootbase:latest`
- Always gets the latest version
- Requires ~13.5 GB download
- Takes 10-30 minutes

### Option 2: Use Existing .sif File

If you already have the Singularity image (from a colleague or USB drive), provide the path when prompted.

## Troubleshooting

### GUI doesn't appear
```bash
xhost +local:
echo $DISPLAY  # Should show :0 or similar
```

### Image build fails
- Check internet connection
- Verify ~15 GB free disk space
- Try building manually:
  ```bash
  apptainer build chronoroot.sif docker://ngaggion/chronorootbase:latest
  ```
- If the process is killed, check ram usage or try on a different machine.
- This was tested on a 16 GB RAM machine as the minimum specs.

### Desktop entries don't appear
```bash
update-desktop-database ~/.local/share/applications
```

### Git pull fails
```bash
cd ~/.local/chronoroot/ChronoRoot2
git status  # Check for local changes
git stash   # Stash changes if needed
git pull
```

### Custom Installation Directory

When running `install.sh`, you can specify a different directory when prompted (default is `~/.local/chronoroot`).

## Support

For issues and questions:
- [GitHub Issues](https://github.com/ngaggion/ChronoRoot2/issues)
- [Main Documentation](https://github.com/ngaggion/ChronoRoot2)

## Technical Details

### What the Installer Does

1. Detects Apptainer/Singularity
2. Clones `https://github.com/ngaggion/ChronoRoot2.git`
3. Builds Singularity image from Docker Hub
4. Creates launcher scripts pointing to cloned repo
5. Creates desktop entries with icons from repo
6. Copies uninstall script to installation directory
7. Saves configuration to `~/.config/chronoroot/config.json`

### Configuration File

Location: `~/.config/chronoroot/config.json`

Example:
```json
{
  "install_dir": "/home/user/.local/chronoroot",
  "repo_dir": "/home/user/.local/chronoroot/ChronoRoot2",
  "image_path": "/home/user/.local/chronoroot/Image_ChronoRoot.sif",
  "container_cmd": "apptainer",
  "install_date": "2024-12-15T14:30:00Z",
  "docker_image": "ngaggion/chronorootbase:latest"
}
```

### Launcher Scripts

Each launcher script runs:
```bash
apptainer exec Image_ChronoRoot.sif \
  bash -c "conda activate ChronoRoot && cd /path/to/ChronoRoot2/app/ && python run.py"
```

This means any changes in the repository take effect immediately - no reinstallation needed!
