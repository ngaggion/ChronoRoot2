# ChronoRoot Apptainer/Singularity Installation

Quick installation system for Linux using Apptainer or Singularity containers.

## Prerequisites

- **Apptainer** or **Singularity** installed
- Ubuntu 20.04+ or compatible Linux distribution
- ~20 GB free disk space
- X11 display server

Install Apptainer:
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

## Quick Start

### First Installation

```bash
git clone https://github.com/ngaggion/ChronoRoot2.git
cd ChronoRoot2
bash apptainerInstaller/install.sh
```

Follow the prompts to:
1. Choose installation directory (default: `~/.local/chronoroot`)
2. Build Singularity image from Docker Hub OR provide existing .sif file
3. Wait for installation to complete

### Update Application Code

When new code is released (bug fixes, new features):

**Option 1: From installation directory (recommended after install)**
```bash
bash ~/.local/chronoroot/update.sh
```

**Option 2: From repository**
```bash
cd ChronoRoot2
git pull
bash apptainerInstaller/update.sh
```

The update script automatically:
- Reads the original repository location from config
- Pulls latest changes from git (if repository still exists)
- Copies updated files to installation directory
- Does NOT re-download the 13.5 GB Singularity image

**Note:** Keep your original ChronoRoot2 repository - the update script needs it to get the latest code!

### Uninstall

```bash
bash ~/.local/chronoroot/uninstall.sh
```

Or from the repository:

```bash
bash apptainerInstaller/uninstall.sh
```

## What Gets Installed

- **Location**: `~/.local/chronoroot/` (or your chosen directory)
- **Singularity Image**: `Image_ChronoRoot.sif` (~13.5 GB)
- **Applications**: chronoRootApp, chronoRootScreeningApp, segmentationApp
- **Desktop Entries**: `~/.local/share/applications/ChronoRoot*.desktop`
- **Launchers**: Bash scripts to run each application

## Launching Applications

After installation, launch from:

1. **Application Menu**: Search for "ChronoRoot"
2. **Command Line**:
   ```bash
   ~/.local/chronoroot/ChronoRootApp.sh
   ~/.local/chronoroot/ChronoRootScreeningApp.sh
   ~/.local/chronoroot/ChronoRootSegmentationApp.sh
   ```

## Singularity Image Options

### Option 1: Build from Docker Hub (Recommended)

The installer builds the image automatically using:
```bash
apptainer build --remote Image_ChronoRoot.sif docker://ngaggion/chronorootbase:latest
```

- Always gets latest version
- Requires ~13.5 GB download
- Takes 10-30 minutes

### Option 2: Use Existing .sif File

If you already have the image (from a colleague or USB drive), provide the path when prompted.

## When to Use Each Script

### Use `install.sh` for:
- First-time installation
- Rebuilding the Singularity image
- Major version updates
- Changing installation directory

### Use `update.sh` for:
- Regular code updates
- Bug fixes
- New features
- Any time you don't need to rebuild the image

**Rule of thumb**: Use `update.sh` for 95% of updates!

## Troubleshooting

### GUI doesn't appear
```bash
xhost +local:
echo $DISPLAY  # Should show :0 or similar
```

### Permission denied
```bash
chmod +x apptainerInstaller/*.sh
```

### Image build fails
- Check internet connection
- Verify sufficient disk space
- Try building manually:
  ```bash
  apptainer build --remote chronoroot.sif docker://ngaggion/chronorootbase:latest
  ```

### Desktop entries don't appear
```bash
update-desktop-database ~/.local/share/applications
```

### What if I delete the ChronoRoot2 repository?

The installation itself is self-contained and will continue to work. However, you won't be able to update the application code using `update.sh` since it needs the repository to pull changes.

**Solutions:**
- Keep the repository (recommended)
- Clone the repository again when you need to update
- Or just re-run the full `install.sh` to get the latest version

## Technical Details

### What `install.sh` does:
1. Detects Apptainer/Singularity
2. Creates installation directory
3. Builds/copies Singularity image
4. Copies application files from repo
5. Copies logos to assets folder
6. Copies update.sh and uninstall.sh to installation directory
7. Creates launcher scripts with resolved paths
8. Creates desktop entries pointing to logos in assets
9. Saves configuration to `~/.config/chronoroot/config.json`

### What `update.sh` does:
1. Reads installation location from config
2. Pulls latest git changes (optional)
3. Updates application directories
4. Updates logos in assets folder
5. Verifies installation integrity
6. Updates config timestamp
7. **Preserves** Singularity image

### File Structure After Installation:
```
~/.local/chronoroot/
├── Image_ChronoRoot.sif
├── chronoRootApp/
├── chronoRootScreeningApp/
├── segmentationApp/
├── assets/
│   ├── logo.jpg
│   ├── logo_screening.jpg
│   └── logo_seg.jpg
├── ChronoRootApp.sh
├── ChronoRootScreeningApp.sh
├── ChronoRootSegmentationApp.sh
├── update.sh
└── uninstall.sh

~/.local/share/applications/
├── ChronoRoot.desktop
├── ChronoRootScreening.desktop
└── ChronoRootSegmentation.desktop

~/.config/chronoroot/
└── config.json
```

## Configuration File

Location: `~/.config/chronoroot/config.json`

Contains:
- Installation directory path
- Singularity image path
- Container command (apptainer/singularity)
- Installation date
- Last update date
- Repository root (for icons)

## Support

For issues, see the main repository:
- [GitHub Issues](https://github.com/ngaggion/ChronoRoot2/issues)
- [Main README](../README.md)