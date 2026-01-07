#!/bin/bash

# ============================================================================
#  ChronoRoot Master Installer (Linux Native)
# ============================================================================

set -e

# --- Colors ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- Configuration ---
REPO_URL="https://github.com/ngaggion/ChronoRoot2.git"
DEFAULT_INSTALL_DIR="$HOME/.local/chronoroot"
DESKTOP_ENTRY_DIR="$HOME/.local/share/applications"
CONFIG_FILE="$HOME/.config/chronoroot/config.json"

main() {
    clear
    echo "============================================"
    echo "   ChronoRoot Linux Installer"
    echo "============================================"
    echo ""

    # 1. Dependencies Check
    print_info "Checking system dependencies..."

    # Check for Apptainer/Singularity
    if command_exists apptainer; then
        CONTAINER_CMD="apptainer"
    elif command_exists singularity; then
        CONTAINER_CMD="singularity"
    else
        print_warning "Apptainer not found. Attempting to install..."
        if command_exists apt; then
            sudo apt update && sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:apptainer/ppa
            sudo apt update && sudo apt install -y apptainer
            CONTAINER_CMD="apptainer"
        else
            print_error "Could not auto-install Apptainer. Please install it manually."
            exit 1
        fi
    fi

    # Check for Git LFS
    if ! command_exists git-lfs; then
        print_warning "Git LFS missing. Installing..."
        sudo apt update && sudo apt install -y git-lfs
        git lfs install
    else
        git lfs install 2>/dev/null || true
    fi

    # GPU Detection
    GPU_FLAG=""
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected. enabling --nv flag."
        GPU_FLAG="--nv"
    fi

    # 2. Setup Directory
    echo ""
    read -p "Installation directory (Default: $DEFAULT_INSTALL_DIR): " user_install_dir
    INSTALL_DIR="${user_install_dir:-$DEFAULT_INSTALL_DIR}"
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    mkdir -p "$INSTALL_DIR"

    # 3. Clone / Update Repo
    REPO_DIR="$INSTALL_DIR/ChronoRoot2"
    if [ ! -d "$REPO_DIR" ]; then
        print_info "Cloning repository..."
        git clone "$REPO_URL" "$REPO_DIR"
    else
        print_info "Updating repository..."
        (cd "$REPO_DIR" && git pull)
    fi

    # 4. Image Build
    IMAGE_PATH="$INSTALL_DIR/Image_ChronoRoot.sif"
    
    if [ ! -f "$IMAGE_PATH" ]; then
        echo ""
        echo "Select Image Version:"
        echo "1) Standard (13.5 GB)"
        echo "2) Full with Demo Data (23.5 GB)"
        echo "3) I have my own .sif file"
        read -p "Choice [1-3]: " img_choice
        
        case $img_choice in
            1)
                print_info "Building Standard Image (nodemo)..."
                $CONTAINER_CMD build "$IMAGE_PATH" "docker://ngaggion/chronorootbase:nodemo"
                ;;
            2)
                print_info "Building Full Image (with demo data)..."
                $CONTAINER_CMD build "$IMAGE_PATH" "docker://ngaggion/chronorootbase:full"
                ;;
            3)
                read -p "Enter path to existing .sif file: " existing_sif
                existing_sif="${existing_sif/#\~/$HOME}"
                if [ -f "$existing_sif" ]; then
                    cp "$existing_sif" "$IMAGE_PATH"
                    print_success "Image copied."
                else
                    print_error "File not found."
                    exit 1
                fi
                ;;
            *)
                print_error "Invalid choice."
                exit 1
                ;;
        esac
    else
        print_success "Singularity image already exists. Skipping build."
    fi

    # 5. Create Launchers & Desktop Entries
    print_info "Creating Launchers and Menu Shortcuts..."
    mkdir -p "$DESKTOP_ENTRY_DIR"

    # Function to create both the shell script wrapper and the .desktop file
    create_app_shortcuts() {
        local pretty_name="$1"      # Name shown in Menu (e.g. "ChronoRoot Screening")
        local file_base_name="$2"   # Filename base (e.g. "ChronoRootScreening")
        local python_dir="$3"       # Folder inside repo (e.g. "chronoRootScreeningApp")
        local icon_file="$4"        # Icon filename (e.g. "logo_screening.jpg")

        local wrapper_script="$INSTALL_DIR/${file_base_name}.sh"
        local desktop_file="$DESKTOP_ENTRY_DIR/${file_base_name}.desktop"
        local icon_path="$REPO_DIR/$icon_file"

        # A. Create the Wrapper Script
        cat > "$wrapper_script" << EOF
#!/bin/bash
# Allow X server connection
xhost +local: > /dev/null 2>&1

echo "Starting $pretty_name..."
$CONTAINER_CMD exec $GPU_FLAG --bind /tmp/.X11-unix --env DISPLAY=\$DISPLAY "$IMAGE_PATH" \\
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ChronoRoot && cd $REPO_DIR/$python_dir && python run.py"

# Cleanup X permissions (optional, but good practice)
xhost -local: > /dev/null 2>&1
EOF
        chmod +x "$wrapper_script"

        # B. Create the .desktop file (Start Menu Entry)
        cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$pretty_name
Comment=Launch $pretty_name container
Exec=$wrapper_script
Icon=$icon_path
Terminal=true
Categories=Science;Education;
StartupNotify=true
EOF
        chmod +x "$desktop_file"
        print_success "Created shortcuts for $pretty_name"
    }

    # Generate the 3 apps
    # Args: "Display Name" "ScriptBaseName" "RepoSubDir" "IconFile"
    create_app_shortcuts "ChronoRoot App" "ChronoRootApp" "chronoRootApp" "logo.jpg"
    create_app_shortcuts "ChronoRoot Screening" "ChronoRootScreening" "chronoRootScreeningApp" "logo_screening.jpg"
    create_app_shortcuts "ChronoRoot Segmentation" "ChronoRootSegmentation" "segmentationApp" "logo_seg.jpg"

    # 6. Install Uninstall Script
    if [ -f "$REPO_DIR/apptainerInstaller/uninstall.sh" ]; then
        cp "$REPO_DIR/apptainerInstaller/uninstall.sh" "$INSTALL_DIR/uninstall.sh"
        chmod +x "$INSTALL_DIR/uninstall.sh"
    fi

    # 7. Save Config
    mkdir -p "$(dirname "$CONFIG_FILE")"
    cat > "$CONFIG_FILE" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "repo_dir": "$REPO_DIR",
  "image_path": "$IMAGE_PATH",
  "container_cmd": "$CONTAINER_CMD",
  "install_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    # 8. Refresh Desktop Database
    if command_exists update-desktop-database; then
        update-desktop-database "$DESKTOP_ENTRY_DIR" 2>/dev/null || true
    fi

    echo ""
    echo "============================================"
    print_success "Installation Complete!"
    echo "============================================"
    echo "You can launch the apps from your Application Menu (search 'ChronoRoot')."
    echo ""
}

main "$@"