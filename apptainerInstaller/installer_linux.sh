#!/bin/bash

# ============================================================================
#  ChronoRoot Master Installer
# ============================================================================

set -e

# --- Visual Styling ---
BOLD='\033[1m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Helper Functions ---
section_title() {
    echo -e "\n${BOLD}--- $1 ---${NC}"
}

print_status() { echo -e "${BLUE}[STATUS]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# Function to check available disk space in GB
check_disk_space() {
    local required_gb=$1
    local target_dir=$2
    # Get available space in KB and convert to GB
    local available_kb=$(df -Pk "$target_dir" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space."
        echo "Required: ${required_gb} GB"
        echo "Available: ${available_gb} GB"
        exit 1
    else
        print_success "Disk space check passed (${available_gb} GB available)."
    fi
}

# --- Configuration ---
REPO_URL="https://github.com/ngaggion/ChronoRoot2.git"
DEFAULT_INSTALL_DIR="$HOME/.local/chronoroot"
DESKTOP_ENTRY_DIR="$HOME/.local/share/applications"
CONFIG_FILE="$HOME/.config/chronoroot/config.json"

main() {
    clear
    echo -e "${BOLD}ChronoRoot Installation Wizard${NC}"
    echo "This script will configure the environment and create desktop shortcuts."

    # 1. System Requirements Check
    section_title "1. Checking System Requirements"
    
    # Apptainer/Singularity Check
    if command_exists apptainer; then
        CONTAINER_CMD="apptainer"
        print_success "Container runtime: Apptainer detected."
    elif command_exists singularity; then
        CONTAINER_CMD="singularity"
        print_success "Container runtime: Singularity detected."
    else
        print_warning "Apptainer is required."
        read -p "Attempt to install Apptainer via APT? (y/n): " install_app
        if [[ $install_app =~ ^[Yy]$ ]]; then
            sudo apt update && sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:apptainer/ppa
            sudo apt update && sudo apt install -y apptainer
            CONTAINER_CMD="apptainer"
        else
            print_error "Installation cannot proceed without a container runtime."
            exit 1
        fi
    fi

    # Git LFS Check
    if ! command_exists git-lfs; then
        print_warning "Git LFS (Large File Support) is missing."
        sudo apt update && sudo apt install -y git-lfs
        git lfs install
    else
        git lfs install 2>/dev/null || true
        print_success "Git LFS is active."
    fi

    # GPU Detection
    GPU_FLAG=""
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected. Hardware acceleration enabled."
        GPU_FLAG="--nv"
    fi

    # 2. Path Configuration
    section_title "2. Installation Directory"
    echo -e "Choose the directory where ChronoRoot will be stored."
    echo -e "Default path: ${BLUE}$DEFAULT_INSTALL_DIR${NC}"
    echo -e "${BOLD}Action:${NC} Press ${BOLD}ENTER${NC} to use the default path, or type a new path below:"
    
    read -p "> " user_install_dir
    INSTALL_DIR="${user_install_dir:-$DEFAULT_INSTALL_DIR}"
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    
    mkdir -p "$INSTALL_DIR"
    print_success "Target directory: $INSTALL_DIR"

    # 3. Repository Setup
    section_title "3. Downloading Repository"
    REPO_DIR="$INSTALL_DIR/ChronoRoot2"
    if [ ! -d "$REPO_DIR" ]; then
        print_status "Cloning ChronoRoot source files..."
        git clone "$REPO_URL" "$REPO_DIR"
    else
        print_status "Repository exists. Updating files..."
        (cd "$REPO_DIR" && git pull)
    fi

    # 4. Container Image Configuration
    section_title "4. Container Image Setup"
    IMAGE_PATH="$INSTALL_DIR/Image_ChronoRoot.sif"
    
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Please select an image version to download:"
        echo "1) Standard (~13.5 GB)"
        echo "2) Full with Demo Data (~23.5 GB)"
        echo "3) Provide path to an existing .sif file"
        
        read -p "Selection [1-3]: " img_choice
        
        case $img_choice in
            1)
                check_disk_space 15 "$INSTALL_DIR"
                print_status "Downloading Standard Image..."
                $CONTAINER_CMD build "$IMAGE_PATH" "docker://ngaggion/chronorootbase:nodemo"
                ;;
            2)
                check_disk_space 25 "$INSTALL_DIR"
                print_status "Downloading Full Image..."
                $CONTAINER_CMD build "$IMAGE_PATH" "docker://ngaggion/chronorootbase:full"
                ;;
            3)
                check_disk_space 25 "$INSTALL_DIR"
                read -p "Type or drag-and-drop the .sif file path: " existing_sif
                existing_sif=$(echo "$existing_sif" | tr -d "'\"" | xargs)
                existing_sif="${existing_sif/#\~/$HOME}"
                if [ -f "$existing_sif" ]; then
                    cp "$existing_sif" "$IMAGE_PATH"
                    print_success "Image imported."
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
        print_success "Container image is already present."
    fi

    # 5. Shortcuts and Integration
    section_title "5. Creating System Shortcuts"
    mkdir -p "$DESKTOP_ENTRY_DIR"

    create_app_shortcuts() {
        local pretty_name="$1"
        local file_base_name="$2"
        local python_dir="$3"
        local icon_file="$4"

        local wrapper_script="$INSTALL_DIR/${file_base_name}.sh"
        local desktop_file="$DESKTOP_ENTRY_DIR/${file_base_name}.desktop"

        # Create Wrapper
        cat > "$wrapper_script" << EOF
#!/bin/bash
xhost +local: > /dev/null 2>&1
$CONTAINER_CMD exec $GPU_FLAG --bind /tmp/.X11-unix --env DISPLAY=\$DISPLAY --bind /run/user/$(id -u):/run/user/$(id -u) "$IMAGE_PATH" \\
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ChronoRoot && cd $REPO_DIR/$python_dir && python run.py"
xhost -local: > /dev/null 2>&1
EOF
        chmod +x "$wrapper_script"

        # Create Desktop Entry
        cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$pretty_name
Exec=$wrapper_script
Icon=$REPO_DIR/$icon_file
Terminal=true
Categories=Science;Education;
EOF
        chmod +x "$desktop_file"
        print_success "Created: $pretty_name"
    }

    create_app_shortcuts "ChronoRoot App" "ChronoRootApp" "chronoRootApp" "logo.ico"
    create_app_shortcuts "ChronoRoot Screening" "ChronoRootScreening" "chronoRootScreeningApp" "logo_screening.ico"
    create_app_shortcuts "ChronoRoot Segmentation" "ChronoRootSegmentation" "segmentationApp" "logo_seg.ico"

    # Uninstall Script
    if [ -f "$REPO_DIR/apptainerInstaller/uninstall.sh" ]; then
        cp "$REPO_DIR/apptainerInstaller/uninstall.sh" "$INSTALL_DIR/uninstall.sh"
        chmod +x "$INSTALL_DIR/uninstall.sh"
    fi

    # Save Config
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

    if command_exists update-desktop-database; then
        update-desktop-database "$DESKTOP_ENTRY_DIR" 2>/dev/null || true
    fi

    section_title "Installation Complete"
    print_success "ChronoRoot is successfully installed."
    echo "You can now find the applications in your system menu by searching 'ChronoRoot'."
    echo ""
}

main "$@"