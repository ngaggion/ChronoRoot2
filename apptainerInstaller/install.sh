#!/bin/bash

# ChronoRoot Installation Script
# Can be downloaded and run standalone via:
# wget https://raw.githubusercontent.com/ngaggion/ChronoRoot2/main/apptainerInstaller/install.sh
# bash install.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
REPO_URL="https://github.com/ngaggion/ChronoRoot2.git"
DOCKER_IMAGE="ngaggion/chronorootbase:latest"
DEFAULT_INSTALL_DIR="$HOME/.local/chronoroot"
DESKTOP_DIR="$HOME/.local/share/applications"
CONFIG_FILE="$HOME/.config/chronoroot/config.json"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_container_runtime() {
    if command_exists apptainer; then
        echo "apptainer"
    elif command_exists singularity; then
        echo "singularity"
    else
        echo ""
    fi
}

main() {
    clear
    echo "============================================"
    echo "  ChronoRoot Installation Script"
    echo "============================================"
    echo ""

    # Check for container runtime
    print_info "Checking for Apptainer/Singularity..."
    CONTAINER_CMD=$(detect_container_runtime)
    
    if [ -z "$CONTAINER_CMD" ]; then
        print_error "Neither Apptainer nor Singularity found!"
        print_error "Please install Apptainer: https://apptainer.org/docs/admin/main/installation.html"
        exit 1
    fi
    print_success "Found: $CONTAINER_CMD"
    echo ""

    # Check for git
    if ! command_exists git; then
        print_error "Git not found!"
        print_error "Please install git: sudo apt install git"
        exit 1
    fi

    # Check for git-lfs
    print_info "Checking for Git LFS..."
    if ! command_exists git-lfs; then
        print_warning "Git LFS not found!"
        echo "Git LFS is required to download large files from the repository."
        read -p "Install Git LFS now? [Y/n]: " install_lfs
        
        if [[ ! "$install_lfs" =~ ^[Nn]$ ]]; then
            print_info "Installing Git LFS..."
            if sudo apt install -y git-lfs; then
                git lfs install
                print_success "Git LFS installed!"
            else
                print_error "Failed to install Git LFS. Please install manually:"
                print_error "  sudo apt install git-lfs"
                print_error "  git lfs install"
                exit 1
            fi
        else
            print_error "Git LFS is required. Installation cancelled."
            exit 1
        fi
    else
        print_success "Git LFS found"
        # Make sure it's initialized for this user
        git lfs install 2>/dev/null || true
    fi
    echo ""

    # Ask for installation directory
    echo "Installation directory (press Enter for default: $DEFAULT_INSTALL_DIR):"
    read -r user_install_dir
    INSTALL_DIR="${user_install_dir:-$DEFAULT_INSTALL_DIR}"
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    
    print_info "Will install to: $INSTALL_DIR"
    echo ""

    # Check if installation already exists
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory already exists!"
        echo "Options:"
        echo "  1) Remove and reinstall"
        echo "  2) Cancel"
        read -p "Choose [1-2]: " choice
        
        case $choice in
            1)
                print_info "Removing existing installation..."
                rm -rf "$INSTALL_DIR"
                ;;
            *)
                print_info "Installation cancelled."
                exit 0
                ;;
        esac
    fi

    mkdir -p "$INSTALL_DIR"
    
    # Clone repository
    print_info "Cloning ChronoRoot repository..."
    REPO_DIR="$INSTALL_DIR/ChronoRoot2"
    
    if git clone "$REPO_URL" "$REPO_DIR"; then
        print_success "Repository cloned successfully!"
    else
        print_error "Failed to clone repository!"
        exit 1
    fi
    echo ""
    
    # Handle Singularity image
    IMAGE_PATH="$INSTALL_DIR/Image_ChronoRoot.sif"
    
    echo "Singularity Image Options:"
    echo "  1) Build from Docker Hub (ngaggion/chronorootbase:latest) - ~13.5 GB"
    echo "  2) Provide path to existing .sif file"
    read -p "Choose [1-2]: " image_choice
    
    case $image_choice in
        1)
            print_info "Building Singularity image from Docker Hub..."
            print_warning "This will download ~13.5 GB and may take some time..."
            echo ""
            
            if $CONTAINER_CMD build "$IMAGE_PATH" "docker://$DOCKER_IMAGE"; then
                print_success "Image built successfully!"
            else
                print_error "Failed to build image!"
                exit 1
            fi
            ;;
        2)
            read -p "Enter path to existing .sif file: " existing_image
            existing_image="${existing_image/#\~/$HOME}"
            
            if [ ! -f "$existing_image" ]; then
                print_error "File not found: $existing_image"
                exit 1
            fi
            
            print_info "Copying image to installation directory..."
            cp "$existing_image" "$IMAGE_PATH"
            print_success "Image copied successfully!"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
    
    echo ""
    print_info "Creating launcher scripts..."
    
    create_launcher() {
        local app_name=$1
        local app_dir=$2
        local launcher_path="$INSTALL_DIR/${app_name}.sh"
        
        cat > "$launcher_path" << EOF
#!/bin/bash
xhost +local: > /dev/null 2>&1
ImagePath="$IMAGE_PATH"
$CONTAINER_CMD exec --bind /tmp/.X11-unix --env DISPLAY=\$DISPLAY \$ImagePath \\
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ChronoRoot && cd $REPO_DIR/$app_dir/ && python run.py"
xhost -local: > /dev/null 2>&1
EOF
        
        chmod +x "$launcher_path"
        print_success "Created: $launcher_path"
    }
    
    create_launcher "ChronoRootApp" "chronoRootApp"
    create_launcher "ChronoRootScreeningApp" "chronoRootScreeningApp"
    create_launcher "ChronoRootSegmentationApp" "segmentationApp"
    
    echo ""
    print_info "Creating desktop entries..."
    mkdir -p "$DESKTOP_DIR"
    
    create_desktop_entry() {
        local name=$1
        local exec_name=$2
        local icon_file=$3
        local desktop_file="$DESKTOP_DIR/ChronoRoot${name}.desktop"
        local icon_path="$REPO_DIR/$icon_file"
        
        cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ChronoRoot $name
Exec=bash $INSTALL_DIR/${exec_name}.sh
Icon=$icon_path
Terminal=true
Categories=Science;Education;
EOF
        
        chmod +x "$desktop_file"
        print_success "Created: $desktop_file"
    }
    
    create_desktop_entry "" "ChronoRootApp" "logo.jpg"
    create_desktop_entry "Screening" "ChronoRootScreeningApp" "logo_screening.jpg"
    create_desktop_entry "Segmentation" "ChronoRootSegmentationApp" "logo_seg.jpg"
    
    # Copy uninstall script
    echo ""
    print_info "Installing uninstall script..."
    if [ -f "$REPO_DIR/apptainerInstaller/uninstall.sh" ]; then
        cp "$REPO_DIR/apptainerInstaller/uninstall.sh" "$INSTALL_DIR/uninstall.sh"
        chmod +x "$INSTALL_DIR/uninstall.sh"
        print_success "Uninstall script installed"
    fi
    
    echo ""
    print_info "Saving configuration..."
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    cat > "$CONFIG_FILE" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "repo_dir": "$REPO_DIR",
  "image_path": "$IMAGE_PATH",
  "container_cmd": "$CONTAINER_CMD",
  "install_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "docker_image": "$DOCKER_IMAGE"
}
EOF
    
    print_success "Configuration saved to: $CONFIG_FILE"
    
    echo ""
    echo "============================================"
    print_success "Installation Complete!"
    echo "============================================"
    echo ""
    echo "ChronoRoot has been installed to: $INSTALL_DIR"
    echo "Repository location: $REPO_DIR"
    echo ""
    echo "Launch applications:"
    echo "  - From your application menu (search for 'ChronoRoot')"
    echo "  - Or run:"
    echo "    $INSTALL_DIR/ChronoRootApp.sh"
    echo "    $INSTALL_DIR/ChronoRootScreeningApp.sh"
    echo "    $INSTALL_DIR/ChronoRootSegmentationApp.sh"
    echo ""
    echo "To update:"
    echo "  cd $REPO_DIR"
    echo "  git pull"
    echo ""
    echo "To uninstall:"
    echo "  bash $INSTALL_DIR/uninstall.sh"
    echo ""
}

main "$@"
