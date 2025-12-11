#!/bin/bash

# ChronoRoot Installation Script

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

get_repo_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$(dirname "$script_dir")"
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

    # Get repository root
    REPO_ROOT=$(get_repo_root)
    print_info "Repository location: $REPO_ROOT"
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
        echo "  2) Update existing installation"
        echo "  3) Cancel"
        read -p "Choose [1-3]: " choice
        
        case $choice in
            1)
                print_info "Removing existing installation..."
                rm -rf "$INSTALL_DIR"
                ;;
            2)
                print_info "Updating existing installation..."
                UPDATE_MODE=true
                ;;
            *)
                print_info "Installation cancelled."
                exit 0
                ;;
        esac
    fi

    mkdir -p "$INSTALL_DIR"
    
    # Handle Singularity image
    IMAGE_PATH="$INSTALL_DIR/Image_ChronoRoot.sif"
    
    if [ -f "$IMAGE_PATH" ] && [ "$UPDATE_MODE" = true ]; then
        print_info "Using existing Singularity image: $IMAGE_PATH"
        print_warning "To rebuild the image, remove it first or choose 'Remove and reinstall'"
        echo ""
    else
        echo ""
        echo "Singularity Image Options:"
        echo "  1) Build from Docker Hub (ngaggion/chronorootbase:latest) - ~13.5 GB"
        echo "  2) Provide path to existing .sif file"
        read -p "Choose [1-2]: " image_choice
        
        case $image_choice in
            1)
                print_info "Building Singularity image from Docker Hub..."
                print_warning "This will download ~13.5 GB and may take some time..."
                echo ""
                
                if $CONTAINER_CMD build --remote "$IMAGE_PATH" "docker://$DOCKER_IMAGE"; then
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
    fi
    
    echo ""
    print_info "Copying application files..."
    
    cp -r "$REPO_ROOT/chronoRootApp" "$INSTALL_DIR/"
    cp -r "$REPO_ROOT/chronoRootScreeningApp" "$INSTALL_DIR/"
    cp -r "$REPO_ROOT/segmentationApp" "$INSTALL_DIR/"
    
    # Copy logos to assets
    mkdir -p "$INSTALL_DIR/assets"
    [ -f "$REPO_ROOT/logo.jpg" ] && cp "$REPO_ROOT/logo.jpg" "$INSTALL_DIR/assets/"
    [ -f "$REPO_ROOT/logo_screening.jpg" ] && cp "$REPO_ROOT/logo_screening.jpg" "$INSTALL_DIR/assets/"
    [ -f "$REPO_ROOT/logo_seg.jpg" ] && cp "$REPO_ROOT/logo_seg.jpg" "$INSTALL_DIR/assets/"
    
    # Copy installer scripts
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cp "$SCRIPT_DIR/update.sh" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/uninstall.sh" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/update.sh" "$INSTALL_DIR/uninstall.sh"
    
    print_success "Application files copied!"
    echo ""
    
    # Create launcher scripts
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
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ChronoRoot && cd $INSTALL_DIR/$app_dir/ && python run.py"
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
        local icon_path="$INSTALL_DIR/assets/$icon_file"
        
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
    
    echo ""
    print_info "Saving configuration..."
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    cat > "$CONFIG_FILE" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "image_path": "$IMAGE_PATH",
  "container_cmd": "$CONTAINER_CMD",
  "install_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "docker_image": "$DOCKER_IMAGE",
  "repo_root": "$REPO_ROOT"
}
EOF
    
    print_success "Configuration saved to: $CONFIG_FILE"
    
    echo ""
    echo "============================================"
    print_success "Installation Complete!"
    echo "============================================"
    echo ""
    echo "ChronoRoot has been installed to: $INSTALL_DIR"
    echo ""
    echo "You can now launch the applications:"
    echo "  - From your application menu (search for 'ChronoRoot')"
    echo "  - Or run directly:"
    echo "    $INSTALL_DIR/ChronoRootApp.sh"
    echo "    $INSTALL_DIR/ChronoRootScreeningApp.sh"
    echo "    $INSTALL_DIR/ChronoRootSegmentationApp.sh"
    echo ""
    echo "To update: bash $INSTALL_DIR/update.sh"
    echo "To uninstall: bash $INSTALL_DIR/uninstall.sh"
    echo ""
}

main "$@"