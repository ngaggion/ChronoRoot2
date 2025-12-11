#!/bin/bash

# ChronoRoot Update Script

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

CONFIG_FILE="$HOME/.config/chronoroot/config.json"

main() {
    clear
    echo "============================================"
    echo "  ChronoRoot Update Script"
    echo "============================================"
    echo ""

    # Check if ChronoRoot is installed
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "ChronoRoot installation not found!"
        print_error "Please run install.sh first to install ChronoRoot."
        exit 1
    fi

    # Read installation directory from config
    INSTALL_DIR=$(grep -o '"install_dir": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    REPO_ROOT=$(grep -o '"repo_root": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    
    if [ -z "$INSTALL_DIR" ]; then
        print_error "Could not read installation directory from config!"
        exit 1
    fi

    if [ ! -d "$INSTALL_DIR" ]; then
        print_error "Installation directory not found: $INSTALL_DIR"
        exit 1
    fi

    print_info "Found installation at: $INSTALL_DIR"
    echo ""

    # Get repository root from config
    if [ -z "$REPO_ROOT" ]; then
        print_error "Could not read repository location from config!"
        print_error "Please run update.sh from the original repository:"
        print_error "  cd /path/to/ChronoRoot2"
        print_error "  bash apptainerInstaller/update.sh"
        exit 1
    fi

    if [ ! -d "$REPO_ROOT" ]; then
        print_error "Repository directory not found: $REPO_ROOT"
        print_error "The original ChronoRoot2 repository may have been moved or deleted."
        exit 1
    fi
    
    print_info "Repository location: $REPO_ROOT"
    echo ""

    # Check if we're in a git repository and pull
    if [ -d "$REPO_ROOT/.git" ]; then
        print_info "Pulling latest changes from Git..."
        cd "$REPO_ROOT"
        
        if git diff-index --quiet HEAD --; then
            git pull
        else
            print_warning "Local changes detected. Stashing them..."
            git stash
            git pull
            print_info "Local changes stashed. Use 'git stash pop' to restore them."
        fi
        
        print_success "Repository updated!"
        echo ""
    else
        print_warning "Not a git repository. Using local files at $REPO_ROOT"
        echo ""
    fi

    # Backup option
    read -p "Create backup before updating? [y/N]: " backup_choice
    if [[ "$backup_choice" =~ ^[Yy]$ ]]; then
        BACKUP_DIR="$INSTALL_DIR.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "Creating backup at: $BACKUP_DIR"
        cp -r "$INSTALL_DIR" "$BACKUP_DIR"
        print_success "Backup created!"
        echo ""
    fi

    # Update application files
    print_info "Updating application files..."
    
    if [ -d "$REPO_ROOT/chronoRootApp" ]; then
        print_info "  Updating chronoRootApp..."
        rm -rf "$INSTALL_DIR/chronoRootApp"
        cp -r "$REPO_ROOT/chronoRootApp" "$INSTALL_DIR/"
        print_success "  chronoRootApp updated!"
    fi
    
    if [ -d "$REPO_ROOT/chronoRootScreeningApp" ]; then
        print_info "  Updating chronoRootScreeningApp..."
        rm -rf "$INSTALL_DIR/chronoRootScreeningApp"
        cp -r "$REPO_ROOT/chronoRootScreeningApp" "$INSTALL_DIR/"
        print_success "  chronoRootScreeningApp updated!"
    fi
    
    if [ -d "$REPO_ROOT/segmentationApp" ]; then
        print_info "  Updating segmentationApp..."
        rm -rf "$INSTALL_DIR/segmentationApp"
        cp -r "$REPO_ROOT/segmentationApp" "$INSTALL_DIR/"
        print_success "  segmentationApp updated!"
    fi

    # Update logos
    if [ -f "$REPO_ROOT/logo.jpg" ] || [ -f "$REPO_ROOT/logo_screening.jpg" ] || [ -f "$REPO_ROOT/logo_seg.jpg" ]; then
        print_info "  Updating logos..."
        mkdir -p "$INSTALL_DIR/assets"
        [ -f "$REPO_ROOT/logo.jpg" ] && cp "$REPO_ROOT/logo.jpg" "$INSTALL_DIR/assets/"
        [ -f "$REPO_ROOT/logo_screening.jpg" ] && cp "$REPO_ROOT/logo_screening.jpg" "$INSTALL_DIR/assets/"
        [ -f "$REPO_ROOT/logo_seg.jpg" ] && cp "$REPO_ROOT/logo_seg.jpg" "$INSTALL_DIR/assets/"
        print_success "  Logos updated!"
    fi

    echo ""
    print_info "Verifying installation..."
    
    if [ -f "$INSTALL_DIR/Image_ChronoRoot.sif" ] && \
       [ -f "$INSTALL_DIR/ChronoRootApp.sh" ] && \
       [ -d "$INSTALL_DIR/chronoRootApp" ]; then
        print_success "Installation verification passed!"
    else
        print_error "Installation verification failed!"
        print_error "Some critical files are missing. You may need to run install.sh"
        exit 1
    fi

    # Update config with new update date
    UPDATE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    # Read existing config
    CONTAINER_CMD=$(grep -o '"container_cmd": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    IMAGE_PATH=$(grep -o '"image_path": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    DOCKER_IMAGE=$(grep -o '"docker_image": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    INSTALL_DATE=$(grep -o '"install_date": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    
    # Write updated config
    cat > "$CONFIG_FILE" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "image_path": "$IMAGE_PATH",
  "container_cmd": "$CONTAINER_CMD",
  "install_date": "$INSTALL_DATE",
  "last_update": "$UPDATE_DATE",
  "docker_image": "$DOCKER_IMAGE",
  "repo_root": "$REPO_ROOT"
}
EOF

    echo ""
    echo "============================================"
    print_success "Update Complete!"
    echo "============================================"
    echo ""
    echo "Application files have been updated."
    echo "The Singularity image was not modified (no re-download needed)."
    echo ""
    
    if [[ "$backup_choice" =~ ^[Yy]$ ]]; then
        echo "Backup location: $BACKUP_DIR"
        echo ""
    fi
}

main "$@"