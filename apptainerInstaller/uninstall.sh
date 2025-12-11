#!/bin/bash

# ChronoRoot Uninstallation Script

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
DESKTOP_DIR="$HOME/.local/share/applications"

main() {
    clear
    echo "============================================"
    echo "  ChronoRoot Uninstallation Script"
    echo "============================================"
    echo ""

    # Check if configuration exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "ChronoRoot configuration not found!"
        print_error "Cannot determine installation location."
        echo ""
        echo "Manual removal:"
        echo "  1. Remove installation directory (typically ~/.local/chronoroot)"
        echo "  2. Remove desktop files from $DESKTOP_DIR"
        echo "  3. Remove config: $CONFIG_FILE"
        exit 1
    fi

    # Read configuration
    INSTALL_DIR=$(grep -o '"install_dir": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    
    if [ -z "$INSTALL_DIR" ]; then
        print_error "Could not read installation directory from config!"
        exit 1
    fi

    print_info "Found installation at: $INSTALL_DIR"
    echo ""
    
    print_warning "This will remove:"
    echo "  - Application files: $INSTALL_DIR"
    echo "  - Desktop entries: $DESKTOP_DIR/ChronoRoot*.desktop"
    echo "  - Configuration: $CONFIG_FILE"
    echo ""
    
    read -p "Are you sure you want to uninstall ChronoRoot? [y/N]: " confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        print_info "Uninstallation cancelled."
        exit 0
    fi

    echo ""
    print_info "Removing application files..."
    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
        print_success "Removed: $INSTALL_DIR"
    else
        print_warning "Directory not found: $INSTALL_DIR"
    fi

    print_info "Removing desktop entries..."
    for desktop_file in "$DESKTOP_DIR"/ChronoRoot*.desktop; do
        if [ -f "$desktop_file" ]; then
            rm -f "$desktop_file"
            print_success "Removed: $desktop_file"
        fi
    done

    print_info "Removing configuration..."
    if [ -f "$CONFIG_FILE" ]; then
        rm -f "$CONFIG_FILE"
        rmdir "$(dirname "$CONFIG_FILE")" 2>/dev/null || true
        print_success "Removed: $CONFIG_FILE"
    fi

    echo ""
    echo "============================================"
    print_success "Uninstallation Complete!"
    echo "============================================"
    echo ""
}

main "$@"