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

    # Try to read config
    if [ -f "$CONFIG_FILE" ]; then
        INSTALL_DIR=$(grep -o '"install_dir": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    fi
    
    # If no config or couldn't read, try to detect from script location
    if [ -z "$INSTALL_DIR" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        INSTALL_DIR="$SCRIPT_DIR"
        print_warning "Config file not found, using script location: $INSTALL_DIR"
    else
        print_info "Found installation at: $INSTALL_DIR"
    fi
    
    echo ""
    print_warning "This will remove:"
    echo "  - Installation directory: $INSTALL_DIR"
    echo "  - Desktop entries: $DESKTOP_DIR/ChronoRoot*.desktop"
    echo "  - Configuration: $CONFIG_FILE"
    echo ""
    
    read -p "Are you sure you want to uninstall ChronoRoot? [y/N]: " confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        print_info "Uninstallation cancelled."
        exit 0
    fi

    echo ""
    print_info "Removing installation directory..."
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