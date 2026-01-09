#!/bin/bash

# ============================================================================
#  ChronoRoot WSL Native Installer (Conda-based)
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
section_title() { echo -e "\n${BOLD}--- $1 ---${NC}"; }
print_status() { echo -e "${BLUE}[STATUS]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- Configuration ---
REPO_URL="https://github.com/ngaggion/ChronoRoot2.git"
DEFAULT_INSTALL_DIR="$HOME/.local/chronoroot"

main() {
    clear
    echo -e "${BOLD}ChronoRoot WSL Native Installer${NC}"
    echo "============================================"
    echo ""

    # 1. Environment Check
    section_title "1. System Checks"
    
    if ! grep -qEi "(Microsoft|WSL)" /proc/version; then
        print_error "This script must be run inside WSL."
        exit 1
    fi

    if ! command_exists conda; then
        print_error "Conda not found. Please install Miniconda or Anaconda in WSL first."
        exit 1
    fi
    print_success "WSL and Conda detected."

    # GPU Check
    HAS_GPU=false
    if command_exists nvidia-smi && nvidia-smi > /dev/null 2>&1; then
        print_success "NVIDIA GPU detected."
        HAS_GPU=true
    else
        print_status "No active NVIDIA GPU detected. Using CPU-only mode."
    fi

    # 2. Setup Directory
    section_title "2. Directory Setup"
    read -p "Installation directory (Default: $DEFAULT_INSTALL_DIR): " user_install_dir
    INSTALL_DIR="${user_install_dir:-$DEFAULT_INSTALL_DIR}"
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    mkdir -p "$INSTALL_DIR"
    print_status "Installing to: $INSTALL_DIR"

    # 3. Repository Setup
    section_title "3. Downloading Repository"
    REPO_DIR="$INSTALL_DIR/ChronoRoot2"
    
    if [ ! -d "$REPO_DIR" ]; then
        print_status "Cloning repository..."
        git clone "$REPO_URL" "$REPO_DIR"
    else
        print_status "Updating repository..."
        (cd "$REPO_DIR" && git pull)
    fi

    # 4. Conda Environment Setup
    section_title "4. Conda Environment Setup"
    echo "Which version would you like to install?"
    echo -e "1) ${BOLD}Full${NC} (Segmentation + Analysis, requires GPU)"
    echo -e "2) ${BOLD}Lite${NC} (Analysis only, perfect for laptops)"
    read -p "Selection [1-2]: " env_choice

    # Safety check for Full without GPU
    if [ "$env_choice" == "1" ] && [ "$HAS_GPU" = false ]; then
        print_warning "Full version requires a GPU for segmentation."
        read -p "Proceed anyway? [y/N]: " proceed
        [[ ! $proceed =~ ^[Yy]$ ]] && exit 1
    fi

    ENV_NAME="chronoroot_wsl"
    if [ "$env_choice" == "1" ]; then
        ENV_FILE="$REPO_DIR/environment.yml"
        print_status "Selected: Full Environment"
    else
        ENV_FILE="$REPO_DIR/environment_no_nnunet.yml"
        print_status "Selected: Lite Environment"
    fi

    # Create/Update Environment
    if conda env list | grep -q "$ENV_NAME"; then
        print_status "Updating existing environment..."
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        print_status "Creating new environment (this may take a while)..."
        conda env create -n "$ENV_NAME" -f "$ENV_FILE"
    fi

    # 5. Windows Integration
    section_title "5. Configuring Windows Shortcuts"
    
    # Windows Paths
    [ -z "$WSL_DISTRO_NAME" ] && WSL_DISTRO_NAME="Ubuntu"
    WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
    WIN_USER_PROFILE="/mnt/c/Users/$WIN_USER"
    WIN_ICON_STORE="$WIN_USER_PROFILE/AppData/Local/ChronoRootIcons"
    START_MENU_DIR="$WIN_USER_PROFILE/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/ChronoRoot"
    DESKTOP_DIR="$WIN_USER_PROFILE/Desktop"

    mkdir -p "$WIN_ICON_STORE"
    mkdir -p "$START_MENU_DIR"

    # Get Conda Base Path for sourcing
    CONDA_BASE=$(conda info --base)

    create_wsl_shortcut() {
        local name="$1"
        local folder="$2"
        local icon_name="$3"
        local script_name="${name}.sh"
        
        # A. Create Linux Wrapper Script
        local sh_file="$INSTALL_DIR/$script_name"
        cat > "$sh_file" << EOF
#!/bin/bash
[ -z "\$DISPLAY" ] && export DISPLAY=:0
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME
cd "$REPO_DIR/$folder"
python run.py
EOF
        chmod +x "$sh_file"

        # B. Copy Icon to Windows
        local source_icon="$REPO_DIR/$icon_name"
        local dest_icon="$WIN_ICON_STORE/$icon_name"
        local win_icon_path=""

        if [ -f "$source_icon" ]; then
            cp "$source_icon" "$dest_icon"
            win_icon_path=$(wslpath -w "$dest_icon")
        else
            print_warning "Icon $icon_name not found."
        fi

        # C. Create Windows Batch Launcher (Hidden)
        local bat_path="$INSTALL_DIR/${name}.bat"
        # Note: We use printf to ensure CRLF line endings for Windows
        printf "@echo off\r\n" > "$bat_path"
        printf "title $name\r\n" >> "$bat_path"
        printf "wsl -d $WSL_DISTRO_NAME bash -l -c \"$sh_file\"\r\n" >> "$bat_path"
        printf "if %%errorlevel%% neq 0 pause\r\n" >> "$bat_path"

        local win_bat_path=$(wslpath -w "$bat_path")

        # D. Generate Shortcuts via PowerShell
        print_status "Generating shortcuts for $name..."
        
        for target_dir in "$DESKTOP_DIR" "$START_MENU_DIR"; do
            local win_target_dir=$(wslpath -w "$target_dir")
            
            powershell.exe -Command "
            \$WshShell = New-Object -ComObject WScript.Shell;
            \$Shortcut = \$WshShell.CreateShortcut('$win_target_dir\\${name}.lnk');
            \$Shortcut.TargetPath = 'C:\\Windows\\System32\\cmd.exe';
            \$Shortcut.Arguments = '/c \"$win_bat_path\"';
            \$Shortcut.Description = 'Launch $name (WSL)';
            \$Shortcut.WindowStyle = 1; 
            if ('$win_icon_path' -ne '') { \$Shortcut.IconLocation = '$win_icon_path'; }
            \$Shortcut.Save();"
        done
    }

    # Generate Shortcuts
    create_wsl_shortcut "ChronoRootApp" "chronoRootApp" "logo.ico"
    create_wsl_shortcut "ChronoRootScreening" "chronoRootScreeningApp" "logo_screening.ico"

    if [ "$env_choice" == "1" ]; then
        create_wsl_shortcut "ChronoRootSegmentation" "segmentationApp" "logo_seg.ico"
    fi

    # 6. Cleanup
    if command_exists dos2unix; then
        dos2unix "$INSTALL_DIR"/*.sh >/dev/null 2>&1 || true
    fi
    
    # Force Icon Refresh
    cmd.exe /c "ie4uinit.exe -show" 2>/dev/null || true

    echo ""
    print_success "Installation Complete!"
    echo "Shortcuts have been added to your Desktop and Start Menu."
}

main "$@"