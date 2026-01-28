#!/bin/bash

# ============================================================================
#  ChronoRoot Master Installer (WSL -> Windows) - CMD WRAPPER EDITION
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

main() {
    clear
    echo "============================================"
    echo "   ChronoRoot WSL -> Windows Installer"
    echo "============================================"
    echo ""

    # 1. Environment Check
    if ! grep -qEi "(Microsoft|WSL)" /proc/version; then
        print_error "This script must be run inside WSL."
        exit 1
    fi

    # 2. Dependencies
    print_info "Checking dependencies..."
    
    if ! command_exists apptainer; then
        print_warning "Apptainer missing. Installing..."
        sudo apt update && sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:apptainer/ppa
        sudo apt update && sudo apt install -y apptainer
    fi

    if ! command_exists git-lfs; then
        print_warning "Git LFS missing. Installing..."
        sudo apt update && sudo apt install -y git-lfs
        git lfs install
    fi

    if ! command_exists dos2unix; then
        sudo apt install -y dos2unix
    fi

    GPU_FLAG=""
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected."
        GPU_FLAG="--nv"
    fi

    # 3. Setup Directory
    echo ""
    read -p "Installation directory (Default: $DEFAULT_INSTALL_DIR): " user_install_dir
    INSTALL_DIR="${user_install_dir:-$DEFAULT_INSTALL_DIR}"
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    mkdir -p "$INSTALL_DIR"

    # 4. Clone / Update Repo
    REPO_DIR="$INSTALL_DIR/ChronoRoot2"
    if [ ! -d "$REPO_DIR" ]; then
        print_info "Cloning repository..."
        git clone "$REPO_URL" "$REPO_DIR"
    else
        print_info "Updating repository..."
        (cd "$REPO_DIR" && git pull)
    fi

    # 5. Image Build
    IMAGE_PATH="$INSTALL_DIR/Image_ChronoRoot.sif"
    if [ ! -f "$IMAGE_PATH" ]; then
        echo ""
        echo "Select Image Version:"
        echo "1) Standard (13.5 GB)"
        echo "2) Full with Demo Data (23.5 GB)"
        read -p "Choice [1-2]: " img_choice
        DOCKER_TAG="nodemo"
        [[ "$img_choice" == "2" ]] && DOCKER_TAG="full"
        
        print_info "Building image..."
        apptainer build "$IMAGE_PATH" "docker://ngaggion/chronorootbase:$DOCKER_TAG"
    else
        print_success "Image already exists."
    fi

    # 6. Download Weights (NEW SECTION)
    echo ""
    echo "The segmentation models are required for the Full analysis pipeline."
    read -p "Would you like to download/update the segmentation weights now? [Y/n]: " download_choice
    
    # Default to 'Yes' if empty
    if [[ -z "$download_choice" || "$download_choice" =~ ^[Yy]$ ]]; then
        WEIGHTS_SCRIPT="$REPO_DIR/segmentationApp/download_weights.sh"
        if [ -f "$WEIGHTS_SCRIPT" ]; then
            chmod +x "$WEIGHTS_SCRIPT"
            print_info "Syncing models from Hugging Face..."

            # Create temp directory for the isolated pip install
            TEMP_HF_HOME=".tmp_hf_env"
            mkdir -p "$TEMP_HF_HOME"

            # Execute container
            apptainer exec $GPU_FLAG \
                --bind "$TEMP_HF_HOME:/tmp/hf_env" \
                --bind "/mnt/c:/mnt/c" --bind "/init:/init" --bind "/run/WSL:/run/WSL" \
                --bind "/home/$USER:/home/$USER" --bind "/run/user/$(id -u):/run/user/$(id -u)" \
                --env PIP_USER=true \
                --env PYTHONUSERBASE=/tmp/hf_env \
                "$IMAGE_PATH" \
                bash -c "export PATH=/tmp/hf_env/bin:\$PATH && source /opt/conda/etc/profile.d/conda.sh && conda activate ChronoRoot && pip install huggingface_hub==1.3.4 && $WEIGHTS_SCRIPT"
                
            if [ $? -eq 0 ]; then
                print_success "Weights synchronized successfully."
            else
                print_error "Weight download failed."
            fi

            rm -rf "$TEMP_HF_HOME"
        else
            print_error "Weights download script not found."
        fi
    else
        print_info "Skipping weight download."
    fi

    # 7. Windows Integration (CMD WRAPPER FIX)
    print_info "Configuring Shortcuts & Icons..."

    [ -z "$WSL_DISTRO_NAME" ] && WSL_DISTRO_NAME="Ubuntu"
    
    WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
    WIN_USER_PROFILE="/mnt/c/Users/$WIN_USER"
    WIN_ICON_STORE="$WIN_USER_PROFILE/AppData/Local/ChronoRootIcons"
    START_MENU_DIR="$WIN_USER_PROFILE/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/ChronoRoot"
    DESKTOP_DIR="$WIN_USER_PROFILE/Desktop"

    mkdir -p "$WIN_ICON_STORE"
    mkdir -p "$START_MENU_DIR"

    create_linux_wrapper() {
        local name=$1
        local folder=$2
        local sh_file="$INSTALL_DIR/${name}.sh"

        cat > "$sh_file" << EOF
#!/bin/bash
[ -z "\$DISPLAY" ] && export DISPLAY=:0
echo "Starting $name..."

# 1. Initialize Bind Paths (WSL Essentials)
# We strictly bind C: and WSL interop paths first
BINDS="--bind /mnt/c:/mnt/c --bind /init:/init --bind /run/WSL:/run/WSL"

# 2. Add User & Runtime paths
BINDS="\$BINDS --bind /home/\$USER:/home/\$USER --bind /run/user/\$(id -u):/run/user/\$(id -u)"

# 3. Dynamic Optional Paths (Net/Media/Mnt)
# Helpful if you have mounted network drives or external disks in WSL
if [ -d "/net" ]; then BINDS="\$BINDS --bind /net:/net"; fi
if [ -d "/media" ]; then BINDS="\$BINDS --bind /media:/media"; fi

# WSL Special: /mnt usually exists, but binding it whole allows access to D:, E:, etc.
if [ -d "/mnt" ]; then BINDS="\$BINDS --bind /mnt:/mnt"; fi

# 4. Execute
apptainer exec $GPU_FLAG \$BINDS --env DISPLAY=\$DISPLAY --env WSL_INTEROP=\$WSL_INTEROP "$IMAGE_PATH" \\
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ChronoRoot && cd $REPO_DIR/$folder && python run.py"
EOF
        chmod +x "$sh_file"
    }

    create_windows_assets() {
        local name=$1
        local icon_name=$2
        local script_name="${name}.sh"
        
        # 1. Copy Icon to Windows
        local source_icon="$REPO_DIR/$icon_name"
        local dest_icon="$WIN_ICON_STORE/$icon_name"
        local win_icon_path=""

        if [ -f "$source_icon" ]; then
            cp "$source_icon" "$dest_icon"
            win_icon_path=$(wslpath -w "$dest_icon")
        else
            echo "   [Warning] Icon $icon_name not found."
        fi

        # 2. Create Hidden Batch File
        local linux_bat_path="$INSTALL_DIR/${name}.bat"
        printf "@echo off\r\n" > "$linux_bat_path"
        printf "title $name\r\n" >> "$linux_bat_path"
        printf "cd /d %%USERPROFILE%%\r\n" >> "$linux_bat_path"
        printf "wsl -d $WSL_DISTRO_NAME bash -l -c \"$INSTALL_DIR/$script_name\"\r\n" >> "$linux_bat_path"
        printf "if %%errorlevel%% neq 0 pause\r\n" >> "$linux_bat_path"

        local win_bat_path=$(wslpath -w "$linux_bat_path")

        # 3. Create Shortcuts (CMD WRAPPER STRATEGY)
        # Target: C:\Windows\System32\cmd.exe
        # Arguments: /c "Path\To\Script.bat"
        # This tricks Windows into thinking it's a local app, allowing the icon.
        
        for target_dir in "$DESKTOP_DIR" "$START_MENU_DIR"; do
            local win_target_dir=$(wslpath -w "$target_dir")
            
            powershell.exe -Command "
            \$WshShell = New-Object -ComObject WScript.Shell;
            \$Shortcut = \$WshShell.CreateShortcut('$win_target_dir\\${name}.lnk');
            
            \$Shortcut.TargetPath = 'C:\\Windows\\System32\\cmd.exe';
            \$Shortcut.Arguments = '/c \"$win_bat_path\"';
            \$Shortcut.Description = 'Launch $name';
            \$Shortcut.WindowStyle = 1; 
            
            if ('$win_icon_path' -ne '') { 
                \$Shortcut.IconLocation = '$win_icon_path'; 
            }
            
            \$Shortcut.Save();"
        done
    }

    create_linux_wrapper "ChronoRootApp" "chronoRootApp"
    create_windows_assets "ChronoRootApp" "logo.ico"

    create_linux_wrapper "ChronoRootScreening" "chronoRootScreeningApp"
    create_windows_assets "ChronoRootScreening" "logo_screening.ico"

    create_linux_wrapper "ChronoRootSegmentation" "segmentationApp"
    create_windows_assets "ChronoRootSegmentation" "logo_seg.ico"

    dos2unix "$INSTALL_DIR"/*.sh 2>/dev/null

    # 8. Force Icon Refresh
    print_info "Flushing Windows Icon Cache..."
    cmd.exe /c "ie4uinit.exe -show" 2>/dev/null || true

    echo ""
    print_success "DONE! Icons should now be visible on Desktop."
}

main "$@"