#!/bin/bash

# ============================================================================
#  ChronoRoot Local Installer (Conda-based)
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
INSTALL_DIR="$HOME/.local/chronoroot"
DESKTOP_ENTRY_DIR="$HOME/.local/share/applications"

main() {
    clear
    echo -e "${BOLD}ChronoRoot Local Installation Wizard${NC}"
    
    # 1. Prerequisite & GPU Check
    section_title "1. Checking System Hardware & Prerequisites"
    
    if ! command_exists conda; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    print_success "Conda detected."

    HAS_GPU=false
    if command_exists nvidia-smi; then
        if nvidia-smi > /dev/null 2>&1; then
            print_success "NVIDIA GPU detected and functional."
            HAS_GPU=true
        else
            print_warning "NVIDIA GPU found but drivers are not responding."
        fi
    else
        print_status "No NVIDIA GPU detected. Using CPU-only mode."
    fi

    # 2. Setup Directory
    section_title "2. Directory Setup"
    mkdir -p "$INSTALL_DIR"
    print_status "Installation path: $INSTALL_DIR"

    # 3. Repository Setup
    section_title "3. Downloading Repository"
    cd "$INSTALL_DIR"
    if [ ! -d "ChronoRoot2" ]; then
        print_status "Cloning repository..."
        git clone "$REPO_URL" ChronoRoot2
    else
        print_status "Updating existing repository..."
        (cd ChronoRoot2 && git pull)
    fi
    REPO_DIR="$INSTALL_DIR/ChronoRoot2"

    # 4. Environment & Capability Selection
    section_title "4. Configuration Selection"
    
    echo "Choose your installation type:"
    echo -e "1) ${BOLD}Full Node${NC} (Executes Segmentation + Analysis. Requires GPU)."
    echo -e "2) ${BOLD}Lite Node${NC} (Analysis only. Runs on standard Laptops)."
    read -p "Selection [1-2]: " env_choice

    # Logic Variables
    INSTALL_SEG_GUI=false
    DOWNLOAD_WEIGHTS=false
    ENV_FILE=""

    if [ "$env_choice" == "1" ]; then
        # -- FULL MODE --
        # Safety check for GPU
        if [ "$HAS_GPU" = false ]; then
            print_warning "Full version requires a GPU for processing."
            read -p "Proceed anyway? [y/N]: " proceed
            [[ ! $proceed =~ ^[Yy]$ ]] && exit 1
        fi
        ENV_FILE="$REPO_DIR/environment.yml"
        INSTALL_SEG_GUI=true
        DOWNLOAD_WEIGHTS=true
        print_status "Selected: Full Installation"

    else
        # -- LITE MODE --
        ENV_FILE="$REPO_DIR/environment_no_nnunet.yml"
        print_status "Selected: Lite Installation"

        # Ask about Monitoring GUI
        echo -e "\n${BOLD}Optional: Segmentation Monitoring${NC}"
        echo "Do you want to install the Segmentation GUI?"
        echo "Useful for monitoring jobs running on a cluster/shared drive."
        read -p "Install Segmentation GUI? [Y/n]: " monitor_choice
        
        if [[ $monitor_choice =~ ^[Yy]$ ]] || [[ -z $monitor_choice ]]; then
            INSTALL_SEG_GUI=true
            print_status "Segmentation GUI will be installed (Monitoring Mode)."
        else
            print_status "Segmentation GUI skipped."
        fi
    fi

    # 5. Conda Environment Install
    section_title "5. Conda Environment Setup"
    ENV_NAME="ChronoRoot"
    
    if conda env list | grep -q "$ENV_NAME"; then
        print_status "Updating existing environment ($ENV_NAME)..."
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        print_status "Creating new environment ($ENV_NAME)..."
        conda env create -n "$ENV_NAME" -f "$ENV_FILE"
    fi

    # 6. Download Weights (Only if Full Node)
    if [ "$DOWNLOAD_WEIGHTS" = true ]; then
        section_title "6. Downloading Segmentation Weights"
        
        WEIGHTS_SCRIPT="$REPO_DIR/segmentationApp/download_weights.sh"
        
        if [ -f "$WEIGHTS_SCRIPT" ]; then
            chmod +x "$WEIGHTS_SCRIPT"
            print_status "Syncing models from Hugging Face..."
            # Run inside conda to access 'hf' or install it if missing
            conda run --no-capture-output -n "$ENV_NAME" /bin/bash "$WEIGHTS_SCRIPT"
        else
            print_warning "download_weights.sh not found. Skipping."
        fi
    else
        echo -e "\n${BLUE}[INFO]${NC} Skipping weight download (Monitoring/Lite mode selected)."
    fi

    # 7. Launcher Generation
    section_title "7. Creating Desktop Launchers"
    CONDA_BASE=$(conda info --base)
    
    create_local_shortcut() {
        local name="$1"
        local file_name="$2"
        local subdir="$3"
        local icon="$4"
        local wrapper="$INSTALL_DIR/${file_name}.sh"
        local desktop="$DESKTOP_ENTRY_DIR/${file_name}.desktop"

        # Create Wrapper
        cat > "$wrapper" << EOF
#!/bin/bash

umask 002
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME
cd "$REPO_DIR/$subdir"
python run.py
EOF
        chmod +x "$wrapper"

        # Create Desktop Entry
        cat > "$desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$name
Exec=$wrapper
Icon=$REPO_DIR/$icon
Terminal=true
Categories=Science;Education;
EOF
        chmod +x "$desktop"
        print_success "Created: $name"
    }

    # Always install Core Apps
    create_local_shortcut "ChronoRoot App" "ChronoRootApp" "chronoRootApp" "logo.ico"
    create_local_shortcut "ChronoRoot Screening" "ChronoRootScreening" "chronoRootScreeningApp" "logo_screening.ico"
    
    # Conditionally install Segmentation GUI
    if [ "$INSTALL_SEG_GUI" = true ]; then
        create_local_shortcut "ChronoRoot Segmentation" "ChronoRootSegmentation" "segmentationApp" "logo_seg.ico"
    fi

    # Refresh desktop database
    if command_exists update-desktop-database; then
        update-desktop-database "$DESKTOP_ENTRY_DIR" 2>/dev/null || true
    fi

    section_title "Installation Complete"
    print_success "ChronoRoot is ready at $INSTALL_DIR"
    echo "Check your application menu for the new shortcuts."
}

main "$@"