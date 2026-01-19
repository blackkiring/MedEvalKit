#!/bin/bash
# install_deps.sh - Install transformers and vllm from local submodules
#
# Usage:
#   ./install_deps.sh              # Install both
#   ./install_deps.sh transformers # Install only transformers
#   ./install_deps.sh vllm         # Install only vllm
#   ./install_deps.sh --update     # Update submodules and reinstall

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/deps"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if submodules are initialized
check_submodules() {
    if [ ! -d "${DEPS_DIR}/transformers/.git" ] || [ ! -d "${DEPS_DIR}/vllm/.git" ]; then
        log_info "Initializing submodules..."
        git submodule update --init --recursive
    fi
}

# Update submodules to latest
update_submodules() {
    log_info "Updating submodules to latest..."
    git submodule update --remote --merge
}

# Install transformers from submodule
install_transformers() {
    log_info "Installing transformers from submodule..."

    if [ ! -d "${DEPS_DIR}/transformers" ]; then
        log_error "transformers submodule not found at ${DEPS_DIR}/transformers"
        exit 1
    fi

    # Uninstall existing transformers first
    pip uninstall -y transformers 2>/dev/null || true

    # Install from local directory in editable mode
    pip install -e "${DEPS_DIR}/transformers" --no-build-isolation

    log_info "transformers installed successfully!"
}

# Install vllm from submodule
install_vllm() {
    log_info "Installing vllm from submodule..."

    if [ ! -d "${DEPS_DIR}/vllm" ]; then
        log_error "vllm submodule not found at ${DEPS_DIR}/vllm"
        exit 1
    fi

    # Uninstall existing vllm first
    pip uninstall -y vllm 2>/dev/null || true

    # Set environment variables for vllm build
    export MAX_JOBS=${MAX_JOBS:-4}
    export VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED:-1}

    # Install from local directory
    # Note: vllm compilation can take a long time
    log_warn "vLLM installation may take 10-30 minutes for compilation..."
    pip install -e "${DEPS_DIR}/vllm" --no-build-isolation

    log_info "vllm installed successfully!"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [PACKAGE]"
    echo ""
    echo "Options:"
    echo "  --update    Update submodules before installing"
    echo "  --help      Show this help message"
    echo ""
    echo "Packages:"
    echo "  transformers    Install only transformers"
    echo "  vllm            Install only vllm"
    echo "  (none)          Install both packages"
    echo ""
    echo "Examples:"
    echo "  $0                      # Install both packages"
    echo "  $0 transformers         # Install only transformers"
    echo "  $0 --update             # Update and install both"
    echo "  $0 --update vllm        # Update and install vllm"
}

# Main
main() {
    local update=false
    local packages=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --update)
                update=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            transformers|vllm)
                packages+=("$1")
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Default to both packages if none specified
    if [ ${#packages[@]} -eq 0 ]; then
        packages=("transformers" "vllm")
    fi

    # Check and update submodules
    check_submodules

    if [ "$update" = true ]; then
        update_submodules
    fi

    # Install packages
    for pkg in "${packages[@]}"; do
        case "$pkg" in
            transformers)
                install_transformers
                ;;
            vllm)
                install_vllm
                ;;
        esac
    done

    log_info "Installation complete!"

    # Show installed versions
    echo ""
    log_info "Installed versions:"
    python -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null || echo "  transformers: not installed"
    python -c "import vllm; print(f'  vllm: {vllm.__version__}')" 2>/dev/null || echo "  vllm: not installed"
}

main "$@"
