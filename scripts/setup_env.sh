#!/bin/bash

set -e

PROFILE="basic"
NON_INTERACTIVE=0
RECREATE_VENV=0
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --non-interactive)
            NON_INTERACTIVE=1
            shift
            ;;
        --recreate-venv)
            RECREATE_VENV=1
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

install_apt_deps() {
    if ! command -v apt-get >/dev/null 2>&1; then
        return
    fi
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv build-essential git curl wget tmux tree zip unzip
}

create_or_activate_venv() {
    if [[ -d "$VENV_DIR" && "$RECREATE_VENV" -eq 1 ]]; then
        rm -rf "$VENV_DIR"
    fi
    if [[ -d "$VENV_DIR" && "$NON_INTERACTIVE" -eq 0 && "$RECREATE_VENV" -eq 0 ]]; then
        printf "venv already exists, recreate? (y/N): "
        read -r ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        fi
    fi
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
}

install_pytorch() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        CUDA_MAJOR=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
        if [[ "$CUDA_MAJOR" == "11" ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        fi
    else
        pip install torch torchvision torchaudio
    fi
}

install_project_deps() {
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        pip install -r "$PROJECT_DIR/requirements.txt"
    fi
}

verify_runtime() {
    python - << 'PYEOF'
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PYEOF
}

if [[ "$PROFILE" == "ubuntu" || "$PROFILE" == "gpu" || "$PROFILE" == "wsl" ]]; then
    install_apt_deps
fi

if [[ "$PROFILE" == "wsl" ]]; then
    if ! grep -qi microsoft /proc/version 2>/dev/null; then
        echo "WSL profile requires WSL."
        exit 1
    fi
fi

create_or_activate_venv
install_pytorch
install_project_deps
verify_runtime

echo "setup complete: profile=$PROFILE project=$PROJECT_DIR"
