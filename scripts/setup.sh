#!/bin/bash
# DAFA Setup Script - One-click installation

set -e

echo "=========================================="
echo "  DAFA Setup Script"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[1/6] Python version: $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "      Virtual environment created."
else
    echo "      Virtual environment already exists."
fi

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Install PyTorch (detect CUDA)
echo "[4/6] Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
    echo "      Detected CUDA $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" =~ ^12 ]]; then
        echo "      Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    elif [[ "$CUDA_VERSION" =~ ^11 ]]; then
        echo "      Installing PyTorch for CUDA 11.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    else
        echo "      Installing PyTorch (CPU version)..."
        pip install torch torchvision -q
    fi
else
    echo "      No NVIDIA GPU detected, installing PyTorch (CPU version)..."
    pip install torch torchvision -q
fi

# Install other dependencies
echo "[5/6] Installing other dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "      Dependencies installed."

# Verify installation
echo "[6/6] Verifying installation..."

# Check PyTorch
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""

if [ "$PYTORCH_VERSION" = "NOT_INSTALLED" ]; then
    echo "WARNING: PyTorch installation failed!"
    echo "Please install manually:"
    echo "  pip install torch torchvision"
else
    echo "PyTorch version: $PYTORCH_VERSION"
    echo "CUDA available: $CUDA_AVAILABLE"

    if [ "$CUDA_AVAILABLE" = "True" ]; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        GPU_MEMORY=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')")
        echo "GPU: $GPU_NAME ($GPU_MEMORY)"
    fi
fi

echo ""
echo "Quick Start:"
echo "  1. Activate:  source venv/bin/activate"
echo "  2. Test:      bash scripts/run_quick.sh"
echo "  3. Train:     bash scripts/run_experiment.sh"
echo ""
