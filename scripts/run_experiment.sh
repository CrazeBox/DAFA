#!/bin/bash
# Full experiment script - Runs 100 rounds

set -e

echo "=========================================="
echo "  DAFA Full Experiment"
echo "=========================================="

# Check if already in a virtual environment
IN_VENV=false
if [ ! -z "$VIRTUAL_ENV" ]; then
    IN_VENV=true
    echo "Using active virtual environment: $VIRTUAL_ENV"
elif [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    IN_VENV=true
    echo "Using active conda environment: $CONDA_DEFAULT_ENV"
fi

# If not in venv, try to activate venv
if [ "$IN_VENV" = false ]; then
    if [ -d "venv" ]; then
        echo "Activating venv..."
        source venv/bin/activate
    else
        echo "No virtual environment found. Please activate your environment first:"
        echo "  conda activate jd"
        echo "  or: source venv/bin/activate"
        exit 1
    fi
fi

# Verify torch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch not installed in current environment."
    echo "Please install: pip install torch torchvision"
    exit 1
fi

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Detect GPU memory and set optimal config
GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")

if [ "$GPU_MEMORY" -gt 12000000000 ]; then
    PARALLEL=4
    WORKERS=0
    echo "Detected high-memory GPU, using parallel training."
elif [ "$GPU_MEMORY" -gt 8000000000 ]; then
    PARALLEL=2
    WORKERS=2
    echo "Detected mid-range GPU, using moderate parallelism."
else
    PARALLEL=1
    WORKERS=4
    echo "Detected standard GPU, using sequential training."
fi

echo ""
echo "Configuration:"
echo "  - Parallel clients: $PARALLEL"
echo "  - DataLoader workers: $WORKERS"
echo ""

# Run experiment
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 100 \
    --device cuda \
    --use_amp true \
    --num_parallel_clients $PARALLEL \
    --num_workers $WORKERS \
    --eval_every 5

echo ""
echo "=========================================="
echo "  Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/cifar10_dafa_*/"
echo ""
