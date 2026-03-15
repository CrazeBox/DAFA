#!/bin/bash
# Quick test script - Runs 10 rounds (~2 minutes)

set -e

echo "=========================================="
echo "  DAFA Quick Test (10 rounds)"
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

# Run quick test
echo ""
echo "Running quick test with DAFA on CIFAR-10..."
echo ""

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 10 \
    --device cuda \
    --use_amp true \
    --eval_every 2

echo ""
echo "=========================================="
echo "  Quick Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/cifar10_dafa_*/"
echo ""
