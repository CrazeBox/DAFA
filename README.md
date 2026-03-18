# DAFA: Direction-Aware Federated Averaging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of **DAFA (Direction-Aware Federated Averaging)** for federated learning with non-IID data.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/DAFA.git
cd DAFA

# 2. Run setup (creates venv, installs dependencies, verifies GPU)
bash scripts/setup_env.sh --profile basic

# 3. Run quick test (10 rounds, ~2 minutes)
bash scripts/run_quick.sh

# 4. Run full experiment
bash scripts/run_experiment.sh

# 5. Run full study pipeline
bash scripts/study_pipeline.sh results/study_pipeline cuda 42,123,456 100
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ GPU memory for basic experiments

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
bash scripts/setup_env.sh --profile basic
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Usage

### Basic Experiment

```bash
# Activate environment
source venv/bin/activate

# Run single experiment
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 100 \
    --device cuda
```

### Hyperparameter Tuning (Phase 1)

```bash
# Run full tuning (8 methods, ~2 days on RTX 3050)
python scripts/run_phase1_tuning.py \
    --dataset cifar10 \
    --alpha 0.5 \
    --num_rounds 50

# Quick tuning (faster)
python scripts/run_phase1_tuning.py \
    --dataset cifar10 \
    --alpha 0.5 \
    --num_rounds 20 \
    --num_repeats 2
```

### GPU Memory Configuration

| GPU Memory | Recommended Settings |
|------------|---------------------|
| ≥12GB | `--num_parallel_clients 4 --num_workers 0` |
| 8-12GB | `--num_parallel_clients 2 --num_workers 2` |
| 4-8GB | `--num_parallel_clients 1 --num_workers 4` (default) |
| <4GB | Use CPU: `--device cpu` |

## Supported Methods

| Method | Description |
|--------|-------------|
| `fedavg` | Federated Averaging (McMahan et al., 2017) |
| `fedprox` | Federated Proximal (Li et al., 2020) |
| `scaffold` | SCAFFOLD (Karimireddy et al., 2020) |
| `fednova` | FedNova (Wang et al., 2020) |
| `fedavgm` | FedAvg with server momentum |
| `fedadam` | FedAdam (Reddi et al., 2020) |
| `dafa` | Direction-Aware Federated Averaging (Ours) |
| `dir_weight` | Direction-based Weighting |

## Supported Datasets

| Dataset | Classes | Default Clients |
|---------|---------|-----------------|
| CIFAR-10 | 10 | 100 |
| FEMNIST | 62 | 100 |
| Shakespeare | 80 | 100 |

## Project Structure

```
DAFA/
├── scripts/           # Experiment scripts
│   ├── setup_env.sh   # Unified environment setup
│   ├── run_quick.sh   # Quick test
│   ├── run_experiment.py
│   ├── run_five_stages.py
│   ├── extract_best_runs.py
│   ├── plot_results.py
│   ├── run_analysis.py
│   └── study_pipeline.sh
├── src/               # Source code
│   ├── methods/       # Aggregation methods
│   ├── models/        # Neural network models
│   ├── data/          # Data loading utilities
│   └── utils/         # Helper functions
├── configs/           # Configuration files
├── results/           # Experiment results (auto-generated)
└── requirements.txt   # Dependencies
```

## Results

After running experiments, results are saved to `results/`:

```
results/
└── default/
    └── cifar10_dafa_seed42_20240101_120000/
        ├── results.json       # Final metrics
        ├── metadata.json      # Run metadata
        ├── config.json        # Experiment config
        ├── experiment.log     # Training log
        └── checkpoints/       # Model checkpoints
            └── best_model.pt  # Best model weights
```

Pipeline outputs:

```
results/study_pipeline/
├── phase1/
├── five_stages/
│   ├── five_stages_summary.json
│   └── pipeline_status.json
└── summary/
    ├── best_runs.json
    ├── best_runs.csv
    └── plots/
```

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{dafa2024,
  title={Direction-Aware Federated Averaging for Non-IID Data},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- FedProx: Li et al., "Federated Optimization in Heterogeneous Networks"
- SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
