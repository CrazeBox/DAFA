# DAFA Linux Reproduction Guide

This guide is written for a clean Linux environment. It assumes nothing about the author's local workspace and is intended for open-source users who want to reproduce the DAFA experiments from scratch.

## 1. Target outcome

By following this guide, a new user should be able to:

- create the environment
- download the required datasets without manually copying dataset links
- run the staged experiment pipeline
- regenerate the main summaries under `results/five_stages/`
- regenerate summary tables and plots from result files

## 2. Recommended machine

- OS: Ubuntu 20.04+ or another recent Linux distribution
- Python: 3.10 or 3.11
- GPU: NVIDIA CUDA GPU recommended
- Disk: at least `20 GB` free
- RAM: at least `16 GB`

CPU-only execution is possible for smoke tests, but full runs will be slow.

## 3. Clone and set up

```bash
git clone <your-public-repo-url>
cd DAFA
bash scripts/setup_env.sh --profile basic
source venv/bin/activate
```

Quick verification:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python tests/test_ubuntu_environment.py
```

Optional unit tests:

```bash
pytest tests/test_all.py -v
```

## 4. Download datasets

The public release should use the built-in downloader instead of asking users to fetch links manually.

Download all required datasets:

```bash
python scripts/download_datasets.py --datasets cifar10,femnist,shakespeare
```

Download only a subset:

```bash
python scripts/download_datasets.py --datasets femnist,shakespeare
```

Notes:

- `CIFAR-10` is downloaded through `torchvision`.
- `FEMNIST` and `Shakespeare` are downloaded into:
  - `data/femnist/train/all_data.json`
  - `data/femnist/test/all_data.json`
  - `data/shakespeare/train/all_data.json`
  - `data/shakespeare/test/all_data.json`
- the downloader shows terminal progress bars
- `--allow_synthetic_data` is only for debugging and must not be used for formal reproduction
- if the default public mirror for `FEMNIST/Shakespeare` changes, pass an explicit mirror:

```bash
python scripts/download_datasets.py \
  --datasets femnist,shakespeare \
  --femnist_base_url <stable-femnist-mirror> \
  --shakespeare_base_url <stable-shakespeare-mirror>
```

Verify the files:

```bash
find data -maxdepth 3 -type f | sort
```

## 5. Smoke test before long runs

Run a very short CIFAR-10 experiment first:

```bash
python scripts/run_experiment.py \
  --method fedavg \
  --dataset cifar10 \
  --alpha 0.1 \
  --num_rounds 5 \
  --device cuda
```

Expected artifacts:

- `results/.../config.json`
- `results/.../metadata.json`
- `results/.../results.json`
- `results/.../experiment.log`

If this step fails, do not proceed to the full staged pipeline.

## 6. Full staged pipeline

### Stage 1: tuning

```bash
python scripts/run_five_stages.py \
  --stages 1 \
  --device cuda \
  --num_rounds 100 \
  --seeds 42,123,456
```

Expected files:

- `results/five_stages/stage1_tuning/best_hyperparams_generated.yaml`
- `results/five_stages/stage1_tuning/stage1_summary.json`

### Stage 2: main comparison

```bash
python scripts/run_five_stages.py \
  --stages 2 \
  --device cuda \
  --num_rounds 200 \
  --seeds 42,123,456,777,1024
```

Expected file:

- `results/five_stages/stage2_comparison/stage2_summary.json`

### Stage 3: mechanism analysis

```bash
python scripts/run_five_stages.py \
  --stages 3 \
  --device cuda \
  --num_rounds 200 \
  --seeds 42,123,456,777,1024
```

Expected file:

- `results/five_stages/stage3_mechanism/stage3_summary.json`

### Stage 4: ablations

```bash
python scripts/run_five_stages.py \
  --stages 4 \
  --device cuda \
  --num_rounds 200 \
  --seeds 42,123,456,777,1024
```

Expected file:

- `results/five_stages/stage4_ablation/stage4_summary.json`

### Stage 5: robustness and extensions

```bash
python scripts/run_five_stages.py \
  --stages 5 \
  --device cuda \
  --num_rounds 200 \
  --seeds 42,123,456,777,1024
```

Expected file:

- `results/five_stages/stage5_extension/stage5_summary.json`

## 7. Summaries and plots

Generate summarized run tables:

```bash
python scripts/analyze_results.py select-best \
  --results_root results \
  --output_dir results/summary
```

Generate plots from the summary:

```bash
python scripts/analyze_results.py plot \
  --best_runs results/summary/best_runs.json \
  --output_dir results/summary/plots \
  --format pdf
```

Compare a small set of completed runs directly:

```bash
python scripts/analyze_results.py compare \
  --inputs results/run_a results/run_b \
  --output_dir results/compare \
  --format png
```

## 8. Single-run commands

### CIFAR-10

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset cifar10 \
  --alpha 0.1 \
  --device cuda
```

### FEMNIST

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset femnist \
  --device cuda
```

### Shakespeare

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset shakespeare \
  --device cuda
```

## 9. Result layout

Single runs are saved under:

```text
results/<run_group>/<dataset>_<method>_seed<seed>_<timestamp>/
```

Key files:

- `config.json`
- `metadata.json`
- `results.json`
- `experiment.log`
- `checkpoints/`

The staged pipeline writes into:

```text
results/five_stages/
```

## 10. Important caveat for open-source claims

This repo already supports the staged pipeline and dataset bootstrap, but the current codebase still has some gaps relative to the exact paper algorithm and metrics described in `NeurlPS_Paper22.pdf`.

In particular, before claiming "full paper reproduction", verify the status of:

- DAFA warm-up
- EMA-based adaptive clipping and soft scaling
- centralized DSNR using a validation gradient proxy
- exact plotting scripts for all final paper figures
- a stable public mirror for `FEMNIST/Shakespeare` JSON files or an officially maintained generation script

`EXPERIMENT_DESIGN.md` is the source of truth for those release-readiness gaps.
