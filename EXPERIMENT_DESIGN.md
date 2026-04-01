# DAFA Experiment Design

> Version: v3.0
> Updated: 2026-04-01
> Paper source: `NeurlPS_Paper22.pdf`
> Design target: a fresh Linux machine can clone the repo, download data, run the experiments, and regenerate the paper-level artifacts with minimal manual intervention.

## 1. Scope

This document defines the release-grade experiment architecture for DAFA. The target is not "what happens to exist in the current workspace", but "what a new open-source user on Linux must be able to run from scratch".

The design therefore distinguishes three statuses for every requirement:

- `Reusable`: already aligned with the release target and can be used as-is.
- `Light change`: the repo has the right entrypoint, but needs targeted changes before claiming paper-grade reproduction.
- `Missing`: must be implemented before the public release can honestly claim full reproduction of `NeurlPS_Paper22.pdf`.

## 2. Paper-Required Outputs

The paper's experiment section requires the following canonical outputs.

### 2.1 Main paper outputs

1. `Table 1`: final test performance after 200 rounds across:
   - `CIFAR-10, alpha=0.5`
   - `CIFAR-10, alpha=0.1`
   - `FEMNIST, natural partition`
   - `Shakespeare, natural partition`
2. `Figure 1`: convergence curve on `CIFAR-10, alpha=0.1`
   - x-axis: communication rounds
   - y-axis: test accuracy
   - compare at least `FedAvg`, `SCAFFOLD`, `DAFA`, and preferably all main baselines
3. `Figure 2`: mechanism figure on `CIFAR-10, alpha=0.1`
   - left panel: empirical DSNR
   - right panel: client-update variance
4. `Table 2`: temperature `gamma` ablation on `CIFAR-10, alpha=0.1`
   - report accuracy
   - report empirical DSNR
5. `Beta ablation`: momentum coefficient `beta in {0.0, 0.5, 0.9}`
   - report final accuracy
   - report proxy-reliability style metric
6. `Dir-Weight comparison`
   - final accuracy comparison on `CIFAR-10, alpha=0.1`
   - proxy reliability comparison

### 2.2 Supplementary outputs needed for a convincing open-source release

These are not all final camera-ready paper figures, but they are necessary if the repo is meant to be independently audited:

1. Drift-alignment relationship
   - scatter or binned plot of drift magnitude vs. alignment score
   - Pearson correlation summary
2. Magnitude-poisoning robustness
   - `CIFAR-10, alpha=0.1`
   - 10% malicious clients
   - attack scale `100x`
   - compare `FedAvg`, `FedAvgM`, `DAFA`
3. Assumption validation
   - proxy reliability over rounds
   - max clipped norm over rounds
   - soft-scaling attenuation statistics near convergence
4. Efficiency summary
   - wall-clock time per round
   - total runtime per run
   - peak GPU memory if practical
5. Fairness summary
   - at least `bottom_10_accuracy` on classification datasets
   - explicitly marked as supplementary, not the primary optimization target

## 3. Benchmark Matrix

### 3.1 Datasets and models

| Dataset | Partition | Clients | Sampled / round | Local epochs | Model | Primary metric |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | Dirichlet `alpha=0.5` | 100 | 10 | 5 | ResNet-18 | Accuracy |
| CIFAR-10 | Dirichlet `alpha=0.1` | 100 | 10 | 5 | ResNet-18 | Accuracy |
| FEMNIST | Natural partition | 200 | 20 | 10 | 2-layer CNN | Accuracy |
| Shakespeare | Natural partition | 100 | 10 | 2 | 2-layer LSTM, hidden=256, embedding=200 | Perplexity |

### 3.2 Shared training protocol

- Batch size: `32`
- Client optimizer: `SGD`
- Client learning rate: `0.01` with cosine decay
- Server learning rate: `1.0`
- Communication rounds: `200`
- Seeds: `5`
- Report format: `mean +- std`

### 3.3 Required baselines

The paper explicitly requires these baselines:

- `FedAvg`
- `FedProx`
- `SCAFFOLD`
- `FedNova`
- `FedAvgM`
- `FedAdam`
- `Dir-Weight`
- `DAFA`

`FedYogi` is not required by `NeurlPS_Paper22.pdf`. It may be added later as an engineering extension, but it should not block the release.

## 4. Release-Grade Result Directory Standard

Every formal run must produce:

- `config.json`: resolved runtime config
- `metadata.json`: environment and run summary
- `results.json`: round-level history and final metrics
- `experiment.log`: human-readable log
- `checkpoints/`: saved model states

Every stage must additionally produce:

- stage summary JSON
- stage summary CSV
- paper-ready plots in `pdf` and `png`
- a manifest listing the exact run directories used to build each table/figure

Recommended public entrypoints:

- `scripts/run_experiment.py`
- `scripts/run_five_stages.py`
- `scripts/download_datasets.py`
- `scripts/analyze_results.py`

If the public repo cannot regenerate a table or figure from tracked result files plus plotting scripts, the release is not yet reproducible enough.

## 5. Five-Stage Architecture

### Stage 0. Environment and data bootstrap

Goal:
- enable a brand-new Linux user to prepare the environment and datasets with one short sequence of commands

Inputs:
- Python 3.10+ or 3.11
- CUDA-capable GPU recommended, CPU fallback allowed

Outputs:
- ready virtual environment
- downloaded datasets under `data/`
- environment verification log

Required commands:

```bash
bash scripts/setup_env.sh --profile basic
source venv/bin/activate
python scripts/download_datasets.py --datasets cifar10,femnist,shakespeare
python tests/test_ubuntu_environment.py
```

Status:
- setup script: `Reusable`
- one-shot dataset downloader with progress bar: `Light change`
- environment verification: `Reusable`

### Stage 1. Fair tuning

Goal:
- tune only method-specific hyperparameters under a shared protocol

Primary tuning scenario:
- `CIFAR-10, alpha=0.1`

Transfer-check scenario:
- `FEMNIST`

Outputs:
- `best_hyperparams_generated.yaml`
- tuning summary JSON/CSV
- search-space manifest

Status:
- current tuning entrypoint: `Reusable`
- paper-complete DAFA parameter search including warm-up and clipping thresholds: `Missing`

### Stage 2. Main results

Goal:
- generate the paper's main final-performance table and convergence plot

Required outputs:
- `Table 1`
- `Figure 1`
- stage summary JSON/CSV

Status:
- stage runner for final metrics: `Reusable`
- publication-grade convergence plotting: `Light change`
- direct table export by scenario and metric: `Light change`

### Stage 3. Mechanism validation

Goal:
- test the paper's directional-alignment story

Required outputs:
- `Figure 2` DSNR panel
- `Figure 2` variance panel
- drift-alignment correlation summary

Status:
- round-level metric tracking exists: `Reusable`
- paper-definition centralized DSNR with validation subset: `Light change`
- drift/alignment scatter generation: `Light change`

### Stage 4. Structural ablations

Goal:
- isolate what component actually matters inside DAFA

Required outputs:
- `Table 2` for `gamma`
- `beta` ablation summary
- `Dir-Weight vs DAFA` comparison

Status:
- partial ablation runner exists: `Reusable`
- warm-up ablation: `Missing`
- adaptive clipping / threshold ablation: `Missing`

### Stage 5. Robustness and release supplements

Goal:
- provide the extra evidence expected from a serious open-source FL paper repo

Required outputs:
- magnitude-poisoning table
- proxy-reliability curve
- clipping/attenuation curve
- runtime summary
- fairness summary

Status:
- attack knobs exist in the CLI: `Reusable`
- paper-specific poisoning protocol and summary plots: `Light change`
- wall-clock aggregation into stage outputs: `Light change`
- fairness summary table: `Light change`

## 6. Current Code vs Paper Gap Analysis

This section is the most important one for release planning.

### 6.1 What already matches the paper reasonably well

- Dataset defaults in `scripts/run_experiment.py`
- core benchmark roster except optional FedYogi
- five-stage orchestration in `scripts/run_five_stages.py`
- CIFAR-10 / FEMNIST / Shakespeare data loaders
- result serialization and checkpoint layout
- basic DSNR-like and variance tracking hooks

### 6.2 What does not yet match the paper closely enough

1. DAFA implementation gap
   - current `src/methods/dafa.py` uses a simple norm threshold `mu`
   - the paper describes lagged proxy scoring plus EMA-based dynamic thresholds, adaptive clipping, and soft scaling
   - warm-up `gamma_t = gamma * min(1, t / T_warm)` is not implemented
2. Metric-definition gap
   - current code tracks an empirical SNR-style proxy by default
   - the paper's centralized DSNR uses a validation-set approximation of the true global direction
3. Output gap
   - the repo has summary JSONs, but not a stable paper artifact manifest that maps runs to tables/figures
4. Plotting gap
   - current plotting utilities are generic and not yet aligned to the exact paper outputs
5. Data bootstrap gap
   - open-source users need a first-class dataset bootstrap script with progress bars and no manual link copying
   - the historical public JSON URLs for `FEMNIST/Shakespeare` are not stable enough for a long-lived release; a maintained mirror or release asset must be pinned before publication

### 6.3 Release claim that is currently safe

Safe claim now:
- "This repo provides a strong experimental scaffold for DAFA on CIFAR-10, FEMNIST, and Shakespeare, with automatic dataset download, a staged experiment pipeline, and a unified result-analysis entrypoint."

Claim that is not yet safe until the above gaps are closed:
- "This repo fully reproduces every experiment and exact algorithmic detail from `NeurlPS_Paper22.pdf`."

## 7. Fresh-Linux Reproduction Contract

For the public release, a new user should be able to do exactly this:

```bash
git clone <repo-url>
cd DAFA
bash scripts/setup_env.sh --profile basic
source venv/bin/activate
python scripts/download_datasets.py --datasets cifar10,femnist,shakespeare
python tests/test_ubuntu_environment.py
python scripts/run_five_stages.py --stages 1 --device cuda --num_rounds 100 --seeds 42,123,456
python scripts/run_five_stages.py --stages 2,3,4,5 --device cuda --num_rounds 200 --seeds 42,123,456,777,1024
```

And they should obtain:

- a complete `results/five_stages/` tree
- stage summaries
- paper plots
- enough metadata to rerender tables without rerunning training

If any step still depends on private local files, hidden shell history, or manual dataset hunting, the release is not finished.

## 8. Recommended Release Checklist

Before open-sourcing, verify:

1. `scripts/download_datasets.py` works on a clean Linux host and shows progress bars.
2. `docs/REPRODUCTION_GUIDE.md` matches the actual commands and paths.
3. `EXPERIMENT_DESIGN.md` only promises outputs that the repo can regenerate.
4. The DAFA algorithm description in the README does not overclaim beyond the current implementation.
5. At least one CI or manual smoke test covers:
   - environment setup
   - dataset download
   - a 1-5 round CIFAR-10 run
   - result-file generation

## 9. Deliverables for This Iteration

This iteration should leave the repo with:

- an updated `EXPERIMENT_DESIGN.md` centered on fresh-Linux reproducibility
- a Linux reproduction guide that a stranger can follow
- an automatic dataset download entrypoint with progress bars
- explicit documentation of what is already reproducible and what still needs implementation before claiming full paper reproduction
