# DAFA: Directionally Aligned Federated Aggregation

PyTorch implementation of DAFA for federated learning under heterogeneous data distributions.

## Main entrypoints

The repository exposes one canonical path per task:

- `scripts/run_experiment.py`: run a single experiment
- `scripts/run_five_stages.py`: run the staged paper pipeline
- `scripts/download_datasets.py`: download datasets with progress bars
- `scripts/analyze_results.py`: summarize runs, compare runs, and generate plots

## Quick Start

```bash
git clone https://github.com/your-username/DAFA.git
cd DAFA

bash scripts/setup_env.sh --profile basic
source venv/bin/activate

python scripts/download_datasets.py --datasets cifar10,femnist,shakespeare

bash scripts/run_quick.sh
```

## Single Experiment

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset cifar10 \
  --alpha 0.1 \
  --device cuda
```

## Staged Pipeline

Run the paper-style staged pipeline:

```bash
python scripts/run_five_stages.py \
  --stages all \
  --device cuda \
  --num_rounds 200 \
  --seeds 42,123,456,777,1024
```

Stage 1 only:

```bash
python scripts/run_five_stages.py \
  --stages 1 \
  --device cuda \
  --num_rounds 100 \
  --seeds 42,123,456
```

Optional shell wrapper:

```bash
bash scripts/study_pipeline.sh results/study_pipeline cuda 42,123,456 100
```

## Result Summaries and Plots

Build summarized run tables:

```bash
python scripts/analyze_results.py select-best \
  --results_root results \
  --output_dir results/summary
```

Generate plots:

```bash
python scripts/analyze_results.py plot \
  --best_runs results/summary/best_runs.json \
  --output_dir results/summary/plots \
  --format pdf
```

Compare multiple finished runs directly:

```bash
python scripts/analyze_results.py compare \
  --inputs results/run_a results/run_b results/run_c \
  --output_dir results/compare_ablation \
  --format png
```

## Datasets

Supported datasets:

- `CIFAR-10`
- `FEMNIST`
- `Shakespeare`

Use the built-in downloader:

```bash
python scripts/download_datasets.py --datasets cifar10,femnist,shakespeare
```

If the public mirror for `FEMNIST/Shakespeare` changes, override it:

```bash
python scripts/download_datasets.py \
  --datasets femnist,shakespeare \
  --femnist_base_url <stable-femnist-mirror> \
  --shakespeare_base_url <stable-shakespeare-mirror>
```

## Default paper settings

- `CIFAR-10`: `ResNet-18`, 100 clients, 10 sampled, `K=5`, batch size `32`, `T=200`
- `FEMNIST`: `TwoLayerCNN`, 200 clients, 20 sampled, `K=10`, batch size `32`, `T=200`
- `Shakespeare`: 2-layer `LSTM`, `hidden=256`, `embedding=200`, 100 clients, 10 sampled, `K=2`, batch size `32`, `T=200`

## Project structure

```text
scripts/
  setup_env.sh
  run_quick.sh
  run_experiment.py
  run_five_stages.py
  download_datasets.py
  analyze_results.py
  study_pipeline.sh

src/
  methods/
  models/
  data/
  core/
  analysis/
  utils/

docs/
  REPRODUCTION_GUIDE.md

configs/
results/
```

## Reproduction

The main documentation is:

- [EXPERIMENT_DESIGN.md](e:\AIProject\FedJD\DAFA\EXPERIMENT_DESIGN.md)
- [REPRODUCTION_GUIDE.md](e:\AIProject\FedJD\DAFA\docs\REPRODUCTION_GUIDE.md)
- [PROJECT_LAYOUT.md](e:\AIProject\FedJD\DAFA\docs\PROJECT_LAYOUT.md)

## License

MIT. See [LICENSE](LICENSE).
