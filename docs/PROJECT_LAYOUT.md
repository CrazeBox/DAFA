# Project Layout

This repository is organized around four public entrypoints:

- `scripts/run_experiment.py`
- `scripts/run_five_stages.py`
- `scripts/download_datasets.py`
- `scripts/analyze_results.py`

## Top-level tree

```text
DAFA/
  README.md
  EXPERIMENT_DESIGN.md
  requirements.txt
  setup.py
  LICENSE

  configs/
    base_config.yaml
    best_hyperparams.yaml
    datasets/
    methods/

  data/
    README.md

  docs/
    PROJECT_LAYOUT.md
    REPRODUCTION_GUIDE.md
    LINUX_SERVER_EXPERIMENT_RUNBOOK.md

  examples/
    quick_start.py

  scripts/
    __init__.py
    analyze_results.py
    download_datasets.py
    run_experiment.py
    run_experiment.sh
    run_experiment_5e.py
    run_five_stages.py
    run_quick.sh
    setup_env.sh
    study_pipeline.sh

  src/
    analysis/
    core/
    data/
    methods/
    models/
    monitor/
    utils/

  tests/
```

## Release expectations

- `results/` is generated locally and is not part of the published source tree.
- downloaded datasets under `data/` are local runtime assets and are not part of the published source tree.
- `venv/`, logs, checkpoints, and other machine-local artifacts must stay untracked.

## Rule of thumb

If a new file does not directly help a user:

- install the project
- download data
- run experiments
- analyze results

then it should not live at the top level of the repository.
