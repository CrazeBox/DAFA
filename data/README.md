# Data Directory

This directory stores the datasets required by the DAFA experiments.

For a fresh Linux setup, use the built-in downloader instead of downloading files manually:

```bash
source venv/bin/activate
python scripts/download_datasets.py --datasets cifar10,femnist,shakespeare
```

The downloader shows terminal progress bars and writes the datasets to:

- `data/cifar10/`
- `data/femnist/train/all_data.json`
- `data/femnist/test/all_data.json`
- `data/shakespeare/train/all_data.json`
- `data/shakespeare/test/all_data.json`

Notes:

- `CIFAR-10` is handled by `torchvision`.
- `FEMNIST` and `Shakespeare` are pulled from the public LEAF-style JSON sources configured in the data loaders.
- if those upstream URLs move, use `scripts/download_datasets.py --femnist_base_url ... --shakespeare_base_url ...`.
- `--allow_synthetic_data` is only for debugging and must not be used for formal reproduction.
