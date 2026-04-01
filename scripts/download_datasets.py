#!/usr/bin/env python3
"""Download DAFA datasets for a fresh Linux environment."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEMNIST_BASE_URL = "https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/data/femnist"
SHAKESPEARE_BASE_URL = "https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/data/shakespeare"


def download_url_with_progress(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
        ) as progress:
            with open(destination, "wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets required by DAFA experiments")
    parser.add_argument(
        "--datasets",
        type=str,
        default="cifar10,femnist,shakespeare",
        help="Comma-separated list from cifar10,femnist,shakespeare",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root directory where dataset folders will be created",
    )
    parser.add_argument(
        "--allow_synthetic_data",
        action="store_true",
        help="Allow synthetic fallback for FEMNIST/Shakespeare if download fails",
    )
    parser.add_argument(
        "--femnist_base_url",
        type=str,
        default=FEMNIST_BASE_URL,
        help="Base URL containing FEMNIST train/test all_data.json files",
    )
    parser.add_argument(
        "--shakespeare_base_url",
        type=str,
        default=SHAKESPEARE_BASE_URL,
        help="Base URL containing Shakespeare train/test all_data.json files",
    )
    return parser.parse_args()


def download_cifar10(data_root: Path) -> None:
    root = data_root / "cifar10"
    print(f"[cifar10] target={root}")
    if (root / "cifar-10-batches-py").exists():
        print("[cifar10] ready")
        return
    from src.data.cifar10 import CIFAR10Dataset

    CIFAR10Dataset(root=str(root), train=True, download=True)
    CIFAR10Dataset(root=str(root), train=False, download=True)
    print("[cifar10] ready")


def download_femnist(data_root: Path, base_url: str, allow_synthetic_data: bool) -> None:
    root = data_root / "femnist"
    print(f"[femnist] target={root}")
    for split in ["train", "test"]:
        target = root / split / "all_data.json"
        if not target.exists():
            url = f"{base_url.rstrip('/')}/{split}/all_data.json"
            download_url_with_progress(url, target)
    print("[femnist] ready")


def download_shakespeare(data_root: Path, base_url: str, allow_synthetic_data: bool) -> None:
    root = data_root / "shakespeare"
    print(f"[shakespeare] target={root}")
    for split in ["train", "test"]:
        target = root / split / "all_data.json"
        if not target.exists():
            url = f"{base_url.rstrip('/')}/{split}/all_data.json"
            download_url_with_progress(url, target)
    print("[shakespeare] ready")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    requested = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    valid = {"cifar10", "femnist", "shakespeare"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    if "cifar10" in requested:
        download_cifar10(data_root)
    if "femnist" in requested:
        download_femnist(data_root, args.femnist_base_url, args.allow_synthetic_data)
    if "shakespeare" in requested:
        download_shakespeare(data_root, args.shakespeare_base_url, args.allow_synthetic_data)

    print("all requested datasets are ready")


if __name__ == "__main__":
    main()
