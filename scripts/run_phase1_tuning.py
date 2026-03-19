#!/usr/bin/env python3
"""
Phase-1 tuning entrypoint.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-1 tuning experiments")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/phase1_tuning")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_repeats", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    for repeat in range(args.num_repeats):
        output_dir = Path(args.output_dir) / f"repeat_{repeat + 1}"
        cmd = [
            sys.executable,
            str(root / "scripts" / "run_all_experiments.py"),
            "--experiment",
            "sensitivity",
            "--output_dir",
            str(output_dir),
            "--seed",
            str(42 + repeat),
            "--dataset",
            args.dataset,
            "--alpha",
            str(args.alpha),
            "--num_rounds",
            str(args.num_rounds),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
