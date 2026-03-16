#!/usr/bin/env python3
"""
Run Experiment 5E: Federated System Parameter Robustness

This script runs experiments to validate DAFA's robustness under different
local steps (K) and participation fractions (C).

Reference: EXPERIMENT_DESIGN.md - Section 8.6

Usage:
    python scripts/run_experiment_5e.py --setting local_steps
    python scripts/run_experiment_5e.py --setting participation
    python scripts/run_experiment_5e.py --setting all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


LOCAL_STEPS_CONFIGS = [
    {"K": 1, "C": 0.1},
    {"K": 5, "C": 0.1},
    {"K": 10, "C": 0.1},
    {"K": 20, "C": 0.1},
]

PARTICIPATION_CONFIGS = [
    {"K": 5, "C": 0.05},
    {"K": 5, "C": 0.1},
    {"K": 5, "C": 0.2},
]

METHODS = ["fedavg", "dafa"]
SEEDS = [42, 123, 456]
DATASET = "cifar10"
ALPHA = 0.1
NUM_ROUNDS = 100


def run_single_experiment(
    method: str,
    local_epochs: int,
    client_fraction: float,
    seed: int,
    output_dir: Path,
) -> bool:
    """Run a single experiment configuration."""
    
    num_clients_per_round = int(100 * client_fraction)
    
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        f"--method={method}",
        f"--dataset={DATASET}",
        f"--alpha={ALPHA}",
        f"--num_rounds={NUM_ROUNDS}",
        f"--local_epochs={local_epochs}",
        f"--clients_per_round={num_clients_per_round}",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
        "--track_fairness",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {method} | K={local_epochs} | C={client_fraction} | seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        return False


def run_local_steps_experiments(output_dir: Path) -> dict:
    """Run experiments varying local steps (K)."""
    
    results = {}
    
    for config in LOCAL_STEPS_CONFIGS:
        K = config["K"]
        C = config["C"]
        
        for method in METHODS:
            for seed in SEEDS:
                exp_name = f"local_steps_K{K}_C{C}_{method}_seed{seed}"
                exp_output_dir = output_dir / exp_name
                
                success = run_single_experiment(
                    method=method,
                    local_epochs=K,
                    client_fraction=C,
                    seed=seed,
                    output_dir=exp_output_dir,
                )
                
                key = f"K{K}_{method}"
                if key not in results:
                    results[key] = []
                results[key].append(success)
    
    return results


def run_participation_experiments(output_dir: Path) -> dict:
    """Run experiments varying participation fraction (C)."""
    
    results = {}
    
    for config in PARTICIPATION_CONFIGS:
        K = config["K"]
        C = config["C"]
        
        for method in METHODS:
            for seed in SEEDS:
                exp_name = f"participation_K{K}_C{C}_{method}_seed{seed}"
                exp_output_dir = output_dir / exp_name
                
                success = run_single_experiment(
                    method=method,
                    local_epochs=K,
                    client_fraction=C,
                    seed=seed,
                    output_dir=exp_output_dir,
                )
                
                key = f"C{C}_{method}"
                if key not in results:
                    results[key] = []
                results[key].append(success)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5E")
    parser.add_argument(
        "--setting",
        type=str,
        choices=["local_steps", "participation", "all"],
        default="all",
        help="Which experiment setting to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiment_5e",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    
    print(f"\n{'#'*60}")
    print(f"# Experiment 5E: Federated System Parameter Robustness")
    print(f"# Setting: {args.setting}")
    print(f"# Output: {output_dir}")
    print(f"# Timestamp: {timestamp}")
    print(f"{'#'*60}\n")
    
    all_results = {}
    
    if args.setting in ["local_steps", "all"]:
        print("\n>>> Running Local Steps (K) Experiments <<<\n")
        local_steps_results = run_local_steps_experiments(output_dir / "local_steps")
        all_results.update(local_steps_results)
    
    if args.setting in ["participation", "all"]:
        print("\n>>> Running Participation Fraction (C) Experiments <<<\n")
        participation_results = run_participation_experiments(output_dir / "participation")
        all_results.update(participation_results)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 5E RESULTS SUMMARY")
    print("=" * 60)
    
    total = 0
    successful = 0
    
    for key, results in all_results.items():
        success_count = sum(results)
        total_count = len(results)
        total += total_count
        successful += success_count
        print(f"{key}: {success_count}/{total_count} successful")
    
    print(f"\nTotal: {successful}/{total} experiments successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
