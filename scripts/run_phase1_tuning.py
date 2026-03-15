#!/usr/bin/env python3
"""
Phase 1: Automated Hyperparameter Tuning Script

This script performs systematic hyperparameter tuning for all baseline methods
as specified in EXPERIMENT_DESIGN.md Phase 1.

Usage:
    python scripts/run_phase1_tuning.py --dataset cifar10 --alpha 0.5
    python scripts/run_phase1_tuning.py --dataset cifar10 --alpha 0.1
    python scripts/run_phase1_tuning.py --dataset femnist
    python scripts/run_phase1_tuning.py --all  # Run all datasets
"""

import argparse
import os
import sys
import json
import yaml
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import itertools

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TuningConfig:
    method: str
    params: Dict[str, List[Any]]
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[Tuple[str, str]] = None


PHASE1_TUNING_CONFIGS = {
    "fedavg": TuningConfig(
        method="fedavg",
        params={"local_lr": [0.1, 0.01, 0.001]},
    ),
    "fedprox": TuningConfig(
        method="fedprox",
        params={"mu": [0.001, 0.01, 0.1]},
        depends_on=("fedavg", "local_lr"),
    ),
    "scaffold": TuningConfig(
        method="scaffold",
        params={
            "local_lr": [0.1, 0.01, 0.001],
            "server_lr": [0.5, 1.0, 2.0],
        },
    ),
    "fednova": TuningConfig(
        method="fednova",
        params={"local_lr": [0.1, 0.01, 0.001]},
    ),
    "fedavgm": TuningConfig(
        method="fedavgm",
        params={
            "local_lr": [0.1, 0.01, 0.001],
            "server_momentum": [0.5, 0.7, 0.9, 0.99],
        },
    ),
    "fedadam": TuningConfig(
        method="fedadam",
        params={
            "local_lr": [0.1, 0.01, 0.001],
            "server_lr": [0.01, 0.1, 0.3],
        },
    ),
    "dir_weight": TuningConfig(
        method="dir_weight",
        params={
            "local_lr": [0.1, 0.01, 0.001],
            "gamma": [0.5, 1.0, 2.0],
        },
        fixed_params={"beta": 0.0},
    ),
    "dafa": TuningConfig(
        method="dafa",
        params={"local_lr": [0.1, 0.01, 0.001]},
        fixed_params={
            "gamma": 1.0,
            "beta": 0.9,
            "mu": 0.01,
            "use_pi_weighting": True,
        },
    ),
}

DATASET_CONFIGS = {
    "cifar10_0.5": {"dataset": "cifar10", "alpha": 0.5, "num_rounds": 50},
    "cifar10_0.1": {"dataset": "cifar10", "alpha": 0.1, "num_rounds": 50},
    "femnist": {"dataset": "femnist", "num_rounds": 50},
    "shakespeare": {"dataset": "shakespeare", "num_rounds": 50},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Automated Hyperparameter Tuning")
    
    parser.add_argument("--dataset", type=str, choices=["cifar10", "femnist", "shakespeare"],
                       help="Dataset to tune on")
    parser.add_argument("--alpha", type=float, help="Dirichlet alpha for CIFAR-10")
    parser.add_argument("--all", action="store_true", help="Run all dataset configurations")
    
    parser.add_argument("--methods", type=str, nargs="+",
                       choices=list(PHASE1_TUNING_CONFIGS.keys()),
                       help="Methods to tune (default: all)")
    
    parser.add_argument("--num_rounds", type=int, default=50,
                       help="Number of rounds per trial")
    parser.add_argument("--num_repeats", type=int, default=3,
                       help="Number of repetitions per configuration")
    
    parser.add_argument("--output_dir", type=str, default="results/tuning",
                       help="Output directory for tuning results")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    parser.add_argument("--use_amp", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Use automatic mixed precision for faster training")
    parser.add_argument("--num_parallel_clients", type=int, default=1,
                       help="Number of clients to train in parallel (1 for 4GB GPU)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of DataLoader workers (0 for parallel clients)")
    parser.add_argument("--eval_every", type=int, default=5,
                       help="Evaluate every N rounds (larger = faster)")
    
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without running")
    
    parser.add_argument("--continue_from", type=str, default=None,
                       help="Continue from previous tuning results")
    
    return parser.parse_args()


def generate_param_grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(params.keys())
    values = list(params.values())
    
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))
    
    return grid


def get_best_params_from_results(results_dir: Path, method: str, param_name: str) -> Optional[Any]:
    results_file = results_dir / f"{method}_tuning_summary.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, "r") as f:
        data = json.load(f)
    
    best_config = data.get("summary", {}).get("best_config")
    
    if best_config and param_name in best_config:
        return best_config[param_name]
    
    return None


def run_single_trial(
    method: str,
    dataset: str,
    alpha: Optional[float],
    param_config: Dict[str, Any],
    fixed_params: Dict[str, Any],
    num_rounds: int,
    device: str,
    output_dir: Path,
    trial_id: int,
    use_amp: bool = True,
    num_parallel_clients: int = 1,
    num_workers: int = 4,
    eval_every: int = 5,
) -> Dict[str, Any]:
    trial_name = f"trial_{trial_id}_" + "_".join(
        f"{k}={v}" for k, v in param_config.items()
    )
    trial_dir = output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/run_experiment.py",
        "--method", method,
        "--dataset", dataset,
        "--num_rounds", str(num_rounds),
        "--device", device,
        "--output_dir", str(trial_dir),
        "--use_amp", str(use_amp).lower(),
        "--num_parallel_clients", str(num_parallel_clients),
        "--num_workers", str(num_workers),
        "--eval_every", str(eval_every),
        "--use_monitor", "false",
        "--save_every", str(num_rounds + 1),
    ]
    
    if alpha is not None:
        cmd.extend(["--alpha", str(alpha)])
    
    for key, value in param_config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    for key, value in fixed_params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"Trial {trial_id}: {method} on {dataset}")
    print(f"Params: {param_config}")
    print(f"Fixed: {fixed_params}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    trial_result = {
        "trial_id": trial_id,
        "method": method,
        "dataset": dataset,
        "alpha": alpha,
        "params": param_config,
        "fixed_params": fixed_params,
        "return_code": result.returncode,
        "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
        "stderr": result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr,
    }
    
    if result.returncode != 0:
        print(f"  ERROR: Trial failed with return code {result.returncode}")
        print(f"  STDERR: {result.stderr[-1000:]}")
    
    results_file = trial_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            trial_result["results"] = json.load(f)
    else:
        print(f"  WARNING: No results.json found at {results_file}")
    
    with open(trial_dir / "trial_info.json", "w") as f:
        json.dump(trial_result, f, indent=2, default=str)
    
    return trial_result


def run_method_tuning(
    config: TuningConfig,
    dataset: str,
    alpha: Optional[float],
    num_rounds: int,
    num_repeats: int,
    device: str,
    output_dir: Path,
    previous_results: Optional[Any] = None,
    dry_run: bool = False,
    use_amp: bool = True,
    num_parallel_clients: int = 1,
    num_workers: int = 4,
    eval_every: int = 5,
) -> Dict[str, Any]:
    param_grid = generate_param_grid(config.params)
    
    if config.depends_on:
        dep_method, dep_param = config.depends_on
        best_param = get_best_params_from_results(output_dir, dep_method, dep_param)
        
        if best_param is not None:
            print(f"Using best {dep_param}={best_param} from {dep_method} tuning")
            config.fixed_params[dep_param] = best_param
    
    all_results = []
    trial_id = 0
    
    total_trials = len(param_grid) * num_repeats
    print(f"\n{'#'*60}")
    print(f"# Method: {config.method}")
    print(f"# Dataset: {dataset}, Alpha: {alpha}")
    print(f"# Total trials: {total_trials}")
    print(f"# Parameter grid: {len(param_grid)} configs × {num_repeats} repeats")
    print(f"{'#'*60}\n")
    
    for param_config in param_grid:
        for repeat in range(num_repeats):
            trial_id += 1
            
            if dry_run:
                print(f"[DRY RUN] Would run trial {trial_id}/{total_trials}")
                print(f"  Params: {param_config}")
                continue
            
            trial_result = run_single_trial(
                method=config.method,
                dataset=dataset,
                alpha=alpha,
                param_config=param_config,
                fixed_params=config.fixed_params,
                num_rounds=num_rounds,
                device=device,
                output_dir=output_dir / config.method,
                trial_id=trial_id,
                use_amp=use_amp,
                num_parallel_clients=num_parallel_clients,
                num_workers=num_workers,
                eval_every=eval_every,
            )
            
            all_results.append(trial_result)
            
            if trial_result.get("results"):
                acc = trial_result["results"].get("best_accuracy", 0)
                print(f"  Result: accuracy={acc:.4f}")
    
    summary = summarize_results(all_results)
    
    tuning_result = {
        "method": config.method,
        "dataset": dataset,
        "alpha": alpha,
        "num_rounds": num_rounds,
        "num_repeats": num_repeats,
        "param_grid": config.params,
        "fixed_params": config.fixed_params,
        "summary": summary,
        "all_results": all_results,
    }
    
    with open(output_dir / f"{config.method}_tuning_summary.json", "w") as f:
        json.dump(tuning_result, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Method {config.method} tuning complete")
    print(f"Best config: {summary['best_config']}")
    print(f"Best accuracy: {summary['best_accuracy']:.4f} ± {summary['best_std']:.4f}")
    print(f"{'='*60}\n")
    
    return tuning_result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_results = [r for r in results if r.get("results")]
    
    if not valid_results:
        return {"best_config": None, "best_accuracy": 0.0, "best_std": 0.0}
    
    param_groups = {}
    for result in valid_results:
        params_key = str(sorted(result["params"].items()))
        if params_key not in param_groups:
            param_groups[params_key] = []
        param_groups[params_key].append(result)
    
    config_summaries = []
    for params_key, group in param_groups.items():
        accuracies = [
            r["results"].get("best_accuracy", 0)
            for r in group
        ]
        
        mean_acc = sum(accuracies) / len(accuracies)
        std_acc = (
            sum((a - mean_acc)**2 for a in accuracies) / len(accuracies)
        )**0.5 if len(accuracies) > 1 else 0
        
        config_summaries.append({
            "params": group[0]["params"],
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "num_trials": len(group),
        })
    
    config_summaries.sort(key=lambda x: x["mean_accuracy"], reverse=True)
    
    best = config_summaries[0] if config_summaries else None
    
    return {
        "best_config": best["params"] if best else None,
        "best_accuracy": best["mean_accuracy"] if best else 0.0,
        "best_std": best["std_accuracy"] if best else 0.0,
        "all_configs": config_summaries,
    }


def run_phase1_tuning(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.all:
        dataset_configs = DATASET_CONFIGS
    elif args.dataset == "cifar10" and args.alpha:
        config_name = f"cifar10_{args.alpha}"
        dataset_configs = {config_name: DATASET_CONFIGS[config_name]}
    elif args.dataset:
        dataset_configs = {args.dataset: DATASET_CONFIGS.get(args.dataset, {
            "dataset": args.dataset,
            "num_rounds": args.num_rounds
        })}
    else:
        print("Error: Must specify --dataset or --all")
        return
    
    methods = args.methods if args.methods else list(PHASE1_TUNING_CONFIGS.keys())
    
    all_tuning_results = {}
    
    for config_name, dataset_config in dataset_configs.items():
        print(f"\n{'#'*70}")
        print(f"# Dataset Configuration: {config_name}")
        print(f"{'#'*70}")
        
        output_dir = Path(args.output_dir) / config_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = dataset_config["dataset"]
        alpha = dataset_config.get("alpha")
        num_rounds = dataset_config.get("num_rounds", args.num_rounds)
        
        for method in methods:
            if method not in PHASE1_TUNING_CONFIGS:
                print(f"Warning: Unknown method {method}, skipping")
                continue
            
            config = PHASE1_TUNING_CONFIGS[method]
            
            result = run_method_tuning(
                config=config,
                dataset=dataset,
                alpha=alpha,
                num_rounds=num_rounds,
                num_repeats=args.num_repeats,
                device=args.device,
                output_dir=output_dir,
                dry_run=args.dry_run,
                use_amp=args.use_amp,
                num_parallel_clients=args.num_parallel_clients,
                num_workers=args.num_workers,
                eval_every=args.eval_every,
            )
            
            all_tuning_results[f"{config_name}_{method}"] = result
    
    final_summary = {
        "timestamp": timestamp,
        "dataset_configs": dataset_configs,
        "methods": methods,
        "num_repeats": args.num_repeats,
        "results": {
            name: {
                "best_config": r["summary"]["best_config"],
                "best_accuracy": r["summary"]["best_accuracy"],
                "best_std": r["summary"]["best_std"],
            }
            for name, r in all_tuning_results.items()
        }
    }
    
    final_file = Path(args.output_dir) / f"phase1_summary_{timestamp}.json"
    with open(final_file, "w") as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    print(f"\n{'#'*70}")
    print("# PHASE 1 TUNING COMPLETE")
    print(f"{'#'*70}")
    print(f"\nFinal summary saved to: {final_file}")
    print("\nBest configurations:")
    for name, r in final_summary["results"].items():
        print(f"  {name}: {r['best_config']} -> {r['best_accuracy']:.4f}")


def main():
    args = parse_args()
    run_phase1_tuning(args)


if __name__ == "__main__":
    main()
