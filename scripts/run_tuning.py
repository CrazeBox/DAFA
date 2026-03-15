#!/usr/bin/env python3
"""
Hyperparameter tuning script for federated learning methods.

Usage:
    python run_tuning.py --method fedprox --dataset cifar10 --param mu --values 0.001 0.01 0.1
    python run_tuning.py --config configs/tuning/fedprox_tuning.yaml
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import itertools
import subprocess

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for FL methods")
    
    parser.add_argument("--config", type=str, help="Path to tuning config file")
    parser.add_argument("--method", type=str, required=True,
                       help="Aggregation method to tune")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset name")
    
    parser.add_argument("--param", type=str, action="append",
                       help="Parameter name to tune (can specify multiple)")
    parser.add_argument("--values", type=float, action="append",
                       help="Parameter values to try (for single param)")
    
    parser.add_argument("--num_rounds", type=int, default=50,
                       help="Number of rounds per trial")
    parser.add_argument("--num_repeats", type=int, default=3,
                       help="Number of repetitions per configuration")
    
    parser.add_argument("--base_config", type=str, default=None,
                       help="Base configuration file")
    
    parser.add_argument("--output_dir", type=str, default="tuning_results",
                       help="Output directory for tuning results")
    
    parser.add_argument("--parallel", action="store_true",
                       help="Run trials in parallel")
    parser.add_argument("--num_parallel", type=int, default=4,
                       help="Number of parallel trials")
    
    return parser.parse_args()


def generate_param_grid(
    params: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Generate parameter grid from parameter ranges."""
    keys = list(params.keys())
    values = list(params.values())
    
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))
    
    return grid


def run_single_trial(
    method: str,
    dataset: str,
    param_config: Dict[str, Any],
    base_args: Dict[str, Any],
    trial_id: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single hyperparameter trial."""
    trial_name = f"trial_{trial_id}_" + "_".join(
        f"{k}={v}" for k, v in param_config.items()
    )
    trial_dir = output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/run_experiment.py",
        "--method", method,
        "--dataset", dataset,
        "--output_dir", str(trial_dir),
        "--num_rounds", str(base_args.get("num_rounds", 50)),
    ]
    
    for key, value in param_config.items():
        cmd.extend([f"--{key}", str(value)])
    
    for key, value in base_args.items():
        if key not in ["num_rounds", "method", "dataset"]:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Running trial {trial_id}: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    trial_result = {
        "trial_id": trial_id,
        "params": param_config,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    
    results_file = trial_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            trial_result["results"] = json.load(f)
    
    with open(trial_dir / "trial_info.json", "w") as f:
        json.dump(trial_result, f, indent=2, default=str)
    
    return trial_result


def run_tuning(args: argparse.Namespace) -> Dict[str, Any]:
    """Run hyperparameter tuning."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_name = f"{args.method}_{args.dataset}_{timestamp}"
    output_dir = Path(args.output_dir) / tuning_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tuning_config = {
        "method": args.method,
        "dataset": args.dataset,
        "num_rounds": args.num_rounds,
        "num_repeats": args.num_repeats,
        "timestamp": timestamp,
    }
    
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        param_grid = generate_param_grid(config.get("param_grid", {}))
        tuning_config["param_grid"] = config.get("param_grid", {})
        base_args = config.get("base_args", {})
    else:
        if not args.param or not args.values:
            raise ValueError("Must specify --param and --values or --config")
        
        params = {args.param[0]: args.values}
        param_grid = generate_param_grid(params)
        tuning_config["param_grid"] = params
        base_args = {}
    
    base_args["num_rounds"] = args.num_rounds
    
    all_results = []
    trial_id = 0
    
    for param_config in param_grid:
        for repeat in range(args.num_repeats):
            trial_id += 1
            
            print(f"\n{'='*60}")
            print(f"Trial {trial_id}/{len(param_grid) * args.num_repeats}")
            print(f"Params: {param_config}")
            print(f"Repeat: {repeat + 1}/{args.num_repeats}")
            print(f"{'='*60}\n")
            
            trial_result = run_single_trial(
                method=args.method,
                dataset=args.dataset,
                param_config=param_config,
                base_args=base_args,
                trial_id=trial_id,
                output_dir=output_dir,
            )
            
            all_results.append(trial_result)
    
    summary = summarize_results(all_results)
    
    tuning_result = {
        "config": tuning_config,
        "summary": summary,
        "all_results": all_results,
    }
    
    with open(output_dir / "tuning_summary.json", "w") as f:
        json.dump(tuning_result, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best configuration: {summary['best_config']}")
    print(f"Best accuracy: {summary['best_accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return tuning_result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize tuning results."""
    valid_results = [r for r in results if r.get("results")]
    
    if not valid_results:
        return {"best_config": None, "best_accuracy": 0.0}
    
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
        
        config_summaries.append({
            "params": group[0]["params"],
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "std_accuracy": (
                sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) /
                len(accuracies)
            )**0.5 if len(accuracies) > 1 else 0,
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


def main():
    """Main entry point."""
    args = parse_args()
    
    results = run_tuning(args)
    
    print("\nTuning completed successfully!")


if __name__ == "__main__":
    main()
