#!/usr/bin/env python3
"""
Batch experiment runner for comprehensive DAFA evaluation.

This script runs all experiments specified in EXPERIMENT_DESIGN.md:
1. Baseline comparison (Table 1)
2. Hyperparameter sensitivity (Figure 4)
3. Ablation study (Table 2)
4. DSNR analysis (Figure 3)
5. Convergence speed analysis

Usage:
    python run_all_experiments.py --experiment baseline
    python run_all_experiments.py --experiment sensitivity
    python run_all_experiments.py --experiment ablation
    python run_all_experiments.py --experiment all
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from itertools import product
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))


BASE_CONFIG = {
    "num_rounds": 200,
    "num_clients": 100,
    "clients_per_round": 10,
    "local_epochs": 5,
    "batch_size": 32,
    "seed": 42,
    "device": "cuda",
    "eval_every": 10,
    "save_every": 50,
    "track_dsnr": True,
    "track_variance": True,
    "track_convergence": True,
    "convergence_threshold": 0.8,
}

DATASETS = ["cifar10", "femnist", "shakespeare"]

BASELINE_METHODS = ["fedavg", "fedprox", "scaffold", "fednova", "fedavgm", "fedadam", "dafa", "dir_weight"]

NON_IID_LEVELS = [0.1, 0.5, 1.0]

GAMMA_VALUES = [0.5, 1.0, 2.0, 5.0]
BETA_VALUES = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
MU_VALUES = [0.001, 0.01, 0.1]


def get_model_for_dataset(dataset: str) -> str:
    """Get appropriate model for dataset."""
    model_map = {
        "cifar10": "resnet18",
        "femnist": "twolayer_cnn",
        "shakespeare": "lstm",
    }
    return model_map.get(dataset, "resnet18")


def get_lr_for_dataset(dataset: str) -> float:
    """Get default learning rate for dataset."""
    lr_map = {
        "cifar10": 0.01,
        "femnist": 0.01,
        "shakespeare": 0.01,
    }
    return lr_map.get(dataset, 0.01)


def run_single_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment with given configuration."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_experiment.py"),
    ]
    
    for key, value in config.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}")
            cmd.append("true" if value else "false")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    cmd.extend(["--output_dir", str(output_dir)])
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return {"status": "failed", "error": result.stderr}
    
    return {"status": "success", "output": result.stdout}


def run_baseline_experiments(output_dir: Path) -> List[Dict[str, Any]]:
    """Run baseline comparison experiments (Table 1)."""
    results = []
    
    for dataset in DATASETS:
        for method in BASELINE_METHODS:
            for alpha in NON_IID_LEVELS:
                config = BASE_CONFIG.copy()
                config.update({
                    "dataset": dataset,
                    "method": method,
                    "model": get_model_for_dataset(dataset),
                    "local_lr": get_lr_for_dataset(dataset),
                    "alpha": alpha,
                })
                
                if method == "dafa":
                    config.update({
                        "gamma": 1.0,
                        "beta": 0.9,
                        "mu": 0.01,
                    })
                elif method == "dir_weight":
                    config.update({
                        "gamma": 1.0,
                        "mu": 0.01,
                    })
                elif method == "fedprox":
                    config["mu"] = 0.01
                
                exp_name = f"baseline_{dataset}_{method}_alpha{alpha}"
                exp_dir = output_dir / exp_name
                
                print(f"\n{'='*60}")
                print(f"Running baseline experiment: {exp_name}")
                print(f"{'='*60}")
                
                result = run_single_experiment(config, exp_dir)
                result["experiment"] = exp_name
                result["config"] = config
                results.append(result)
    
    return results


def run_sensitivity_experiments(output_dir: Path) -> List[Dict[str, Any]]:
    """Run hyperparameter sensitivity experiments (Figure 4)."""
    results = []
    
    for dataset in ["cifar10"]:
        base_config = BASE_CONFIG.copy()
        base_config.update({
            "dataset": dataset,
            "method": "dafa",
            "model": get_model_for_dataset(dataset),
            "local_lr": get_lr_for_dataset(dataset),
            "alpha": 0.5,
        })
        
        print(f"\n{'='*60}")
        print("Running gamma sensitivity experiments")
        print(f"{'='*60}")
        
        for gamma in GAMMA_VALUES:
            config = base_config.copy()
            config.update({
                "gamma": gamma,
                "beta": 0.9,
                "mu": 0.01,
            })
            
            exp_name = f"sensitivity_gamma_{gamma}"
            exp_dir = output_dir / exp_name
            
            result = run_single_experiment(config, exp_dir)
            result["experiment"] = exp_name
            result["config"] = config
            results.append(result)
        
        print(f"\n{'='*60}")
        print("Running beta sensitivity experiments")
        print(f"{'='*60}")
        
        for beta in BETA_VALUES:
            config = base_config.copy()
            config.update({
                "gamma": 1.0,
                "beta": beta,
                "mu": 0.01,
            })
            
            exp_name = f"sensitivity_beta_{beta}"
            exp_dir = output_dir / exp_name
            
            result = run_single_experiment(config, exp_dir)
            result["experiment"] = exp_name
            result["config"] = config
            results.append(result)
        
        print(f"\n{'='*60}")
        print("Running mu sensitivity experiments")
        print(f"{'='*60}")
        
        for mu in MU_VALUES:
            config = base_config.copy()
            config.update({
                "gamma": 1.0,
                "beta": 0.9,
                "mu": mu,
            })
            
            exp_name = f"sensitivity_mu_{mu}"
            exp_dir = output_dir / exp_name
            
            result = run_single_experiment(config, exp_dir)
            result["experiment"] = exp_name
            result["config"] = config
            results.append(result)
    
    return results


def run_ablation_experiments(output_dir: Path) -> List[Dict[str, Any]]:
    """Run ablation study experiments (Table 2)."""
    results = []
    
    ablation_configs = [
        {
            "name": "dafa_full",
            "method": "dafa",
            "gamma": 1.0,
            "beta": 0.9,
            "mu": 0.01,
            "use_pi_weighting": True,
        },
        {
            "name": "dafa_no_pi",
            "method": "dafa",
            "gamma": 1.0,
            "beta": 0.9,
            "mu": 0.01,
            "use_pi_weighting": False,
        },
        {
            "name": "dafa_no_momentum",
            "method": "dir_weight",
            "gamma": 1.0,
            "mu": 0.01,
            "use_pi_weighting": True,
        },
        {
            "name": "dafa_no_momentum_no_pi",
            "method": "dir_weight",
            "gamma": 1.0,
            "mu": 0.01,
            "use_pi_weighting": False,
        },
        {
            "name": "fedavg_baseline",
            "method": "fedavg",
        },
    ]
    
    for dataset in DATASETS:
        for ablation_config in ablation_configs:
            config = BASE_CONFIG.copy()
            config.update({
                "dataset": dataset,
                "model": get_model_for_dataset(dataset),
                "local_lr": get_lr_for_dataset(dataset),
                "alpha": 0.5,
            })
            config.update(ablation_config)
            del config["name"]
            
            exp_name = f"ablation_{dataset}_{ablation_config['name']}"
            exp_dir = output_dir / exp_name
            
            print(f"\n{'='*60}")
            print(f"Running ablation experiment: {exp_name}")
            print(f"{'='*60}")
            
            result = run_single_experiment(config, exp_dir)
            result["experiment"] = exp_name
            result["config"] = config
            results.append(result)
    
    return results


def run_dsnr_analysis(output_dir: Path) -> List[Dict[str, Any]]:
    """Run DSNR analysis experiments (Figure 3)."""
    results = []
    
    methods = ["fedavg", "dafa"]
    
    for dataset in ["cifar10"]:
        for method in methods:
            for alpha in [0.1, 0.5, 1.0]:
                config = BASE_CONFIG.copy()
                config.update({
                    "dataset": dataset,
                    "method": method,
                    "model": get_model_for_dataset(dataset),
                    "local_lr": get_lr_for_dataset(dataset),
                    "alpha": alpha,
                    "track_dsnr": True,
                })
                
                if method == "dafa":
                    config.update({
                        "gamma": 1.0,
                        "beta": 0.9,
                        "mu": 0.01,
                    })
                
                exp_name = f"dsnr_{method}_alpha{alpha}"
                exp_dir = output_dir / exp_name
                
                print(f"\n{'='*60}")
                print(f"Running DSNR analysis: {exp_name}")
                print(f"{'='*60}")
                
                result = run_single_experiment(config, exp_dir)
                result["experiment"] = exp_name
                result["config"] = config
                results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["baseline", "sensitivity", "ablation", "dsnr", "all"],
                       help="Which experiment to run")
    parser.add_argument("--output_dir", type=str, default="results/batch",
                       help="Output directory for all experiments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=DATASETS,
                       help="Override dataset for all experiments")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Override non-IID alpha for all experiments")
    parser.add_argument("--num_rounds", type=int, default=None,
                       help="Override number of rounds")
    parser.add_argument("--device", type=str, default=None,
                       help="Override device for all experiments")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    BASE_CONFIG["seed"] = args.seed
    if args.num_rounds is not None:
        BASE_CONFIG["num_rounds"] = args.num_rounds
    if args.device is not None:
        BASE_CONFIG["device"] = args.device
    if args.dataset is not None:
        DATASETS[:] = [args.dataset]
    if args.alpha is not None:
        NON_IID_LEVELS[:] = [args.alpha]
    
    all_results = {}
    
    if args.experiment in ["baseline", "all"]:
        print("\n" + "="*80)
        print("RUNNING BASELINE EXPERIMENTS")
        print("="*80)
        results = run_baseline_experiments(output_dir / "baseline")
        all_results["baseline"] = results
    
    if args.experiment in ["sensitivity", "all"]:
        print("\n" + "="*80)
        print("RUNNING SENSITIVITY EXPERIMENTS")
        print("="*80)
        results = run_sensitivity_experiments(output_dir / "sensitivity")
        all_results["sensitivity"] = results
    
    if args.experiment in ["ablation", "all"]:
        print("\n" + "="*80)
        print("RUNNING ABLATION EXPERIMENTS")
        print("="*80)
        results = run_ablation_experiments(output_dir / "ablation")
        all_results["ablation"] = results
    
    if args.experiment in ["dsnr", "all"]:
        print("\n" + "="*80)
        print("RUNNING DSNR ANALYSIS EXPERIMENTS")
        print("="*80)
        results = run_dsnr_analysis(output_dir / "dsnr")
        all_results["dsnr"] = results
    
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    
    total = sum(len(r) for r in all_results.values())
    success = sum(1 for r in all_results.values() for exp in r if exp.get("status") == "success")
    print(f"Total experiments: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {total - success}")


if __name__ == "__main__":
    main()
