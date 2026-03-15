#!/usr/bin/env python3
"""
Main experiment runner for DAFA federated learning experiments.

Usage:
    python run_experiment.py --config configs/experiments/cifar10_fedavg.yaml
    python run_experiment.py --method dafa --dataset cifar10 --num_rounds 100
    python run_experiment.py --method dafa --dataset cifar10 --gamma 1.0 --beta 0.9
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import platform

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import get_cifar10_loaders
from src.data.femnist import get_femnist_loaders
from src.data.shakespeare import get_shakespeare_loaders
from src.models.resnet import ResNet18
from src.models.cnn import TwoLayerCNN, SimpleCNN
from src.models.lstm import ShakespeareLSTM
from src.methods.fedavg import FedAvgAggregator, FedAvgConfig
from src.methods.fedprox import FedProxAggregator, FedProxConfig
from src.methods.scaffold import SCAFFOLDAggregator, SCAFFOLDConfig
from src.methods.fednova import FedNovaAggregator, FedNovaConfig
from src.methods.fedavgm import FedAvgMAggregator, FedAvgMConfig
from src.methods.fedadam import FedAdamAggregator, FedAdamConfig
from src.methods.dafa import DAFAAggregator, DAFAConfig
from src.methods.dir_weight import DirWeightAggregator, DirWeightConfig
from src.core.trainer import FederatedTrainer, TrainerConfig
from src.utils.seed import set_seed
from src.utils.logger import get_logger, setup_logger


logger = get_logger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

DATASET_LOADERS = {
    "cifar10": get_cifar10_loaders,
    "femnist": get_femnist_loaders,
    "shakespeare": get_shakespeare_loaders,
}

MODEL_CREATORS = {
    "resnet18": lambda num_classes: ResNet18(num_classes=num_classes),
    "cnn": lambda num_classes: SimpleCNN(num_classes=num_classes),
    "twolayer_cnn": lambda num_classes: TwoLayerCNN(num_classes=num_classes),
    "lstm": lambda vocab_size: ShakespeareLSTM(vocab_size=vocab_size),
}

AGGREGATOR_CREATORS = {
    "fedavg": lambda config: FedAvgAggregator(FedAvgConfig(**config)),
    "fedprox": lambda config: FedProxAggregator(FedProxConfig(**config)),
    "scaffold": lambda config: SCAFFOLDAggregator(SCAFFOLDConfig(**config)),
    "fednova": lambda config: FedNovaAggregator(FedNovaConfig(**config)),
    "fedavgm": lambda config: FedAvgMAggregator(FedAvgMConfig(**config)),
    "fedadam": lambda config: FedAdamAggregator(FedAdamConfig(**config)),
    "dafa": lambda config: DAFAAggregator(DAFAConfig(**config)),
    "dir_weight": lambda config: DirWeightAggregator(DirWeightConfig(**config)),
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run federated learning experiment")
    
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--method", type=str, default="fedavg",
                       choices=list(AGGREGATOR_CREATORS.keys()),
                       help="Aggregation method")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=list(DATASET_LOADERS.keys()),
                       help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18",
                       help="Model architecture")
    
    parser.add_argument("--num_rounds", type=int, default=100,
                       help="Number of training rounds")
    parser.add_argument("--num_clients", type=int, default=100,
                       help="Number of total clients")
    parser.add_argument("--clients_per_round", type=int, default=10,
                       help="Number of clients per round")
    parser.add_argument("--local_epochs", type=int, default=5,
                       help="Number of local epochs")
    parser.add_argument("--local_lr", type=float, default=0.01,
                       help="Local learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for local training")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Dirichlet concentration parameter for non-IID")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory for datasets (default: project/data)")
    parser.add_argument("--download", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Whether to download dataset if not exists")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers")
    
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    parser.add_argument("--eval_every", type=int, default=10,
                       help="Evaluate every N rounds")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N rounds")
    
    parser.add_argument("--mu", type=float, default=0.01,
                       help="DAFA/Dir-Weight norm threshold or FedProx proximal coefficient")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="DAFA/Dir-Weight temperature parameter")
    parser.add_argument("--beta", type=float, default=0.9,
                       help="DAFA momentum coefficient")
    parser.add_argument("--use_pi_weighting", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Whether to use data size weighting p_i")
    parser.add_argument("--server_lr", type=float, default=1.0,
                       help="Server learning rate")
    parser.add_argument("--server_momentum", type=float, default=0.9,
                       help="Server momentum for FedAvgM")
    
    parser.add_argument("--track_dsnr", action="store_true", default=True,
                       help="Track DSNR metrics")
    parser.add_argument("--track_variance", action="store_true", default=True,
                       help="Track update variance")
    parser.add_argument("--track_convergence", action="store_true", default=True,
                       help="Track convergence speed")
    parser.add_argument("--convergence_threshold", type=float, default=0.8,
                       help="Accuracy threshold for convergence")
    
    parser.add_argument("--use_monitor", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Use real-time monitoring panel instead of progress bar")
    parser.add_argument("--monitor_refresh_rate", type=float, default=0.5,
                       help="Monitor refresh rate in seconds")
    
    parser.add_argument("--use_amp", type=lambda x: x.lower() == 'true',
                       default=True,
                       help="Use automatic mixed precision (AMP) for faster training")
    parser.add_argument("--num_parallel_clients", type=int, default=4,
                       help="Number of clients to train in parallel")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_default_data_dir() -> Path:
    """Get default data directory based on platform."""
    return PROJECT_ROOT / "data"


def get_dataset(args: argparse.Namespace) -> tuple:
    """Get dataset loaders based on arguments."""
    dataset_name = args.dataset
    
    if args.data_root:
        data_dir = Path(args.data_root)
    else:
        data_dir = get_default_data_dir()
    
    if dataset_name == "cifar10":
        cifar10_root = data_dir if "cifar10" in str(data_dir) else data_dir / "cifar10"
        client_loaders, test_loader, data_manager = get_cifar10_loaders(
            root=str(cifar10_root),
            num_clients=args.num_clients,
            alpha=args.alpha,
            batch_size=args.batch_size,
            seed=args.seed,
            download=args.download,
            num_workers=args.num_workers,
        )
        num_classes = 10
    elif dataset_name == "femnist":
        femnist_root = data_dir if "femnist" in str(data_dir) else data_dir / "femnist"
        client_loaders, test_loader, data_manager = get_femnist_loaders(
            root=str(femnist_root),
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            seed=args.seed,
            download=args.download,
            num_workers=args.num_workers,
        )
        num_classes = 62
    elif dataset_name == "shakespeare":
        shakespeare_root = data_dir if "shakespeare" in str(data_dir) else data_dir / "shakespeare"
        client_loaders, test_loader, data_manager = get_shakespeare_loaders(
            root=str(shakespeare_root),
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            seed=args.seed,
            download=args.download,
            num_workers=args.num_workers,
        )
        num_classes = 80
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return client_loaders, test_loader, num_classes


def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """Get model based on name and number of classes."""
    if model_name not in MODEL_CREATORS:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_CREATORS[model_name](num_classes)


def get_aggregator(method: str, args: argparse.Namespace) -> Any:
    """Get aggregator based on method name."""
    config = {
        "device": args.device,
        "use_data_size_weighting": True,
    }
    
    if method == "fedprox":
        config["mu"] = args.mu
    elif method == "scaffold":
        config["server_lr"] = args.server_lr
    elif method == "fedavgm":
        config["server_momentum"] = args.server_momentum
        config["server_lr"] = args.server_lr
    elif method == "fedadam":
        config["server_lr"] = args.server_lr
    elif method == "dafa":
        config["gamma"] = args.gamma
        config["beta"] = args.beta
        config["mu"] = args.mu
        config["use_pi_weighting"] = args.use_pi_weighting
        config["server_lr"] = args.server_lr
    elif method == "dir_weight":
        config["gamma"] = args.gamma
        config["mu"] = args.mu
        config["use_pi_weighting"] = args.use_pi_weighting
        config["server_lr"] = args.server_lr
    
    return AGGREGATOR_CREATORS[method](config)


def create_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """Create trainer configuration from arguments."""
    return TrainerConfig(
        num_rounds=args.num_rounds,
        num_clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        local_batch_size=args.batch_size,
        server_lr=args.server_lr,
        eval_every=args.eval_every,
        save_every=args.save_every,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        log_dir=args.output_dir,
        results_dir=args.output_dir,
        track_dsnr=args.track_dsnr,
        track_variance=args.track_variance,
        track_convergence_speed=args.track_convergence,
        convergence_threshold=args.convergence_threshold,
        use_monitor=args.use_monitor,
        monitor_refresh_rate=args.monitor_refresh_rate,
        use_amp=args.use_amp,
        num_parallel_clients=args.num_parallel_clients,
    )


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the federated learning experiment."""
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.dataset}_{args.method}_{timestamp}"
        output_dir = output_dir / experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = output_dir.name
    
    setup_logger(
        log_file=str(output_dir / "experiment.log"),
        level="DEBUG" if args.verbose else "INFO",
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Arguments: {vars(args)}")
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Loading dataset...")
    client_loaders, test_loader, num_classes = get_dataset(args)
    logger.info(f"Dataset loaded: {len(client_loaders)} clients")
    
    logger.info("Creating model...")
    model = get_model(args.model, num_classes)
    model = model.to(args.device)
    logger.info(f"Model created: {args.model}")
    
    logger.info("Creating aggregator...")
    aggregator = get_aggregator(args.method, args)
    logger.info(f"Aggregator created: {args.method}")
    
    if args.method in ["dafa", "dir_weight"]:
        logger.info(f"  - gamma: {args.gamma}")
        logger.info(f"  - mu: {args.mu}")
        logger.info(f"  - use_pi_weighting: {args.use_pi_weighting}")
        if args.method == "dafa":
            logger.info(f"  - beta: {args.beta}")
    
    trainer_config = create_trainer_config(args)
    trainer_config.checkpoint_dir = str(output_dir / "checkpoints")
    
    logger.info("Initializing trainer...")
    trainer = FederatedTrainer(
        model=model,
        aggregator=aggregator,
        client_loaders=client_loaders,
        test_loader=test_loader,
        config=trainer_config,
    )
    
    logger.info("Starting training...")
    results = trainer.train()
    
    results_path = output_dir / "results.json"
    trainer.save_results(str(results_path))
    
    if args.track_dsnr and results.get("dsnr_summary"):
        logger.info(f"DSNR summary: {results['dsnr_summary']}")
    
    if results.get("convergence_round"):
        logger.info(f"Convergence reached at round: {results['convergence_round']}")
    
    logger.info(f"Experiment completed. Results saved to {results_path}")
    logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    if not torch.cuda.is_available() and args.device == "cuda":
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    if IS_LINUX and args.num_workers == 0:
        if args.num_parallel_clients > 1:
            args.num_workers = 0
            logger.info("Parallel client training enabled, setting num_workers=0 to avoid resource conflicts")
        else:
            args.num_workers = 4
    
    results = run_experiment(args)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Total rounds: {results['final_round']}")
    print(f"Total time: {results['total_time']:.2f}s")
    
    if results.get("convergence_round"):
        print(f"Convergence round: {results['convergence_round']}")
    
    if results.get("dsnr_summary"):
        print(f"DSNR mean: {results['dsnr_summary']['mean']:.4f}")


if __name__ == "__main__":
    main()
