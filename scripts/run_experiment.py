#!/usr/bin/env python3
"""
Main experiment runner for DAFA federated learning experiments.

Usage:
    python run_experiment.py --config configs/base_config.yaml
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.logger import get_logger, setup_logger


logger = get_logger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
DATASET_NAMES = ("cifar10", "femnist", "shakespeare")
METHOD_NAMES = ("fedavg", "fedprox", "scaffold", "fednova", "fedavgm", "fedadam", "dafa", "dir_weight")


def parse_bool(value: str) -> bool:
    """Parse boolean values from CLI."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _collect_config_values(config_obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(config_obj, dict):
        for key, value in config_obj.items():
            if isinstance(value, dict):
                _collect_config_values(value, out)
            else:
                out[key] = value

DATASET_PROFILES: Dict[str, Dict[str, Any]] = {
    "cifar10": {
        "model": "resnet18",
        "num_rounds": 200,
        "num_clients": 100,
        "clients_per_round": 10,
        "local_epochs": 5,
        "local_lr": 0.01,
        "batch_size": 32,
        "task_type": "classification",
        "client_lr_scheduler": "cosine",
    },
    "femnist": {
        "model": "twolayer_cnn",
        "num_rounds": 200,
        "num_clients": 200,
        "clients_per_round": 20,
        "local_epochs": 10,
        "local_lr": 0.01,
        "batch_size": 32,
        "task_type": "classification",
        "client_lr_scheduler": "cosine",
    },
    "shakespeare": {
        "model": "lstm",
        "num_rounds": 200,
        "num_clients": 100,
        "clients_per_round": 10,
        "local_epochs": 2,
        "local_lr": 0.01,
        "batch_size": 32,
        "task_type": "language_modeling",
        "client_lr_scheduler": "cosine",
        "seq_length": 80,
        "embedding_dim": 200,
        "hidden_size": 256,
        "num_layers": 2,
    },
}

RUNTIME: Dict[str, Any] = {}


def ensure_runtime_imports() -> None:
    """Import training dependencies lazily so CLI help works without torch."""
    global RUNTIME
    if RUNTIME:
        return

    import torch

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

    RUNTIME = {
        "torch": torch,
        "get_cifar10_loaders": get_cifar10_loaders,
        "get_femnist_loaders": get_femnist_loaders,
        "get_shakespeare_loaders": get_shakespeare_loaders,
        "ShakespeareLSTM": ShakespeareLSTM,
        "FederatedTrainer": FederatedTrainer,
        "TrainerConfig": TrainerConfig,
        "set_seed": set_seed,
        "MODEL_CREATORS": {
            "resnet18": lambda num_classes: ResNet18(num_classes=num_classes),
            "cnn": lambda num_classes: SimpleCNN(num_classes=num_classes),
            "twolayer_cnn": lambda num_classes: TwoLayerCNN(num_classes=num_classes),
            "lstm": lambda vocab_size: ShakespeareLSTM(vocab_size=vocab_size),
        },
        "AGGREGATOR_CREATORS": {
            "fedavg": lambda config: FedAvgAggregator(FedAvgConfig(**config)),
            "fedprox": lambda config: FedProxAggregator(FedProxConfig(**config)),
            "scaffold": lambda config: SCAFFOLDAggregator(SCAFFOLDConfig(**config)),
            "fednova": lambda config: FedNovaAggregator(FedNovaConfig(**config)),
            "fedavgm": lambda config: FedAvgMAggregator(FedAvgMConfig(**config)),
            "fedadam": lambda config: FedAdamAggregator(FedAdamConfig(**config)),
            "dafa": lambda config: DAFAAggregator(DAFAConfig(**config)),
            "dir_weight": lambda config: DirWeightAggregator(DirWeightConfig(**config)),
        },
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run federated learning experiment")
    
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--method", type=str, default="fedavg",
                       choices=list(METHOD_NAMES),
                       help="Aggregation method")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=list(DATASET_NAMES),
                       help="Dataset name")
    parser.add_argument("--model", type=str, default=None,
                       help="Model architecture")
    
    parser.add_argument("--num_rounds", type=int, default=None,
                       help="Number of training rounds")
    parser.add_argument("--num_clients", type=int, default=None,
                       help="Number of total clients")
    parser.add_argument("--clients_per_round", type=int, default=None,
                       help="Number of clients per round")
    parser.add_argument("--local_epochs", type=int, default=None,
                       help="Number of local epochs")
    parser.add_argument("--local_lr", type=float, default=None,
                       help="Local learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for local training")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Dirichlet concentration parameter for non-IID")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory for datasets (default: project/data)")
    parser.add_argument("--download", type=parse_bool,
                       default=True,
                       help="Whether to download dataset if not exists")
    parser.add_argument("--allow_synthetic_data", type=parse_bool,
                       default=False,
                       help="Allow synthetic FEMNIST/Shakespeare fallback data for debugging")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers")
    parser.add_argument("--seq_length", type=int, default=None,
                       help="Sequence length for Shakespeare")
    parser.add_argument("--embedding_dim", type=int, default=None,
                       help="Embedding dimension for Shakespeare LSTM")
    parser.add_argument("--hidden_size", type=int, default=None,
                       help="Hidden size for Shakespeare LSTM")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of LSTM layers for Shakespeare")
    
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--run_group", type=str, default="default",
                       help="Experiment group name for result organization")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Optional explicit run name")
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
    parser.add_argument("--use_pi_weighting", type=parse_bool,
                       default=True,
                       help="Whether to use data size weighting p_i")
    parser.add_argument("--server_lr", type=float, default=1.0,
                       help="Server learning rate")
    parser.add_argument("--server_momentum", type=float, default=0.9,
                       help="Server momentum for FedAvgM")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Beta1 for FedAdam")
    parser.add_argument("--beta2", type=float, default=0.999,
                       help="Beta2 for FedAdam")
    parser.add_argument("--tau", type=float, default=1e-3,
                       help="Tau for FedAdam")
    
    parser.add_argument("--track_dsnr", type=parse_bool, default=True,
                       help="Track DSNR metrics")
    parser.add_argument("--track_variance", type=parse_bool, default=True,
                       help="Track update variance")
    parser.add_argument("--track_convergence", type=parse_bool, default=True,
                       help="Track convergence speed")
    parser.add_argument("--convergence_threshold", type=float, default=0.8,
                       help="Accuracy threshold for convergence")
    parser.add_argument("--track_fairness", type=parse_bool, default=True,
                       help="Track fairness metrics")
    parser.add_argument("--fairness_eval_freq", type=int, default=10,
                       help="Evaluate fairness every N rounds")
    parser.add_argument("--task_type", type=str, default=None,
                       choices=["classification", "language_modeling"],
                       help="Evaluation task type")
    parser.add_argument("--client_lr_scheduler", type=str, default=None,
                       choices=["constant", "cosine"],
                       help="Client learning-rate scheduler")
    parser.add_argument("--cosine_decay_T_max", type=int, default=None,
                       help="T_max for cosine learning-rate decay")
    parser.add_argument("--cosine_decay_eta_min", type=float, default=0.0,
                       help="Minimum learning rate for cosine decay")
    
    parser.add_argument("--use_monitor", type=parse_bool,
                       default=True,
                       help="Use real-time monitoring panel instead of progress bar")
    parser.add_argument("--monitor_refresh_rate", type=float, default=0.5,
                       help="Monitor refresh rate in seconds")
    
    parser.add_argument("--use_amp", type=parse_bool,
                       default=True,
                       help="Use automatic mixed precision (AMP) for faster training")
    parser.add_argument("--num_parallel_clients", type=int, default=4,
                       help="Number of clients to train in parallel")
    parser.add_argument("--malicious_client_fraction", type=float, default=0.0,
                       help="Fraction of selected clients treated as malicious")
    parser.add_argument("--attack_type", type=str, default="none",
                       choices=["none", "reverse", "random"],
                       help="Adversarial attack type for malicious clients")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_config_to_args(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    flattened: Dict[str, Any] = {}
    _collect_config_values(config, flattened)
    aliases: Dict[str, str] = {
        "learning_rate": "local_lr",
        "lr": "local_lr",
        "rounds": "num_rounds",
        "eval_freq": "eval_every",
        "save_freq": "save_every",
        "save_dir": "checkpoint_dir",
        "local_batch_size": "batch_size",
        "lr_scheduler": "client_lr_scheduler",
        "sampled_clients": "num_clients",
    }
    applied = 0
    for key, value in flattened.items():
        mapped_key = aliases.get(key, key)
        if hasattr(args, mapped_key):
            setattr(args, mapped_key, value)
            applied += 1
    if "client_fraction" in flattened and hasattr(args, "num_clients") and hasattr(args, "clients_per_round"):
        client_fraction = float(flattened["client_fraction"])
        args.clients_per_round = max(1, int(round(float(args.num_clients) * client_fraction)))
        applied += 1
    return applied


def resolve_dataset_defaults(args: argparse.Namespace) -> None:
    """Fill unset CLI arguments with dataset defaults from the paper setup."""
    profile = DATASET_PROFILES[args.dataset]

    for key, value in profile.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)

    if args.clients_per_round is None and args.num_clients is not None:
        args.clients_per_round = max(1, int(round(args.num_clients * 0.1)))

    if args.cosine_decay_T_max is None:
        args.cosine_decay_T_max = args.num_rounds


def get_default_data_dir() -> Path:
    """Get default data directory based on platform."""
    return PROJECT_ROOT / "data"


def get_dataset(args: argparse.Namespace) -> tuple:
    """Get dataset loaders based on arguments."""
    ensure_runtime_imports()
    dataset_name = args.dataset
    
    if args.data_root:
        data_dir = Path(args.data_root)
    else:
        data_dir = get_default_data_dir()
    
    if dataset_name == "cifar10":
        cifar10_root = data_dir if "cifar10" in str(data_dir) else data_dir / "cifar10"
        client_loaders, val_loader, test_loader, data_manager = RUNTIME["get_cifar10_loaders"](
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
        client_loaders, val_loader, test_loader, data_manager = RUNTIME["get_femnist_loaders"](
            root=str(femnist_root),
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            seed=args.seed,
            download=args.download,
            allow_synthetic_data=args.allow_synthetic_data,
            num_workers=args.num_workers,
        )
        num_classes = 62
    elif dataset_name == "shakespeare":
        shakespeare_root = data_dir if "shakespeare" in str(data_dir) else data_dir / "shakespeare"
        client_loaders, val_loader, test_loader, data_manager = RUNTIME["get_shakespeare_loaders"](
            root=str(shakespeare_root),
            num_clients=args.num_clients,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            seed=args.seed,
            download=args.download,
            allow_synthetic_data=args.allow_synthetic_data,
            num_workers=args.num_workers,
        )
        num_classes = 80
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return client_loaders, val_loader, test_loader, num_classes


def get_model(model_name: str, num_classes: int, args: argparse.Namespace):
    """Get model based on name and number of classes."""
    ensure_runtime_imports()
    model_creators = RUNTIME["MODEL_CREATORS"]
    if model_name not in model_creators:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "lstm":
        return RUNTIME["ShakespeareLSTM"](
            vocab_size=num_classes,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )

    return model_creators[model_name](num_classes)


def get_aggregator(method: str, args: argparse.Namespace) -> Any:
    """Get aggregator based on method name."""
    ensure_runtime_imports()
    config = {
        "device": args.device,
        "use_data_size_weighting": True,
    }
    
    if method == "fedprox":
        config["proximal_mu"] = args.mu
    elif method == "scaffold":
        config["server_lr"] = args.server_lr
    elif method == "fedavgm":
        config["server_momentum"] = args.server_momentum
        config["server_lr"] = args.server_lr
    elif method == "fedadam":
        config["server_lr"] = args.server_lr
        config["beta1"] = args.beta1
        config["beta2"] = args.beta2
        config["tau"] = args.tau
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
    
    return RUNTIME["AGGREGATOR_CREATORS"][method](config)


def create_trainer_config(args: argparse.Namespace):
    """Create trainer configuration from arguments."""
    ensure_runtime_imports()
    return RUNTIME["TrainerConfig"](
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
        track_fairness=args.track_fairness,
        fairness_eval_freq=args.fairness_eval_freq,
        task_type=args.task_type,
        client_lr_scheduler=args.client_lr_scheduler,
        cosine_decay_T_max=args.cosine_decay_T_max,
        cosine_decay_eta_min=args.cosine_decay_eta_min,
        use_monitor=args.use_monitor,
        monitor_refresh_rate=args.monitor_refresh_rate,
        use_amp=args.use_amp,
        num_parallel_clients=args.num_parallel_clients,
        malicious_client_fraction=args.malicious_client_fraction,
        attack_type=args.attack_type,
    )


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the federated learning experiment."""
    ensure_runtime_imports()
    RUNTIME["set_seed"](args.seed)
    
    output_base = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_{args.method}_seed{args.seed}_{timestamp}"
    output_dir = output_base / args.run_group / run_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = output_dir.name
    
    setup_logger(
        log_file=str(output_dir / "experiment.log"),
        level="DEBUG" if args.verbose else "INFO",
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Run group: {args.run_group}")
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    metadata = {
        "run_name": run_name,
        "run_group": args.run_group,
        "status": "running",
        "seed": args.seed,
        "method": args.method,
        "dataset": args.dataset,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Loading dataset...")
    client_loaders, val_loader, test_loader, num_classes = get_dataset(args)
    logger.info(f"Dataset loaded: {len(client_loaders)} clients")
    
    logger.info("Creating model...")
    model = get_model(args.model, num_classes, args)
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
    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.is_absolute():
        trainer_config.checkpoint_dir = str(checkpoint_dir)
    else:
        trainer_config.checkpoint_dir = str(output_dir / checkpoint_dir)
    
    logger.info("Initializing trainer...")
    trainer = RUNTIME["FederatedTrainer"](
        model=model,
        aggregator=aggregator,
        client_loaders=client_loaders,
        val_loader=val_loader,
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
    primary_metric_name = results.get("primary_metric_name", "accuracy")
    logger.info(f"Best {primary_metric_name}: {results['best_primary_metric']:.4f}")
    
    metadata.update({
        "status": "completed",
        "best_accuracy": results["best_accuracy"],
        "best_perplexity": results.get("best_perplexity"),
        "best_primary_metric": results.get("best_primary_metric"),
        "primary_metric_name": primary_metric_name,
        "final_round": results["final_round"],
        "total_time": results["total_time"],
    })
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.config:
        config = load_config(args.config)
        applied = apply_config_to_args(args, config)
        logger.info(f"Applied {applied} config entries from {args.config}")

    resolve_dataset_defaults(args)
    ensure_runtime_imports()
    torch = RUNTIME["torch"]
    
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
    print(f"Best {results.get('primary_metric_name', 'accuracy')}: {results['best_primary_metric']:.4f}")
    print(f"Total rounds: {results['final_round']}")
    print(f"Total time: {results['total_time']:.2f}s")
    
    if results.get("convergence_round"):
        print(f"Convergence round: {results['convergence_round']}")
    
    if results.get("dsnr_summary"):
        print(f"DSNR mean: {results['dsnr_summary']['mean']:.4f}")


if __name__ == "__main__":
    main()
