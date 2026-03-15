"""Core federated learning components."""

from .trainer import FederatedTrainer, LocalTrainer, TrainerConfig

__all__ = [
    "FederatedTrainer",
    "LocalTrainer",
    "TrainerConfig",
]
