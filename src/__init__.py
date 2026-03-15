"""DAFA: Directionally Aligned Federated Aggregation Framework."""

__version__ = "1.0.0"
__author__ = "DAFA Research Team"

from src.methods import (
    FedAvgAggregator,
    FedProxAggregator,
    SCAFFOLDAggregator,
    FedNovaAggregator,
    FedAvgMAggregator,
    FedAdamAggregator,
    DAFAAggregator,
)

from src.models import (
    ResNet18,
    SimpleCNN,
    TwoLayerCNN,
    LSTMModel,
    ShakespeareLSTM,
)

from src.core import FederatedTrainer, TrainerConfig

__all__ = [
    "FedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator",
    "FedNovaAggregator",
    "FedAvgMAggregator",
    "FedAdamAggregator",
    "DAFAAggregator",
    "ResNet18",
    "SimpleCNN",
    "TwoLayerCNN",
    "LSTMModel",
    "ShakespeareLSTM",
    "FederatedTrainer",
    "TrainerConfig",
]
