"""Federated learning aggregation methods."""

from .base import BaseAggregator, AggregatorConfig
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .scaffold import SCAFFOLDAggregator
from .fednova import FedNovaAggregator
from .fedavgm import FedAvgMAggregator
from .fedadam import FedAdamAggregator
from .dafa import DAFAAggregator
from .dir_weight import DirWeightAggregator

__all__ = [
    "BaseAggregator",
    "AggregatorConfig",
    "FedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator",
    "FedNovaAggregator",
    "FedAvgMAggregator",
    "FedAdamAggregator",
    "DAFAAggregator",
    "DirWeightAggregator",
]
