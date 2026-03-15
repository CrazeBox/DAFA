"""Data processing modules for DAFA experiment framework."""

from .partition import DirichletPartitioner, NaturalPartitioner, PartitionManager
from .cifar10 import CIFAR10Dataset, get_cifar10_loaders, LazyClientLoaderDict
from .femnist import FEMNISTDataset, get_femnist_loaders
from .shakespeare import ShakespeareDataset, get_shakespeare_loaders

__all__ = [
    "DirichletPartitioner",
    "NaturalPartitioner",
    "PartitionManager",
    "CIFAR10Dataset",
    "get_cifar10_loaders",
    "FEMNISTDataset",
    "get_femnist_loaders",
    "ShakespeareDataset",
    "get_shakespeare_loaders",
    "LazyClientLoaderDict",
]
