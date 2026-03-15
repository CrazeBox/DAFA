"""CIFAR-10 dataset handling for federated learning."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from .partition import PartitionManager, DirichletPartitioner


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_transforms(
    train: bool = True,
    mean: Tuple[float, ...] = CIFAR10_MEAN,
    std: Tuple[float, ...] = CIFAR10_STD,
) -> transforms.Compose:
    """
    Get CIFAR-10 data transforms.
    
    Args:
        train: Whether to use training transforms
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper for federated learning."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            root: Root directory for data
            train: Whether to use training set
            download: Whether to download if not exists
            transform: Optional transform to apply
        """
        self.root = Path(root)
        self.train = train
        
        if transform is None:
            transform = get_cifar10_transforms(train=train)
        
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
        
        self.labels = np.array(self.dataset.targets)
        self.data = self.dataset.data
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]
    
    def get_labels(self) -> np.ndarray:
        """Get all labels."""
        return self.labels
    
    def get_client_dataset(
        self,
        client_indices: List[int],
    ) -> Subset:
        """
        Get dataset subset for a client.
        
        Args:
            client_indices: List of sample indices for the client
        
        Returns:
            Subset of the dataset
        """
        return Subset(self.dataset, client_indices)


class CIFAR10Federated:
    """Federated CIFAR-10 data manager."""
    
    def __init__(
        self,
        root: str = "data/cifar10",
        num_clients: int = 100,
        alpha: float = 0.5,
        seed: int = 42,
        download: bool = True,
    ):
        """
        Initialize federated CIFAR-10 manager.
        
        Args:
            root: Root directory for data
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter
            seed: Random seed
            download: Whether to download if not exists
        """
        self.root = Path(root)
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        
        self.train_dataset = CIFAR10Dataset(
            root=root,
            train=True,
            download=download,
        )
        
        self.test_dataset = CIFAR10Dataset(
            root=root,
            train=False,
            download=download,
        )
        
        self.partition_manager = PartitionManager(
            partition_type="dirichlet",
            alpha=alpha,
            min_samples_per_client=10,
            seed=seed,
        )
        
        self.client_partitions: Optional[Dict[int, List[int]]] = None
    
    def create_partitions(self) -> Dict[int, List[int]]:
        """
        Create data partitions for all clients.
        
        Returns:
            Dictionary mapping client_id to sample indices
        """
        self.client_partitions = self.partition_manager.create_partition(
            labels=self.train_dataset.get_labels(),
            num_clients=self.num_clients,
        )
        return self.client_partitions
    
    def get_client_dataloader(
        self,
        client_id: int,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Get DataLoader for a specific client.
        
        Args:
            client_id: Client ID
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
        
        Returns:
            DataLoader for the client
        """
        if self.client_partitions is None:
            self.create_partitions()
        
        client_indices = self.client_partitions.get(client_id, [])
        if not client_indices:
            raise ValueError(f"No data for client {client_id}")
        
        client_dataset = self.train_dataset.get_client_dataset(client_indices)
        
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    def get_test_dataloader(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Get test DataLoader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
        
        Returns:
            Test DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    def get_partition_stats(self) -> Dict[str, float]:
        """Get partition statistics."""
        return self.partition_manager.get_partition_stats()
    
    def get_client_stats(self, client_id: int) -> Dict[str, int]:
        """
        Get statistics for a specific client.
        
        Args:
            client_id: Client ID
        
        Returns:
            Dictionary with client statistics
        """
        if self.client_partitions is None:
            self.create_partitions()
        
        indices = self.client_partitions.get(client_id, [])
        labels = self.train_dataset.get_labels()[indices]
        
        class_counts = {}
        for label in labels:
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1
        
        return {
            "num_samples": len(indices),
            "num_classes": len(class_counts),
            "class_distribution": class_counts,
        }


def get_cifar10_loaders(
    root: str = "data/cifar10",
    num_clients: int = 100,
    alpha: float = 0.5,
    batch_size: int = 64,
    seed: int = 42,
    download: bool = True,
    num_workers: int = 0,
    lazy_init: bool = True,
) -> Tuple[Dict[int, DataLoader], DataLoader, CIFAR10Federated]:
    """
    Get CIFAR-10 data loaders for federated learning.
    
    Args:
        root: Root directory for data
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        batch_size: Batch size
        seed: Random seed
        download: Whether to download if not exists
        num_workers: Number of data loading workers
        lazy_init: If True, return a lazy dict that creates DataLoaders on demand
    
    Returns:
        Tuple of (client_loaders, test_loader, data_manager)
    """
    manager = CIFAR10Federated(
        root=root,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        download=download,
    )
    
    manager.create_partitions()
    
    if lazy_init:
        client_loaders = LazyClientLoaderDict(
            manager=manager,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        client_loaders = {}
        for client_id in range(num_clients):
            client_loaders[client_id] = manager.get_client_dataloader(
                client_id=client_id,
                batch_size=batch_size,
                num_workers=num_workers,
            )
    
    test_loader = manager.get_test_dataloader(
        batch_size=batch_size * 2,
        num_workers=num_workers,
    )
    
    return client_loaders, test_loader, manager


class LazyClientLoaderDict:
    """Lazy dictionary that creates DataLoaders on demand."""
    
    def __init__(
        self,
        manager,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self._manager = manager
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._cache: Dict[int, DataLoader] = {}
        self._num_clients = getattr(manager, 'num_clients', len(manager.client_partitions) if hasattr(manager, 'client_partitions') else 0)
    
    def __getitem__(self, client_id: int) -> DataLoader:
        if client_id not in self._cache:
            self._cache[client_id] = self._manager.get_client_dataloader(
                client_id=client_id,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
            )
        return self._cache[client_id]
    
    def __contains__(self, client_id: int) -> bool:
        return 0 <= client_id < self._num_clients
    
    def __len__(self) -> int:
        return self._num_clients
    
    def keys(self):
        return range(self._num_clients)
    
    def items(self):
        for client_id in range(self._num_clients):
            yield client_id, self[client_id]
    
    def get(self, client_id: int, default=None):
        try:
            return self[client_id]
        except (KeyError, ValueError):
            return default
