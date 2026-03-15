"""Data partitioning utilities for federated learning."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class BasePartitioner(ABC):
    """Base class for data partitioners."""
    
    @abstractmethod
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        **kwargs,
    ) -> Dict[int, List[int]]:
        """
        Partition data indices among clients.
        
        Args:
            labels: Array of labels for each sample
            num_clients: Number of clients
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        pass


class DirichletPartitioner(BasePartitioner):
    """
    Dirichlet-based non-IID data partitioner.
    
    Implements the Dirichlet distribution-based partitioning method
    for creating heterogeneous data distributions across clients.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        min_samples_per_client: int = 10,
        seed: int = 42,
    ):
        """
        Initialize Dirichlet partitioner.
        
        Args:
            alpha: Concentration parameter for Dirichlet distribution.
                   Smaller alpha = more heterogeneous distribution.
            min_samples_per_client: Minimum samples per client
            seed: Random seed
        """
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        **kwargs,
    ) -> Dict[int, List[int]]:
        """
        Partition data using Dirichlet distribution.
        
        Args:
            labels: Array of labels for each sample
            num_clients: Number of clients
        
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        num_samples = len(labels)
        num_classes = len(np.unique(labels))
        
        client_indices = {i: [] for i in range(num_clients)}
        
        for k in range(num_classes):
            class_indices = np.where(labels == k)[0]
            self.rng.shuffle(class_indices)
            
            proportions = self.rng.dirichlet(
                self.alpha * np.ones(num_clients)
            )
            
            proportions = (proportions * len(class_indices)).astype(int)
            
            diff = len(class_indices) - proportions.sum()
            if diff > 0:
                proportions[self.rng.choice(num_clients, diff)] += 1
            elif diff < 0:
                mask = proportions > 0
                proportions[mask] -= 1
                proportions = np.maximum(proportions, 0)
            
            start_idx = 0
            for client_id, count in enumerate(proportions):
                if count > 0:
                    client_indices[client_id].extend(
                        class_indices[start_idx:start_idx + count].tolist()
                    )
                    start_idx += count
        
        for client_id in range(num_clients):
            if len(client_indices[client_id]) < self.min_samples_per_client:
                self._redistribute_samples(client_indices, client_id, num_clients)
        
        return client_indices
    
    def _redistribute_samples(
        self,
        client_indices: Dict[int, List[int]],
        client_id: int,
        num_clients: int,
    ) -> None:
        """Redistribute samples to meet minimum requirement."""
        needed = self.min_samples_per_client - len(client_indices[client_id])
        
        candidates = []
        for other_id in range(num_clients):
            if other_id != client_id:
                surplus = len(client_indices[other_id]) - self.min_samples_per_client
                if surplus > 0:
                    candidates.append((other_id, surplus))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for other_id, surplus in candidates:
            if needed <= 0:
                break
            
            take = min(needed, surplus)
            client_indices[client_id].extend(
                client_indices[other_id][-take:]
            )
            client_indices[other_id] = client_indices[other_id][:-take]
            needed -= take


class NaturalPartitioner(BasePartitioner):
    """
    Natural partitioner for naturally partitioned datasets.
    
    Used for datasets like FEMNIST and Shakespeare where
    data is already partitioned by user/writer.
    """
    
    def __init__(
        self,
        min_samples_per_client: int = 10,
        max_clients: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize natural partitioner.
        
        Args:
            min_samples_per_client: Minimum samples per client
            max_clients: Maximum number of clients to use
            seed: Random seed
        """
        self.min_samples_per_client = min_samples_per_client
        self.max_clients = max_clients
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        user_ids: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[int, List[int]]:
        """
        Partition data based on natural user IDs.
        
        Args:
            labels: Array of labels for each sample
            num_clients: Number of clients
            user_ids: Array of user IDs for each sample
        
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        if user_ids is None:
            raise ValueError("user_ids must be provided for natural partitioning")
        
        unique_users = np.unique(user_ids)
        
        if self.max_clients is not None:
            if len(unique_users) > self.max_clients:
                self.rng.shuffle(unique_users)
                unique_users = unique_users[:self.max_clients]
        
        user_to_indices = {}
        for user_id in unique_users:
            user_indices = np.where(user_ids == user_id)[0]
            if len(user_indices) >= self.min_samples_per_client:
                user_to_indices[user_id] = user_indices.tolist()
        
        client_indices = {}
        for client_id, user_id in enumerate(sorted(user_to_indices.keys())):
            if client_id >= num_clients:
                break
            client_indices[client_id] = user_to_indices[user_id]
        
        return client_indices


class IIDPartitioner(BasePartitioner):
    """IID data partitioner for baseline experiments."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize IID partitioner.
        
        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        **kwargs,
    ) -> Dict[int, List[int]]:
        """
        Partition data uniformly at random.
        
        Args:
            labels: Array of labels for each sample
            num_clients: Number of clients
        
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        num_samples = len(labels)
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)
        
        samples_per_client = num_samples // num_clients
        client_indices = {}
        
        for client_id in range(num_clients):
            start = client_id * samples_per_client
            if client_id == num_clients - 1:
                client_indices[client_id] = indices[start:].tolist()
            else:
                client_indices[client_id] = indices[start:start + samples_per_client].tolist()
        
        return client_indices


class PartitionManager:
    """Manager for handling data partitions."""
    
    def __init__(
        self,
        partition_type: str = "dirichlet",
        **partition_kwargs,
    ):
        """
        Initialize partition manager.
        
        Args:
            partition_type: Type of partitioner ("dirichlet", "natural", "iid")
            **partition_kwargs: Arguments for the partitioner
        """
        self.partition_type = partition_type
        self.partitioner = self._create_partitioner(partition_type, **partition_kwargs)
        self.partition_cache: Optional[Dict[int, List[int]]] = None
    
    def _create_partitioner(
        self,
        partition_type: str,
        **kwargs,
    ) -> BasePartitioner:
        """Create partitioner based on type."""
        if partition_type == "dirichlet":
            return DirichletPartitioner(**kwargs)
        elif partition_type == "natural":
            return NaturalPartitioner(**kwargs)
        elif partition_type == "iid":
            return IIDPartitioner(**kwargs)
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")
    
    def create_partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        user_ids: Optional[np.ndarray] = None,
    ) -> Dict[int, List[int]]:
        """
        Create and cache data partition.
        
        Args:
            labels: Array of labels for each sample
            num_clients: Number of clients
            user_ids: Array of user IDs (for natural partitioning)
        
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        self.partition_cache = self.partitioner.partition(
            labels,
            num_clients,
            user_ids=user_ids,
        )
        return self.partition_cache
    
    def get_client_indices(self, client_id: int) -> List[int]:
        """Get indices for a specific client."""
        if self.partition_cache is None:
            raise RuntimeError("Partition not created yet")
        return self.partition_cache.get(client_id, [])
    
    def get_partition_stats(self) -> Dict[str, float]:
        """Get statistics about the current partition."""
        if self.partition_cache is None:
            return {}
        
        sizes = [len(indices) for indices in self.partition_cache.values()]
        return {
            "num_clients": len(sizes),
            "total_samples": sum(sizes),
            "min_samples": min(sizes),
            "max_samples": max(sizes),
            "mean_samples": np.mean(sizes),
            "std_samples": np.std(sizes),
        }
    
    def save_partition(self, filepath: str) -> None:
        """Save partition to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({k: v for k, v in self.partition_cache.items()}, f)
    
    def load_partition(self, filepath: str) -> None:
        """Load partition from file."""
        import json
        with open(filepath, 'r') as f:
            loaded = json.load(f)
            self.partition_cache = {int(k): v for k, v in loaded.items()}
