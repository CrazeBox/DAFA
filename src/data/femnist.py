"""FEMNIST dataset handling for federated learning."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from PIL import Image
import urllib.request
import zipfile
import io


FEMNIST_URL = "https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/data/femnist/"


class FEMNISTDataset(Dataset):
    """FEMNIST dataset wrapper for federated learning."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        allow_synthetic_data: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize FEMNIST dataset.
        
        Args:
            root: Root directory for data
            train: Whether to use training set
            download: Whether to download if not exists
            transform: Optional transform to apply
        """
        self.root = Path(root)
        self.train = train
        self.allow_synthetic_data = allow_synthetic_data
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        self.data_dir = self.root / ("train" if train else "test")
        
        if download and not self._check_data_exists():
            self._download_data()
        
        self.images: List[np.ndarray] = []
        self.labels: List[int] = []
        self.user_ids: List[str] = []
        
        self._load_data()
        
        self.labels_array = np.array(self.labels)
        self.user_ids_array = np.array(self.user_ids)
    
    def _check_data_exists(self) -> bool:
        """Check if data already exists."""
        return self.data_dir.exists() and any(self.data_dir.iterdir())
    
    def _download_data(self) -> None:
        """Download FEMNIST data."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading FEMNIST data to {self.root}...")
        
        for split in ["train", "test"]:
            split_dir = self.root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            all_data_path = split_dir / "all_data.json"
            if not all_data_path.exists():
                url = f"{FEMNIST_URL}{split}/all_data.json"
                try:
                    urllib.request.urlretrieve(url, all_data_path)
                    print(f"Downloaded {split} data")
                except Exception as e:
                    if self.allow_synthetic_data:
                        print(f"Failed to download {split} data: {e}")
                        self._create_synthetic_data(split_dir)
                    else:
                        raise RuntimeError(
                            f"Failed to download FEMNIST {split} split and synthetic fallback is disabled: {e}"
                        ) from e
    
    def _create_synthetic_data(self, split_dir: Path) -> None:
        """Create synthetic FEMNIST-like data for testing."""
        print(f"Creating synthetic FEMNIST data in {split_dir}...")
        
        num_users = 100 if "train" in str(split_dir) else 20
        samples_per_user = 50
        
        all_data = {"users": [], "num_samples": [], "user_data": {}}
        
        for user_id in range(num_users):
            user_name = f"f_{user_id:04d}"
            all_data["users"].append(user_name)
            all_data["num_samples"].append(samples_per_user)
            
            images = np.random.randint(0, 256, (samples_per_user, 28, 28), dtype=np.uint8)
            labels = np.random.randint(0, 62, samples_per_user).tolist()
            
            all_data["user_data"][user_name] = {
                "x": images.reshape(samples_per_user, -1).tolist(),
                "y": labels,
            }
        
        with open(split_dir / "all_data.json", "w") as f:
            json.dump(all_data, f)
    
    def _load_data(self) -> None:
        """Load data from disk."""
        all_data_path = self.data_dir / "all_data.json"
        
        if not all_data_path.exists():
            if self.allow_synthetic_data:
                print(f"Data file not found: {all_data_path}")
                print("Creating synthetic data...")
                self._create_synthetic_data(self.data_dir)
            else:
                raise FileNotFoundError(
                    f"Missing FEMNIST data file: {all_data_path}. "
                    "Provide the real dataset or rerun with synthetic fallback enabled."
                )
        
        with open(all_data_path, "r") as f:
            data = json.load(f)
        
        for user_id, user_name in enumerate(data["users"]):
            user_data = data["user_data"][user_name]
            
            for img_flat, label in zip(user_data["x"], user_data["y"]):
                img = np.array(img_flat, dtype=np.uint8).reshape(28, 28)
                self.images.append(img)
                self.labels.append(label)
                self.user_ids.append(user_name)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx])
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]
    
    def get_labels(self) -> np.ndarray:
        """Get all labels."""
        return self.labels_array
    
    def get_user_ids(self) -> np.ndarray:
        """Get all user IDs."""
        return self.user_ids_array
    
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
        return Subset(self, client_indices)


class FEMNISTFederated:
    """Federated FEMNIST data manager."""
    
    def __init__(
        self,
        root: str = "data/femnist",
        num_clients: int = 100,
        seed: int = 42,
        download: bool = True,
        allow_synthetic_data: bool = False,
        val_ratio: float = 0.1,
    ):
        """
        Initialize federated FEMNIST manager.
        
        Args:
            root: Root directory for data
            num_clients: Number of clients to sample
            seed: Random seed
            download: Whether to download if not exists
            val_ratio: Ratio of training data to use for validation
        """
        self.root = Path(root)
        self.num_clients = num_clients
        self.seed = seed
        self.val_ratio = val_ratio
        self.rng = np.random.RandomState(seed)
        
        self.train_dataset = FEMNISTDataset(
            root=root,
            train=True,
            download=download,
            allow_synthetic_data=allow_synthetic_data,
        )
        
        self.test_dataset = FEMNISTDataset(
            root=root,
            train=False,
            download=download,
            allow_synthetic_data=allow_synthetic_data,
        )
        
        self._create_validation_set()
        
        self.client_partitions: Optional[Dict[int, List[int]]] = None
    
    def _create_validation_set(self):
        """Create validation set from training data."""
        np.random.seed(self.seed)
        
        num_train = len(self.train_dataset)
        indices = np.random.permutation(num_train)
        
        num_val = int(num_train * self.val_ratio)
        self.val_indices = indices[:num_val]
        self.train_indices = indices[num_val:]
        
        self.val_dataset = Subset(self.train_dataset, self.val_indices)
        self.train_dataset_for_partition = Subset(self.train_dataset, self.train_indices)
    
    def create_partitions(self) -> Dict[int, List[int]]:
        """
        Create data partitions for all clients based on train_indices.
        
        Returns:
            Dictionary mapping client_id to sample indices
        """
        user_ids = self.train_dataset.get_user_ids()
        unique_users = np.unique(user_ids)
        
        if len(unique_users) > self.num_clients:
            self.rng.shuffle(unique_users)
            unique_users = unique_users[:self.num_clients]
        
        self.client_partitions = {}
        
        train_indices_set = set(self.train_indices)
        
        for client_id, user_name in enumerate(unique_users):
            if client_id >= self.num_clients:
                break
            
            user_indices = np.where(user_ids == user_name)[0]
            user_indices_in_train = [idx for idx in user_indices if idx in train_indices_set]
            if user_indices_in_train:
                self.client_partitions[client_id] = user_indices_in_train
        
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
    
    def get_val_dataloader(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Get validation DataLoader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
        
        Returns:
            Validation DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    def get_partition_stats(self) -> Dict[str, float]:
        """Get partition statistics."""
        if self.client_partitions is None:
            return {}
        
        sizes = [len(indices) for indices in self.client_partitions.values()]
        return {
            "num_clients": len(sizes),
            "total_samples": sum(sizes),
            "min_samples": min(sizes),
            "max_samples": max(sizes),
            "mean_samples": np.mean(sizes),
            "std_samples": np.std(sizes),
        }


def get_femnist_loaders(
    root: str = "data/femnist",
    num_clients: int = 100,
    batch_size: int = 64,
    seed: int = 42,
    download: bool = True,
    allow_synthetic_data: bool = False,
    num_workers: int = 0,
    lazy_init: bool = True,
    val_ratio: float = 0.1,
) -> Tuple[Dict[int, DataLoader], DataLoader, DataLoader, FEMNISTFederated]:
    """
    Get FEMNIST data loaders for federated learning.
    
    Args:
        root: Root directory for data
        num_clients: Number of clients
        batch_size: Batch size
        seed: Random seed
        download: Whether to download if not exists
        num_workers: Number of data loading workers
        lazy_init: If True, return a lazy dict that creates DataLoaders on demand
        val_ratio: Ratio of training data to use for validation
    
    Returns:
        Tuple of (client_loaders, val_loader, test_loader, data_manager)
    """
    from .cifar10 import LazyClientLoaderDict
    
    manager = FEMNISTFederated(
        root=root,
        num_clients=num_clients,
        seed=seed,
        download=download,
        allow_synthetic_data=allow_synthetic_data,
        val_ratio=val_ratio,
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
        for client_id in range(min(num_clients, len(manager.client_partitions))):
            client_loaders[client_id] = manager.get_client_dataloader(
                client_id=client_id,
                batch_size=batch_size,
                num_workers=num_workers,
            )
    
    test_loader = manager.get_test_dataloader(
        batch_size=batch_size * 2,
        num_workers=num_workers,
    )
    
    val_loader = manager.get_val_dataloader(
        batch_size=batch_size * 2,
        num_workers=num_workers,
    )
    
    return client_loaders, val_loader, test_loader, manager
