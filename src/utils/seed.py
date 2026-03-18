"""Random seed utilities for reproducibility."""

import os
import random
import numpy as np
import torch
from typing import Optional

_current_seed: int = 42


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode for PyTorch
    """
    global _current_seed
    _current_seed = seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def get_seed() -> int:
    """Get current random seed."""
    return _current_seed


def get_generator(device: Optional[str] = None) -> torch.Generator:
    """
    Get a PyTorch generator with current seed.
    
    Args:
        device: Device for the generator (default: current device)
    
    Returns:
        torch.Generator with current seed
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(_current_seed)
    return generator
