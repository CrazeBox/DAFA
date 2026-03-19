"""Base aggregator class for federated learning."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from torch.nn.utils import parameters_to_vector, vector_to_parameters


@dataclass
class AggregatorConfig:
    """Configuration for aggregators."""
    
    name: str = "base"
    use_data_size_weighting: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ClientUpdate:
    """Container for client update information."""
    
    client_id: int
    update: torch.Tensor
    num_samples: int
    loss: float = 0.0
    num_steps: int = 1
    extra_info: Dict[str, Any] = field(default_factory=dict)


class BaseAggregator(ABC):
    """Base class for federated aggregation methods."""
    
    def __init__(self, config: AggregatorConfig):
        """
        Initialize aggregator.
        
        Args:
            config: Aggregator configuration
        """
        self.config = config
        self.device = config.device
        self.round_num = 0
        self.last_aggregated_update: Optional[torch.Tensor] = None
        self.last_proxy_direction: Optional[torch.Tensor] = None
    
    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates to produce new global model.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
            **kwargs: Additional arguments
        
        Returns:
            Updated global model
        """
        pass
    
    def get_weights(
        self,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute aggregation weights for each client.
        
        Args:
            client_updates: List of client updates
            **kwargs: Additional arguments
        
        Returns:
            Tensor of weights (sum to 1)
        """
        if self.config.use_data_size_weighting:
            total_samples = sum(u.num_samples for u in client_updates)
            weights = torch.tensor(
                [u.num_samples / total_samples for u in client_updates],
                device=self.device,
            )
        else:
            weights = torch.ones(len(client_updates), device=self.device) / len(client_updates)
        
        return weights
    
    def get_model_params(self, model: nn.Module) -> torch.Tensor:
        """Get flattened model parameters."""
        return parameters_to_vector(model.parameters()).detach()
    
    def set_model_params(self, model: nn.Module, params: torch.Tensor) -> None:
        """Set model parameters from flattened tensor."""
        vector_to_parameters(params, model.parameters())
    
    def get_update_tensor(self, update: ClientUpdate, device: torch.device) -> torch.Tensor:
        """Get update tensor on the specified device."""
        tensor = update.update
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor
    
    def get_model_update(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
    ) -> torch.Tensor:
        """Compute update as difference between two models."""
        params_before = self.get_model_params(model_before)
        params_after = self.get_model_params(model_after)
        return params_after - params_before
    
    def apply_update(
        self,
        model: nn.Module,
        update: torch.Tensor,
        lr: float = 1.0,
    ) -> None:
        """Apply update to model with optional learning rate."""
        params = self.get_model_params(model)
        new_params = params + lr * update
        self.set_model_params(model, new_params)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get aggregator state for checkpointing."""
        return {
            "round_num": self.round_num,
            "config": self.config.__dict__,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        self.round_num = state.get("round_num", 0)
    
    def step(self) -> None:
        """Increment round counter."""
        self.round_num += 1
    
    def reset(self) -> None:
        """Reset aggregator state."""
        self.round_num = 0
        self.last_aggregated_update = None
        self.last_proxy_direction = None
