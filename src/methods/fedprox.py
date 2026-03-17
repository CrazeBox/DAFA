"""FedProx: Federated Proximal implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import BaseAggregator, AggregatorConfig, ClientUpdate
from .fedavg import FedAvgAggregator


@dataclass
class FedProxConfig(AggregatorConfig):
    """Configuration for FedProx aggregator."""
    
    name: str = "fedprox"
    proximal_mu: float = 0.01


class FedProxAggregator(FedAvgAggregator):
    """
    Federated Proximal (FedProx) aggregator.
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks",
    MLSys 2020.
    
    FedProx adds a proximal term to the local objective:
    h(w; w^t) = (mu/2) ||w - w^t||^2
    
    The aggregation is the same as FedAvg, but the local training differs.
    """
    
    def __init__(self, config: FedProxConfig):
        super().__init__(config)
        self.proximal_mu = config.proximal_mu
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates (same as FedAvg).
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
        
        Returns:
            Updated global model
        """
        return super().aggregate(global_model, client_updates, **kwargs)
    
    def get_proximal_term(
        self,
        local_params: torch.Tensor,
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute proximal term for local training.
        
        Args:
            local_params: Current local model parameters
            global_params: Global model parameters
        
        Returns:
            Proximal term value
        """
        return (self.proximal_mu / 2) * torch.norm(local_params - global_params) ** 2
    
    def state_dict(self) -> Dict[str, Any]:
        """Get aggregator state for checkpointing."""
        state = super().state_dict()
        state["proximal_mu"] = self.proximal_mu
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        super().load_state_dict(state)
        self.proximal_mu = state.get("proximal_mu", 0.01)
