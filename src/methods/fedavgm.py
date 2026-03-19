"""FedAvgM: FedAvg with Server Momentum implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import BaseAggregator, AggregatorConfig, ClientUpdate


@dataclass
class FedAvgMConfig(AggregatorConfig):
    """Configuration for FedAvgM aggregator."""
    
    name: str = "fedavgm"
    server_momentum: float = 0.9
    server_lr: float = 1.0


class FedAvgMAggregator(BaseAggregator):
    """
    FedAvg with Server Momentum (FedAvgM) aggregator.
    
    Reference: Hsu et al., "Measuring the Effects of Non-Identical Data
    Distribution for Federated Visual Classification", arXiv 2019.
    
    FedAvgM applies momentum on the server side to improve convergence
    in heterogeneous settings.
    """
    
    def __init__(self, config: FedAvgMConfig):
        super().__init__(config)
        self.server_momentum = config.server_momentum
        self.server_lr = config.server_lr
        self.velocity: Optional[torch.Tensor] = None
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates with server-side momentum.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
        
        Returns:
            Updated global model
        """
        if not client_updates:
            return global_model
        
        global_params = self.get_model_params(global_model)
        
        if self.velocity is None:
            self.velocity = torch.zeros_like(global_params)
        elif self.velocity.device != global_params.device:
            self.velocity = self.velocity.to(global_params.device)
        
        weights = self.get_weights(client_updates)
        
        aggregated_update = torch.zeros_like(global_params)
        for i, update in enumerate(client_updates):
            update_tensor = self.get_update_tensor(update, global_params.device)
            aggregated_update += weights[i] * update_tensor
        
        self.velocity = self.server_momentum * self.velocity + (1 - self.server_momentum) * aggregated_update

        self.last_aggregated_update = self.velocity.detach().clone()
        self.last_proxy_direction = self.velocity.detach().clone()
        
        new_params = global_params + self.server_lr * self.velocity
        self.set_model_params(global_model, new_params)
        
        self.step()
        
        return global_model
    
    def state_dict(self) -> Dict[str, Any]:
        """Get aggregator state for checkpointing."""
        state = super().state_dict()
        state["server_momentum"] = self.server_momentum
        state["server_lr"] = self.server_lr
        state["velocity"] = self.velocity.cpu() if self.velocity is not None else None
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        super().load_state_dict(state)
        self.server_momentum = state.get("server_momentum", 0.9)
        self.server_lr = state.get("server_lr", 1.0)
        
        if state.get("velocity") is not None:
            self.velocity = state["velocity"].to(self.device)
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self.velocity = None
