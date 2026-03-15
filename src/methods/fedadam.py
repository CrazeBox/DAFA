"""FedAdam: Federated Adam implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math

from .base import BaseAggregator, AggregatorConfig, ClientUpdate


@dataclass
class FedAdamConfig(AggregatorConfig):
    """Configuration for FedAdam aggregator."""
    
    name: str = "fedadam"
    server_lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    tau: float = 1e-3


class FedAdamAggregator(BaseAggregator):
    """
    Federated Adam (FedAdam) aggregator.
    
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    
    FedAdam uses Adam optimizer on the server side for adaptive learning
    rates during aggregation.
    """
    
    def __init__(self, config: FedAdamConfig):
        super().__init__(config)
        self.server_lr = config.server_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        self.tau = config.tau
        
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates using Adam.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
        
        Returns:
            Updated global model
        """
        if not client_updates:
            return global_model
        
        global_params = self.get_model_params(global_model)
        
        if self.m is None:
            self.m = torch.zeros_like(global_params)
        elif self.m.device != global_params.device:
            self.m = self.m.to(global_params.device)
        if self.v is None:
            self.v = torch.zeros_like(global_params)
        elif self.v.device != global_params.device:
            self.v = self.v.to(global_params.device)
        
        weights = self.get_weights(client_updates)
        
        aggregated_update = torch.zeros_like(global_params)
        for i, update in enumerate(client_updates):
            update_tensor = self.get_update_tensor(update, global_params.device)
            aggregated_update += weights[i] * update_tensor
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * aggregated_update
        self.v = self.beta2 * self.v + (1 - self.beta2) * (aggregated_update ** 2)
        
        t = self.round_num + 1
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)
        
        adaptive_lr = self.server_lr / (torch.sqrt(v_hat) + self.tau)
        
        new_params = global_params + adaptive_lr * m_hat
        self.set_model_params(global_model, new_params)
        
        self.step()
        
        return global_model
    
    def state_dict(self) -> Dict[str, Any]:
        """Get aggregator state for checkpointing."""
        state = super().state_dict()
        state["server_lr"] = self.server_lr
        state["beta1"] = self.beta1
        state["beta2"] = self.beta2
        state["epsilon"] = self.epsilon
        state["tau"] = self.tau
        state["m"] = self.m.cpu() if self.m is not None else None
        state["v"] = self.v.cpu() if self.v is not None else None
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        super().load_state_dict(state)
        self.server_lr = state.get("server_lr", 0.001)
        self.beta1 = state.get("beta1", 0.9)
        self.beta2 = state.get("beta2", 0.999)
        self.epsilon = state.get("epsilon", 1e-8)
        self.tau = state.get("tau", 1e-3)
        
        if state.get("m") is not None:
            self.m = state["m"].to(self.device)
        if state.get("v") is not None:
            self.v = state["v"].to(self.device)
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self.m = None
        self.v = None
