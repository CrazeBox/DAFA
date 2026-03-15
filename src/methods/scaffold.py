"""SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from copy import deepcopy

from .base import BaseAggregator, AggregatorConfig, ClientUpdate


@dataclass
class SCAFFOLDConfig(AggregatorConfig):
    """Configuration for SCAFFOLD aggregator."""
    
    name: str = "scaffold"
    server_lr: float = 1.0


class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD aggregator with control variates.
    
    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning", ICML 2020.
    
    SCAFFOLD uses control variates to correct client drift in heterogeneous
    federated learning settings.
    """
    
    def __init__(self, config: SCAFFOLDConfig):
        super().__init__(config)
        self.server_lr = config.server_lr
        self.global_control: Optional[torch.Tensor] = None
        self.client_controls: Dict[int, torch.Tensor] = {}
    
    def initialize_controls(self, model: nn.Module) -> None:
        """Initialize control variates."""
        params = self.get_model_params(model)
        self.global_control = torch.zeros_like(params)
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates with control variate correction.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates with control info
        
        Returns:
            Updated global model
        """
        if not client_updates:
            return global_model
        
        if self.global_control is None:
            self.initialize_controls(global_model)
        
        global_params = self.get_model_params(global_model)
        
        weights = self.get_weights(client_updates)
        
        delta_updates = torch.zeros_like(global_params)
        delta_controls = torch.zeros_like(global_params)
        
        for i, update in enumerate(client_updates):
            client_id = update.client_id
            
            update_tensor = self.get_update_tensor(update, global_params.device)
            delta_updates += weights[i] * update_tensor
            
            new_control = update.extra_info.get("new_control", torch.zeros_like(global_params))
            if new_control.device != global_params.device:
                new_control = new_control.to(global_params.device)
            old_control = self.client_controls.get(client_id, torch.zeros_like(global_params))
            if old_control.device != global_params.device:
                old_control = old_control.to(global_params.device)
            
            delta_controls += weights[i] * (new_control - old_control)
            
            self.client_controls[client_id] = new_control.clone()
        
        new_params = global_params + self.server_lr * delta_updates
        self.set_model_params(global_model, new_params)
        
        self.global_control = self.global_control + delta_controls
        
        self.step()
        
        return global_model
    
    def get_client_control(self, client_id: int, model: nn.Module) -> torch.Tensor:
        """Get control variate for a client."""
        if self.global_control is None:
            self.initialize_controls(model)
        
        if client_id not in self.client_controls:
            params = self.get_model_params(model)
            self.client_controls[client_id] = torch.zeros_like(params)
        
        return self.client_controls[client_id].clone()
    
    def get_global_control(self, model: nn.Module) -> torch.Tensor:
        """Get global control variate."""
        if self.global_control is None:
            self.initialize_controls(model)
        
        return self.global_control.clone()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get aggregator state for checkpointing."""
        state = super().state_dict()
        state["server_lr"] = self.server_lr
        state["global_control"] = self.global_control.cpu() if self.global_control is not None else None
        state["client_controls"] = {
            k: v.cpu() for k, v in self.client_controls.items()
        }
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        super().load_state_dict(state)
        self.server_lr = state.get("server_lr", 1.0)
        
        if state.get("global_control") is not None:
            self.global_control = state["global_control"].to(self.device)
        
        self.client_controls = {
            int(k): v.to(self.device)
            for k, v in state.get("client_controls", {}).items()
        }
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self.global_control = None
        self.client_controls = {}
