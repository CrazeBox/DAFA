"""FedNova: Federated Normalized Averaging implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import BaseAggregator, AggregatorConfig, ClientUpdate


@dataclass
class FedNovaConfig(AggregatorConfig):
    """Configuration for FedNova aggregator."""
    
    name: str = "fednova"


class FedNovaAggregator(BaseAggregator):
    """
    Federated Normalized Averaging (FedNova) aggregator.
    
    Reference: Wang et al., "Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization", NeurIPS 2020.
    
    FedNova normalizes local updates by the number of local steps to ensure
    objective consistency across heterogeneous clients.
    """
    
    def __init__(self, config: FedNovaConfig):
        super().__init__(config)
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates with normalized averaging.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates with num_steps info
        
        Returns:
            Updated global model
        """
        if not client_updates:
            return global_model
        
        global_params = self.get_model_params(global_model)
        
        total_samples = sum(u.num_samples for u in client_updates)
        
        effective_steps = 0.0
        weighted_update = torch.zeros_like(global_params)
        
        for update in client_updates:
            data_weight = update.num_samples / total_samples
            num_steps = update.num_steps
            
            update_tensor = self.get_update_tensor(update, global_params.device)
            normalized_update = update_tensor / num_steps
            
            weighted_update += data_weight * normalized_update
            
            effective_steps += data_weight * num_steps
        
        final_update = effective_steps * weighted_update

        self.last_aggregated_update = final_update.detach().clone()
        self.last_proxy_direction = final_update.detach().clone()
        
        new_params = global_params + final_update
        self.set_model_params(global_model, new_params)
        
        self.step()
        
        return global_model
    
    def get_weights(
        self,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute aggregation weights based on data size.
        
        Args:
            client_updates: List of client updates
        
        Returns:
            Normalized weights tensor
        """
        total_samples = sum(u.num_samples for u in client_updates)
        weights = torch.tensor(
            [u.num_samples / total_samples for u in client_updates],
            device=self.device,
            dtype=torch.float32,
        )
        return weights
    
    def compute_effective_steps(
        self,
        client_updates: List[ClientUpdate],
    ) -> float:
        """
        Compute effective number of steps for aggregation.
        
        Args:
            client_updates: List of client updates
        
        Returns:
            Effective steps value
        """
        total_samples = sum(u.num_samples for u in client_updates)
        effective_steps = 0.0
        
        for update in client_updates:
            data_weight = update.num_samples / total_samples
            effective_steps += data_weight * update.num_steps
        
        return effective_steps
