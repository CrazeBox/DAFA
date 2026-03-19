"""FedAvg: Federated Averaging implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from .base import BaseAggregator, AggregatorConfig, ClientUpdate


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) aggregator.
    
    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.
    """
    
    def __init__(self, config: AggregatorConfig):
        super().__init__(config)
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates using weighted averaging.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
        
        Returns:
            Updated global model
        """
        if not client_updates:
            return global_model
        
        weights = self.get_weights(client_updates)
        
        global_params = self.get_model_params(global_model)
        
        aggregated_update = torch.zeros_like(global_params)
        
        for i, update in enumerate(client_updates):
            update_tensor = self.get_update_tensor(update, global_params.device)
            aggregated_update += weights[i] * update_tensor
        
        self.last_aggregated_update = aggregated_update.detach().clone()
        self.last_proxy_direction = aggregated_update.detach().clone()

        new_params = global_params + aggregated_update
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
        if self.config.use_data_size_weighting:
            total_samples = sum(u.num_samples for u in client_updates)
            weights = torch.tensor(
                [u.num_samples / total_samples for u in client_updates],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            n = len(client_updates)
            weights = torch.ones(n, device=self.device, dtype=torch.float32) / n
        
        return weights


class FedAvgConfig(AggregatorConfig):
    """Configuration for FedAvg aggregator."""
    
    name: str = "fedavg"
