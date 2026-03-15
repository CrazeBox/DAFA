"""DAFA: Directionally Aligned Federated Aggregation implementation.

Reference: NeurIPS Paper4 - Algorithm 1

Key components:
1. Momentum proxy: m_t = (1-β) * Δ_FedAvg + β * m_{t-1}
2. Proxy direction: v* = m_t / ||m_t||
3. Alignment scores: s_i = <v_i, v*> where v_i = Δ_i / ||Δ_i||
4. Softmax weighting: w_i = p_i * exp(γ * s_i) / Σ p_j * exp(γ * s_j)
5. Graceful degradation: s_i = 0 if ||Δ_i|| < μ
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

from .base import BaseAggregator, AggregatorConfig, ClientUpdate


@dataclass
class DAFAConfig(AggregatorConfig):
    """Configuration for DAFA aggregator.
    
    Attributes:
        gamma: Temperature parameter for softmax weighting (default: 1.0)
               Higher gamma = stronger alignment preference
        beta: Momentum coefficient for proxy direction (default: 0.9)
              Higher beta = more temporal smoothing
        mu: Norm threshold for graceful degradation (default: 0.01)
            Updates with ||Δ_i|| < μ get s_i = 0
        use_pi_weighting: Whether to include data size weighting p_i (default: True)
        server_lr: Server-side learning rate (default: 1.0)
    """
    
    name: str = "dafa"
    gamma: float = 1.0
    beta: float = 0.9
    mu: float = 0.01
    use_pi_weighting: bool = True
    server_lr: float = 1.0


class DAFAAggregator(BaseAggregator):
    """
    Directionally Aligned Federated Aggregation (DAFA) aggregator.
    
    DAFA weights client updates based on their directional alignment
    with a momentum-smoothed proxy direction, using softmax weighting
    with temperature parameter γ.
    
    Algorithm (from paper):
    1. Compute FedAvg update: Δ_FedAvg = Σ p_i * Δ_i / Σ p_i
    2. Update momentum proxy: m_t = (1-β) * Δ_FedAvg + β * m_{t-1}
    3. Compute proxy direction: v* = m_t / ||m_t||
    4. For each client i:
       - If ||Δ_i|| >= μ: s_i = <Δ_i/||Δ_i||, v*>
       - Else: s_i = 0 (graceful degradation)
    5. Compute weights: w_i = p_i * exp(γ * s_i) / Σ p_j * exp(γ * s_j)
    6. Aggregate: Δ_DAFA = Σ w_i * Δ_i
    7. Update global model: x_{t+1} = x_t + η * Δ_DAFA
    """
    
    def __init__(self, config: DAFAConfig):
        super().__init__(config)
        self.gamma = config.gamma
        self.beta = config.beta
        self.mu = config.mu
        self.use_pi_weighting = config.use_pi_weighting
        self.server_lr = config.server_lr
        
        self.momentum: Optional[torch.Tensor] = None
        
        self.history: Dict[str, List[Any]] = {
            "dsnr": [],
            "alignment_scores": [],
            "alignment_mean": [],
            "alignment_std": [],
            "update_variance": [],
            "proxy_reliability": [],
            "num_filtered": [],
            "weights_entropy": [],
        }
    
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[ClientUpdate],
        **kwargs,
    ) -> nn.Module:
        """
        Aggregate client updates using DAFA algorithm.
        
        Args:
            global_model: Current global model
            client_updates: List of client updates
        
        Returns:
            Updated global model
        """
        if not client_updates:
            return global_model
        
        global_params = self.get_model_params(global_model)
        
        updates = [self.get_update_tensor(u, global_params.device) for u in client_updates]
        data_sizes = [u.num_samples for u in client_updates]
        
        delta_fedavg, weights_fedavg = self._compute_fedavg_update(updates, data_sizes)
        
        if self.momentum is None:
            self.momentum = delta_fedavg.clone()
        else:
            if self.momentum.device != delta_fedavg.device:
                self.momentum = self.momentum.to(delta_fedavg.device)
            self.momentum = (1 - self.beta) * delta_fedavg + self.beta * self.momentum
        
        proxy_direction = self.momentum / (self.momentum.norm() + 1e-10)
        
        alignment_scores, filtered_mask = self._compute_alignment_scores(
            updates, proxy_direction
        )
        
        dafa_weights = self._compute_dafa_weights(
            alignment_scores, data_sizes, filtered_mask
        )
        
        aggregated_update = torch.zeros_like(global_params)
        for i, update in enumerate(client_updates):
            update_tensor = self.get_update_tensor(update, global_params.device)
            aggregated_update += dafa_weights[i] * update_tensor
        
        new_params = global_params + self.server_lr * aggregated_update
        self.set_model_params(global_model, new_params)
        
        self._record_metrics(
            updates, aggregated_update, proxy_direction,
            alignment_scores, dafa_weights, filtered_mask
        )
        
        self.step()
        
        return global_model
    
    def _compute_fedavg_update(
        self,
        updates: List[torch.Tensor],
        data_sizes: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FedAvg update direction.
        
        Args:
            updates: List of client update tensors
            data_sizes: List of client data sizes
        
        Returns:
            Tuple of (fedavg_update, weights)
        """
        total_samples = sum(data_sizes)
        weights = torch.tensor(
            [s / total_samples for s in data_sizes],
            device=updates[0].device,
            dtype=torch.float32,
        )
        
        fedavg_update = torch.zeros_like(updates[0])
        for i, update in enumerate(updates):
            fedavg_update += weights[i] * update
        
        return fedavg_update, weights
    
    def _compute_alignment_scores(
        self,
        updates: List[torch.Tensor],
        proxy_direction: torch.Tensor,
    ) -> Tuple[List[float], List[bool]]:
        """
        Compute alignment scores for each update.
        
        Alignment score s_i = <v_i, v*> where v_i = Δ_i / ||Δ_i||
        If ||Δ_i|| < μ, then s_i = 0 (graceful degradation).
        
        Args:
            updates: List of client update tensors
            proxy_direction: Normalized proxy direction v*
        
        Returns:
            Tuple of (alignment_scores, filtered_mask)
        """
        alignment_scores = []
        filtered_mask = []
        
        for update in updates:
            norm = update.norm().item()
            
            if norm >= self.mu:
                v_i = update / (norm + 1e-10)
                score = torch.dot(v_i, proxy_direction).item()
                score = max(-1.0, min(1.0, score))
                filtered_mask.append(False)
            else:
                score = 0.0
                filtered_mask.append(True)
            
            alignment_scores.append(score)
        
        return alignment_scores, filtered_mask
    
    def _compute_dafa_weights(
        self,
        alignment_scores: List[float],
        data_sizes: List[int],
        filtered_mask: List[bool],
    ) -> torch.Tensor:
        """
        Compute DAFA weights using softmax with temperature.
        
        w_i = p_i * exp(γ * s_i) / Σ p_j * exp(γ * s_j)
        
        Args:
            alignment_scores: List of alignment scores
            data_sizes: List of client data sizes
            filtered_mask: List indicating if client was filtered
        
        Returns:
            Normalized weight tensor
        """
        device = torch.device(self.device)
        scores_tensor = torch.tensor(alignment_scores, device=device, dtype=torch.float32)
        
        if self.use_pi_weighting:
            p_i = torch.tensor(data_sizes, device=device, dtype=torch.float32)
            p_i = p_i / p_i.sum()
        else:
            p_i = torch.ones(len(data_sizes), device=device, dtype=torch.float32) / len(data_sizes)
        
        exp_scores = torch.exp(self.gamma * scores_tensor)
        weights = p_i * exp_scores
        
        weights = weights / (weights.sum() + 1e-10)
        
        return weights
    
    def _record_metrics(
        self,
        updates: List[torch.Tensor],
        aggregated_update: torch.Tensor,
        proxy_direction: torch.Tensor,
        alignment_scores: List[float],
        weights: torch.Tensor,
        filtered_mask: List[bool],
    ) -> None:
        """Record metrics for analysis."""
        dsnr = self.compute_dsnr(updates, aggregated_update)
        self.history["dsnr"].append(dsnr)
        
        scores_array = torch.tensor(alignment_scores)
        self.history["alignment_scores"].append(alignment_scores)
        self.history["alignment_mean"].append(scores_array.mean().item())
        self.history["alignment_std"].append(scores_array.std().item() if len(scores_array) > 1 else 0.0)
        
        variance = self._compute_update_variance(updates, aggregated_update)
        self.history["update_variance"].append(variance)
        
        num_filtered = sum(filtered_mask)
        self.history["num_filtered"].append(num_filtered)
        
        weights_entropy = self._compute_weights_entropy(weights)
        self.history["weights_entropy"].append(weights_entropy)
    
    def compute_dsnr(
        self,
        updates: List[torch.Tensor],
        aggregated_update: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute Directional Signal-to-Noise Ratio.
        
        DSNR = ||Δ_agg|| / std(||Δ_i - Δ_agg||)
        
        Args:
            updates: List of client update tensors
            aggregated_update: Aggregated update (computed if None)
        
        Returns:
            DSNR value
        """
        if not updates:
            return 0.0
        
        if aggregated_update is None:
            aggregated_update = torch.stack(updates).mean(dim=0)
        
        signal = aggregated_update.norm().item()
        
        deviations = [u - aggregated_update for u in updates]
        deviation_norms = torch.tensor([d.norm().item() for d in deviations])
        noise = deviation_norms.std().item()
        
        if noise < 1e-10:
            return float('inf')
        
        return signal / noise
    
    def compute_decentralized_dsnr(
        self,
        updates: List[torch.Tensor],
        aggregated_update: torch.Tensor,
    ) -> float:
        """
        Compute decentralized DSNR proxy (privacy-preserving).
        
        DSNR_decentralized = <Δ_agg, m_t>² / (1/|S_t| * Σ ||Δ_i - Δ_agg||²)
        
        This metric uses the momentum vector instead of true global direction,
        making it suitable for privacy-sensitive scenarios.
        
        Args:
            updates: List of client update tensors
            aggregated_update: Aggregated update
        
        Returns:
            Decentralized DSNR value
        """
        if self.momentum is None:
            return 0.0
        
        signal = torch.dot(aggregated_update, self.momentum).item() ** 2
        
        deviations = [u - aggregated_update for u in updates]
        noise = sum(d.norm().item() ** 2 for d in deviations) / len(deviations)
        
        return signal / (noise + 1e-10)
    
    def _compute_update_variance(
        self,
        updates: List[torch.Tensor],
        mean_update: torch.Tensor,
    ) -> float:
        """Compute variance of client updates."""
        if not updates:
            return 0.0
        
        deviations = [u - mean_update for u in updates]
        variance = sum(d.norm().item() ** 2 for d in deviations) / len(deviations)
        
        return variance
    
    def _compute_weights_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy of weight distribution."""
        weights = weights + 1e-10
        entropy = -(weights * torch.log(weights)).sum().item()
        max_entropy = math.log(len(weights))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def get_proxy_direction(self) -> Optional[torch.Tensor]:
        """Get current proxy direction v*."""
        if self.momentum is None:
            return None
        return self.momentum / (self.momentum.norm() + 1e-10)
    
    def get_alignment_scores(self) -> List[List[float]]:
        """Get history of alignment scores."""
        return self.history["alignment_scores"]
    
    def get_dsnr_history(self) -> List[float]:
        """Get history of DSNR values."""
        return self.history["dsnr"]
    
    def get_variance_history(self) -> List[float]:
        """Get history of update variance."""
        return self.history["update_variance"]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get aggregator state for checkpointing."""
        state = super().state_dict()
        state["gamma"] = self.gamma
        state["beta"] = self.beta
        state["mu"] = self.mu
        state["use_pi_weighting"] = self.use_pi_weighting
        state["server_lr"] = self.server_lr
        state["momentum"] = self.momentum.cpu() if self.momentum is not None else None
        state["history"] = self.history
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        super().load_state_dict(state)
        self.gamma = state.get("gamma", 1.0)
        self.beta = state.get("beta", 0.9)
        self.mu = state.get("mu", 0.01)
        self.use_pi_weighting = state.get("use_pi_weighting", True)
        self.server_lr = state.get("server_lr", 1.0)
        
        if state.get("momentum") is not None:
            self.momentum = state["momentum"].to(self.device)
        
        self.history = state.get("history", self.history)
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self.momentum = None
        self.history = {
            "dsnr": [],
            "alignment_scores": [],
            "alignment_mean": [],
            "alignment_std": [],
            "update_variance": [],
            "proxy_reliability": [],
            "num_filtered": [],
            "weights_entropy": [],
        }


def compute_update_norm_stats(updates: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics of update norms.
    
    Args:
        updates: List of client update tensors
    
    Returns:
        Dictionary with norm statistics
    """
    if not updates:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    norms = torch.tensor([u.norm().item() for u in updates])
    
    return {
        "mean": norms.mean().item(),
        "std": norms.std().item() if len(norms) > 1 else 0.0,
        "min": norms.min().item(),
        "max": norms.max().item(),
    }
