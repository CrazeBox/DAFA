"""Analysis module for federated learning experiments.

This module provides comprehensive analysis tools for:
1. DSNR (Directional Signal-to-Noise Ratio) computation
2. Decentralized DSNR proxy (privacy-preserving)
3. Drift-alignment correlation analysis
4. Update variance and heterogeneity metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from torch.utils.data import DataLoader


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    
    dsnr: float
    decentralized_dsnr: float
    alignment_scores: np.ndarray
    update_variance: float
    update_norms: np.ndarray
    correlation_matrix: Optional[np.ndarray] = None
    drift_magnitudes: Optional[np.ndarray] = None
    proxy_reliability: Optional[float] = None


class DSNRAnalyzer:
    """
    Analyzer for Directional Signal-to-Noise Ratio (DSNR).
    
    DSNR measures the quality of client updates by comparing the signal
    (alignment with global direction) to noise (deviation from global direction).
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize DSNR analyzer.
        
        Args:
            normalize: Whether to normalize updates before analysis
        """
        self.normalize = normalize
    
    def compute_dsnr(
        self,
        updates: List[torch.Tensor],
        aggregated_update: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute DSNR for a set of updates.
        
        DSNR = ||Δ_agg|| / std(||Δ_i - Δ_agg||)
        
        Args:
            updates: List of client update tensors
            aggregated_update: Pre-computed aggregated update
        
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
        momentum: torch.Tensor,
    ) -> float:
        """
        Compute decentralized DSNR proxy (privacy-preserving).
        
        DSNR_decentralized = <Δ_agg, m_t>² / (1/|S_t| * Σ ||Δ_i - Δ_agg||²)
        
        This metric uses the momentum vector instead of true global direction,
        making it suitable for privacy-sensitive scenarios.
        
        Args:
            updates: List of client update tensors
            aggregated_update: Aggregated update
            momentum: Server momentum vector m_t
        
        Returns:
            Decentralized DSNR value
        """
        if momentum is None:
            return 0.0
        
        agg_device = aggregated_update.device
        momentum_tensor = momentum if momentum.device == agg_device else momentum.to(agg_device)
        
        signal = torch.dot(aggregated_update, momentum_tensor).item() ** 2
        
        deviations = [u - aggregated_update for u in updates]
        noise = sum(d.norm().item() ** 2 for d in deviations) / len(deviations)
        
        return signal / (noise + 1e-10)
    
    def compute_centralized_dsnr(
        self,
        updates: List[torch.Tensor],
        aggregated_update: torch.Tensor,
        true_global_direction: torch.Tensor,
    ) -> float:
        """
        Compute centralized DSNR with true global direction.
        
        DSNR_centralized = <Δ_agg, v_true>² / (||Δ_agg - v_true||² + ε)
        
        Args:
            updates: List of client update tensors
            aggregated_update: Aggregated update
            true_global_direction: True global gradient direction
        
        Returns:
            Centralized DSNR value
        """
        signal = torch.dot(aggregated_update, true_global_direction).item() ** 2
        
        deviation = aggregated_update - true_global_direction
        noise = deviation.norm().item() ** 2
        
        return signal / (noise + 1e-10)
    
    def compute_alignment_scores(
        self,
        updates: List[torch.Tensor],
        proxy_direction: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute alignment scores for each update.
        
        Alignment score = cos(update, proxy_direction)
        
        Args:
            updates: List of client update tensors
            proxy_direction: Proxy direction v*
        
        Returns:
            Array of alignment scores
        """
        if not updates:
            return np.array([])
        
        scores = []
        for update in updates:
            norm = update.norm().item()
            if norm > 1e-10:
                v_i = update / norm
                score = torch.dot(v_i, proxy_direction).item()
            else:
                score = 0.0
            scores.append(score)
        
        return np.array(scores)
    
    def analyze_round(
        self,
        updates: List[torch.Tensor],
        aggregated_update: Optional[torch.Tensor] = None,
        momentum: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis for a round.
        
        Args:
            updates: List of client update tensors
            aggregated_update: Pre-computed aggregated update
            momentum: Server momentum vector
        
        Returns:
            Dictionary with analysis results
        """
        if not updates:
            return {}
        
        if aggregated_update is None:
            aggregated_update = torch.stack(updates).mean(dim=0)
        
        dsnr = self.compute_dsnr(updates, aggregated_update)
        
        decentralized_dsnr = 0.0
        if momentum is not None:
            decentralized_dsnr = self.compute_decentralized_dsnr(
                updates, aggregated_update, momentum
            )
        
        proxy_direction = aggregated_update / (aggregated_update.norm() + 1e-10)
        alignment_scores = self.compute_alignment_scores(updates, proxy_direction)
        
        norms = torch.tensor([u.norm().item() for u in updates])
        
        stacked = torch.stack(updates)
        variance = stacked.var().item()
        
        return {
            "dsnr": dsnr,
            "decentralized_dsnr": decentralized_dsnr,
            "alignment_scores": alignment_scores.tolist(),
            "alignment_mean": float(alignment_scores.mean()),
            "alignment_std": float(alignment_scores.std()) if len(alignment_scores) > 1 else 0.0,
            "update_variance": variance,
            "update_norms": norms.tolist(),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()) if len(norms) > 1 else 0.0,
            "num_clients": len(updates),
        }


class DriftAlignmentAnalyzer:
    """
    Analyzer for drift-alignment correlation.
    
    This analyzer computes the correlation between alignment scores
    and drift magnitudes to validate Theorem 3's claim that
    Γ_DAFA < Γ_FedAvg.
    
    Reference: EXPERIMENT_DESIGN - Experiment 5A
    """
    
    def __init__(
        self,
        validation_loader: DataLoader,
        device: str = "cuda",
        validation_ratio: float = 0.02,
    ):
        """
        Initialize drift-alignment analyzer.
        
        Args:
            validation_loader: DataLoader for validation set (small subset)
            device: Device to use
            validation_ratio: Ratio of validation data to use
        """
        self.validation_loader = validation_loader
        self.device = device
        self.validation_ratio = validation_ratio
        
        self.history: List[Dict[str, Any]] = []
    
    def compute_true_global_gradient(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute true global gradient on validation set.
        
        This approximates the true global direction v_true using
        a full-batch gradient on a small centralized validation subset.
        
        Args:
            model: Current global model
        
        Returns:
            Flattened gradient tensor
        """
        model.eval()
        model.zero_grad()
        
        total_grad = None
        num_batches = 0
        max_batches = max(1, int(len(self.validation_loader) * self.validation_ratio))
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.validation_loader):
                if batch_idx >= max_batches:
                    break
                
                data = data.to(self.device)
                target = target.to(self.device)
                
                model_copy = model
                model_copy.zero_grad()
                
                output = model_copy(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                
                grad = torch.cat([p.grad.view(-1) for p in model_copy.parameters() if p.grad is not None])
                
                if total_grad is None:
                    total_grad = grad.clone()
                else:
                    total_grad += grad
                
                num_batches += 1
        
        model.zero_grad()
        
        if total_grad is None:
            return None
        
        return total_grad / num_batches
    
    def compute_proxy_reliability(
        self,
        proxy_direction: torch.Tensor,
        true_direction: torch.Tensor,
    ) -> float:
        """
        Compute proxy reliability ρ = <v*, v_true>.
        
        Args:
            proxy_direction: Proxy direction v*
            true_direction: True global direction v_true
        
        Returns:
            Proxy reliability value in [-1, 1]
        """
        if proxy_direction is None or true_direction is None:
            return 0.0
        
        proxy_norm = proxy_direction / (proxy_direction.norm() + 1e-10)
        true_norm = true_direction / (true_direction.norm() + 1e-10)
        
        return torch.dot(proxy_norm, true_norm).item()
    
    def analyze_round(
        self,
        model: nn.Module,
        client_updates: List[torch.Tensor],
        proxy_direction: torch.Tensor,
        round_num: int,
    ) -> Dict[str, Any]:
        """
        Analyze drift-alignment correlation for a round.
        
        Computes:
        1. True global direction v_true
        2. Alignment scores s_i = <v_i, v*>
        3. Drift magnitudes ||d_i|| = ||Δ_i - v_true||
        4. Pearson correlation between s_i and ||d_i||
        
        Args:
            model: Current global model
            client_updates: List of client update tensors
            proxy_direction: Proxy direction v*
            round_num: Current round number
        
        Returns:
            Dictionary with analysis results
        """
        true_direction = self.compute_true_global_gradient(model)
        
        if true_direction is None:
            return {"round": round_num, "error": "Could not compute true gradient"}
        
        proxy_reliability = self.compute_proxy_reliability(proxy_direction, true_direction)
        
        alignment_scores = []
        drift_magnitudes = []
        
        for update in client_updates:
            norm = update.norm().item()
            if norm > 1e-10:
                v_i = update / norm
                s_i = torch.dot(v_i, proxy_direction).item()
            else:
                s_i = 0.0
            
            drift = (update - true_direction).norm().item()
            
            alignment_scores.append(s_i)
            drift_magnitudes.append(drift)
        
        from scipy import stats
        if len(alignment_scores) >= 3:
            correlation, p_value = stats.pearsonr(alignment_scores, drift_magnitudes)
        else:
            correlation, p_value = 0.0, 1.0
        
        result = {
            "round": round_num,
            "proxy_reliability": proxy_reliability,
            "alignment_scores": alignment_scores,
            "drift_magnitudes": drift_magnitudes,
            "correlation": correlation,
            "p_value": p_value,
            "alignment_mean": float(np.mean(alignment_scores)),
            "alignment_std": float(np.std(alignment_scores)),
            "drift_mean": float(np.mean(drift_magnitudes)),
            "drift_std": float(np.std(drift_magnitudes)),
        }
        
        self.history.append(result)
        
        return result
    
    def get_scatter_data(self) -> Dict[str, List[float]]:
        """
        Get aggregated scatter plot data across all rounds.
        
        Returns:
            Dictionary with 'alignment_scores' and 'drift_magnitudes' lists
        """
        all_alignment = []
        all_drift = []
        
        for record in self.history:
            all_alignment.extend(record.get("alignment_scores", []))
            all_drift.extend(record.get("drift_magnitudes", []))
        
        return {
            "alignment_scores": all_alignment,
            "drift_magnitudes": all_drift,
        }
    
    def compute_overall_correlation(self) -> Tuple[float, float]:
        """
        Compute overall correlation across all rounds.
        
        Returns:
            Tuple of (correlation, p_value)
        """
        scatter_data = self.get_scatter_data()
        
        if len(scatter_data["alignment_scores"]) < 3:
            return 0.0, 1.0
        
        from scipy import stats
        return stats.pearsonr(
            scatter_data["alignment_scores"],
            scatter_data["drift_magnitudes"]
        )


class VarianceAnalyzer:
    """Analyzer for update variance and heterogeneity."""
    
    def __init__(self):
        """Initialize variance analyzer."""
        pass
    
    def compute_pairwise_distances(
        self,
        updates: List[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute pairwise distances between updates.
        
        Args:
            updates: List of client update tensors
        
        Returns:
            Distance matrix
        """
        n = len(updates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = (updates[i] - updates[j]).norm().item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def compute_update_divergence(
        self,
        updates: List[torch.Tensor],
        aggregated_update: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute divergence metrics for updates.
        
        Args:
            updates: List of client update tensors
            aggregated_update: Pre-computed aggregated update
        
        Returns:
            Dictionary with divergence metrics
        """
        if not updates:
            return {}
        
        if aggregated_update is None:
            aggregated_update = torch.stack(updates).mean(dim=0)
        
        deviations = [u - aggregated_update for u in updates]
        deviation_norms = torch.tensor([d.norm().item() for d in deviations])
        
        return {
            "mean_deviation": float(deviation_norms.mean()),
            "max_deviation": float(deviation_norms.max()),
            "std_deviation": float(deviation_norms.std()) if len(deviation_norms) > 1 else 0.0,
        }
    
    def compute_gradient_variance(
        self,
        updates: List[torch.Tensor],
    ) -> float:
        """
        Compute total gradient variance.
        
        Args:
            updates: List of client update tensors
        
        Returns:
            Total variance
        """
        if not updates:
            return 0.0
        
        stacked = torch.stack(updates)
        return stacked.var().item()


class CorrelationAnalyzer:
    """Analyzer for correlations between metrics."""
    
    def __init__(self):
        """Initialize correlation analyzer."""
        pass
    
    def compute_alignment_drift_correlation(
        self,
        alignment_scores: List[float],
        drift_magnitudes: List[float],
    ) -> Tuple[float, float]:
        """
        Compute correlation between alignment scores and drift magnitudes.
        
        Args:
            alignment_scores: List of alignment scores
            drift_magnitudes: List of drift magnitudes
        
        Returns:
            Tuple of (correlation, p-value)
        """
        from scipy import stats
        
        if len(alignment_scores) != len(drift_magnitudes):
            raise ValueError("Lists must have same length")
        
        if len(alignment_scores) < 3:
            return 0.0, 1.0
        
        correlation, p_value = stats.pearsonr(alignment_scores, drift_magnitudes)
        
        return correlation, p_value
    
    def compute_metric_correlations(
        self,
        metrics_history: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute correlations between different metrics over time.
        
        Args:
            metrics_history: List of metric dictionaries from each round
        
        Returns:
            Dictionary of correlation matrices
        """
        if not metrics_history:
            return {}
        
        from scipy import stats
        
        keys = list(metrics_history[0].keys())
        
        correlations = {}
        
        for i, key1 in enumerate(keys):
            correlations[key1] = {}
            values1 = [m.get(key1, 0) for m in metrics_history]
            
            for j, key2 in enumerate(keys):
                values2 = [m.get(key2, 0) for m in metrics_history]
                
                if len(values1) >= 3 and len(values2) >= 3:
                    try:
                        corr, _ = stats.pearsonr(values1, values2)
                        correlations[key1][key2] = corr
                    except:
                        correlations[key1][key2] = 0.0
                else:
                    correlations[key1][key2] = 0.0
        
        return correlations


class ExperimentAnalyzer:
    """Comprehensive analyzer for federated learning experiments."""
    
    def __init__(
        self,
        output_dir: str = "analysis_results",
        validation_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        """
        Initialize experiment analyzer.
        
        Args:
            output_dir: Directory to save analysis results
            validation_loader: Optional validation loader for drift analysis
            device: Device to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dsnr_analyzer = DSNRAnalyzer()
        self.variance_analyzer = VarianceAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        if validation_loader is not None:
            self.drift_analyzer = DriftAlignmentAnalyzer(
                validation_loader, device=device
            )
        else:
            self.drift_analyzer = None
        
        self.round_analyses: List[Dict[str, Any]] = []
    
    def analyze_round(
        self,
        updates: List[torch.Tensor],
        round_num: int,
        aggregated_update: Optional[torch.Tensor] = None,
        momentum: Optional[torch.Tensor] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis for a round.
        
        Args:
            updates: List of client update tensors
            round_num: Current round number
            aggregated_update: Pre-computed aggregated update
            momentum: Server momentum vector
            extra_metrics: Additional metrics to include
        
        Returns:
            Analysis results dictionary
        """
        dsnr_results = self.dsnr_analyzer.analyze_round(
            updates, aggregated_update, momentum
        )
        variance_results = self.variance_analyzer.compute_update_divergence(
            updates, aggregated_update
        )
        
        results = {
            "round": round_num,
            **dsnr_results,
            **variance_results,
        }
        
        if extra_metrics:
            results.update(extra_metrics)
        
        self.round_analyses.append(results)
        
        return results
    
    def analyze_drift_alignment(
        self,
        model: nn.Module,
        client_updates: List[torch.Tensor],
        proxy_direction: torch.Tensor,
        round_num: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze drift-alignment correlation for a round.
        
        Args:
            model: Current global model
            client_updates: List of client update tensors
            proxy_direction: Proxy direction v*
            round_num: Current round number
        
        Returns:
            Drift analysis results or None if analyzer not initialized
        """
        if self.drift_analyzer is None:
            return None
        
        return self.drift_analyzer.analyze_round(
            model, client_updates, proxy_direction, round_num
        )
    
    def compute_final_analysis(self) -> Dict[str, Any]:
        """
        Compute final analysis across all rounds.
        
        Returns:
            Final analysis results
        """
        if not self.round_analyses:
            return {}
        
        dsnr_values = [r.get("dsnr", 0) for r in self.round_analyses]
        decentralized_dsnr_values = [
            r.get("decentralized_dsnr", 0) for r in self.round_analyses
        ]
        alignment_means = [r.get("alignment_mean", 0) for r in self.round_analyses]
        variances = [r.get("update_variance", 0) for r in self.round_analyses]
        
        final = {
            "num_rounds": len(self.round_analyses),
            "dsnr": {
                "mean": float(np.mean(dsnr_values)),
                "std": float(np.std(dsnr_values)),
                "min": float(np.min(dsnr_values)),
                "max": float(np.max(dsnr_values)),
            },
            "decentralized_dsnr": {
                "mean": float(np.mean(decentralized_dsnr_values)),
                "std": float(np.std(decentralized_dsnr_values)),
            },
            "alignment": {
                "mean": float(np.mean(alignment_means)),
                "std": float(np.std(alignment_means)),
            },
            "variance": {
                "mean": float(np.mean(variances)),
                "std": float(np.std(variances)),
            },
        }
        
        if len(self.round_analyses) >= 3:
            metrics_for_corr = [
                {
                    "dsnr": r.get("dsnr", 0),
                    "decentralized_dsnr": r.get("decentralized_dsnr", 0),
                    "alignment": r.get("alignment_mean", 0),
                    "variance": r.get("update_variance", 0),
                }
                for r in self.round_analyses
            ]
            final["correlations"] = self.correlation_analyzer.compute_metric_correlations(
                metrics_for_corr
            )
        
        if self.drift_analyzer is not None:
            overall_corr, p_value = self.drift_analyzer.compute_overall_correlation()
            final["drift_alignment"] = {
                "correlation": overall_corr,
                "p_value": p_value,
                "num_samples": len(self.drift_analyzer.get_scatter_data()["alignment_scores"]),
            }
        
        return final
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save analysis results to file.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            from datetime import datetime
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        results = {
            "round_analyses": self.round_analyses,
            "final_analysis": self.compute_final_analysis(),
        }
        
        if self.drift_analyzer is not None:
            results["drift_scatter_data"] = self.drift_analyzer.get_scatter_data()
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        
        return str(output_path)
    
    def load_results(self, filepath: str) -> None:
        """
        Load analysis results from file.
        
        Args:
            filepath: Path to results file
        """
        with open(filepath, "r") as f:
            results = json.load(f)
        
        self.round_analyses = results.get("round_analyses", [])
