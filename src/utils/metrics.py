"""Metrics utilities for DAFA experiment framework."""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field


@dataclass
class AverageMeter:
    """Computes and stores the average and current value."""
    
    name: str = ""
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value.
        
        Args:
            val: New value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple = (1,),
) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = predictions.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def compute_perplexity(
    loss: float,
    base: float = np.e,
) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        base: Base for exponentiation (default: e)
    
    Returns:
        Perplexity value
    """
    return base ** loss


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task: str = "classification",
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute various metrics for model evaluation.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        task: Task type (classification, regression, etc.)
        num_classes: Number of classes (for classification)
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    if task == "classification":
        acc1, acc5 = compute_accuracy(predictions, targets, topk=(1, 5))
        metrics["accuracy"] = acc1
        metrics["top5_accuracy"] = acc5
        
        if num_classes and num_classes > 5:
            pred_labels = predictions.argmax(dim=1)
            metrics["balanced_accuracy"] = compute_balanced_accuracy(
                pred_labels.cpu().numpy(),
                targets.cpu().numpy(),
                num_classes,
            )
    
    return metrics


def compute_balanced_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> float:
    """
    Compute balanced accuracy (average per-class accuracy).
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Balanced accuracy value
    """
    per_class_acc = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_acc = (predictions[mask] == c).mean()
            per_class_acc.append(class_acc)
    
    return np.mean(per_class_acc) if per_class_acc else 0.0


def compute_fairness(
    client_accuracies: List[float],
    metric: str = "std",
) -> float:
    """
    Compute fairness metric across clients.
    
    Args:
        client_accuracies: List of per-client accuracies
        metric: Fairness metric type (std, min, range)
    
    Returns:
        Fairness metric value
    """
    if not client_accuracies:
        return 0.0
    
    acc_array = np.array(client_accuracies)
    
    if metric == "std":
        return float(np.std(acc_array))
    elif metric == "min":
        return float(np.min(acc_array))
    elif metric == "range":
        return float(np.max(acc_array) - np.min(acc_array))
    else:
        raise ValueError(f"Unknown fairness metric: {metric}")


class MetricsTracker:
    """Track and aggregate metrics over time."""
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize metrics tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names
        self.meters = {name: AverageMeter(name) for name in metric_names}
        self.history = {name: [] for name in metric_names}
    
    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            n: Number of samples
        """
        for name, value in metrics.items():
            if name in self.meters:
                self.meters[name].update(value, n)
    
    def record(self) -> Dict[str, float]:
        """
        Record current averages to history.
        
        Returns:
            Dictionary of current averages
        """
        current = {}
        for name, meter in self.meters.items():
            current[name] = meter.avg
            self.history[name].append(meter.avg)
        return current
    
    def reset(self) -> None:
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get history for a specific metric.
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            List of historical values
        """
        return self.history.get(metric_name, [])
    
    def get_best(self, metric_name: str, mode: str = "max") -> float:
        """
        Get best value for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: "max" or "min"
        
        Returns:
            Best value
        """
        history = self.get_history(metric_name)
        if not history:
            return 0.0
        
        if mode == "max":
            return max(history)
        elif mode == "min":
            return min(history)
        else:
            raise ValueError(f"Unknown mode: {mode}")
