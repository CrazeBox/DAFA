"""Checkpoint utilities for saving and loading experiment state."""

import os
import json
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class CheckpointState:
    """Checkpoint state container."""
    
    round: int
    global_model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    best_metric: float = 0.0
    metrics_history: Optional[Dict[str, List[float]]] = None
    config: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    extra_state: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def save_checkpoint(
    state: CheckpointState,
    save_dir: str,
    filename: Optional[str] = None,
    is_best: bool = False,
    best_filename: str = "best_model.pt",
) -> str:
    """
    Save checkpoint to disk.
    
    Args:
        state: CheckpointState object containing all state
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename (default: checkpoint_round_{round}.pt)
        is_best: Whether this is the best model so far
        best_filename: Filename for best model
    
    Returns:
        Path to saved checkpoint
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_round_{state.round}.pt"
    
    filepath = save_dir / filename
    torch.save(asdict(state), filepath)
    
    if is_best:
        best_path = save_dir / best_filename
        torch.save(asdict(state), best_path)
    
    return str(filepath)


def load_checkpoint(
    filepath: str,
    map_location: Optional[str] = None,
) -> CheckpointState:
    """
    Load checkpoint from disk.
    
    Args:
        filepath: Path to checkpoint file
        map_location: Device to map tensors to
    
    Returns:
        CheckpointState object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    state_dict = torch.load(filepath, map_location=map_location)
    return CheckpointState(**state_dict)


class CheckpointManager:
    """Manager for handling multiple checkpoints with rotation."""
    
    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        best_metric_mode: str = "max",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to only save best checkpoints
            best_metric_mode: "max" or "min" for best metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric_mode = best_metric_mode
        self.best_metric = float("-inf") if best_metric_mode == "max" else float("inf")
        self.checkpoints: List[str] = []
        
        self._load_checkpoint_list()
    
    def _load_checkpoint_list(self) -> None:
        """Load list of existing checkpoints."""
        self.checkpoints = sorted([
            f for f in os.listdir(self.save_dir)
            if f.startswith("checkpoint_round_") and f.endswith(".pt")
        ])
    
    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.best_metric_mode == "max":
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    def save(
        self,
        state: Any,
        round_num: Optional[int] = None,
        is_best: bool = False,
        current_metric: Optional[float] = None,
        force_save: bool = False,
    ) -> Optional[str]:
        """
        Save checkpoint if conditions are met.
        
        Args:
            state: State dict or CheckpointState to save
            round_num: Current round number (used if state is dict)
            is_best: Whether this is the best model
            current_metric: Current metric value
            force_save: Force save regardless of conditions
        
        Returns:
            Path to saved checkpoint or None
        """
        if isinstance(state, dict):
            if round_num is None:
                round_num = state.get("round", 0)
            checkpoint_state = CheckpointState(
                round=round_num,
                global_model_state=state,
                best_metric=current_metric or 0.0,
            )
        else:
            checkpoint_state = state
        
        if current_metric is not None and self._is_better(current_metric):
            self.best_metric = current_metric
            is_best = True
        
        if self.save_best_only and not is_best and not force_save:
            return None
        
        filepath = save_checkpoint(
            checkpoint_state,
            str(self.save_dir),
            is_best=is_best,
        )
        
        self.checkpoints.append(os.path.basename(filepath))
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def load(
        self,
        filepath: Optional[str] = None,
        map_location: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.
        
        Args:
            filepath: Path to checkpoint file (if None, load latest)
            map_location: Device to map tensors to
        
        Returns:
            State dict or None if not found
        """
        if filepath is None:
            result = self.load_latest(map_location)
            if result is None:
                return None
            if isinstance(result, CheckpointState):
                return asdict(result)
            return result
        
        if not os.path.exists(filepath):
            return None
        
        state_dict = torch.load(filepath, map_location=map_location)
        
        if isinstance(state_dict, dict):
            if "global_model_state" in state_dict:
                inner_state = state_dict.get("global_model_state")
                if isinstance(inner_state, dict) and "round" in inner_state:
                    return inner_state
            return state_dict
        
        return state_dict
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = self.save_dir / old_checkpoint
            if old_path.exists():
                os.remove(old_path)
    
    def load_latest(self, map_location: Optional[str] = None) -> Optional[CheckpointState]:
        """
        Load the latest checkpoint.
        
        Args:
            map_location: Device to map tensors to
        
        Returns:
            CheckpointState or None if no checkpoints exist
        """
        self._load_checkpoint_list()
        
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        return load_checkpoint(str(self.save_dir / latest), map_location)
    
    def load_best(self, map_location: Optional[str] = None) -> Optional[CheckpointState]:
        """
        Load the best checkpoint.
        
        Args:
            map_location: Device to map tensors to
        
        Returns:
            CheckpointState or None if no best checkpoint exists
        """
        best_path = self.save_dir / "best_model.pt"
        if best_path.exists():
            return load_checkpoint(str(best_path), map_location)
        return None
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all checkpoints."""
        info = []
        for ckpt_name in self.checkpoints:
            ckpt_path = self.save_dir / ckpt_name
            if ckpt_path.exists():
                stat = os.stat(ckpt_path)
                info.append({
                    "name": ckpt_name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return info


def save_experiment_state(
    save_dir: str,
    round: int,
    global_model: torch.nn.Module,
    aggregator_state: Optional[Dict] = None,
    client_states: Optional[Dict] = None,
    metrics_history: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> str:
    """
    Save complete experiment state for resuming.
    
    Args:
        save_dir: Directory to save state
        round: Current round number
        global_model: Global model
        aggregator_state: Aggregator state dict
        client_states: Client states dict
        metrics_history: Metrics history dict
        config: Experiment config
    
    Returns:
        Path to saved state
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    state = {
        "round": round,
        "global_model_state": global_model.state_dict(),
        "aggregator_state": aggregator_state,
        "client_states": client_states,
        "metrics_history": metrics_history,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    
    filepath = save_dir / f"experiment_state_round_{round}.pt"
    torch.save(state, filepath)
    
    latest_path = save_dir / "latest_state.pt"
    torch.save(state, latest_path)
    
    return str(filepath)


def load_experiment_state(
    filepath: str,
    global_model: torch.nn.Module,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load experiment state for resuming.
    
    Args:
        filepath: Path to state file
        global_model: Global model to load state into
        map_location: Device to map tensors to
    
    Returns:
        Dictionary containing loaded state
    """
    state = torch.load(filepath, map_location=map_location)
    
    global_model.load_state_dict(state["global_model_state"])
    
    return {
        "round": state["round"],
        "aggregator_state": state.get("aggregator_state"),
        "client_states": state.get("client_states"),
        "metrics_history": state.get("metrics_history"),
        "config": state.get("config"),
        "timestamp": state.get("timestamp"),
    }
