"""
Real-time monitoring panel for federated learning experiments.

This module provides a comprehensive real-time data monitoring interface
that replaces traditional progress bars with a dynamic data display list.
"""

import time
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque
import json
import os
import sys


class DataStatus(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    DECLINE = "decline"
    INFO = "info"
    LOADING = "loading"
    ERROR = "error"


class LoadState(Enum):
    LOADING = "loading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DataItem:
    timestamp: datetime
    round_num: int
    accuracy: float
    loss: float
    dsnr: Optional[float] = None
    variance: Optional[float] = None
    status: DataStatus = DataStatus.INFO
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "round_num": self.round_num,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "dsnr": self.dsnr,
            "variance": self.variance,
            "status": self.status.value,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataItem":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            round_num=data["round_num"],
            accuracy=data["accuracy"],
            loss=data["loss"],
            dsnr=data.get("dsnr"),
            variance=data.get("variance"),
            status=DataStatus(data.get("status", "info")),
            extra=data.get("extra", {}),
        )


@dataclass
class ClientStatus:
    client_id: int
    state: str
    progress: float = 0.0
    loss: float = 0.0
    samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "state": self.state,
            "progress": self.progress,
            "loss": self.loss,
            "samples": self.samples,
        }


@dataclass
class TrainingOverview:
    current_round: int = 0
    total_rounds: int = 100
    best_accuracy: float = 0.0
    current_accuracy: float = 0.0
    current_loss: float = 0.0
    current_dsnr: float = 0.0
    elapsed_time: float = 0.0
    remaining_time: float = 0.0
    speed: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        return (self.current_round / self.total_rounds) * 100 if self.total_rounds > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "best_accuracy": self.best_accuracy,
            "current_accuracy": self.current_accuracy,
            "current_loss": self.current_loss,
            "current_dsnr": self.current_dsnr,
            "elapsed_time": self.elapsed_time,
            "remaining_time": self.remaining_time,
            "speed": self.speed,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory": self.gpu_memory,
            "progress_percent": self.progress_percent,
        }


class DataFilter:
    def __init__(self):
        self.status_filter: Optional[List[DataStatus]] = None
        self.min_accuracy: Optional[float] = None
        self.max_accuracy: Optional[float] = None
        self.min_round: Optional[int] = None
        self.max_round: Optional[int] = None
        self.time_range: Optional[tuple] = None
    
    def apply(self, items: List[DataItem]) -> List[DataItem]:
        filtered = items
        
        if self.status_filter:
            filtered = [item for item in filtered if item.status in self.status_filter]
        
        if self.min_accuracy is not None:
            filtered = [item for item in filtered if item.accuracy >= self.min_accuracy]
        
        if self.max_accuracy is not None:
            filtered = [item for item in filtered if item.accuracy <= self.max_accuracy]
        
        if self.min_round is not None:
            filtered = [item for item in filtered if item.round_num >= self.min_round]
        
        if self.max_round is not None:
            filtered = [item for item in filtered if item.round_num <= self.max_round]
        
        return filtered


class DataSorter:
    SORT_BY_TIME = "time"
    SORT_BY_ROUND = "round"
    SORT_BY_ACCURACY = "accuracy"
    SORT_BY_LOSS = "loss"
    SORT_BY_DSNR = "dsnr"
    
    def __init__(self, sort_by: str = "time", ascending: bool = False):
        self.sort_by = sort_by
        self.ascending = ascending
    
    def apply(self, items: List[DataItem]) -> List[DataItem]:
        if not items:
            return items
        
        if self.sort_by == self.SORT_BY_TIME:
            key = lambda x: x.timestamp
        elif self.sort_by == self.SORT_BY_ROUND:
            key = lambda x: x.round_num
        elif self.sort_by == self.SORT_BY_ACCURACY:
            key = lambda x: x.accuracy
        elif self.sort_by == self.SORT_BY_LOSS:
            key = lambda x: x.loss
        elif self.sort_by == self.SORT_BY_DSNR:
            key = lambda x: x.dsnr or 0
        else:
            key = lambda x: x.timestamp
        
        return sorted(items, key=key, reverse=not self.ascending)


class RealtimeMonitor:
    def __init__(
        self,
        max_items: int = 1000,
        page_size: int = 20,
        auto_scroll: bool = True,
    ):
        self.max_items = max_items
        self.page_size = page_size
        self.auto_scroll = auto_scroll
        
        self._data: deque = deque(maxlen=max_items)
        self._overview: TrainingOverview = TrainingOverview()
        self._clients: Dict[int, ClientStatus] = {}
        self._filter = DataFilter()
        self._sorter = DataSorter()
        
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._is_paused = False
        self._load_state = LoadState.COMPLETED
        self._start_time = time.time()
        
        self._prev_accuracy = 0.0
    
    def add_data(self, item: DataItem) -> None:
        with self._lock:
            if self._is_paused:
                return
            
            if self._data:
                self._prev_accuracy = self._data[-1].accuracy
            
            if item.accuracy > self._prev_accuracy:
                item.status = DataStatus.SUCCESS
            elif item.accuracy < self._prev_accuracy:
                item.status = DataStatus.DECLINE
            else:
                item.status = DataStatus.WARNING
            
            self._data.append(item)
            
            self._overview.current_round = item.round_num
            self._overview.current_accuracy = item.accuracy
            self._overview.current_loss = item.loss
            self._overview.current_dsnr = item.dsnr or 0.0
            
            if item.accuracy > self._overview.best_accuracy:
                self._overview.best_accuracy = item.accuracy
            
            self._update_timing()
            self._notify_callbacks()
    
    def update_client(self, client_id: int, status: ClientStatus) -> None:
        with self._lock:
            self._clients[client_id] = status
            self._notify_callbacks()
    
    def update_overview(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._overview, key):
                    setattr(self._overview, key, value)
            self._notify_callbacks()
    
    def set_total_rounds(self, total: int) -> None:
        with self._lock:
            self._overview.total_rounds = total
    
    def set_gpu_info(self, utilization: float, memory: float) -> None:
        with self._lock:
            self._overview.gpu_utilization = utilization
            self._overview.gpu_memory = memory
    
    def _update_timing(self) -> None:
        elapsed = time.time() - self._start_time
        self._overview.elapsed_time = elapsed
        
        if self._overview.current_round > 0:
            speed = self._overview.current_round / elapsed if elapsed > 0 else 0
            self._overview.speed = speed
            
            remaining_rounds = self._overview.total_rounds - self._overview.current_round
            self._overview.remaining_time = remaining_rounds / speed if speed > 0 else 0
    
    def get_data(self, page: int = 0, apply_filter: bool = True, apply_sort: bool = True) -> List[DataItem]:
        with self._lock:
            items = list(self._data)
        
        if apply_filter:
            items = self._filter.apply(items)
        
        if apply_sort:
            items = self._sorter.apply(items)
        
        start_idx = page * self.page_size
        end_idx = start_idx + self.page_size
        
        return items[start_idx:end_idx]
    
    def get_all_data(self) -> List[DataItem]:
        with self._lock:
            return list(self._data)
    
    def get_overview(self) -> TrainingOverview:
        with self._lock:
            return self._overview
    
    def get_clients(self) -> Dict[int, ClientStatus]:
        with self._lock:
            return dict(self._clients)
    
    def set_filter(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._filter, key):
                    setattr(self._filter, key, value)
    
    def set_sorter(self, sort_by: str, ascending: bool = False) -> None:
        with self._lock:
            self._sorter = DataSorter(sort_by, ascending)
    
    def pause(self) -> None:
        self._is_paused = True
    
    def resume(self) -> None:
        self._is_paused = False
    
    def toggle_pause(self) -> bool:
        self._is_paused = not self._is_paused
        return self._is_paused
    
    @property
    def is_paused(self) -> bool:
        return self._is_paused
    
    def set_load_state(self, state: LoadState) -> None:
        self._load_state = state
    
    @property
    def load_state(self) -> LoadState:
        return self._load_state
    
    def register_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception:
                pass
    
    def export_data(self, format: str = "json", filepath: Optional[str] = None) -> str:
        data = {
            "overview": self._overview.to_dict(),
            "data": [item.to_dict() for item in self._data],
            "clients": {str(k): v.to_dict() for k, v in self._clients.items()},
        }
        
        if format == "json":
            output = json.dumps(data, indent=2, ensure_ascii=False)
        elif format == "csv":
            import io
            import csv
            
            output_buffer = io.StringIO()
            writer = csv.writer(output_buffer)
            writer.writerow(["timestamp", "round", "accuracy", "loss", "dsnr", "status"])
            
            for item in self._data:
                writer.writerow([
                    item.timestamp.isoformat(),
                    item.round_num,
                    item.accuracy,
                    item.loss,
                    item.dsnr or "",
                    item.status.value,
                ])
            
            output = output_buffer.getvalue()
        else:
            output = str(data)
        
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(output)
        
        return output
    
    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._clients.clear()
            self._overview = TrainingOverview()
            self._start_time = time.time()
            self._prev_accuracy = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._data:
                return {"count": 0}
            
            accuracies = [item.accuracy for item in self._data]
            losses = [item.loss for item in self._data]
            dsnrs = [item.dsnr for item in self._data if item.dsnr is not None]
            
            return {
                "count": len(self._data),
                "accuracy": {
                    "min": min(accuracies),
                    "max": max(accuracies),
                    "mean": sum(accuracies) / len(accuracies),
                },
                "loss": {
                    "min": min(losses),
                    "max": max(losses),
                    "mean": sum(losses) / len(losses),
                },
                "dsnr": {
                    "min": min(dsnrs) if dsnrs else None,
                    "max": max(dsnrs) if dsnrs else None,
                    "mean": sum(dsnrs) / len(dsnrs) if dsnrs else None,
                },
            }
