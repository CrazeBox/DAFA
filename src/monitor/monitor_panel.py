"""
Monitor panel manager for real-time training visualization.

Integrates the realtime monitor with the terminal renderer to provide
a complete monitoring solution.
"""

import os
import sys
import time
import threading
from typing import Optional, Callable, Any
from datetime import datetime

from .realtime_monitor import (
    RealtimeMonitor,
    DataItem,
    DataStatus,
    ClientStatus,
    TrainingOverview,
    LoadState,
)
from .terminal_renderer import TerminalRenderer, ResponsiveLayout


class MonitorPanel:
    def __init__(
        self,
        total_rounds: int = 100,
        max_items: int = 1000,
        page_size: int = 20,
        refresh_rate: float = 0.1,
        enable_colors: bool = True,
    ):
        self.monitor = RealtimeMonitor(
            max_items=max_items,
            page_size=page_size,
        )
        self.monitor.set_total_rounds(total_rounds)
        
        self.renderer = TerminalRenderer()
        self.layout = ResponsiveLayout(self.renderer)
        
        self.refresh_rate = refresh_rate
        self._running = False
        self._render_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._on_data_callback: Optional[Callable] = None
        self._start_time = time.time()
    
    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self.renderer.reset()
        
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()
    
    def stop(self) -> None:
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=1.0)
        
        sys.stdout.write(self.renderer.show_cursor())
        sys.stdout.flush()
    
    def _render_loop(self) -> None:
        while self._running:
            try:
                output = self.layout.render(self.monitor)
                sys.stdout.write(output)
                sys.stdout.flush()
            except Exception as e:
                pass
            
            time.sleep(self.refresh_rate)
    
    def add_round_data(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        dsnr: Optional[float] = None,
        variance: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> None:
        item = DataItem(
            timestamp=datetime.now(),
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            dsnr=dsnr,
            variance=variance,
            extra=extra or {},
        )
        
        self.monitor.add_data(item)
        
        if self._on_data_callback:
            self._on_data_callback(item)
    
    def update_client(
        self,
        client_id: int,
        state: str,
        progress: float = 0.0,
        loss: float = 0.0,
        samples: int = 0,
    ) -> None:
        status = ClientStatus(
            client_id=client_id,
            state=state,
            progress=progress,
            loss=loss,
            samples=samples,
        )
        self.monitor.update_client(client_id, status)
    
    def update_gpu(self, utilization: float, memory: float) -> None:
        self.monitor.set_gpu_info(utilization, memory)
    
    def update_overview(self, **kwargs) -> None:
        self.monitor.update_overview(**kwargs)
    
    def set_total_rounds(self, total: int) -> None:
        self.monitor.set_total_rounds(total)
    
    def pause(self) -> None:
        self.monitor.pause()
    
    def resume(self) -> None:
        self.monitor.resume()
    
    def toggle_pause(self) -> bool:
        return self.monitor.toggle_pause()
    
    def set_filter(
        self,
        status: Optional[list] = None,
        min_accuracy: Optional[float] = None,
        max_accuracy: Optional[float] = None,
        min_round: Optional[int] = None,
        max_round: Optional[int] = None,
    ) -> None:
        self.monitor.set_filter(
            status_filter=status,
            min_accuracy=min_accuracy,
            max_accuracy=max_accuracy,
            min_round=min_round,
            max_round=max_round,
        )
    
    def set_sorter(self, sort_by: str = "time", ascending: bool = False) -> None:
        self.monitor.set_sorter(sort_by, ascending)
    
    def export_data(self, format: str = "json", filepath: Optional[str] = None) -> str:
        return self.monitor.export_data(format, filepath)
    
    def get_stats(self) -> dict:
        return self.monitor.get_stats()
    
    def get_data(self, page: int = 0, apply_filter: bool = True, apply_sort: bool = True):
        return self.monitor.get_data(page, apply_filter, apply_sort)
    
    def clear(self) -> None:
        self.monitor.clear()
    
    def on_data(self, callback: Callable) -> None:
        self._on_data_callback = callback
    
    def __enter__(self) -> "MonitorPanel":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class TrainingMonitorWrapper:
    def __init__(self, panel: MonitorPanel):
        self.panel = panel
        self._prev_accuracy = 0.0
        self._client_states: dict = {}
    
    def on_round_start(self, round_num: int, selected_clients: list) -> None:
        for client_id in selected_clients:
            self.panel.update_client(
                client_id=client_id,
                state="waiting",
                progress=0.0,
            )
    
    def on_client_start(self, client_id: int) -> None:
        self.panel.update_client(
            client_id=client_id,
            state="training",
            progress=0.0,
        )
    
    def on_client_progress(self, client_id: int, progress: float, loss: float = 0.0) -> None:
        self.panel.update_client(
            client_id=client_id,
            state="training",
            progress=progress,
            loss=loss,
        )
    
    def on_client_complete(self, client_id: int, loss: float, samples: int) -> None:
        self.panel.update_client(
            client_id=client_id,
            state="completed",
            progress=1.0,
            loss=loss,
            samples=samples,
        )
    
    def on_round_complete(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        dsnr: Optional[float] = None,
        variance: Optional[float] = None,
    ) -> None:
        self.panel.add_round_data(
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            dsnr=dsnr,
            variance=variance,
        )
        
        self._prev_accuracy = accuracy
    
    def on_training_complete(self, best_accuracy: float, total_time: float) -> None:
        self.panel.update_overview(
            best_accuracy=best_accuracy,
        )
    
    def update_gpu_info(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                utilization = torch.cuda.utilization(0)
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory
                memory_percent = (memory_allocated / memory_total) * 100
                
                self.panel.update_gpu(utilization, memory_percent)
        except Exception:
            pass


def create_monitor(
    total_rounds: int = 100,
    num_clients: int = 10,
    **kwargs,
) -> MonitorPanel:
    panel = MonitorPanel(
        total_rounds=total_rounds,
        **kwargs,
    )
    
    for i in range(num_clients):
        panel.update_client(
            client_id=i,
            state="waiting",
        )
    
    return panel
