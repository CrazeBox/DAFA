"""
Terminal UI renderer for real-time monitoring panel.

Provides a compact, fixed-size terminal interface with smooth refresh.
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from colorama import init as colorama_init, Fore, Back, Style
    colorama_init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    Fore = Back = Style = type('Dummy', (), {'__getattr__': lambda s, n: ''})()

try:
    import shutil
    TERMINAL_WIDTH = shutil.get_terminal_size().columns
    TERMINAL_HEIGHT = shutil.get_terminal_size().lines
except Exception:
    TERMINAL_WIDTH = 100
    TERMINAL_HEIGHT = 30


class ColorScheme:
    PRIMARY = Fore.CYAN if HAS_COLORAMA else ""
    SUCCESS = Fore.GREEN if HAS_COLORAMA else ""
    WARNING = Fore.YELLOW if HAS_COLORAMA else ""
    ERROR = Fore.RED if HAS_COLORAMA else ""
    INFO = Fore.BLUE if HAS_COLORAMA else ""
    MUTED = Fore.LIGHTBLACK_EX if HAS_COLORAMA else ""
    BRIGHT = Style.BRIGHT if HAS_COLORAMA else ""
    RESET = Style.RESET_ALL if HAS_COLORAMA else ""


class TerminalRenderer:
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        self.width = width or TERMINAL_WIDTH
        self.height = height or TERMINAL_HEIGHT
        self._animation_frame = 0
        self._initialized = False
        self._max_data_rows = 5
    
    def clear_screen(self) -> str:
        return "\033[2J\033[H" if HAS_COLORAMA else "\n" * 50
    
    def move_cursor_top(self) -> str:
        return "\033[H" if HAS_COLORAMA else ""
    
    def hide_cursor(self) -> str:
        return "\033[?25l" if HAS_COLORAMA else ""
    
    def show_cursor(self) -> str:
        return "\033[?25h" if HAS_COLORAMA else ""
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
    
    def _get_status_icon(self, status) -> str:
        from .realtime_monitor import DataStatus
        
        icons = {
            DataStatus.SUCCESS: f"{ColorScheme.SUCCESS}●{ColorScheme.RESET}",
            DataStatus.WARNING: f"{ColorScheme.WARNING}●{ColorScheme.RESET}",
            DataStatus.DECLINE: f"{ColorScheme.ERROR}▼{ColorScheme.RESET}",
            DataStatus.INFO: f"{ColorScheme.INFO}●{ColorScheme.RESET}",
        }
        return icons.get(status, "●")
    
    def render(self, overview: Any, items: List[Any], clients: Dict[int, Any], 
               is_paused: bool, stats: Dict[str, Any]) -> str:
        self._animation_frame += 1
        
        lines = []
        
        status = f"{ColorScheme.WARNING}[PAUSED]{ColorScheme.RESET}" if is_paused else f"{ColorScheme.SUCCESS}[LIVE]{ColorScheme.RESET}"
        header = f"{ColorScheme.BRIGHT}{'═' * self.width}{ColorScheme.RESET}"
        title = f"│ {ColorScheme.PRIMARY}{ColorScheme.BRIGHT}▶ DAFA 联邦学习监控{ColorScheme.RESET}"
        title += " " * (self.width - len("│ ▶ DAFA 联邦学习监控") - 10)
        title += f"{status} │"
        
        lines.append(header)
        lines.append(title)
        lines.append(header)
        
        progress_bar = self._render_progress_bar(overview.progress_percent / 100, 20)
        elapsed = self._format_time(overview.elapsed_time)
        remaining = self._format_time(overview.remaining_time)
        
        line1 = f"│ Round: {overview.current_round:>3}/{overview.total_rounds} {progress_bar} {overview.progress_percent:>5.1f}% │ Time: {elapsed} / {remaining} │"
        acc_color = ColorScheme.SUCCESS if overview.current_accuracy > 0.5 else ColorScheme.WARNING
        line2 = f"│ {acc_color}ACC: {overview.current_accuracy*100:>6.2f}%{ColorScheme.RESET} │ Loss: {overview.current_loss:>7.4f} │ DSNR: {overview.current_dsnr or 0:>5.2f} │ GPU: {overview.gpu_utilization:>3.0f}% │"
        
        lines.append(line1)
        lines.append(line2)
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        lines.append(f"│ {ColorScheme.INFO}最新数据:{ColorScheme.RESET}" + " " * (self.width - 12) + "│")
        
        display_items = items[:self._max_data_rows]
        if not display_items:
            lines.append(f"│ {ColorScheme.MUTED}{'等待训练数据...':^{self.width - 4}}{ColorScheme.RESET} │")
        else:
            for item in display_items:
                icon = self._get_status_icon(item.status)
                time_str = item.timestamp.strftime("%H:%M:%S")
                dsnr_str = f"{item.dsnr:.1f}" if item.dsnr else "N/A"
                acc_color = ColorScheme.SUCCESS if item.accuracy > overview.current_accuracy * 0.9 else ""
                
                data_line = f"│ {icon} R{item.round_num:>3} {time_str} │ {acc_color}ACC:{item.accuracy*100:>5.1f}%{ColorScheme.RESET} │ Loss:{item.loss:.3f} │ DSNR:{dsnr_str:>4} │"
                data_line += " " * (self.width - len(data_line) + 20) + "│"
                lines.append(data_line)
        
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        client_items = list(clients.items())[:8]
        if client_items:
            client_strs = []
            for cid, c in client_items:
                if c.state == "completed":
                    ico = f"{ColorScheme.SUCCESS}●{ColorScheme.RESET}"
                elif c.state == "training":
                    ico = f"{ColorScheme.WARNING}◐{ColorScheme.RESET}"
                else:
                    ico = f"{ColorScheme.MUTED}○{ColorScheme.RESET}"
                client_strs.append(f"{ico}C{cid}")
            
            client_line = "│ Clients: " + " ".join(client_strs)
            client_line += " " * (self.width - len(client_line) - 1) + "│"
            lines.append(client_line)
        
        acc_stats = stats.get("accuracy", {})
        count = stats.get('count', 0)
        if count > 0:
            stats_line = f"│ Stats: {count} rounds │ ACC: {acc_stats.get('min', 0)*100:.1f}%~{acc_stats.get('max', 0)*100:.1f}% (avg: {acc_stats.get('mean', 0)*100:.1f}%) │"
        else:
            stats_line = f"│ Stats: No data yet │"
        stats_line += " " * (self.width - len(stats_line) - 1) + "│"
        lines.append(stats_line)
        
        lines.append(f"│ {ColorScheme.MUTED}Press Ctrl+C to stop{ColorScheme.RESET}" + " " * (self.width - 25) + "│")
        lines.append(header)
        
        content = '\n'.join(lines)
        
        if not self._initialized:
            self._initialized = True
            return self.clear_screen() + self.hide_cursor() + content
        else:
            return self.move_cursor_top() + content
    
    def _render_progress_bar(self, progress: float, width: int) -> str:
        filled = int(progress * width)
        bar = f"{ColorScheme.SUCCESS}{'█' * filled}{ColorScheme.RESET}{'░' * (width - filled)}"
        return bar
    
    def update_terminal_size(self) -> None:
        try:
            import shutil
            self.width = shutil.get_terminal_size().columns
            self.height = shutil.get_terminal_size().lines
        except Exception:
            pass
    
    def reset(self) -> None:
        self._initialized = False


class ResponsiveLayout:
    def __init__(self, renderer: TerminalRenderer):
        self.renderer = renderer
    
    def render(self, monitor: Any) -> str:
        self.renderer.update_terminal_size()
        
        items = monitor.get_data(page=0, apply_filter=False, apply_sort=False)
        items = sorted(items, key=lambda x: x.round_num, reverse=True)[:self.renderer._max_data_rows]
        
        overview = monitor.get_overview()
        clients = monitor.get_clients()
        stats = monitor.get_stats()
        
        return self.renderer.render(
            overview=overview,
            items=items,
            clients=clients,
            is_paused=monitor.is_paused,
            stats=stats,
        )
