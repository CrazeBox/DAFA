"""
Real-time monitoring module for DAFA federated learning experiments.

This module provides a comprehensive real-time data monitoring interface
that replaces traditional progress bars with dynamic data display lists.

Features:
- Real-time data updates with visual feedback
- Data filtering and sorting
- Client status tracking
- Responsive terminal UI
- Data export (JSON/CSV)
"""

from .realtime_monitor import (
    RealtimeMonitor,
    DataItem,
    DataStatus,
    ClientStatus,
    TrainingOverview,
    DataFilter,
    DataSorter,
    LoadState,
)

from .terminal_renderer import (
    TerminalRenderer,
    ResponsiveLayout,
    ColorScheme,
)

from .monitor_panel import (
    MonitorPanel,
    TrainingMonitorWrapper,
    create_monitor,
)

__all__ = [
    "RealtimeMonitor",
    "DataItem",
    "DataStatus",
    "ClientStatus",
    "TrainingOverview",
    "DataFilter",
    "DataSorter",
    "LoadState",
    "TerminalRenderer",
    "ResponsiveLayout",
    "ColorScheme",
    "MonitorPanel",
    "TrainingMonitorWrapper",
    "create_monitor",
]
