"""Logging utilities for DAFA experiment framework."""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from pathlib import Path


_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "dafa",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (default: auto-generated)
        level: Logging level
        format_string: Log format string
    
    Returns:
        Configured logger instance
    """
    global _logger
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"experiment_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (default: use default logger)
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    if _logger is None:
        return setup_logger()
    return _logger


class ExperimentLogger:
    """Context manager for experiment logging."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "results/logs",
        config: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.config = config or {}
        self.logger = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=f"exp_{self.experiment_name}",
            log_dir=self.log_dir,
            log_file=log_file,
        )
        
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        if self.config:
            self.logger.info(f"Configuration: {self.config}")
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Experiment completed successfully. Duration: {duration}")
        else:
            self.logger.error(f"Experiment failed with error: {exc_val}")
            self.logger.error(f"Duration before failure: {duration}")
        
        return False
