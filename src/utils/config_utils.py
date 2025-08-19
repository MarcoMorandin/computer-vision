"""Enhanced configuration utilities for comprehensive Hydra configuration management."""

import os
import sys
import logging
import platform
import psutil
import torch
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from omegaconf import DictConfig, OmegaConf


def get_all_output_paths(config: DictConfig) -> List[Path]:
    """Extract all output paths from configuration structure.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        List of all output paths to be created
    """
    paths = []
    
    def extract_paths(obj: Any, parent_key: str = "") -> None:
        if isinstance(obj, DictConfig):
            for key, value in obj.items():
                if key in ['root', 'visualizations', 'data', 'annotations'] or key.endswith('_dir'):
                    if isinstance(value, str) and value:
                        paths.append(Path(value))
                elif isinstance(value, (DictConfig, dict)):
                    extract_paths(value, key)
                elif isinstance(value, str) and value and ('/' in value or '\\' in value):
                    # Potential path string
                    if not value.startswith('http') and not value.endswith('.pt'):
                        paths.append(Path(value))
    
    # Extract from output paths
    if hasattr(config, 'paths') and hasattr(config.paths, 'output'):
        extract_paths(config.paths.output)
    
    # Add temp and log directories if they exist
    if hasattr(config, 'paths'):
        if hasattr(config.paths, 'temp'):
            extract_paths(config.paths.temp)
        if hasattr(config.paths, 'logs'):
            extract_paths(config.paths.logs)
    
    return list(set(paths))  # Remove duplicates


def setup_logging(config: Optional[DictConfig] = None) -> logging.Logger:
    """Setup comprehensive logging based on configuration.
    
    Args:
        config: Hydra configuration object with logging settings
        
    Returns:
        Configured logger instance
    """
    # Determine log level

    level_name = config.logging.root.level    
    level = getattr(logging, level_name, logging.INFO)
    
    # Get logger name from config or use default

    logger_name = config.project.name    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()  # Clear existing handlers
    logger.propagate = config.logging.root.propagate

    console_handler = logging.StreamHandler(sys.stdout)
        
    # Use format from config if available
    formatter = logging.Formatter("[%(levelname)s] - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_section(logger: logging.Logger, title: str, width: int = 80) -> None:
    """Log a section separator with title.
    
    Args:
        logger: Logger instance
        title: Section title
        width: Width of separator line
    """
    separator = "=" * width
    logger.info(separator)
    logger.info(title)
    logger.info(separator)
