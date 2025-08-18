"""Configuration utilities for Hydra configuration management and device selection."""

import os
import sys
import logging
from pathlib import Path
from typing import Any, List
from omegaconf import DictConfig


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
                if key in ['root', 'visualizations'] or key.endswith('_dir'):
                    if isinstance(value, str):
                        paths.append(Path(value))
                elif isinstance(value, (DictConfig, dict)):
                    extract_paths(value, key)
                elif isinstance(value, str) and '/' in value:
                    # Potential path string
                    paths.append(Path(value))
    
    extract_paths(config.paths.output)
    return paths


def setup_logging() -> logging.Logger:
    """Setup logging based on configuration.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        Configured logger instance
    """
    # Get log level from config or environment
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
  
    logger = logging.getLogger("CV Pipeline")
    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to root logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
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
