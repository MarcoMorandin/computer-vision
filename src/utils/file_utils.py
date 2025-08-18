"""File and path utilities for filename parsing and path operations."""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union


def extract_frame_number(file_name: str) -> Optional[int]:
    """Extract frame number from filename.
    
    Args:
        file_name: Input filename
        
    Returns:
        Frame number if found, None otherwise
    """
    match = re.search(r'frame_(\d+)', file_name)
    return int(match.group(1)) if match else None


def extract_camera_number(file_name: str) -> Optional[str]:
    """Extract camera number from filename.
    
    Args:
        file_name: Input filename
        
    Returns:
        Camera number as string if found, None otherwise
    """
    match = re.search(r'(?:out|cam)(\d+)', file_name)
    return match.group(1) if match else None


def extract_frame_cam_from_filename(file_name: str) -> Optional[Tuple[str, str]]:
    """Extract frame_id and cam_id from filename.
    
    Args:
        file_name: Input filename
        
    Returns:
        Tuple of (frame_id, cam_id) if found, None otherwise
    """
    match = re.search(r'out(\d+)_frame_(\d+)', file_name)
    if match:
        cam_id = match.group(1)
        frame_id = str(int(match.group(2)))
        return frame_id, cam_id
    return None


def ensure_directories(paths: Union[List[Path], List[str]]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
