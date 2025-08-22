"""File and path utilities for filename parsing and path operations."""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union


def extract_frame_number(file_name: str) -> Optional[int]:
    """Extract a frame number from a filename.

    Parameters
    ----------
    file_name : str
        Input filename.

    Returns
    -------
    int | None
        Frame number if found; otherwise None.
    """
    match = re.search(r"frame_(\d+)", file_name)
    return int(match.group(1)) if match else None


def extract_camera_number(file_name: str) -> Optional[str]:
    """Extract a camera number from a filename.

    Parameters
    ----------
    file_name : str
        Input filename.

    Returns
    -------
    str | None
        Camera identifier (digits) if found; otherwise None.
    """
    match = re.search(r"(?:out|cam)(\d+)", file_name)
    return match.group(1) if match else None


def extract_frame_cam_from_filename(file_name: str) -> Optional[Tuple[str, str]]:
    """Extract frame and camera identifiers from a filename.

    Parameters
    ----------
    file_name : str
        Input filename.

    Returns
    -------
    tuple[str, str] | None
        (frame_id, cam_id) if matched; otherwise None.
    """
    match = re.search(r"out(\d+)_frame_(\d+)", file_name)
    if match:
        cam_id = match.group(1)
        frame_id = str(int(match.group(2)))
        return frame_id, cam_id
    return None


def ensure_directories(paths: Union[List[Path], List[str]]) -> None:
    """Create directories if they don't already exist.

    Parameters
    ----------
    paths : list[pathlib.Path] | list[str]
        Paths to create with parents.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
