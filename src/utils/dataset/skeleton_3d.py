import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path


class SkeletonManager3D:
    """Manager for 3D skeleton data."""
    
    def __init__(self, skeleton_data: Optional[Dict[str, Any]] = None):
        """
        Initialize SkeletonManager3D.
        
        Args:
            skeleton_data: Dictionary mapping frame_id -> keypoints_3d or None for empty
        """
        self.data = skeleton_data if skeleton_data is not None else {}
    
    def save(self, file_path: str) -> None:
        """Save skeleton data to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_frame(self, frame_id: str, keypoints_3d: List[Optional[List[float]]]) -> None:
        """Add a frame with 3D keypoints."""
        self.data[str(frame_id)] = keypoints_3d
    
    def get_frame(self, frame_id: str) -> Optional[List[Optional[List[float]]]]:
        """Get 3D keypoints for a specific frame."""
        return self.data.get(str(frame_id))
    
    def get_frame_ids(self) -> List[str]:
        """Get all available frame IDs."""
        return list(self.data.keys())