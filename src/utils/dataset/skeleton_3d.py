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
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SkeletonManager3D':
        """Load skeleton data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Ensure frame IDs are strings
        data = {str(k): v for k, v in data.items()}
        return cls(data)
    
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
    
    def get_all_frames(self) -> Dict[str, List[Optional[List[float]]]]:
        """Get all frames data."""
        return self.data.copy()
    
    def get_frame_ids(self) -> List[str]:
        """Get all available frame IDs."""
        return list(self.data.keys())
    
    def has_frame(self, frame_id: str) -> bool:
        """Check if frame exists."""
        return str(frame_id) in self.data
    
    def remove_frame(self, frame_id: str) -> None:
        """Remove a frame."""
        self.data.pop(str(frame_id), None)
    
    def clear(self) -> None:
        """Clear all data."""
        self.data.clear()
    
    def get_num_frames(self) -> int:
        """Get number of frames."""
        return len(self.data)
    
    def get_num_keypoints(self) -> Optional[int]:
        """Get number of keypoints (from first available frame)."""
        if not self.data:
            return None
        first_frame = next(iter(self.data.values()))
        return len(first_frame) if first_frame else 0
