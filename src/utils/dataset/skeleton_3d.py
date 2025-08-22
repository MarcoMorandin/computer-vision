"""3D skeleton storage utilities.

Provides a container to store and retrieve per-frame 3D keypoints.
"""

import json
import os
from typing import Dict, List, Optional, Any


class SkeletonManager3D:
    """Manager for 3D skeleton data."""

    def __init__(self, skeleton_data: Optional[Dict[str, Any]] = None):
        """Initialize SkeletonManager3D.

        Parameters
        ----------
        skeleton_data : dict[str, Any] | None
            Mapping frame_id -> keypoints_3d (list of [x, y, z] or None). If None,
            an empty store is created.
        """
        self.data = skeleton_data if skeleton_data is not None else {}

    def save(self, file_path: str) -> None:
        """Save skeleton data to a JSON file on disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_frame(
        self, frame_id: str, keypoints_3d: List[Optional[List[float]]]
    ) -> None:
        """Add a frame with 3D keypoints.

        Parameters
        ----------
        frame_id : str
            Frame identifier.
        keypoints_3d : list[Optional[list[float]]]
            Sequence of [x, y, z] entries (or None) ordered by keypoint name.
        """
        self.data[str(frame_id)] = keypoints_3d

    def get_frame(self, frame_id: str) -> Optional[List[Optional[List[float]]]]:
        """Get 3D keypoints for a specific frame.

        Parameters
        ----------
        frame_id : str
            Frame identifier.
        """
        return self.data.get(str(frame_id))

    def get_frame_ids(self) -> List[str]:
        """Get all available frame IDs (order is arbitrary)."""
        return list(self.data.keys())
