"""Camera management utilities.

Loads calibrated cameras from disk and provides convenience helpers to
access projection matrices and project 3D points across all cameras.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np

from .camera_calibration import CameraCalibration


class CameraManager:
    """Holds multiple CameraCalibration instances.

    Provides helpers to load cameras and query their properties.
    """

    def __init__(self):
        """Initialize an empty camera registry."""
        self.cameras: Dict[int, CameraCalibration] = {}

    def get_projection_matrix(self, camera_id: int) -> np.ndarray:
        """Return the 3x4 projection matrix for a camera.

        Parameters
        ----------
        camera_id : int
            Identifier of the camera (e.g., parsed from folder name cam_{id}).

        Returns
        -------
        np.ndarray
            3x4 projection matrix.
        """
        return self.cameras.get(camera_id).get_projection_matrix()

    def load_cameras(
        self,
        base_dir: str,
    ) -> None:
        """Load all cameras from a base directory.

        Expects subdirectories named cam_*/calib/camera_calib.json.

        Parameters
        ----------
        base_dir : str
            Root directory containing camera folders.
        """
        for cam_dir in Path(base_dir).glob("cam_*"):
            calib_path = cam_dir / "calib" / "camera_calib.json"

            camera_id = int(cam_dir.name.split("_")[-1])

            calibration = CameraCalibration.from_json(
                calib_path,
                camera_id=camera_id,
            )
            self.cameras[camera_id] = calibration

    def get_camera_ids(self) -> List[int]:
        """List loaded camera identifiers."""
        return list(self.cameras.keys())

    def get_camera(self, camera_id: int) -> CameraCalibration:
        """Fetch a loaded CameraCalibration by id.

        Note: Raises if the id is unknown when dereferenced by the caller.
        """
        return self.cameras.get(camera_id)

    def __str__(self) -> str:
        return f"CameraManager({len(self.cameras)} cameras: {sorted(list(self.cameras.keys()))})"
