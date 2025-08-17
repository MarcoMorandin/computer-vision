from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np

from .camera_calibration import CameraCalibration

class CameraManager:

    def __init__(self):
        self.cameras: Dict[int, CameraCalibration] = {}

    def get_projection_matrix(self, camera_id: int) -> np.ndarray:
        return self.cameras.get(camera_id).get_projection_matrix()
    
    def load_cameras(
        self, 
        base_dir: str,
    ) -> None:

        for cam_dir in Path(base_dir).glob("cam_*"):
            calib_path = cam_dir / "calib" / "camera_calib.json"
            
            camera_id = int(cam_dir.name.split('_')[-1])
            
            calibration = CameraCalibration.from_json(
                calib_path, 
                camera_id=camera_id,
            )
            self.cameras[camera_id] = calibration

    def get_camera_ids(self) -> List[int]:
        return list(self.cameras.keys())
    
    def get_camera(self, camera_id: int) -> CameraCalibration:
        return self.cameras.get(camera_id)

    def get_projection_matrices(self) -> Dict[int, np.ndarray]:
        return {cam_id: cam.P for cam_id, cam in self.cameras.items()}
    
    def project_points_all_cameras(
        self, 
        points_3d: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        projections = {}
        for cam_id, camera in self.cameras.items():
            projections[cam_id] = camera.project_points(points_3d)
        return projections
    
    def __str__(self) -> str:
        return f"CameraManager({len(self.cameras)} cameras: {sorted(list(self.cameras.keys()))})"
