from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """
    Attributes:
        mtx: (3,3) intrinsic camera matrix
        dist: (N,) distortion coefficients  
        rvec: (3,) rotation vector (Rodrigues format)
        tvec: (3,) translation vector
        R: (3,3) rotation matrix (computed from rvec)
        P: (3,4) projection matrix K[R|t]
        camera_id: Optional camera identifier
    """
    
    mtx: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    P: np.ndarray
    camera_id: int

    @classmethod
    def from_json(
        self, 
        path: str, 
        camera_id: int,
    ) -> "CameraCalibration":
        
        with open(path, 'r') as f:
            calib = json.load(f)
        
        self.mtx = np.array(calib.get("mtx"), dtype=np.float64).reshape(3, 3)
        self.dist = np.array(calib['dist'], dtype=np.float64).flatten()
        self.rvec = np.array(calib['rvecs'], dtype=np.float64).flatten()
        self.tvec = np.array(calib['tvecs'], dtype=np.float64).flatten()

        self.R, _ = cv2.Rodrigues(self.rvec)
        self.P = self.mtx @ np.hstack([self.R, self.tvec.reshape(3, 1)])
        
        return self(
            mtx=self.mtx,
            dist=self.dist,
            rvec=self.rvec,
            tvec=self.tvec,
            R=self.R,
            P=self.P,
            camera_id=camera_id
        )
        
    def get_calibration_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get intrinsic camera matrix and distortion coefficients.
        
        Returns:
            Tuple of (mtx, dist) where mtx is the intrinsic matrix and dist is the distortion coefficients.
        """
        return self.mtx, self.dist
    
    def project_points(
        self, 
        points_3d: np.ndarray, 
    ) -> np.ndarray:
        
        projected, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3), 
            self.rvec, 
            self.tvec, 
            self.K, 
            self.dist
        )
        return projected.reshape(-1, 2)
    
    def get_projection_matrix(self) -> np.ndarray:
        return self.P
    
    def __str__(self) -> str:
        return (
            f"CameraCalibration(id={self.camera_id})"
        )