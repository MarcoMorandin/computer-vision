"""Camera calibration dataclass and utilities.

Stores intrinsics/extrinsics and derived matrices for a single camera,
with helpers to load from JSON and project 3D points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Holds intrinsics/extrinsics and projection for a single camera.

    Attributes
    ----------
    mtx : np.ndarray
        (3, 3) intrinsic camera matrix
    dist : np.ndarray
        (N,) distortion coefficients.
    rvec : np.ndarray
        (3,) rotation vector in Rodrigues form.
    tvec : np.ndarray
        (3,) translation vector.
    R : np.ndarray
        (3, 3) rotation matrix computed from rvec.
    P : np.ndarray
        (3, 4) projection matrix formed as mtx @ [R | t].
    camera_id : int
        Identifier for this camera.
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
        """Load calibration from a JSON file.

        Parameters
        ----------
        path : str
            Path to a camera_calib.json file containing fields mtx, dist, rvecs, tvecs.
        camera_id : int
            Identifier for the created camera instance.

        Returns
        -------
        CameraCalibration
            New immutable instance populated from file contents.
        """
        with open(path, "r") as f:
            calib = json.load(f)

        self.mtx = np.array(calib.get("mtx"), dtype=np.float64).reshape(3, 3)
        self.dist = np.array(calib["dist"], dtype=np.float64).flatten()
        self.rvec = np.array(calib["rvecs"], dtype=np.float64).flatten()
        self.tvec = np.array(calib["tvecs"], dtype=np.float64).flatten()

        self.R, _ = cv2.Rodrigues(self.rvec)
        self.P = self.mtx @ np.hstack([self.R, self.tvec.reshape(3, 1)])

        return self(
            mtx=self.mtx,
            dist=self.dist,
            rvec=self.rvec,
            tvec=self.tvec,
            R=self.R,
            P=self.P,
            camera_id=camera_id,
        )

    def get_calibration_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the intrinsic matrix and distortion coefficients.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mtx, dist) where mtx is 3x3 intrinsics and dist is the distortion vector.
        """
        return self.mtx, self.dist

    def get_projection_matrix(self) -> np.ndarray:
        return self.P

    def __str__(self) -> str:
        return f"CameraCalibration(id={self.camera_id})"
