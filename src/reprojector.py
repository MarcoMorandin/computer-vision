"""Reproject 3D skeletons back to 2D image coordinates.

Uses camera projection matrices to project 3D keypoints and updates a
COCO dataset with the resulting 2D annotations.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from tqdm import tqdm

from .utils.camera.camera_manager import CameraManager
from .utils.dataset.coco_utils import COCOManager
from .utils.dataset.skeleton_3d import SkeletonManager3D
from .utils.file_utils import extract_frame_cam_from_filename


class SkeletonReprojector:
    """
    A class for reprojecting 3D skeletons to 2D using camera calibration parameters
    and updating existing COCO dataset annotations.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        coco_manager: COCOManager,
        skeleton_manager: SkeletonManager3D,
    ):
        """
        Initialize the SkeletonReprojector.

        Parameters
        ----------
            camera_manager: Initialized CameraManager object
            coco_manager: Existing COCOManager object to update
            skeleton_manager: SkeletonManager3D object with 3D keypoints data
        """
        self.camera_manager = camera_manager
        self.coco_manager = coco_manager
        self.skeleton_manager = skeleton_manager

    def _to_np_keypoints_3d(
        self, kps: Sequence[Optional[Sequence[float]]]
    ) -> np.ndarray:
        """Convert optional 3D keypoints to a dense (N, 3) float array.

        Parameters
        ----------
        kps : Sequence[Optional[Sequence[float]]]
            Sequence of length N containing [x, y, z] or None per keypoint.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) with NaNs where keypoints are missing.
        """
        arr = [kp if kp is not None else (np.nan, np.nan, np.nan) for kp in kps]
        return np.asarray(arr, dtype=np.float32)

    def reproject(self) -> COCOManager:
        """Reproject 3D skeleton data to 2D and update an existing COCO dataset.

        Returns
        -------
        COCOManager
            The same manager instance with reprojected 2D keypoints per image.
        """
        # Get existing images and create mapping
        images = self.coco_manager.get_images()
        image_mapping = self._create_image_mapping(images)

        # Clear existing annotations and rebuild with reprojected data
        self.coco_manager.clear_annotations()

        # Process each frame from skeleton manager
        for frame_id in tqdm(
            self.skeleton_manager.get_frame_ids(),
            desc="Reprojecting frames",
            unit="frame",
        ):
            keypoints_3d = self.skeleton_manager.get_frame(frame_id)
            if keypoints_3d:
                self._process_frame(frame_id, keypoints_3d, image_mapping)

        return self.coco_manager

    def _create_image_mapping(self, images: List[Dict]) -> Dict[Tuple[str, str], int]:
        """Create a mapping from (frame_id, cam_id) to COCO image_id.

        Parameters
        ----------
        images : list[dict]
            COCO image entries (each must include file_name and id).

        Returns
        -------
        dict[tuple[str, str], int]
            Keyed by (frame_id, cam_id) derived from filenames.
        """
        mapping = {}
        for img in images:
            file_name = img.get("file_name", "")
            frame_cam = extract_frame_cam_from_filename(file_name)
            if frame_cam:
                frame_id, cam_id = frame_cam
                mapping[(frame_id, cam_id)] = img["id"]
        return mapping

    def _process_frame(
        self,
        frame_id: str,
        keypoints_3d: List[Optional[List[float]]],
        image_mapping: Dict[Tuple[str, str], int],
    ) -> None:
        """Project a single frame of 3D keypoints to each camera and add annotations.

        Parameters
        ----------
        frame_id : str
            Frame identifier.
        keypoints_3d : list[Optional[list[float]]]
            3D keypoints for the frame; entries may be None.
        image_mapping : dict[tuple[str, str], int]
            Lookup of (frame_id, cam_id) -> image_id.
        """
        pts3d = self._to_np_keypoints_3d(keypoints_3d)

        # Get available cameras
        camera_ids = self.camera_manager.get_camera_ids()
        for cam_id in camera_ids:
            image_id = image_mapping[(frame_id, str(cam_id))]

            # Get calibration and project points
            calibration = self.camera_manager.get_camera(int(cam_id))
            P = calibration.get_projection_matrix()

            uv = self._project_points(pts3d, P)
            valid = np.isfinite(uv).all(axis=1)

            # Convert to COCO keypoints format
            keypoints_2d = []
            for i, (u, v) in enumerate(uv):
                if valid[i]:
                    keypoints_2d.extend([float(u), float(v), 2])  # visible
                else:
                    keypoints_2d.extend([0, 0, 0])  # not visible

            # Calculate bbox and area from visible keypoints
            visible_points = [(uv[i][0], uv[i][1]) for i in range(len(uv)) if valid[i]]
            if not visible_points:
                continue

            # Get person category ID
            person_cat = self.coco_manager.get_person_category()
            category_id = person_cat["id"]

            # Add annotation using existing method
            self.coco_manager.add_annotation(
                image_id=image_id, category_id=category_id, keypoints=keypoints_2d
            )

    def _project_points(self, points_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using a 3x4 projection matrix.

        Parameters
        ----------
        points_3d : np.ndarray
            Array of shape (N, 3) representing 3D coordinates.
        P : np.ndarray
            3x4 camera projection matrix.

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) containing projected pixel coordinates.
        """
        if points_3d.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Homogenize
        X_h = np.concatenate(
            [points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)], axis=1
        )
        x = (P @ X_h.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = x[:, :2] / x[:, 2:3]
        return uv.astype(np.float32)
