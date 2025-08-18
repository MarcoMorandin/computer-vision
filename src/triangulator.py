"""3D pose triangulation module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import least_squares
from tqdm import tqdm

from .utils.camera.camera_manager import CameraManager
from .utils.dataset.coco_utils import COCOManager
from .utils.dataset.skeleton_3d import SkeletonManager3D
from .utils.file_utils import extract_frame_number, extract_camera_number


class PlayerTriangulator:
    """
    A class for triangulating 3D player poses from 2D keypoints across multiple camera views.
    """
    
    def __init__(self, camera_manager: CameraManager, coco_manager: COCOManager):
        """
        Initialize the PlayerTriangulator.
        
        Args:
            camera_manager: Initialized CameraManager object
            coco_manager: Initialized COCOManager object with 2D keypoints
        """
        self.camera_manager = camera_manager
        self.coco_manager = coco_manager
    
    
    def _triangulate_point_dlt(self, points_2d: List[Tuple[float, float]], 
                              projection_matrices: List[np.ndarray]) -> np.ndarray:
        """Triangulate a 3D point using Direct Linear Transform (DLT)."""
        num_views = len(points_2d)
        A = np.zeros((2 * num_views, 4))
        
        for i in range(num_views):
            x, y = points_2d[i]
            P = projection_matrices[i]
            
            A[2*i] = x * P[2] - P[0]
            A[2*i + 1] = y * P[2] - P[1]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Normalize homogeneous coordinate
        
        return X[:3]
    
    def _triangulate_keypoint(self, points_2d_by_camera: Dict[str, Tuple[float, float]], 
                             min_views: int = 2) -> Optional[np.ndarray]:
        """Triangulate a single keypoint from multiple camera views."""
        if len(points_2d_by_camera) < min_views:
            return None
        
        # Get projection matrices for cameras that have this keypoint
        camera_ids = list(points_2d_by_camera.keys())
        projection_matrices = []
        points_2d = []
        
        for cam_id in camera_ids:
            calibration = self.camera_manager.get_camera(int(cam_id))
            P = calibration.get_projection_matrix()
            projection_matrices.append(P)
            points_2d.append(points_2d_by_camera[cam_id])
        
        return self._triangulate_point_dlt(points_2d, projection_matrices)
    
    def _bundle_adjust_point(self, X0: np.ndarray, points_2d: List[Tuple[float, float]], 
                            camera_ids: List[str]) -> np.ndarray:
        """Bundle adjustment for a single 3D point to refine triangulation."""
        def residuals(X):
            X_homogeneous = np.append(X, 1)
            errors = []
            
            for i, cam_id in enumerate(camera_ids):
                calibration = self.camera_manager.get_camera(int(cam_id))
                P = calibration.get_projection_matrix()
                x_proj = P @ X_homogeneous
                x_proj = x_proj[:2] / x_proj[2]
                
                errors.extend(points_2d[i] - x_proj)
            
            return errors
        
        result = least_squares(residuals, X0, method='lm')
        return result.x
    
    def _group_annotations_by_frame(self) -> Dict[int, Dict[str, List[Tuple[float, float, int]]]]:
        """Group COCO annotations by frame number."""
        images = self.coco_manager.get_images()
        frames_data = {}
        
        # Get all annotations
        for image_info in images:
            annotations = self.coco_manager.get_annotations_by_image_id(image_info["id"])
            for ann in annotations:
                if "keypoints" not in ann:
                    continue
                
                file_name = image_info["file_name"]
                frame_num = extract_frame_number(file_name)
                cam_num = extract_camera_number(file_name)
                
                if frame_num is None or cam_num is None:
                    continue
                
                if frame_num not in frames_data:
                    frames_data[frame_num] = {}
                
                # Reshape keypoints from [x1, y1, v1, x2, y2, v2, ...] to [(x1, y1, v1), ...]
                keypoints = ann["keypoints"]
                num_keypoints = len(keypoints) // 3
                
                keypoints_reshaped = []
                for i in range(num_keypoints):
                    x = keypoints[i * 3]
                    y = keypoints[i * 3 + 1]
                    v = keypoints[i * 3 + 2]
                    keypoints_reshaped.append((x, y, v))
                
                frames_data[frame_num][cam_num] = keypoints_reshaped
        
        return frames_data
    
    def triangulate(self, use_bundle_adjustment: bool = True) -> SkeletonManager3D:
        """
        Triangulate 3D poses for all frames.
        
        Args:
            use_bundle_adjustment: Whether to use bundle adjustment for refinement
            
        Returns:
            SkeletonManager3D with triangulated 3D poses
        """
        frames_data = self._group_annotations_by_frame()
        skeleton_manager = SkeletonManager3D()
        

        for frame_num, frame_data in tqdm(sorted(frames_data.items()), desc="Triangulating player", unit="frame"):
            
            # Skip frames with insufficient camera views
            if len(frame_data) < 2:
                print(f"  Skipping frame {frame_num}: insufficient camera views")
                continue
            
            # Get number of keypoints from first camera
            first_cam = list(frame_data.keys())[0]
            num_keypoints = len(frame_data[first_cam])
            
            # Triangulate each keypoint
            keypoints_3d = []
            for kp_idx in range(num_keypoints):
                # Collect 2D observations from all cameras
                points_2d_by_camera = {}
                
                for cam_num, keypoints in frame_data.items():
                    x, y, v = keypoints[kp_idx]
                    # Only use visible keypoints
                    if v > 0:
                        points_2d_by_camera[cam_num] = (x, y)
                
                # Triangulate keypoint
                point_3d = self._triangulate_keypoint(points_2d_by_camera)
                
                # Refine with bundle adjustment if requested
                if point_3d is not None and use_bundle_adjustment and len(points_2d_by_camera) >= 2:
                    camera_ids = list(points_2d_by_camera.keys())
                    points_2d = [points_2d_by_camera[cam] for cam in camera_ids]
                    point_3d = self._bundle_adjust_point(point_3d, points_2d, camera_ids)
                
                keypoints_3d.append(point_3d.tolist() if point_3d is not None else None)
            
            skeleton_manager.add_frame(str(frame_num), keypoints_3d)
        
        return skeleton_manager

    def __str__(self):
        return f"PlayerTriangulator(camera_manager={self.camera_manager}, coco_manager={self.coco_manager})"