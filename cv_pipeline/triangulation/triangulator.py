"""3D triangulation implementation"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..core.base import Triangulator, CameraParameters


class MultiViewTriangulator(Triangulator):
    """Multi-view triangulation implementation"""
    
    def __init__(self):
        self.camera_params = {}
        self.projection_matrices = {}
        self.min_cameras = 2
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self, camera_params: Dict[str, CameraParameters], config: Dict[str, Any] = None) -> bool:
        """Initialize triangulator with camera parameters"""
        try:
            self.camera_params = camera_params
            self.min_cameras = config.get('min_cameras', 2) if config else 2
            
            # Compute projection matrices for each camera
            for cam_id, params in camera_params.items():
                R = params.rotation_matrix
                t = params.translation_vector.reshape(-1, 1)
                P = params.camera_matrix @ np.hstack([R, t])
                self.projection_matrices[cam_id] = P
                
                self.logger.debug(f"Camera {cam_id} projection matrix computed")
            
            self.logger.info(f"Initialized triangulator with {len(camera_params)} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize triangulator: {e}")
            return False
    
    def triangulate_poses(self, multi_view_annotations: Dict[str, Dict[str, Any]]) -> Dict[str, List[Optional[List[float]]]]:
        """Triangulate 3D poses from multi-view annotations"""
        poses_3d = {}
        
        try:
            # Group data by frame
            for frame_id, cameras_data in multi_view_annotations.items():
                self.logger.debug(f"Processing frame {frame_id}")
                
                # Skip frames with insufficient camera views
                if len(cameras_data) < self.min_cameras:
                    self.logger.debug(f"Skipping frame {frame_id}: insufficient cameras ({len(cameras_data)} < {self.min_cameras})")
                    continue
                
                # Extract keypoints from all cameras for this frame
                frame_keypoints = self._extract_frame_keypoints(cameras_data)
                
                if not frame_keypoints:
                    continue
                
                # Get number of keypoints from first camera
                first_cam = list(frame_keypoints.keys())[0]
                num_keypoints = len(frame_keypoints[first_cam])
                
                # Triangulate each keypoint
                keypoints_3d = []
                for kp_idx in range(num_keypoints):
                    point_3d = self._triangulate_keypoint(frame_keypoints, kp_idx)
                    keypoints_3d.append(point_3d.tolist() if point_3d is not None else None)
                
                poses_3d[frame_id] = keypoints_3d
                
            self.logger.info(f"Triangulated poses for {len(poses_3d)} frames")
            return poses_3d
            
        except Exception as e:
            self.logger.error(f"Failed to triangulate poses: {e}")
            return {}
    
    def _extract_frame_keypoints(self, cameras_data: Dict[str, Any]) -> Dict[str, List[Tuple[float, float, float]]]:
        """Extract keypoints from camera data for a frame"""
        frame_keypoints = {}
        
        for cam_id, data in cameras_data.items():
            if cam_id not in self.camera_params:
                continue
            
            # Handle different data structures
            annotations = None
            if isinstance(data, dict):
                if 'annotations' in data:
                    annotations = data['annotations']
                elif 'annotations' in data and isinstance(data['annotations'], list):
                    annotations = data['annotations']
                else:
                    # Assume the data itself is annotations
                    annotations = data
            else:
                annotations = data
            
            # Extract keypoints from annotations
            keypoints = self._extract_keypoints_from_annotations(annotations)
            if keypoints:
                frame_keypoints[cam_id] = keypoints
        
        return frame_keypoints
    
    def _extract_keypoints_from_annotations(self, annotations: Any) -> Optional[List[Tuple[float, float, float]]]:
        """Extract keypoints from annotation data"""
        if not annotations:
            return None
        
        # Handle different annotation formats
        if isinstance(annotations, dict):
            if 'keypoints' in annotations:
                keypoints_flat = annotations['keypoints']
            else:
                return None
        elif isinstance(annotations, list):
            if len(annotations) > 0 and isinstance(annotations[0], dict):
                # Take first annotation (assuming single person)
                ann = annotations[0]
                if 'keypoints' in ann:
                    keypoints_flat = ann['keypoints']
                else:
                    return None
            else:
                # Assume it's already a flat list
                keypoints_flat = annotations
        else:
            return None
        
        if not keypoints_flat or len(keypoints_flat) % 3 != 0:
            return None
        
        # Convert flat list to (x, y, v) tuples
        keypoints = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
            keypoints.append((float(x), float(y), float(v)))
        
        return keypoints
    
    def _triangulate_keypoint(self, frame_keypoints: Dict[str, List[Tuple[float, float, float]]], 
                            kp_idx: int) -> Optional[np.ndarray]:
        """Triangulate a single keypoint from multiple views"""
        # Collect 2D observations for this keypoint
        points_2d_by_camera = {}
        
        for cam_id, keypoints in frame_keypoints.items():
            if kp_idx < len(keypoints):
                x, y, v = keypoints[kp_idx]
                if v > 0:  # Only use visible keypoints
                    points_2d_by_camera[cam_id] = (x, y)
        
        # Need at least min_cameras observations
        if len(points_2d_by_camera) < self.min_cameras:
            return None
        
        # Triangulate using DLT
        point_3d = self._triangulate_point_dlt(points_2d_by_camera)
        
        # Refine with bundle adjustment if we have a good initial estimate
        if point_3d is not None:
            try:
                point_3d = self._bundle_adjust_point(point_3d, points_2d_by_camera)
            except:
                # If bundle adjustment fails, use DLT result
                pass
        
        return point_3d
    
    def _triangulate_point_dlt(self, points_2d_by_camera: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        """Triangulate 3D point using Direct Linear Transform (DLT)"""
        try:
            # Build system of equations
            A = []
            for cam_id, (x, y) in points_2d_by_camera.items():
                if cam_id not in self.projection_matrices:
                    continue
                
                P = self.projection_matrices[cam_id]
                
                # Each point gives us 2 equations
                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])
            
            if len(A) < 4:  # Need at least 2 cameras (4 equations) 
                return None
            
            A = np.array(A)
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X_homogeneous = Vt[-1, :]
            
            # Convert from homogeneous to 3D coordinates
            if abs(X_homogeneous[3]) < 1e-6:
                return None
            
            X_3d = X_homogeneous[:3] / X_homogeneous[3]
            
            # Basic sanity checks
            if np.any(np.isnan(X_3d)) or np.any(np.isinf(X_3d)):
                return None
            
            # Check if point is reasonable (within reasonable bounds)
            if np.linalg.norm(X_3d) > 1000:  # 1000 units from origin
                return None
            
            return X_3d
            
        except Exception as e:
            self.logger.debug(f"DLT triangulation failed: {e}")
            return None
    
    def _bundle_adjust_point(self, X0: np.ndarray, points_2d_by_camera: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Refine 3D point using bundle adjustment"""
        def residuals(X):
            errors = []
            for cam_id, (x_obs, y_obs) in points_2d_by_camera.items():
                if cam_id not in self.projection_matrices:
                    continue
                
                P = self.projection_matrices[cam_id]
                
                # Project 3D point to 2D
                X_h = np.append(X, 1)  # homogeneous coordinates
                x_proj = P @ X_h
                
                if abs(x_proj[2]) < 1e-6:
                    continue
                
                x_proj = x_proj[:2] / x_proj[2]
                
                # Compute reprojection error
                errors.extend([x_obs - x_proj[0], y_obs - x_proj[1]])
            
            return np.array(errors)
        
        try:
            result = least_squares(residuals, X0, method='lm')
            if result.success:
                return result.x
            else:
                return X0
        except:
            return X0
