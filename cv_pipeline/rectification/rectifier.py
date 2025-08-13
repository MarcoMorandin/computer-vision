"""Image rectification implementation"""

from typing import Dict, Any, Tuple
import logging

import cv2
import numpy as np

from ..core.base import Rectifier, CameraParameters


class CameraRectifier(Rectifier):
    """Camera rectification using OpenCV"""
    
    def __init__(self):
        self.camera_params = None
        self.map1 = None
        self.map2 = None
        self.new_camera_matrix = None
        self.roi = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self, camera_params: CameraParameters) -> bool:
        """Initialize rectifier with camera parameters"""
        try:
            self.camera_params = camera_params
            
            # Get optimal new camera matrix
            # For now, use a reasonable image size; this could be made configurable
            image_size = (1920, 1080)  # Default size, should be actual image size
            
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                camera_params.camera_matrix,
                camera_params.distortion_coeffs,
                image_size,
                1,  # alpha=1 keeps all pixels
                image_size
            )
            
            # Compute rectification maps
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                camera_params.camera_matrix,
                camera_params.distortion_coeffs,
                None,  # No rotation for simple undistortion
                self.new_camera_matrix,
                image_size,
                cv2.CV_16SC2
            )
            
            self.logger.info(f"Initialized rectifier for camera {camera_params.camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rectifier: {e}")
            return False
    
    def rectify_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rectify a single frame"""
        if self.map1 is None or self.map2 is None:
            self.logger.warning("Rectifier not initialized, returning original frame")
            return frame
        
        try:
            # Update maps if frame size differs from initialization
            if frame.shape[:2] != (self.roi[3], self.roi[2]):
                self._update_maps_for_frame_size(frame.shape[:2])
            
            # Apply rectification
            rectified_frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
            
            # Crop to ROI if needed
            if self.roi[2] > 0 and self.roi[3] > 0:
                x, y, w, h = self.roi
                rectified_frame = rectified_frame[y:y+h, x:x+w]
            
            return rectified_frame
            
        except Exception as e:
            self.logger.error(f"Failed to rectify frame: {e}")
            return frame
    
    def rectify_annotations(self, annotations: Dict[str, Any], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Rectify annotations (keypoints, bounding boxes) to match rectified frame"""
        if not annotations or self.new_camera_matrix is None:
            return annotations
        
        try:
            rectified_annotations = annotations.copy()
            
            # Process each annotation
            if 'annotations' in rectified_annotations:
                for ann in rectified_annotations['annotations']:
                    # Rectify keypoints if present
                    if 'keypoints' in ann and ann['keypoints']:
                        ann['keypoints'] = self._rectify_keypoints(ann['keypoints'])
                    
                    # Rectify bounding box if present
                    if 'bbox' in ann and ann['bbox']:
                        ann['bbox'] = self._rectify_bbox(ann['bbox'])
            
            return rectified_annotations
            
        except Exception as e:
            self.logger.error(f"Failed to rectify annotations: {e}")
            return annotations
    
    def _update_maps_for_frame_size(self, frame_shape: Tuple[int, int]) -> None:
        """Update rectification maps for different frame size"""
        height, width = frame_shape
        image_size = (width, height)
        
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_params.camera_matrix,
            self.camera_params.distortion_coeffs,
            image_size,
            1,
            image_size
        )
        
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_params.camera_matrix,
            self.camera_params.distortion_coeffs,
            None,
            self.new_camera_matrix,
            image_size,
            cv2.CV_16SC2
        )
    
    def _rectify_keypoints(self, keypoints: list) -> list:
        """Rectify keypoints coordinates"""
        # Keypoints format: [x1, y1, v1, x2, y2, v2, ...]
        rectified_keypoints = []
        
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            
            if v > 0:  # Only rectify visible keypoints
                # Convert to homogeneous coordinates
                point = np.array([[x, y]], dtype=np.float32)
                
                # Apply undistortion
                rectified_point = cv2.undistortPoints(
                    point,
                    self.camera_params.camera_matrix,
                    self.camera_params.distortion_coeffs,
                    None,
                    self.new_camera_matrix
                )
                
                rectified_x = float(rectified_point[0, 0, 0])
                rectified_y = float(rectified_point[0, 0, 1])
                
                # Apply ROI offset if needed
                if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
                    rectified_x -= self.roi[0]
                    rectified_y -= self.roi[1]
                
                rectified_keypoints.extend([rectified_x, rectified_y, v])
            else:
                rectified_keypoints.extend([x, y, v])
        
        return rectified_keypoints
    
    def _rectify_bbox(self, bbox: list) -> list:
        """Rectify bounding box coordinates"""
        # COCO bbox format: [x, y, width, height]
        x, y, w, h = bbox
        
        # Get corner points
        corners = np.array([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply undistortion
        rectified_corners = cv2.undistortPoints(
            corners,
            self.camera_params.camera_matrix,
            self.camera_params.distortion_coeffs,
            None,
            self.new_camera_matrix
        ).reshape(-1, 2)
        
        # Apply ROI offset if needed
        if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
            rectified_corners[:, 0] -= self.roi[0]
            rectified_corners[:, 1] -= self.roi[1]
        
        # Calculate new bounding box
        min_x = float(np.min(rectified_corners[:, 0]))
        min_y = float(np.min(rectified_corners[:, 1]))
        max_x = float(np.max(rectified_corners[:, 0]))
        max_y = float(np.max(rectified_corners[:, 1]))
        
        return [min_x, min_y, max_x - min_x, max_y - min_y]
