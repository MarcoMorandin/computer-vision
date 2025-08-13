"""MediaPipe pose estimation implementation."""

from typing import Dict, Any, List
import numpy as np
import cv2

from ..core.base import PoseEstimationModel
from .base_estimator import BasePoseEstimator

# Optional dependencies
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class MediaPipePoseEstimator(BasePoseEstimator):
    """MediaPipe pose estimation implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pose = None
        
    def get_model_type(self) -> PoseEstimationModel:
        return PoseEstimationModel.MEDIAPIPE
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize MediaPipe pose model"""
        try:
            if not HAS_MEDIAPIPE:
                raise ImportError("mediapipe not available. Install with: pip install mediapipe")
            
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                model_complexity=config.get('model_complexity', 2),
                enable_segmentation=config.get('enable_segmentation', False),
                static_image_mode=config.get('static_image_mode', False),
                min_detection_confidence=config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=config.get('min_tracking_confidence', 0.5)
            )
            
            self.confidence_threshold = config.get('confidence_threshold', 0.25)
            
            self.logger.info("Initialized MediaPipe Pose model")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe model: {e}")
            return False
    
    def estimate_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate poses using MediaPipe"""
        if self.pose is None:
            return self._empty_annotations()
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            annotations = []
            
            if results.pose_landmarks:
                # MediaPipe pose landmarks mapping
                landmarks = results.pose_landmarks.landmark
                
                # Map MediaPipe landmarks to COCO keypoints
                mp_to_coco = {
                    'nose': 0,  # nose
                    'left_eye': 2,  # left eye inner
                    'right_eye': 5,  # right eye inner
                    'left_ear': 7,  # left ear
                    'right_ear': 8,  # right ear
                    'left_shoulder': 11,  # left shoulder
                    'right_shoulder': 12,  # right shoulder
                    'left_elbow': 13,  # left elbow
                    'right_elbow': 14,  # right elbow
                    'left_wrist': 15,  # left wrist
                    'right_wrist': 16,  # right wrist
                    'left_hip': 23,  # left hip
                    'right_hip': 24,  # right hip
                    'left_knee': 25,  # left knee
                    'right_knee': 26,  # right knee
                    'left_ankle': 27,  # left ankle
                    'right_ankle': 28,  # right ankle
                }
                
                # Build COCO keypoints dictionary
                coco_kpts_person = {}
                h, w = frame.shape[:2]
                
                for coco_name, mp_idx in mp_to_coco.items():
                    landmark = landmarks[mp_idx]
                    coco_kpts_person[coco_name] = {
                        "x": float(landmark.x * w),
                        "y": float(landmark.y * h),
                        "confidence": float(landmark.visibility if hasattr(landmark, 'visibility') else 1.0)
                    }
                
                # Map to custom skeleton
                custom_keypoint_names = self.get_keypoint_names()
                custom_keypoints = self._remap_to_custom_skeleton(coco_kpts_person, custom_keypoint_names)
                keypoints_flat = self._keypoints_to_flat_array(custom_keypoints, custom_keypoint_names)
                
                # Calculate bounding box from keypoints
                valid_points = []
                for i in range(0, len(keypoints_flat), 3):
                    if keypoints_flat[i + 2] > 0:  # if visible
                        valid_points.append([keypoints_flat[i], keypoints_flat[i + 1]])
                
                if valid_points:
                    valid_points = np.array(valid_points)
                    x_min, y_min = np.min(valid_points, axis=0)
                    x_max, y_max = np.max(valid_points, axis=0)
                    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                else:
                    bbox = [0, 0, w, h]  # Default to full frame
                
                annotation = self._create_coco_annotation(keypoints_flat, bbox, 1, 1)
                annotations.append(annotation)
            
            return {
                'annotations': annotations,
                'categories': [{'id': 1, 'name': 'person', 'keypoints': self.get_keypoint_names()}]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate poses with MediaPipe: {e}")
            return self._empty_annotations()
    
    def get_keypoint_names(self) -> List[str]:
        """Get keypoint names"""
        return [
            "Head", "Neck", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand",
            "Spine", "Hips", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"
        ]
    
    def _empty_annotations(self) -> Dict[str, Any]:
        """Return empty annotations structure"""
        return {
            'annotations': [],
            'categories': [{'id': 1, 'name': 'person', 'keypoints': self.get_keypoint_names()}]
        }
