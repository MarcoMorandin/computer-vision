"""YOLO pose estimation implementation"""

from typing import Dict, Any, List

import numpy as np

from ..core.base import PoseEstimationModel
from .base_estimator import BasePoseEstimator

# Optional dependencies
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class YOLOPoseEstimator(BasePoseEstimator):
    """YOLO pose estimation implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        
    def get_model_type(self) -> PoseEstimationModel:
        return PoseEstimationModel.YOLO
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize YOLO pose model"""
        try:
            if not HAS_ULTRALYTICS:
                raise ImportError("ultralytics not available. Install with: pip install ultralytics")
            
            model_name = config.get('model_name', 'yolo11l-pose.pt')
            self.confidence_threshold = config.get('confidence_threshold', 0.25)
            
            self.model = YOLO(model_name)
            
            self.logger.info(f"Initialized YOLO pose model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    def estimate_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate poses using YOLO"""
        if self.model is None:
            return self._empty_annotations()
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, imgsz=1280, verbose=False)
            
            # Extract keypoints and boxes
            result = results[0]
            keypoints_tensor = result.keypoints
            boxes_tensor = result.boxes
            
            if keypoints_tensor is None or boxes_tensor is None:
                return self._empty_annotations()
            
            annotations = []
            coco_keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            # Process each detected person
            for person_idx in range(keypoints_tensor.shape[0]):
                kpts_xy = keypoints_tensor.xy[person_idx].cpu().numpy()
                kpts_conf = keypoints_tensor.conf[person_idx].cpu().numpy()
                
                # Build COCO keypoints dictionary
                coco_kpts_person = {}
                for i, name in enumerate(coco_keypoint_names):
                    coco_kpts_person[name] = {
                        "x": float(kpts_xy[i, 0]),
                        "y": float(kpts_xy[i, 1]),
                        "confidence": float(kpts_conf[i])
                    }
                
                # Map to custom skeleton (without feet/toes)
                custom_keypoint_names = self.get_keypoint_names()
                custom_keypoints = self._remap_to_custom_skeleton(coco_kpts_person, custom_keypoint_names)
                keypoints_flat = self._keypoints_to_flat_array(custom_keypoints, custom_keypoint_names)
                
                # Convert bbox format
                bbox_xywh = boxes_tensor.xywh[person_idx].cpu().numpy()
                top_left_x = float(bbox_xywh[0] - bbox_xywh[2] / 2)
                top_left_y = float(bbox_xywh[1] - bbox_xywh[3] / 2)
                bbox = [round(top_left_x, 2), round(top_left_y, 2), 
                       round(float(bbox_xywh[2]), 2), round(float(bbox_xywh[3]), 2)]
                
                annotation = self._create_coco_annotation(keypoints_flat, bbox, 1, person_idx + 1)
                annotations.append(annotation)
            
            return {
                'annotations': annotations,
                'categories': [{'id': 1, 'name': 'person', 'keypoints': self.get_keypoint_names()}]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate poses with YOLO: {e}")
            return self._empty_annotations()
    
    def get_keypoint_names(self) -> List[str]:
        """Get keypoint names (excluding feet/toes)"""
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
