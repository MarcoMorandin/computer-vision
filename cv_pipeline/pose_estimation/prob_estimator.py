"""ProbPose estimation implementation."""

from typing import Dict, Any, List
import numpy as np
import cv2

from ..core.base import PoseEstimationModel
from .base_estimator import BasePoseEstimator

# Optional dependencies
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from mmpose.apis import init_model, inference_topdown
    HAS_MMPOSE = True
except ImportError:
    HAS_MMPOSE = False


class ProbPoseEstimator(BasePoseEstimator):
    """ProbPose estimation implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = None
        self.pose_model = None
        
    def get_model_type(self) -> PoseEstimationModel:
        return PoseEstimationModel.PROB_POSE
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize ProbPose model"""
        try:
            if not HAS_ULTRALYTICS:
                raise ImportError("ultralytics not available. Install with: pip install ultralytics")
            if not HAS_MMPOSE:
                raise ImportError("mmpose not available. Install with: pip install mmpose")
            
            # Need mmengine for MMPose
            try:
                from mmengine.config import Config
            except ImportError:
                raise ImportError("mmengine not available. Install with: pip install mmengine")
            
            # Initialize YOLO detector
            detector_model = config.get('detector_model', 'yolo11l.pt')
            self.detector = YOLO(detector_model)
            
            # Initialize ProbPose model
            config_path = config.get('config_path', 'pose_estimation_probpose/ProbPose/configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py')
            checkpoint_path = config.get('checkpoint_path', 'pose_estimation_probpose/ProbPose.pth')
            
            cfg = Config.fromfile(config_path)
            self.pose_model = init_model(cfg, checkpoint=checkpoint_path)
            
            self.confidence_threshold = config.get('confidence_threshold', 0.25)
            
            self.logger.info("Initialized ProbPose model")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ProbPose model: {e}")
            return False
    
    def estimate_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate poses using ProbPose"""
        if self.detector is None or self.pose_model is None:
            return self._empty_annotations()
        
        try:
            if not HAS_MMPOSE:
                self.logger.error("MMPose not available for ProbPose estimation")
                return self._empty_annotations()
            
            # Need PIL for image conversion
            try:
                from PIL import Image
            except ImportError:
                self.logger.error("PIL not available. Install with: pip install Pillow")
                return self._empty_annotations()
            
            # Convert to PIL for YOLO detection
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect persons
            det_results = self.detector(pil_img, conf=self.confidence_threshold, verbose=False)
            det = det_results[0]
            
            if det.boxes is None or det.boxes.xyxy.shape[0] == 0:
                return self._empty_annotations()
            
            boxes_xyxy = det.boxes.xyxy.cpu().numpy()
            classes = det.boxes.cls.cpu().numpy().astype(int)
            person_mask = classes == 0
            boxes_xyxy = boxes_xyxy[person_mask]
            
            if boxes_xyxy.shape[0] == 0:
                return self._empty_annotations()
            
            # Run ProbPose inference
            frame_path = "/tmp/temp_frame.jpg"  # Temporary file for MMPose
            cv2.imwrite(frame_path, frame)
            
            pose_samples = inference_topdown(self.pose_model, frame_path, bboxes=boxes_xyxy, bbox_format="xyxy")
            
            annotations = []
            coco_keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            for person_idx, (pose_sample, bbox_xyxy) in enumerate(zip(pose_samples, boxes_xyxy)):
                # Extract keypoints from pose sample
                keypoints = pose_sample.pred_instances.keypoints[0]  # Shape: (17, 2)
                scores = pose_sample.pred_instances.keypoint_scores[0]  # Shape: (17,)
                
                # Build COCO keypoints dictionary
                coco_kpts_person = {}
                for i, name in enumerate(coco_keypoint_names):
                    if i < len(keypoints):
                        coco_kpts_person[name] = {
                            "x": float(keypoints[i, 0]),
                            "y": float(keypoints[i, 1]),
                            "confidence": float(scores[i])
                        }
                
                # Map to custom skeleton
                custom_keypoint_names = self.get_keypoint_names()
                custom_keypoints = self._remap_to_custom_skeleton(coco_kpts_person, custom_keypoint_names)
                keypoints_flat = self._keypoints_to_flat_array(custom_keypoints, custom_keypoint_names)
                
                # Create bbox in COCO format
                x1, y1, x2, y2 = bbox_xyxy
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                annotation = self._create_coco_annotation(keypoints_flat, bbox, 1, person_idx + 1)
                annotations.append(annotation)
            
            return {
                'annotations': annotations,
                'categories': [{'id': 1, 'name': 'person', 'keypoints': self.get_keypoint_names()}]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate poses with ProbPose: {e}")
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
