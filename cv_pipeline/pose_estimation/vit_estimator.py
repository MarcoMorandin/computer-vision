"""ViT Pose estimation implementation."""

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
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ViTPoseEstimator(BasePoseEstimator):
    """ViT Pose estimation implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = None
        self.pose_processor = None
        self.pose_model = None
        
    def get_model_type(self) -> PoseEstimationModel:
        return PoseEstimationModel.VIT_POSE
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize ViT Pose model"""
        try:
            if not HAS_ULTRALYTICS:
                raise ImportError("ultralytics not available. Install with: pip install ultralytics")
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers not available. Install with: pip install transformers")
            
            # Need torch for ViT Pose
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch not available. Install with: pip install torch")
            
            # Initialize YOLO detector for person detection
            detector_model = config.get('detector_model', 'yolo11l.pt')
            self.detector = YOLO(detector_model)
            
            # Initialize ViT Pose model
            model_name = config.get('model_name', 'microsoft/vitpose-base-simple')
            self.pose_processor = AutoProcessor.from_pretrained(model_name)
            self.pose_model = VitPoseForPoseEstimation.from_pretrained(model_name)
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pose_model.to(device)
            
            self.confidence_threshold = config.get('confidence_threshold', 0.25)
            
            self.logger.info(f"Initialized ViT Pose model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ViT Pose model: {e}")
            return False
    
    def estimate_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate poses using ViT Pose"""
        if self.detector is None or self.pose_model is None:
            return self._empty_annotations()
        
        try:
            # Check for torch dependency
            try:
                import torch
            except ImportError:
                self.logger.error("PyTorch not available for ViT Pose estimation")
                return self._empty_annotations()
            
            # Check for PIL dependency
            try:
                from PIL import Image
            except ImportError:
                self.logger.error("PIL not available. Install with: pip install Pillow")
                return self._empty_annotations()
            
            # Convert frame to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect persons
            det_results = self.detector(pil_img, conf=self.confidence_threshold, verbose=False)
            det = det_results[0]
            
            if det.boxes is None or det.boxes.xyxy.shape[0] == 0:
                return self._empty_annotations()
            
            # Filter for person class (class 0)
            boxes_xyxy = det.boxes.xyxy.cpu().numpy()
            classes = det.boxes.cls.cpu().numpy().astype(int)
            person_mask = classes == 0
            boxes_xyxy = boxes_xyxy[person_mask]
            
            if boxes_xyxy.shape[0] == 0:
                return self._empty_annotations()
            
            annotations = []
            device = next(self.pose_model.parameters()).device
            
            # Process each detected person
            for person_idx, bbox_xyxy in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = bbox_xyxy.astype(int)
                
                # Crop person region
                person_crop = pil_img.crop((x1, y1, x2, y2))
                
                # Process with ViT Pose
                inputs = self.pose_processor(person_crop, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.pose_model(**inputs)
                
                # Extract keypoints
                pose_result = outputs.pose_results[0] if hasattr(outputs, 'pose_results') else outputs
                keypoints = pose_result['keypoints'] if isinstance(pose_result, dict) else pose_result
                
                # Map keypoints back to original image coordinates
                keypoints = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
                
                # Scale keypoints to original image
                crop_width = x2 - x1
                crop_height = y2 - y1
                
                coco_keypoint_names = [
                    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                ]
                
                coco_kpts_person = {}
                for i, name in enumerate(coco_keypoint_names):
                    if i < len(keypoints):
                        kpt = keypoints[i]
                        x_scaled = (kpt[0] / person_crop.width) * crop_width + x1
                        y_scaled = (kpt[1] / person_crop.height) * crop_height + y1
                        confidence = kpt[2] if len(kpt) > 2 else 1.0
                        
                        coco_kpts_person[name] = {
                            "x": float(x_scaled),
                            "y": float(y_scaled),
                            "confidence": float(confidence)
                        }
                
                # Map to custom skeleton
                custom_keypoint_names = self.get_keypoint_names()
                custom_keypoints = self._remap_to_custom_skeleton(coco_kpts_person, custom_keypoint_names)
                keypoints_flat = self._keypoints_to_flat_array(custom_keypoints, custom_keypoint_names)
                
                # Create bbox in COCO format
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                annotation = self._create_coco_annotation(keypoints_flat, bbox, 1, person_idx + 1)
                annotations.append(annotation)
            
            return {
                'annotations': annotations,
                'categories': [{'id': 1, 'name': 'person', 'keypoints': self.get_keypoint_names()}]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate poses with ViT Pose: {e}")
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
