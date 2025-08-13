"""Base pose estimator with common functionality"""

import logging
from typing import Dict, Any, List

from ..core.base import PoseEstimator


class BasePoseEstimator(PoseEstimator):
    """Base class for pose estimators with common functionality"""
    
    def __init__(self, **kwargs):
        # Ignore extra kwargs passed by Hydra (like 'type', etc.)
        self.confidence_threshold = 0.25
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _remap_to_custom_skeleton(self, coco_kpts: Dict[str, Dict[str, float]], 
                                custom_keypoint_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Remap COCO keypoints to custom skeleton format"""
        custom_kpts = {name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in custom_keypoint_names}

        def get_kpt(name):
            return coco_kpts.get(name, {"x": 0.0, "y": 0.0, "confidence": 0.0})

        # Direct mappings
        custom_kpts["RHip"] = get_kpt("right_hip")
        custom_kpts["RKnee"] = get_kpt("right_knee")
        custom_kpts["RAnkle"] = get_kpt("right_ankle")
        custom_kpts["LHip"] = get_kpt("left_hip")
        custom_kpts["LKnee"] = get_kpt("left_knee")
        custom_kpts["LAnkle"] = get_kpt("left_ankle")
        custom_kpts["RShoulder"] = get_kpt("right_shoulder")
        custom_kpts["RElbow"] = get_kpt("right_elbow")
        custom_kpts["LShoulder"] = get_kpt("left_shoulder")
        custom_kpts["LElbow"] = get_kpt("left_elbow")
        custom_kpts["Head"] = get_kpt("nose")
        custom_kpts["RHand"] = get_kpt("right_wrist")
        custom_kpts["LHand"] = get_kpt("left_wrist")

        # Virtual keypoints
        l_hip, r_hip = custom_kpts["LHip"], custom_kpts["RHip"]
        if l_hip["confidence"] > 0 and r_hip["confidence"] > 0:
            custom_kpts["Hips"] = {
                "x": (l_hip["x"] + r_hip["x"]) / 2,
                "y": (l_hip["y"] + r_hip["y"]) / 2,
                "confidence": min(l_hip["confidence"], r_hip["confidence"])
            }

        l_shoulder, r_shoulder = custom_kpts["LShoulder"], custom_kpts["RShoulder"]
        if l_shoulder["confidence"] > 0 and r_shoulder["confidence"] > 0:
            custom_kpts["Neck"] = {
                "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
                "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
                "confidence": min(l_shoulder["confidence"], r_shoulder["confidence"])
            }

        hips, neck = custom_kpts.get("Hips", {"confidence": 0}), custom_kpts.get("Neck", {"confidence": 0})
        if hips["confidence"] > 0 and neck["confidence"] > 0:
            custom_kpts["Spine"] = {
                "x": (hips["x"] + neck["x"]) / 2,
                "y": (hips["y"] + neck["y"]) / 2,
                "confidence": min(hips["confidence"], neck["confidence"])
            }

        return custom_kpts

    def _keypoints_to_flat_array(self, custom_kpts: Dict[str, Dict[str, float]], 
                               custom_keypoint_names: List[str]) -> List[float]:
        """Convert keypoints dictionary to flat array for COCO format"""
        keypoints_flat = []
        for name in custom_keypoint_names:
            kpt = custom_kpts[name]
            x, y, conf = kpt['x'], kpt['y'], kpt['confidence']
            visibility = 2 if conf > 0.1 else 0
            keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
        return keypoints_flat

    def _create_coco_annotation(self, keypoints_flat: List[float], bbox: List[float], 
                              img_id: int, ann_id: int) -> Dict[str, Any]:
        """Create COCO-style annotation"""
        return {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,  # person category
            "bbox": bbox,
            "area": bbox[2] * bbox[3] if len(bbox) >= 4 else 0,
            "segmentation": [],
            "iscrowd": 0,
            "keypoints": keypoints_flat,
            "num_keypoints": sum(1 for v in keypoints_flat[2::3] if v > 0)
        }

    def _empty_annotations(self) -> Dict[str, Any]:
        """Return empty annotations structure"""
        return {
            "annotations": [],
            "keypoint_names": [
                "Hips", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Spine",
                "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
            ]
        }
