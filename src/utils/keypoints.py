"""Keypoint utilities for pose estimation models and virtual keypoint calculations."""

from typing import Dict, List


def calculate_virtual_keypoints(keypoints_dict: Dict[str, Dict[str, float]]) -> None:
    """Calculate virtual keypoints (Hips, Neck, Spine) from existing keypoints.
    
    This function modifies the input dictionary in-place.
    
    Args:
        keypoints_dict: Dictionary mapping keypoint names to {x, y, confidence}
    """
    # Hips (midpoint of left and right hip)
    if ("Hips" in keypoints_dict and 
        "LHip" in keypoints_dict and 
        "RHip" in keypoints_dict):
        l_hip, r_hip = keypoints_dict["LHip"], keypoints_dict["RHip"]
        if l_hip["confidence"] > 0 and r_hip["confidence"] > 0:
            keypoints_dict["Hips"] = {
                "x": (l_hip["x"] + r_hip["x"]) / 2,
                "y": (l_hip["y"] + r_hip["y"]) / 2,
                "confidence": min(l_hip["confidence"], r_hip["confidence"]),
            }

    # Neck (midpoint of left and right shoulder)
    if ("Neck" in keypoints_dict and 
        "LShoulder" in keypoints_dict and 
        "RShoulder" in keypoints_dict):
        l_shoulder, r_shoulder = keypoints_dict["LShoulder"], keypoints_dict["RShoulder"]
        if l_shoulder["confidence"] > 0 and r_shoulder["confidence"] > 0:
            keypoints_dict["Neck"] = {
                "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
                "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
                "confidence": min(l_shoulder["confidence"], r_shoulder["confidence"]),
            }

    # Spine (midpoint of hips and neck)
    if ("Spine" in keypoints_dict and 
        "Hips" in keypoints_dict and 
        "Neck" in keypoints_dict):
        hips, neck = keypoints_dict["Hips"], keypoints_dict["Neck"]
        if hips["confidence"] > 0 and neck["confidence"] > 0:
            keypoints_dict["Spine"] = {
                "x": (hips["x"] + neck["x"]) / 2,
                "y": (hips["y"] + neck["y"]) / 2,
                "confidence": min(hips["confidence"], neck["confidence"]),
            }


def create_yolo_keypoint_mapping(kept_keypoint_names: List[str]) -> Dict[str, int]:
    """Create mapping from custom keypoint names to YOLO indices.
    
    Args:
        kept_keypoint_names: List of keypoint names to keep
        
    Returns:
        Dictionary mapping keypoint names to YOLO indices
    """
    # Base YOLO keypoint mapping
    base_mapping = {
        "Nose": 0,
        "LEye": 1,
        "REye": 2,
        "LEar": 3,
        "REar": 4,
        "LShoulder": 5,
        "RShoulder": 6,
        "LElbow": 7,
        "RElbow": 8,
        "LWrist": 9,
        "RWrist": 10,
        "LHip": 11,
        "RHip": 12,
        "LKnee": 13,
        "RKnee": 14,
        "LAnkle": 15,
        "RAnkle": 16,
        # Alternative mappings
        "Head": 0,  # Map Head to Nose (closest equivalent)
        "RHand": 10,  # Map RHand to RWrist
        "LHand": 9,  # Map LHand to LWrist
    }

    # Create mapping only for kept keypoints
    mapping = {}
    for keypoint_name in kept_keypoint_names:
        if keypoint_name in base_mapping:
            mapping[keypoint_name] = base_mapping[keypoint_name]
        elif keypoint_name in ["Hips", "Neck", "Spine"]:
            # Virtual keypoints - will be calculated later
            mapping[keypoint_name] = -1  # Special marker for virtual keypoints
        
    return mapping


def create_vitpose_keypoint_mapping(kept_keypoint_names: List[str]) -> Dict[str, int]:
    """Create mapping from custom keypoint names to ViTPose (COCO-17) indices.
    
    Args:
        kept_keypoint_names: List of keypoint names to keep
        
    Returns:
        Dictionary mapping keypoint names to ViTPose indices
    """
    # ViTPose COCO-17 indices
    vit_indices = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
    }

    # Map custom keypoint names to ViTPose indices
    base_mapping = {
        "LShoulder": vit_indices["left_shoulder"],
        "RShoulder": vit_indices["right_shoulder"],
        "LElbow": vit_indices["left_elbow"],
        "RElbow": vit_indices["right_elbow"],
        "LHand": vit_indices["left_wrist"],
        "RHand": vit_indices["right_wrist"],
        "LWrist": vit_indices["left_wrist"],
        "RWrist": vit_indices["right_wrist"],
        "LHip": vit_indices["left_hip"],
        "RHip": vit_indices["right_hip"],
        "LKnee": vit_indices["left_knee"],
        "RKnee": vit_indices["right_knee"],
        "LAnkle": vit_indices["left_ankle"],
        "RAnkle": vit_indices["right_ankle"],
        "Head": vit_indices["nose"],
        "Nose": vit_indices["nose"],
        "LEye": vit_indices["left_eye"],
        "REye": vit_indices["right_eye"],
        "LEar": vit_indices["left_ear"],
        "REar": vit_indices["right_ear"],
        # Virtual keypoints
        "Hips": -1,
        "Neck": -1,
        "Spine": -1
    }

    # Create mapping only for kept keypoints
    mapping = {}
    for keypoint_name in kept_keypoint_names:
        if keypoint_name in base_mapping:
            mapping[keypoint_name] = base_mapping[keypoint_name]
        else:
            # Not in mapping - default to -2 (no source)
            mapping[keypoint_name] = -2
            
    return mapping


def get_yolo_keypoint_names() -> List[str]:
    """Get the standard YOLO keypoint names in order.
    
    Returns:
        List of YOLO keypoint names
    """
    return [
        "Nose",
        "LEye",
        "REye",
        "LEar",
        "REar",
        "LShoulder",
        "RShoulder",
        "LElbow",
        "RElbow",
        "LWrist",
        "RWrist",
        "LHip",
        "RHip",
        "LKnee",
        "RKnee",
        "LAnkle",
        "RAnkle",
    ]


def get_vitpose_keypoint_names() -> List[str]:
    """Get the standard ViTPose (COCO-17) keypoint names in order.
    
    Returns:
        List of ViTPose keypoint names
    """
    return [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]


def init_keypoints_dict(keypoint_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Initialize keypoints dictionary with default values.
    
    Args:
        keypoint_names: List of keypoint names
        
    Returns:
        Dictionary with initialized keypoint entries
    """
    return {
        name: {"x": 0.0, "y": 0.0, "confidence": 0.0}
        for name in keypoint_names
    }
