"""Keypoint utilities for pose estimation models and virtual keypoint calculations."""

from typing import Dict, List


def calculate_virtual_keypoints(keypoints_dict: Dict[str, Dict[str, float]]) -> None:
    """Calculate virtual keypoints (Hips, Neck, Spine) from existing keypoints.

    Notes
    -----
    This function modifies the input dictionary in-place and expects the
    following keys to be present: "LHip", "RHip", "LShoulder", "RShoulder".

    Parameters
    ----------
    keypoints_dict : dict[str, dict[str, float]]
        Mapping from keypoint name to a dict with fields {x, y, confidence}.
    """
    # Hips (midpoint of left and right hip)
    l_hip, r_hip = keypoints_dict["LHip"], keypoints_dict["RHip"]
    keypoints_dict["Hips"] = {
        "x": (l_hip["x"] + r_hip["x"]) / 2,
        "y": (l_hip["y"] + r_hip["y"]) / 2,
        "confidence": min(l_hip["confidence"], r_hip["confidence"]),
    }

    # Neck (midpoint of left and right shoulder)

    l_shoulder, r_shoulder = keypoints_dict["LShoulder"], keypoints_dict["RShoulder"]
    keypoints_dict["Neck"] = {
        "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
        "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
        "confidence": min(l_shoulder["confidence"], r_shoulder["confidence"]),
    }

    # Spine (midpoint of hips and neck)
    hips, neck = keypoints_dict["Hips"], keypoints_dict["Neck"]
    keypoints_dict["Spine"] = {
        "x": (hips["x"] + neck["x"]) / 2,
        "y": (hips["y"] + neck["y"]) / 2,
        "confidence": min(hips["confidence"], neck["confidence"]),
    }


def create_keypoint_mapping(kept_keypoint_names: List[str]) -> Dict[str, int]:
    """Create mapping from custom keypoint names to COCO indices.

    Parameters
    ----------
    kept_keypoint_names : list[str]
        Ordered list of keypoint names.

    Returns
    -------
    dict[str, int]
        Maps custom names to COCO indices. A value of -1 denotes a virtual
        keypoint (no direct COCO counterpart).
    """
    # Base COCO keypoint mapping
    base_mapping = {
        "Nose": 0,
        "Head": 0,  # Map Head to Nose (closest equivalent)
        "LEye": 1,
        "REye": 2,
        "LEar": 3,
        "REar": 4,
        "LShoulder": 5,
        "RShoulder": 6,
        "LElbow": 7,
        "RElbow": 8,
        "LWrist": 9,
        "LHand": 9,  # Map LHand to LWrist
        "RWrist": 10,
        "RHand": 10,  # Map RHand to RWrist
        "LHip": 11,
        "RHip": 12,
        "LKnee": 13,
        "RKnee": 14,
        "LAnkle": 15,
        "RAnkle": 16,
        "Hips": -1,
        "Neck": -1,
        "Spine": -1,
    }

    # Create mapping only for kept keypoints
    mapping = {}
    for keypoint_name in kept_keypoint_names:
        if keypoint_name in base_mapping:
            mapping[keypoint_name] = base_mapping[keypoint_name]

    return mapping
