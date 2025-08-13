"""Utility functions for the CV pipeline"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np


def load_camera_calibration(calib_path: Union[str, Path]) -> Dict[str, Any]:
    """Load camera calibration data from JSON file"""
    with open(calib_path, 'r') as f:
        return json.load(f)


def extract_camera_id_from_filename(filename: str) -> Optional[str]:
    """Extract camera ID from filename
    
    Supports formats like:
    - out1_frame_0001.jpg -> "1"
    - cam_2_image.png -> "2" 
    - video_cam3.mp4 -> "3"
    """
    patterns = [
        r'out(\d+)',
        r'cam_?(\d+)',
        r'camera_?(\d+)',
        r'c(\d+)',
        r'(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_frame_id_from_filename(filename: str) -> Optional[int]:
    """Extract frame ID from filename
    
    Supports formats like:
    - out1_frame_0001.jpg -> 1
    - image_frame_123.png -> 123
    - video_0045.mp4 -> 45
    """
    patterns = [
        r'frame_(\d+)',
        r'(\d{4,})',  # 4 or more digits
        r'_(\d+)\.'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return None


def parse_filename(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse both frame ID and camera ID from filename"""
    frame_id = extract_frame_id_from_filename(filename)
    camera_id = extract_camera_id_from_filename(filename)
    return frame_id, camera_id


def keypoints_to_coco_format(keypoints: List[Dict[str, Any]], keypoint_names: List[str]) -> List[float]:
    """Convert keypoints dictionary to COCO flat array format
    
    Args:
        keypoints: List of keypoint dictionaries with 'x', 'y', 'confidence' keys
        keypoint_names: Ordered list of keypoint names
        
    Returns:
        Flat list in format [x1, y1, v1, x2, y2, v2, ...]
    """
    flat_array = []
    
    for name in keypoint_names:
        kp = keypoints.get(name, {'x': 0, 'y': 0, 'confidence': 0})
        x, y, conf = kp['x'], kp['y'], kp['confidence']
        visibility = 2 if conf > 0.1 else 0  # COCO visibility format
        flat_array.extend([float(x), float(y), int(visibility)])
    
    return flat_array


def coco_format_to_keypoints(flat_array: List[float], keypoint_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Convert COCO flat array to keypoints dictionary
    
    Args:
        flat_array: Flat list in format [x1, y1, v1, x2, y2, v2, ...]
        keypoint_names: Ordered list of keypoint names
        
    Returns:
        Dictionary with keypoint names as keys and {'x', 'y', 'confidence'} as values
    """
    keypoints = {}
    
    for i, name in enumerate(keypoint_names):
        idx = i * 3
        if idx + 2 < len(flat_array):
            x, y, v = flat_array[idx], flat_array[idx + 1], flat_array[idx + 2]
            confidence = 1.0 if v > 0 else 0.0
            keypoints[name] = {
                'x': float(x),
                'y': float(y),
                'confidence': confidence
            }
        else:
            keypoints[name] = {'x': 0.0, 'y': 0.0, 'confidence': 0.0}
    
    return keypoints


def calculate_bbox_from_keypoints(keypoints: List[float]) -> List[float]:
    """Calculate bounding box from keypoints in COCO format
    
    Args:
        keypoints: Flat list in format [x1, y1, v1, x2, y2, v2, ...]
        
    Returns:
        Bounding box in COCO format [x, y, width, height]
    """
    valid_points = []
    
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints) and keypoints[i + 2] > 0:  # if visible
            valid_points.append([keypoints[i], keypoints[i + 1]])
    
    if not valid_points:
        return [0, 0, 0, 0]
    
    valid_points = np.array(valid_points)
    x_min, y_min = np.min(valid_points, axis=0)
    x_max, y_max = np.max(valid_points, axis=0)
    
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def validate_coco_annotation(annotation: Dict[str, Any]) -> bool:
    """Validate COCO annotation structure"""
    required_fields = ['id', 'image_id', 'category_id', 'bbox', 'area']
    
    for field in required_fields:
        if field not in annotation:
            return False
    
    # Validate bbox format
    bbox = annotation['bbox']
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    
    # Validate keypoints if present
    if 'keypoints' in annotation:
        keypoints = annotation['keypoints']
        if not isinstance(keypoints, list) or len(keypoints) % 3 != 0:
            return False
    
    return True


def merge_coco_datasets(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple COCO datasets into one
    
    Args:
        datasets: List of COCO dataset dictionaries
        
    Returns:
        Merged COCO dataset
    """
    if not datasets:
        return {}
    
    # Start with the first dataset as base
    merged = {
        'info': datasets[0].get('info', {}),
        'licenses': datasets[0].get('licenses', []),
        'categories': datasets[0].get('categories', []),
        'images': [],
        'annotations': []
    }
    
    img_id_offset = 0
    ann_id_offset = 0
    
    for dataset in datasets:
        # Add images with offset IDs
        for img in dataset.get('images', []):
            img_copy = img.copy()
            img_copy['id'] += img_id_offset
            merged['images'].append(img_copy)
        
        # Add annotations with offset IDs
        for ann in dataset.get('annotations', []):
            ann_copy = ann.copy()
            ann_copy['id'] += ann_id_offset
            ann_copy['image_id'] += img_id_offset
            merged['annotations'].append(ann_copy)
        
        # Update offsets
        if dataset.get('images'):
            img_id_offset = max(img['id'] for img in merged['images']) + 1
        if dataset.get('annotations'):
            ann_id_offset = max(ann['id'] for ann in merged['annotations']) + 1
    
    return merged


def filter_annotations_by_confidence(annotations: List[Dict[str, Any]], 
                                   min_confidence: float = 0.1) -> List[Dict[str, Any]]:
    """Filter annotations by minimum keypoint confidence
    
    Args:
        annotations: List of COCO annotations
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered annotations list
    """
    filtered = []
    
    for ann in annotations:
        if 'keypoints' not in ann:
            filtered.append(ann)
            continue
        
        keypoints = ann['keypoints']
        valid_keypoints = 0
        
        # Check keypoint confidences
        for i in range(2, len(keypoints), 3):  # Every 3rd element starting from index 2
            if keypoints[i] > 0:  # Visible keypoint
                valid_keypoints += 1
        
        # Keep annotation if it has enough valid keypoints
        if valid_keypoints > 0:
            filtered.append(ann)
    
    return filtered


def create_skeleton_connections(keypoint_names: List[str]) -> List[List[int]]:
    """Create skeleton connections for visualization
    
    Args:
        keypoint_names: Ordered list of keypoint names
        
    Returns:
        List of connections as [start_idx, end_idx] pairs (1-indexed)
    """
    # Map keypoint names to indices (1-indexed for COCO format)
    name_to_idx = {name: i + 1 for i, name in enumerate(keypoint_names)}
    
    # Define standard skeleton connections
    connections = []
    
    # Basic human skeleton connections
    skeleton_pairs = [
        ('Head', 'Neck'),
        ('Neck', 'LShoulder'),
        ('Neck', 'RShoulder'),
        ('LShoulder', 'LElbow'),
        ('LElbow', 'LHand'),
        ('RShoulder', 'RElbow'),
        ('RElbow', 'RHand'),
        ('Neck', 'Spine'),
        ('Spine', 'Hips'),
        ('Hips', 'LHip'),
        ('Hips', 'RHip'),
        ('LHip', 'LKnee'),
        ('LKnee', 'LAnkle'),
        ('RHip', 'RKnee'),
        ('RKnee', 'RAnkle')
    ]
    
    # Convert to indices
    for start_name, end_name in skeleton_pairs:
        if start_name in name_to_idx and end_name in name_to_idx:
            connections.append([name_to_idx[start_name], name_to_idx[end_name]])
    
    return connections


def normalize_keypoints(keypoints: List[float], image_width: int, image_height: int) -> List[float]:
    """Normalize keypoints to [0, 1] range
    
    Args:
        keypoints: Keypoints in COCO format [x1, y1, v1, ...]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Normalized keypoints
    """
    normalized = []
    
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        
        # Normalize coordinates
        norm_x = x / image_width if image_width > 0 else 0
        norm_y = y / image_height if image_height > 0 else 0
        
        normalized.extend([norm_x, norm_y, v])
    
    return normalized


def denormalize_keypoints(keypoints: List[float], image_width: int, image_height: int) -> List[float]:
    """Denormalize keypoints from [0, 1] range to pixel coordinates
    
    Args:
        keypoints: Normalized keypoints in COCO format [x1, y1, v1, ...]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Denormalized keypoints in pixel coordinates
    """
    denormalized = []
    
    for i in range(0, len(keypoints), 3):
        norm_x, norm_y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        
        # Denormalize coordinates
        x = norm_x * image_width
        y = norm_y * image_height
        
        denormalized.extend([x, y, v])
    
    return denormalized
