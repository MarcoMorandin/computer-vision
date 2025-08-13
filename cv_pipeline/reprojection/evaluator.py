"""Reprojection evaluation implementation"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np

from ..core.base import ReprojectionEvaluator as BaseReprojectionEvaluator, CameraParameters


class ReprojectionEvaluator(BaseReprojectionEvaluator):
    """Reprojection evaluation implementation"""
    
    def __init__(self):
        self.camera_params = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self, camera_params: Dict[str, CameraParameters]) -> bool:
        """Initialize evaluator with camera parameters"""
        try:
            self.camera_params = camera_params
            
            self.logger.info(f"Initialized reprojection evaluator with {len(camera_params)} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reprojection evaluator: {e}")
            return False
    
    def reproject_3d_poses(self, poses_3d: Dict[str, List[Optional[List[float]]]]) -> Dict[str, Dict[str, List[Optional[List[float]]]]]:
        """Reproject 3D poses back to 2D camera views"""
        reprojected_poses = {}
        
        try:
            for frame_id, keypoints_3d in poses_3d.items():
                frame_reprojected = {}
                
                # Convert keypoints to numpy array
                points_3d = self._to_np_keypoints_3d(keypoints_3d)
                
                # Reproject to each camera
                for cam_id, cam_params in self.camera_params.items():
                    reprojected_2d = self._project_points(points_3d, cam_params)
                    frame_reprojected[cam_id] = reprojected_2d
                
                reprojected_poses[frame_id] = frame_reprojected
                
            self.logger.info(f"Reprojected poses for {len(reprojected_poses)} frames")
            return reprojected_poses
            
        except Exception as e:
            self.logger.error(f"Failed to reproject 3D poses: {e}")
            return {}
    
    def evaluate_reprojection_error(self, original_2d: Dict[str, Dict[str, Any]], 
                                   reprojected_2d: Dict[str, Dict[str, List[Optional[List[float]]]]]) -> Dict[str, Any]:
        """Evaluate reprojection error between original and reprojected 2D poses"""
        try:
            total_error = 0.0
            total_keypoints = 0
            errors_by_camera = {}
            errors_by_keypoint = {}
            
            for frame_id in original_2d.keys():
                if frame_id not in reprojected_2d:
                    continue
                
                for cam_id in original_2d[frame_id].keys():
                    if cam_id not in reprojected_2d[frame_id]:
                        continue
                    
                    # Extract original keypoints
                    orig_keypoints = self._extract_keypoints_from_annotation(original_2d[frame_id][cam_id])
                    reproj_keypoints = reprojected_2d[frame_id][cam_id]
                    
                    if not orig_keypoints or not reproj_keypoints:
                        continue
                    
                    # Calculate errors
                    frame_cam_errors = self._calculate_keypoint_errors(orig_keypoints, reproj_keypoints)
                    
                    # Accumulate statistics
                    if cam_id not in errors_by_camera:
                        errors_by_camera[cam_id] = []
                    errors_by_camera[cam_id].extend(frame_cam_errors)
                    
                    for i, error in enumerate(frame_cam_errors):
                        if error is not None:
                            if i not in errors_by_keypoint:
                                errors_by_keypoint[i] = []
                            errors_by_keypoint[i].append(error)
                            
                            total_error += error
                            total_keypoints += 1
            
            # Calculate summary statistics
            results = {
                'mean_error': total_error / total_keypoints if total_keypoints > 0 else 0,
                'total_keypoints_evaluated': total_keypoints,
                'camera_errors': {},
                'keypoint_errors': {}
            }
            
            # Per-camera statistics
            for cam_id, errors in errors_by_camera.items():
                valid_errors = [e for e in errors if e is not None]
                if valid_errors:
                    results['camera_errors'][cam_id] = {
                        'mean_error': float(np.mean(valid_errors)),
                        'std_error': float(np.std(valid_errors)),
                        'median_error': float(np.median(valid_errors)),
                        'max_error': float(np.max(valid_errors)),
                        'num_keypoints': len(valid_errors)
                    }
            
            # Per-keypoint statistics
            for kp_idx, errors in errors_by_keypoint.items():
                results['keypoint_errors'][kp_idx] = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'median_error': float(np.median(errors)),
                    'max_error': float(np.max(errors)),
                    'num_observations': len(errors)
                }
            
            self.logger.info(f"Reprojection evaluation completed. Mean error: {results['mean_error']:.2f} pixels")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate reprojection error: {e}")
            return {'error': str(e)}
    
    def _to_np_keypoints_3d(self, keypoints: List[Optional[List[float]]]) -> np.ndarray:
        """Convert keypoints list to numpy array with NaNs for missing points"""
        arr = []
        for kp in keypoints:
            if kp is not None and len(kp) >= 3:
                arr.append([kp[0], kp[1], kp[2]])
            else:
                arr.append([np.nan, np.nan, np.nan])
        
        return np.array(arr, dtype=np.float32)
    
    def _project_points(self, points_3d: np.ndarray, cam_params: CameraParameters) -> List[Optional[List[float]]]:
        """Project 3D points to 2D using camera parameters"""
        if points_3d.size == 0:
            return []
        
        try:
            # Use projection matrix for direct projection
            P = cam_params.projection_matrix
            
            # Convert to homogeneous coordinates
            points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
            
            # Project points
            points_2d_h = (P @ points_3d_h.T).T
            
            # Convert from homogeneous to 2D coordinates
            with np.errstate(divide='ignore', invalid='ignore'):
                points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
            
            # Convert to list format, handling NaN values
            result = []
            for i, (x, y) in enumerate(points_2d):
                if np.isfinite(x) and np.isfinite(y):
                    result.append([float(x), float(y)])
                else:
                    result.append(None)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to project points: {e}")
            return [None] * len(points_3d)
    
    def _extract_keypoints_from_annotation(self, annotation_data: Any) -> Optional[List[Tuple[float, float, float]]]:
        """Extract keypoints from annotation data"""
        keypoints_flat = None
        
        # Handle different annotation formats
        if isinstance(annotation_data, dict):
            if 'annotations' in annotation_data and isinstance(annotation_data['annotations'], list):
                if len(annotation_data['annotations']) > 0:
                    ann = annotation_data['annotations'][0]  # Take first annotation
                    keypoints_flat = ann.get('keypoints', [])
            elif 'keypoints' in annotation_data:
                keypoints_flat = annotation_data['keypoints']
            else:
                # Try to find keypoints in the data structure
                for value in annotation_data.values():
                    if isinstance(value, dict) and 'keypoints' in value:
                        keypoints_flat = value['keypoints']
                        break
                    elif isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and 'keypoints' in value[0]:
                            keypoints_flat = value[0]['keypoints']
                            break
        elif isinstance(annotation_data, list):
            if len(annotation_data) > 0 and isinstance(annotation_data[0], dict):
                keypoints_flat = annotation_data[0].get('keypoints', [])
            else:
                keypoints_flat = annotation_data
        
        if not keypoints_flat or len(keypoints_flat) % 3 != 0:
            return None
        
        # Convert to (x, y, v) tuples
        keypoints = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
            keypoints.append((float(x), float(y), float(v)))
        
        return keypoints
    
    def _calculate_keypoint_errors(self, original_keypoints: List[Tuple[float, float, float]], 
                                 reprojected_keypoints: List[Optional[List[float]]]) -> List[Optional[float]]:
        """Calculate Euclidean distance errors between original and reprojected keypoints"""
        errors = []
        
        min_len = min(len(original_keypoints), len(reprojected_keypoints))
        
        for i in range(min_len):
            orig_x, orig_y, orig_v = original_keypoints[i]
            reproj_kp = reprojected_keypoints[i]
            
            # Only calculate error for visible keypoints
            if orig_v > 0 and reproj_kp is not None and len(reproj_kp) >= 2:
                reproj_x, reproj_y = reproj_kp[0], reproj_kp[1]
                
                # Calculate Euclidean distance
                error = np.sqrt((orig_x - reproj_x)**2 + (orig_y - reproj_y)**2)
                errors.append(float(error))
            else:
                errors.append(None)
        
        return errors
    
    def save_reprojected_coco(self, reprojected_poses: Dict[str, Dict[str, List[Optional[List[float]]]]], 
                            output_path: Path, template_coco_path: Optional[Path] = None) -> Dict[str, Any]:
        """Save reprojected poses in COCO format"""
        try:
            # Load template COCO file if available
            if template_coco_path and template_coco_path.exists():
                with open(template_coco_path, 'r') as f:
                    template = json.load(f)
            else:
                template = {
                    'info': {},
                    'licenses': [],
                    'categories': [{'id': 1, 'name': 'person', 'keypoints': []}]
                }
            
            # Build COCO structure
            coco_data = {
                'info': template.get('info', {}),
                'licenses': template.get('licenses', []),
                'categories': template.get('categories', []),
                'images': [],
                'annotations': []
            }
            
            # Generate images and annotations
            img_id_counter = 0
            ann_id_counter = 0
            
            for frame_id, cameras in reprojected_poses.items():
                for cam_id, keypoints in cameras.items():
                    # Create image entry
                    file_name = f"out{cam_id}_frame_{int(frame_id):04d}.png"
                    coco_data['images'].append({
                        'id': img_id_counter,
                        'file_name': file_name,
                        'width': 1920,  # Default size
                        'height': 1080
                    })
                    
                    # Create annotation entry
                    keypoints_flat = []
                    valid_points = []
                    
                    for kp in keypoints:
                        if kp is not None and len(kp) >= 2:
                            x, y = kp[0], kp[1]
                            keypoints_flat.extend([float(x), float(y), 2])  # visibility = 2
                            valid_points.append([x, y])
                        else:
                            keypoints_flat.extend([0, 0, 0])
                    
                    # Calculate bounding box
                    if valid_points:
                        valid_points = np.array(valid_points)
                        x_min, y_min = np.min(valid_points, axis=0)
                        x_max, y_max = np.max(valid_points, axis=0)
                        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                        area = float((x_max - x_min) * (y_max - y_min))
                    else:
                        bbox = [0, 0, 0, 0]
                        area = 0
                    
                    coco_data['annotations'].append({
                        'id': ann_id_counter,
                        'image_id': img_id_counter,
                        'category_id': 1,
                        'keypoints': keypoints_flat,
                        'num_keypoints': sum(1 for v in keypoints_flat[2::3] if v > 0),
                        'bbox': bbox,
                        'area': area,
                        'iscrowd': 0
                    })
                    
                    img_id_counter += 1
                    ann_id_counter += 1
            
            # Save COCO file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            self.logger.info(f"Saved reprojected COCO data to {output_path}")
            return coco_data
            
        except Exception as e:
            self.logger.error(f"Failed to save reprojected COCO data: {e}")
            return {}
