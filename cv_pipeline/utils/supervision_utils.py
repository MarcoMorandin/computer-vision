"""Dataset utilities enhanced with supervision library"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from pathlib import Path
import logging

try:
    import supervision as sv
    from supervision import Dataset, Detections
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    sv = None
    # Create dummy classes for when supervision is not available
    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass
    
    class DummyDetections:
        def __init__(self, *args, **kwargs):
            pass
    
    Dataset = DummyDataset
    Detections = DummyDetections

from ..core.base import DataSource, FrameData
from ..utils.helpers import extract_camera_id_from_filename, extract_frame_id_from_filename


class SupervisionDatasetUtils:
    """Enhanced dataset utilities using supervision library"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not HAS_SUPERVISION:
            self.logger.warning("Supervision library not available. Some features may be limited.")
    
    def coco_to_supervision_dataset(self, coco_path: Union[str, Path], 
                                  images_dir: Union[str, Path]):
        """
        Convert COCO dataset to supervision Dataset format
        
        Args:
            coco_path: Path to COCO annotations JSON
            images_dir: Directory containing images
            
        Returns:
            supervision Dataset object
        """
        try:
            dataset = Dataset.from_coco(
                images_directory_path=str(images_dir),
                annotations_path=str(coco_path)
            )
            self.logger.info(f"Loaded COCO dataset with {len(dataset)} images")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load COCO dataset: {e}")
            raise
    
    def supervision_to_coco_format(self, dataset, 
                                 output_path: Union[str, Path],
                                 keypoint_names: Optional[List[str]] = None) -> None:
        """
        Convert supervision Dataset to COCO format with keypoint annotations
        
        Args:
            dataset: supervision Dataset
            output_path: Output COCO JSON file path
            keypoint_names: List of keypoint names for skeleton definition
        """
        if not keypoint_names:
            keypoint_names = [
                "Hips", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Spine",
                "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
            ]
        
        # Create skeleton connections
        skeleton = [
            [1, 2], [2, 3], [3, 4],  # Right leg
            [1, 5], [5, 6], [6, 7],  # Left leg  
            [1, 8], [8, 9], [9, 10], # Torso and head
            [9, 11], [11, 12], [12, 13], # Right arm
            [9, 14], [14, 15], [15, 16]  # Left arm
        ]
        
        coco_data = {
            "info": {
                "description": "Dataset converted from supervision format",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": keypoint_names,
                    "skeleton": skeleton
                }
            ],
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        
        for image_id, (image_name, image) in enumerate(dataset.images.items(), 1):
            # Add image info
            height, width = image.shape[:2]
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })
            
            # Add annotations
            detections = dataset.annotations.get(image_name, Detections.empty())
            
            for det_idx in range(len(detections)):
                # Get bounding box
                bbox = detections.xyxy[det_idx]
                x, y, x2, y2 = bbox
                width_box = x2 - x
                height_box = y2 - y
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(width_box), float(height_box)],
                    "area": float(width_box * height_box),
                    "iscrowd": 0
                }
                
                # Add keypoints if available
                if hasattr(detections, 'data') and 'keypoints' in detections.data:
                    keypoints_list = detections.data['keypoints']
                    if det_idx < len(keypoints_list):
                        keypoints = keypoints_list[det_idx]
                        if isinstance(keypoints, np.ndarray) and len(keypoints) > 0:
                            # Convert to COCO format: [x1, y1, v1, x2, y2, v2, ...]
                            keypoints_flat = []
                            for kp in keypoints:
                                if len(kp) >= 3:
                                    x_kp, y_kp, v_kp = kp[:3]
                                    visibility = 2 if v_kp > 0 else 0
                                    keypoints_flat.extend([float(x_kp), float(y_kp), int(visibility)])
                                else:
                                    keypoints_flat.extend([0.0, 0.0, 0])
                            
                            annotation["keypoints"] = keypoints_flat
                            annotation["num_keypoints"] = sum(1 for v in keypoints_flat[2::3] if v > 0)
                
                coco_data["annotations"].append(annotation)
                annotation_id += 1
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        self.logger.info(f"Saved COCO format dataset to {output_path}")
    
    def merge_datasets(self, datasets: List) :
        """
        Merge multiple supervision datasets
        
        Args:
            datasets: List of supervision Dataset objects
            
        Returns:
            Merged supervision Dataset
        """
        if not datasets:
            return Dataset({}, {})
        
        if len(datasets) == 1:
            return datasets[0]
        
        merged_images = {}
        merged_annotations = {}
        
        for i, dataset in enumerate(datasets):
            for image_name, image in dataset.images.items():
                # Add prefix to avoid name conflicts
                new_image_name = f"dataset_{i}_{image_name}"
                merged_images[new_image_name] = image
                
                # Copy annotations
                if image_name in dataset.annotations:
                    merged_annotations[new_image_name] = dataset.annotations[image_name]
                else:
                    merged_annotations[new_image_name] = Detections.empty()
        
        merged_dataset = Dataset(merged_images, merged_annotations)
        self.logger.info(f"Merged {len(datasets)} datasets into dataset with {len(merged_images)} images")
        
        return merged_dataset
    
    def split_dataset(self, dataset, 
                     split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                     random_seed: Optional[int] = 42) -> Tuple:
        """
        Split dataset into train/validation/test sets
        
        Args:
            dataset: Input supervision Dataset
            split_ratios: (train, val, test) ratios that sum to 1.0
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        import random
        
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Get all image names and shuffle
        image_names = list(dataset.images.keys())
        random.shuffle(image_names)
        
        # Calculate split sizes
        total_images = len(image_names)
        train_size = int(total_images * split_ratios[0])
        val_size = int(total_images * split_ratios[1])
        
        # Split image names
        train_names = image_names[:train_size]
        val_names = image_names[train_size:train_size + val_size]
        test_names = image_names[train_size + val_size:]
        
        # Create split datasets
        def create_split_dataset(names: List[str]) :
            split_images = {name: dataset.images[name] for name in names}
            split_annotations = {name: dataset.annotations.get(name, Detections.empty()) 
                               for name in names}
            return Dataset(split_images, split_annotations)
        
        train_dataset = create_split_dataset(train_names)
        val_dataset = create_split_dataset(val_names)
        test_dataset = create_split_dataset(test_names)
        
        self.logger.info(f"Split dataset: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")
        
        return train_dataset, val_dataset, test_dataset
    
    def filter_dataset_by_quality(self, dataset,
                                min_keypoints: int = 5,
                                min_bbox_area: float = 1000,
                                max_aspect_ratio: float = 5.0) :
        """
        Filter dataset by annotation quality metrics
        
        Args:
            dataset: Input supervision Dataset
            min_keypoints: Minimum number of visible keypoints
            min_bbox_area: Minimum bounding box area
            max_aspect_ratio: Maximum bounding box aspect ratio
            
        Returns:
            Filtered supervision Dataset
        """
        filtered_images = {}
        filtered_annotations = {}
        
        for image_name, image in dataset.images.items():
            detections = dataset.annotations.get(image_name, Detections.empty())
            
            if len(detections) == 0:
                # No annotations, skip
                continue
            
            keep_indices = []
            
            for i in range(len(detections)):
                # Check bounding box quality
                bbox = detections.xyxy[i]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                aspect_ratio = max(width, height) / max(min(width, height), 1e-6)
                
                if area < min_bbox_area or aspect_ratio > max_aspect_ratio:
                    continue
                
                # Check keypoint quality
                if hasattr(detections, 'data') and 'keypoints' in detections.data:
                    keypoints_list = detections.data['keypoints']
                    if i < len(keypoints_list):
                        keypoints = keypoints_list[i]
                        if isinstance(keypoints, np.ndarray) and len(keypoints) > 0:
                            visible_count = np.sum(keypoints[:, 2] > 0)
                            if visible_count < min_keypoints:
                                continue
                
                keep_indices.append(i)
            
            # Keep only quality annotations
            if keep_indices:
                filtered_detections = detections[keep_indices]
                filtered_images[image_name] = image
                filtered_annotations[image_name] = filtered_detections
        
        filtered_dataset = Dataset(filtered_images, filtered_annotations)
        self.logger.info(f"Filtered dataset: {len(dataset.images)} -> {len(filtered_images)} images")
        
        return filtered_dataset
    
    def create_dataset_from_pipeline_results(self, pipeline_results: Dict[str, Any],
                                           images_dir: Union[str, Path],
                                           keypoint_names: Optional[List[str]] = None) :
        """
        Create supervision Dataset from CV pipeline results
        
        Args:
            pipeline_results: Pipeline results dictionary
            images_dir: Directory containing original images
            keypoint_names: List of keypoint names
            
        Returns:
            supervision Dataset object
        """
        if not keypoint_names:
            keypoint_names = [
                "Hips", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Spine",
                "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
            ]
        
        images_path = Path(images_dir)
        dataset_images = {}
        dataset_annotations = {}
        
        # Extract pose annotations from pipeline results
        pose_annotations = pipeline_results.get('pose_annotations', {})
        rectified_annotations = pipeline_results.get('rectified_annotations', {})
        
        # Use pose annotations if available, otherwise rectified annotations
        source_annotations = pose_annotations if pose_annotations else rectified_annotations
        
        for frame_id, cameras in source_annotations.items():
            for camera_id, data in cameras.items():
                # Find corresponding image file
                image_files = list(images_path.glob(f"*{camera_id}*{frame_id}*"))
                if not image_files:
                    # Try different patterns
                    image_files = list(images_path.glob(f"*frame_{frame_id:04d}*"))
                
                if not image_files:
                    continue
                
                image_file = image_files[0]
                image_name = image_file.name
                
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                dataset_images[image_name] = image
                
                # Extract annotations
                annotations = data.get('annotations', data) if isinstance(data, dict) else data
                
                if not annotations:
                    dataset_annotations[image_name] = Detections.empty()
                    continue
                
                # Convert annotations to supervision format
                detections_data = self._convert_annotations_to_detections(
                    annotations, keypoint_names
                )
                
                dataset_annotations[image_name] = detections_data
        
        dataset = Dataset(dataset_images, dataset_annotations)
        self.logger.info(f"Created dataset from pipeline results with {len(dataset_images)} images")
        
        return dataset
    
    def _convert_annotations_to_detections(self, annotations: Any,
                                         keypoint_names: List[str]) :
        """
        Convert various annotation formats to supervision Detections
        
        Args:
            annotations: Annotation data in various formats
            keypoint_names: List of keypoint names
            
        Returns:
            supervision Detections object
        """
        if isinstance(annotations, dict) and 'annotations' in annotations:
            # COCO-style annotations
            ann_list = annotations['annotations']
            
            if not ann_list:
                return Detections.empty()
            
            boxes = []
            keypoints_list = []
            confidences = []
            
            for ann in ann_list:
                # Extract bounding box
                bbox = ann.get('bbox', [0, 0, 100, 100])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
                else:
                    boxes.append([0, 0, 100, 100])
                
                # Extract keypoints
                kpts = ann.get('keypoints', [])
                if kpts:
                    kpts_array = np.array(kpts).reshape(-1, 3)
                    keypoints_list.append(kpts_array)
                else:
                    # Create empty keypoints
                    empty_kpts = np.zeros((len(keypoint_names), 3))
                    keypoints_list.append(empty_kpts)
                
                confidences.append(0.9)  # Default confidence
            
            # Create Detections
            detections = Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences)
            )
            detections.data = {'keypoints': keypoints_list}
            
            return detections
        
        else:
            # Other formats - return empty for now
            return Detections.empty()
    
    def export_dataset_for_training(self, dataset,
                                  output_dir: Union[str, Path],
                                  format_type: str = 'coco',
                                  keypoint_names: Optional[List[str]] = None) -> None:
        """
        Export dataset in format suitable for training
        
        Args:
            dataset: supervision Dataset
            output_dir: Output directory
            format_type: Export format ('coco', 'yolo', 'json')
            keypoint_names: List of keypoint names
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == 'coco':
            # Export as COCO format
            self.supervision_to_coco_format(
                dataset, 
                output_path / 'annotations.json',
                keypoint_names
            )
            
            # Copy images
            images_dir = output_path / 'images'
            images_dir.mkdir(exist_ok=True)
            
            for image_name, image in dataset.images.items():
                cv2.imwrite(str(images_dir / image_name), image)
                
        elif format_type.lower() == 'json':
            # Export as custom JSON format
            self._export_as_json(dataset, output_path / 'dataset.json', keypoint_names)
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        self.logger.info(f"Exported dataset in {format_type} format to {output_path}")
    
    def _export_as_json(self, dataset, output_path: Path,
                       keypoint_names: Optional[List[str]]) -> None:
        """Export dataset as custom JSON format"""
        export_data = {
            'keypoint_names': keypoint_names or [],
            'total_images': len(dataset.images),
            'annotations': {}
        }
        
        for image_name, image in dataset.images.items():
            detections = dataset.annotations.get(image_name, Detections.empty())
            
            image_data = {
                'image_name': image_name,
                'image_shape': image.shape,
                'detections': []
            }
            
            for i in range(len(detections)):
                detection_data = {
                    'bbox': detections.xyxy[i].tolist(),
                    'confidence': float(detections.confidence[i])
                }
                
                # Add keypoints if available
                if hasattr(detections, 'data') and 'keypoints' in detections.data:
                    keypoints_list = detections.data['keypoints']
                    if i < len(keypoints_list):
                        keypoints = keypoints_list[i]
                        if isinstance(keypoints, np.ndarray):
                            detection_data['keypoints'] = keypoints.tolist()
                
                image_data['detections'].append(detection_data)
            
            export_data['annotations'][image_name] = image_data
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)


class SupervisionDataSource(DataSource):
    """Data source wrapper for supervision datasets"""
    
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.camera_ids = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract camera IDs from image names
        for image_name in dataset.images.keys():
            camera_id = extract_camera_id_from_filename(image_name)
            if camera_id and camera_id not in self.camera_ids:
                self.camera_ids.append(camera_id)
        
        if not self.camera_ids:
            self.camera_ids = ["1"]  # Default camera ID
    
    def get_type(self):
        return "supervision_dataset"
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        self.logger.info(f"Initialized supervision data source with {len(self.dataset)} images")
        return True
    
    def get_frames(self) -> Iterator[FrameData]:
        """Get frames from supervision dataset"""
        for image_name, image in self.dataset.images.items():
            detections = self.dataset.annotations.get(image_name, Detections.empty())
            
            # Parse frame and camera IDs
            frame_id = extract_frame_id_from_filename(image_name) or 0
            camera_id = extract_camera_id_from_filename(image_name) or "1"
            
            # Convert detections to annotations format
            annotations = self._detections_to_annotations(detections)
            
            yield FrameData(
                frame=image,
                frame_id=frame_id,
                camera_id=camera_id,
                annotations=annotations,
                metadata={'image_name': image_name}
            )
    
    def _detections_to_annotations(self, detections) -> Dict[str, Any]:
        """Convert supervision detections to annotation format"""
        if len(detections) == 0:
            return {'annotations': []}
        
        annotations = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            x, y, x2, y2 = bbox
            
            annotation = {
                'bbox': [float(x), float(y), float(x2 - x), float(y2 - y)],
                'area': float((x2 - x) * (y2 - y)),
                'category_id': 1
            }
            
            # Add keypoints if available
            if hasattr(detections, 'data') and 'keypoints' in detections.data:
                keypoints_list = detections.data['keypoints']
                if i < len(keypoints_list):
                    keypoints = keypoints_list[i]
                    if isinstance(keypoints, np.ndarray) and len(keypoints) > 0:
                        # Convert to flat format
                        keypoints_flat = []
                        for kp in keypoints:
                            keypoints_flat.extend([float(kp[0]), float(kp[1]), int(kp[2])])
                        annotation['keypoints'] = keypoints_flat
                        annotation['num_keypoints'] = len(keypoints)
            
            annotations.append(annotation)
        
        return {'annotations': annotations}
    
    def has_annotations(self) -> bool:
        return True
    
    def get_camera_ids(self) -> List[str]:
        return self.camera_ids
    
    def cleanup(self) -> None:
        pass
