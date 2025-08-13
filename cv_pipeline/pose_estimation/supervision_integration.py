"""Integration layer for supervision w        # Initialize supervision components
        self.skeleton_visualizer = SupervisionSkeletonVisualizer(
            keypoint_names=self.keypoint_names
        )
        self.pose_annotator = PoseAnnotator(
            keypoint_names=self.keypoint_names
        )
        
        # Initialize supervision annotators if available
        if HAS_SUPERVISION and sv is not None:
            self.box_annotator = sv.BoxAnnotator(
                color=sv.Color.GREEN,
                thickness=2
            )
            
            self.label_annotator = sv.LabelAnnotator(
                color=sv.Color.GREEN,
                text_thickness=1,
                text_scale=0.5
            )
        else:
            self.box_annotator = None
            self.label_annotator = Nones"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import logging

try:
    import supervision as sv
    from supervision import Detections
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    sv = None
    # Create dummy class for when supervision is not available
    class DummyDetections:
        def __init__(self, *args, **kwargs):
            pass
    
    Detections = DummyDetections

from ..visualization import SupervisionSkeletonVisualizer, PoseAnnotator
from ..utils.helpers import calculate_bbox_from_keypoints


class SupervisionPoseEstimatorWrapper:
    """Wrapper to add supervision capabilities to pose estimators"""
    
    def __init__(self, pose_estimator, keypoint_names: Optional[List[str]] = None):
        self.logger = logging.getLogger(__name__)
        
        if not HAS_SUPERVISION:
            self.logger.warning("Supervision library not available. Some features may be limited.")
        
        self.pose_estimator = pose_estimator
        self.keypoint_names = keypoint_names or self._get_default_keypoint_names()
        
        # Initialize supervision components
        self.skeleton_visualizer = SupervisionSkeletonVisualizer(
            keypoint_names=self.keypoint_names
        )
        self.pose_annotator = PoseAnnotator(
            keypoint_names=self.keypoint_names
        )
        
        self.box_annotator = sv.BoxAnnotator(
            color=sv.Color.GREEN,
            thickness=2
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.GREEN,
            text_thickness=2,
            text_scale=0.8
        )
    
    def _get_default_keypoint_names(self) -> List[str]:
        """Get default keypoint names"""
        return [
            "Hips", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Spine",
            "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
        ]
    
    def estimate_poses_with_supervision(self, image: np.ndarray, 
                                      visualize: bool = False,
                                      save_path: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Estimate poses and return both original format and supervision format
        
        Args:
            image: Input image
            visualize: Whether to create visualization
            save_path: Optional path to save visualization
            
        Returns:
            Tuple of (original_annotations, visualization_image)
        """
        # Get original pose estimation results
        original_results = self.pose_estimator.estimate_poses(image)
        
        # Convert to supervision format
        detections = self.annotations_to_supervision_detections(original_results, image.shape[:2])
        
        visualization = None
        if visualize:
            # Create enhanced visualization
            visualization = self.pose_annotator.annotate_pose_detections(
                image, detections,
                show_boxes=True,
                show_labels=True,
                show_keypoints=True,
                show_confidence=True
            )
            
            if save_path:
                cv2.imwrite(str(save_path), visualization)
        
        return original_results, visualization
    
    def annotations_to_supervision_detections(self, annotations: Dict[str, Any], 
                                            image_shape: Tuple[int, int]):
        """
        Convert pose estimation annotations to supervision Detections
        
        Args:
            annotations: Original annotations from pose estimator
            image_shape: (height, width) of image
            
        Returns:
            supervision Detections object
        """
        if not annotations or 'annotations' not in annotations:
            return Detections.empty()
        
        ann_list = annotations['annotations']
        if not ann_list:
            return Detections.empty()
        
        boxes = []
        keypoints_list = []
        confidences = []
        
        for ann in ann_list:
            # Extract bounding box
            bbox = ann.get('bbox')
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                boxes.append([x, y, x + w, y + h])
            else:
                # Calculate bbox from keypoints if available
                kpts = ann.get('keypoints', [])
                if kpts:
                    bbox = calculate_bbox_from_keypoints(kpts)
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
                else:
                    boxes.append([0, 0, 100, 100])  # Default box
            
            # Extract keypoints
            kpts = ann.get('keypoints', [])
            if kpts:
                # Convert from flat format to (N, 3) format
                kpts_array = np.array(kpts).reshape(-1, 3)
                keypoints_list.append(kpts_array)
            else:
                # Create empty keypoints
                empty_kpts = np.zeros((len(self.keypoint_names), 3))
                keypoints_list.append(empty_kpts)
            
            # Extract confidence (use average keypoint confidence)
            if kpts:
                confidences_kpts = kpts[2::3]  # Every 3rd element starting from index 2
                avg_confidence = np.mean([c for c in confidences_kpts if c > 0]) if confidences_kpts else 0.5
                confidences.append(avg_confidence)
            else:
                confidences.append(0.5)  # Default confidence
        
        # Create Detections object
        detections = Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences)
        )
        
        # Add keypoints as additional data
        detections.data = {'keypoints': keypoints_list}
        
        return detections
    
    def create_pose_comparison(self, image1: np.ndarray, image2: np.ndarray,
                             title1: str = "Original", title2: str = "Processed") -> np.ndarray:
        """
        Create side-by-side pose comparison
        
        Args:
            image1: First image
            image2: Second image
            title1: Title for first image
            title2: Title for second image
            
        Returns:
            Combined comparison image
        """
        # Estimate poses for both images
        results1, viz1 = self.estimate_poses_with_supervision(image1, visualize=True)
        results2, viz2 = self.estimate_poses_with_supervision(image2, visualize=True)
        
        if viz1 is None:
            viz1 = image1.copy()
        if viz2 is None:
            viz2 = image2.copy()
        
        # Add titles
        cv2.putText(viz1, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(viz2, title2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ensure same height
        h1, w1 = viz1.shape[:2]
        h2, w2 = viz2.shape[:2]
        
        if h1 != h2:
            if h1 > h2:
                viz2 = cv2.resize(viz2, (int(w2 * h1 / h2), h1))
            else:
                viz1 = cv2.resize(viz1, (int(w1 * h2 / h1), h2))
        
        # Combine horizontally
        combined = np.hstack([viz1, viz2])
        return combined
    
    def analyze_pose_quality_batch(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze pose quality for a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            Dictionary with quality analysis results
        """
        quality_results = {
            'total_images': len(images),
            'images_with_poses': 0,
            'total_poses': 0,
            'quality_metrics': [],
            'summary_stats': {}
        }
        
        all_visibility_ratios = []
        all_completeness_scores = []
        all_symmetry_scores = []
        
        for i, image in enumerate(images):
            results, _ = self.estimate_poses_with_supervision(image, visualize=False)
            
            if results and 'annotations' in results:
                quality_results['images_with_poses'] += 1
                
                for ann in results['annotations']:
                    kpts = ann.get('keypoints', [])
                    if kpts:
                        quality_results['total_poses'] += 1
                        
                        # Convert to (N, 3) format for analysis
                        kpts_array = np.array(kpts).reshape(-1, 3)
                        
                        # Analyze quality
                        pose_quality = self.pose_annotator.analyze_pose_quality(kpts_array)
                        pose_quality['image_index'] = i
                        quality_results['quality_metrics'].append(pose_quality)
                        
                        # Collect statistics
                        all_visibility_ratios.append(pose_quality.get('visibility_ratio', 0))
                        all_completeness_scores.append(pose_quality.get('completeness_score', 0))
                        all_symmetry_scores.append(pose_quality.get('symmetry_score', 0))
        
        # Calculate summary statistics
        if all_visibility_ratios:
            quality_results['summary_stats'] = {
                'avg_visibility_ratio': np.mean(all_visibility_ratios),
                'avg_completeness_score': np.mean(all_completeness_scores),
                'avg_symmetry_score': np.mean(all_symmetry_scores),
                'min_visibility_ratio': np.min(all_visibility_ratios),
                'max_visibility_ratio': np.max(all_visibility_ratios),
                'detection_rate': quality_results['images_with_poses'] / quality_results['total_images']
            }
        
        return quality_results
    
    def create_temporal_analysis(self, image_sequence: List[np.ndarray],
                               output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Create temporal analysis of pose sequence
        
        Args:
            image_sequence: List of images in temporal order
            output_path: Optional path to save result
            
        Returns:
            Grid visualization of pose sequence
        """
        keypoint_sequences = []
        
        # Extract keypoints from each frame
        for image in image_sequence:
            results, _ = self.estimate_poses_with_supervision(image, visualize=False)
            
            if results and 'annotations' in results and results['annotations']:
                # Use first person's keypoints
                ann = results['annotations'][0]
                kpts = ann.get('keypoints', [])
                if kpts:
                    kpts_array = np.array(kpts).reshape(-1, 3)
                    keypoint_sequences.append(kpts_array)
                else:
                    # Create empty keypoints
                    empty_kpts = np.zeros((len(self.keypoint_names), 3))
                    keypoint_sequences.append(empty_kpts)
            else:
                # No pose detected
                empty_kpts = np.zeros((len(self.keypoint_names), 3))
                keypoint_sequences.append(empty_kpts)
        
        # Create temporal visualization
        result = self.pose_annotator.create_temporal_pose_sequence(
            image_sequence, keypoint_sequences
        )
        
        if output_path:
            cv2.imwrite(str(output_path), result)
        
        return result
    
    def export_supervision_dataset(self, images: List[np.ndarray], 
                                 output_dir: Union[str, Path],
                                 image_names: Optional[List[str]] = None) -> None:
        """
        Export images and pose annotations as supervision dataset
        
        Args:
            images: List of images
            output_dir: Output directory
            image_names: Optional list of image names
        """
        from ..utils.supervision_utils import SupervisionDatasetUtils
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not image_names:
            image_names = [f"image_{i:06d}.jpg" for i in range(len(images))]
        
        # Process images and create dataset
        dataset_images = {}
        dataset_annotations = {}
        
        for i, (image, img_name) in enumerate(zip(images, image_names)):
            # Save image
            img_path = output_path / "images" / img_name
            img_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(img_path), image)
            
            # Get pose annotations
            results, _ = self.estimate_poses_with_supervision(image, visualize=False)
            detections = self.annotations_to_supervision_detections(results, image.shape[:2])
            
            dataset_images[img_name] = image
            dataset_annotations[img_name] = detections
        
        # Create supervision dataset
        dataset = sv.Dataset(dataset_images, dataset_annotations)
        
        # Export as COCO format
        utils = SupervisionDatasetUtils()
        utils.supervision_to_coco_format(
            dataset,
            output_path / "annotations.json",
            self.keypoint_names
        )
        
        print(f"Exported supervision dataset to {output_path}")
        print(f"Images: {len(dataset_images)}")
        print(f"Annotations: {sum(len(det) for det in dataset_annotations.values())}")


def enhance_pose_estimator_with_supervision(pose_estimator, 
                                          keypoint_names: Optional[List[str]] = None) -> SupervisionPoseEstimatorWrapper:
    """
    Factory function to enhance any pose estimator with supervision capabilities
    
    Args:
        pose_estimator: Original pose estimator object
        keypoint_names: Optional list of keypoint names
        
    Returns:
        Enhanced pose estimator with supervision capabilities
    """
    return SupervisionPoseEstimatorWrapper(pose_estimator, keypoint_names)
