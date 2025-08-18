"""YOLO-based pose estimation module."""

import os
import cv2
from typing import Dict, List, Any, Optional
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO, settings

from utils.skeleton.pose_plotter_2d import SkeletonDrawer

from ...utils.dataset.coco_utils import COCOManager
from ...utils.keypoints import (
    calculate_virtual_keypoints,
    create_yolo_keypoint_mapping,
    get_yolo_keypoint_names,
    init_keypoints_dict
)
from ...utils.geometry import keypoints_to_flat_array


class YOLOPoseEstimator:
    """YOLO-based pose estimation for updating COCO dataset annotations."""

    def __init__(
        self,
        coco_manager: COCOManager,
        model_weights_path: str = "yolo11l-pose.pt",
        prune_patterns: Optional[List[str]] = None
    ):
        """Initialize the YOLOPoseEstimator.

        Args:
            coco_manager: Initialized COCOManager object
            model_weights_path: Path to YOLO pose model weights
            prune_patterns: Keypoint patterns to prune (default: ["foot", "toe"])
        """
        self.coco_manager = coco_manager
        self.model = YOLO(os.path.abspath(model_weights_path))
        
        # Get standard YOLO keypoint names
        self.yolo_keypoint_names = get_yolo_keypoint_names()
        
        # Prune unwanted keypoints
        prune_patterns = prune_patterns or ["foot", "toe"]
        self.kept_keypoint_names = self.coco_manager.prune_keypoints(prune_patterns)
        
        # Create keypoint mapping
        self.yolo_to_custom_mapping = create_yolo_keypoint_mapping(self.kept_keypoint_names)


    def run_pose_estimation(self, confidence_threshold: float = 0.25) -> COCOManager:
        """Run pose estimation on images and update COCO dataset.

        Args:
            confidence_threshold: Minimum confidence threshold for detections

        Returns:
            Updated COCOManager object with new pose annotations
        """
        images = self.coco_manager.get_images()

        # Clear all existing annotations
        self.coco_manager.clear_annotations()

        # Get person category for annotations
        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]

        for img in tqdm(images, desc="Running YOLO Pose Estimation", unit="image"):
            img_id = img["id"]
            file_name = img["file_name"]

            if not os.path.exists(file_name):
                print(f"  Warning: Image not found at {file_name}")
                continue

            image = cv2.imread(file_name)
            if image is None:
                print(f"  Warning: Could not load image {file_name}")
                continue

            # Run pose estimation
            results = self.model(image, conf=confidence_threshold, verbose=False)

            # Process detections
            self._process_detections(results[0], img_id, category_id)

        return self.coco_manager

    def run_pose_estimation_on_video(
        self,
        video_path: str,
        output_path: str,
        confidence_threshold: float = 0.25
    ) -> COCOManager:
        """
        Run pose estimation on video and create COCO dataset with predictions.

        Args:
            video_path: Path to input video
            output_video_path: Path to save output video with predictions
            video_coco_manager: COCOManager created from video frames
            confidence_threshold: Minimum confidence threshold for detections
            draw_predictions: Whether to draw keypoints and skeleton on video
            
        Returns:
            Updated COCOManager with pose predictions
        """
        
        # Open video
        drawer = SkeletonDrawer(self.coco_manager)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if drawing predictions
        out = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get person category for annotations
        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]
        
        # Get all images from the video COCO dataset
        self.coco_manager.clear_annotations()
        self.coco_manager.clear_images()

        frame_count = 0
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processing frame {frame_count}/{total_frames}")

            results = self.model(frame, conf=confidence_threshold, verbose=True)

            img_id = self.coco_manager.add_image(
                file_name=f"{Path(video_path).stem}_frame_{frame_count:04d}.rf.jpg",
                height=frame.shape[0],
                width=frame.shape[1],
            )

            # Process detections
            keypoint = self._process_detections(results[0], img_id=img_id, category_id=category_id)

            if keypoint is not None:
                frame = drawer.draw_skeleton_on_image(frame, keypoint)

            # Write frame to output video if drawing
            out.write(frame)

        # Cleanup
        cap.release()
        out.release()
        self.coco_manager.save(str(output_path).replace(".mp4", ".json"))

        return self.coco_manager
        
    def _process_detections(self, result, img_id: int = None, category_id: int = None) -> List[float]:
        """Process YOLO detections and add annotations.
        
        Args:
            result: YOLO detection result
            img_id: Image ID for annotation
            category_id: Person category ID
        """
        if result.keypoints is None or result.boxes is None:
            return

        keypoints_tensor = result.keypoints
        boxes_tensor = result.boxes

        # Process each detected person
        for person_idx in range(keypoints_tensor.shape[0]):
            kpts_xy = keypoints_tensor.xy[person_idx].cpu().numpy()
            kpts_conf = keypoints_tensor.conf[person_idx].cpu().numpy()

            # Initialize custom keypoint dictionary
            custom_kpts = init_keypoints_dict(self.kept_keypoint_names)

            # Fill in available YOLO keypoints
            for custom_name, yolo_idx in self.yolo_to_custom_mapping.items():
                if yolo_idx >= 0:  # Not a virtual keypoint
                    custom_kpts[custom_name] = {
                        "x": float(kpts_xy[yolo_idx, 0]),
                        "y": float(kpts_xy[yolo_idx, 1]),
                        "confidence": float(kpts_conf[yolo_idx]),
                    }

            # Calculate virtual keypoints
            calculate_virtual_keypoints(custom_kpts)

            # Convert to COCO format
            keypoints_flat = keypoints_to_flat_array(custom_kpts, self.kept_keypoint_names)

            # Convert bbox from [center_x, center_y, w, h] to [top_left_x, top_left_y, w, h]
            bbox_xywh = boxes_tensor.xywh[person_idx].cpu().numpy()
            bbox = [
                float(bbox_xywh[0] - bbox_xywh[2] / 2),  # top_left_x
                float(bbox_xywh[1] - bbox_xywh[3] / 2),  # top_left_y
                float(bbox_xywh[2]),  # width
                float(bbox_xywh[3]),  # height
            ]
        
            # Add annotation using COCOManager
            self.coco_manager.add_annotation(
                image_id=img_id,
                category_id=category_id,
                keypoints=keypoints_flat,
                bbox=bbox,
                area=bbox[2] * bbox[3],
                num_keypoints=sum(1 for v in keypoints_flat[2::3] if v > 0),
            )
            return keypoints_flat
