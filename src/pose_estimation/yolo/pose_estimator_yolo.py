"""YOLO-based pose estimation module."""

import os
import cv2
from typing import List, Any, Optional
from pathlib import Path

from logging import Logger
from tqdm import tqdm
from ultralytics import YOLO

from utils.skeleton.pose_plotter_2d import SkeletonDrawer

from ...utils.dataset.coco_utils import COCOManager
from ...utils.keypoints import (
    calculate_virtual_keypoints,
    create_keypoint_mapping,
)
from ...utils.geometry import keypoints_to_flat_array


class YOLOPoseEstimator:
    """YOLO-based pose estimation for updating COCO dataset annotations."""

    def __init__(
        self,
        coco_manager: COCOManager,
        config: Optional[Any] = None,
        logger: Logger = None,
    ):
        """Initialize the YOLO pose estimator.

        Parameters
        ----------
        coco_manager : COCOManager
            Dataset manager to read/write annotations.
        config : Any | None
            Configuration with YOLO settings and thresholds.
        logger : logging.Logger | None
            Optional logger for messages.
        """
        self.coco_manager = coco_manager
        self.config = config
        self.logger = logger

        # Determine model weights path
        weights_path = config.models.yolo.pose_model_path
        self.model = YOLO(os.path.abspath(weights_path))

        # Configure device
        self.device = config.models.device

        # Determine prune patterns
        patterns = config.models.keypoints.prune_patterns
        self.kept_keypoint_names = self.coco_manager.prune_keypoints(patterns)

        # Create keypoint mapping
        self.yolo_to_custom_mapping = create_keypoint_mapping(self.kept_keypoint_names)

    def run_pose_estimation(self) -> COCOManager:
        """Run pose estimation on images and update the dataset.

        Returns
        -------
        COCOManager
            Updated dataset with pose annotations. Uses thresholds from config.
        """
        # Determine confidence threshold
        confidence_threshold = self.config.models.yolo.confidence_threshold
        images = self.coco_manager.get_images()

        # Clear all existing annotations
        self.coco_manager.clear_annotations()

        # Get person category for annotations
        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]

        for img in tqdm(images, desc="Running YOLO Pose Estimation", unit="image"):
            img_id = img["id"]
            file_name = img["file_name"]

            image = cv2.imread(file_name)

            results = self.model(
                image, conf=confidence_threshold, verbose=False, device=self.device
            )

            # Process detections
            self._process_detections(results[0], img_id, category_id)

        return self.coco_manager

    def run_pose_estimation_on_video(
        self,
        video_path: str,
        output_path: str,
    ) -> COCOManager:
        """Run pose estimation on a video and write an annotated output.

        Parameters
        ----------
        video_path : str
            Path to input video.
        output_path : str
            Path to save output video with predictions.

        Returns
        -------
        COCOManager
            Updated dataset with pose predictions for frames.
        """
        # Determine confidence threshold
        confidence_threshold = self.config.models.yolo.confidence_threshold

        # Open video
        drawer = SkeletonDrawer(self.coco_manager)
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get video codec from config
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get person category for annotations
        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]

        # Get all images from the video COCO dataset
        self.coco_manager.clear_annotations()
        self.coco_manager.clear_images()

        self.logger.info(f"Processing video: {video_path}")

        for frame_idx in tqdm(
            range(total_frames), desc="Running YoloPose Estimation", unit="frame"
        ):
            _, frame = cap.read()

            results = self.model(
                frame, confidence_threshold, verbose=False, device=self.device
            )

            filename = f"{Path(video_path).stem}_frame_{frame_idx:04d}.rf.jpg"

            img_id = self.coco_manager.add_image(
                file_name=filename,
                height=frame.shape[0],
                width=frame.shape[1],
            )

            # Process detections
            keypoint = self._process_detections(
                next(results), img_id=img_id, category_id=category_id
            )

            if keypoint is not None:
                frame = drawer.draw_skeleton_on_image(frame, keypoint)

            # Write frame to output video if drawing
            out.write(frame)

        # Cleanup
        cap.release()
        out.release()
        self.coco_manager.save(str(output_path).replace(".mp4", ".json"))

        return self.coco_manager

    def _process_detections(
        self, result, img_id: int = None, category_id: int = None
    ) -> List[float]:
        """Process YOLO detections and add annotations.

        Parameters
        ----------
        result : Any
            YOLO detection result object.
        img_id : int | None
            Image id for annotation.
        category_id : int | None
            Person category id.
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
            custom_kpts = {
                name: {"x": 0.0, "y": 0.0, "confidence": 0.0}
                for name in self.kept_keypoint_names
            }

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
            keypoints_flat = keypoints_to_flat_array(
                custom_kpts, self.kept_keypoint_names
            )

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
