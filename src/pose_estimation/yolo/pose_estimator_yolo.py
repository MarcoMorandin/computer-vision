import os
import cv2
from typing import Dict, List, Any
from tqdm import tqdm
from ultralytics import YOLO, settings
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.dataset.coco_utils import COCOManager


class YOLOPoseEstimator:
    """
    A class for running YOLO pose estimation and updating COCO dataset annotations.
    """

    def __init__(
        self,
        coco_manager: COCOManager,
        model_weights_path: str = "yolo11l-pose.pt",
    ):
        """
        Initialize the YOLOPoseEstimator.

        Args:
            coco_manager: Initialized COCOManager object
            model_weights_path: str = "yolo11l-pose.pt",
        """
        self.coco_manager = coco_manager
        self.model = YOLO(os.path.abspath((model_weights_path)))

        # YOLO COCO-17 keypoint names (0-based indexing)
        self.yolo_keypoint_names = [
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

        self.kept_keypoint_names = self.coco_manager.prune_keypoints(["foot", "toe"])

        self.yolo_to_custom_mapping = self._create_keypoint_mapping()

    def _create_keypoint_mapping(self) -> Dict[str, int]:
        """Create mapping from your custom keypoint names to YOLO indices."""
        mapping = {}

        # Map your custom keypoint names to YOLO indices
        keypoint_mapping = {
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
            # Add mappings for custom keypoints to YOLO equivalents
            "Head": 0,  # Map Head to Nose (closest equivalent)
            "RHand": 10,  # Map RHand to RWrist
            "LHand": 9,  # Map LHand to LWrist
        }

        # Only include keypoints that exist in your custom set
        for custom_name in self.kept_keypoint_names:
            if custom_name in keypoint_mapping:
                mapping[custom_name] = keypoint_mapping[custom_name]
            elif custom_name in ["Hips", "Neck", "Spine"]:
                # Virtual keypoints - will be calculated later
                mapping[custom_name] = -1  # Special marker for virtual keypoints

        return mapping

    def _calculate_virtual_keypoints(self, custom_kpts: Dict[str, Dict]) -> None:
        """Calculate virtual keypoints like Hips, Neck, Spine."""
        # Hips (midpoint of left and right hip)
        if "Hips" in custom_kpts and "LHip" in custom_kpts and "RHip" in custom_kpts:
            l_hip, r_hip = custom_kpts["LHip"], custom_kpts["RHip"]
            if l_hip["confidence"] > 0 and r_hip["confidence"] > 0:
                custom_kpts["Hips"] = {
                    "x": (l_hip["x"] + r_hip["x"]) / 2,
                    "y": (l_hip["y"] + r_hip["y"]) / 2,
                    "confidence": min(l_hip["confidence"], r_hip["confidence"]),
                }

        # Neck (midpoint of left and right shoulder)
        if (
            "Neck" in custom_kpts
            and "LShoulder" in custom_kpts
            and "RShoulder" in custom_kpts
        ):
            l_shoulder, r_shoulder = custom_kpts["LShoulder"], custom_kpts["RShoulder"]
            if l_shoulder["confidence"] > 0 and r_shoulder["confidence"] > 0:
                custom_kpts["Neck"] = {
                    "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
                    "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
                    "confidence": min(
                        l_shoulder["confidence"], r_shoulder["confidence"]
                    ),
                }

        # Spine (midpoint of hips and neck)
        if "Spine" in custom_kpts and "Hips" in custom_kpts and "Neck" in custom_kpts:
            hips, neck = custom_kpts["Hips"], custom_kpts["Neck"]
            if hips["confidence"] > 0 and neck["confidence"] > 0:
                custom_kpts["Spine"] = {
                    "x": (hips["x"] + neck["x"]) / 2,
                    "y": (hips["y"] + neck["y"]) / 2,
                    "confidence": min(hips["confidence"], neck["confidence"]),
                }

    def _keypoints_to_flat_array(
        self, custom_kpts: Dict[str, Dict], custom_keypoint_names: List[str]
    ) -> List[float]:
        """Convert dictionary of keypoints to flat array for COCO format."""
        keypoints_flat = []
        for name in custom_keypoint_names:
            kpt = custom_kpts[name]
            x, y, conf = kpt["x"], kpt["y"], kpt["confidence"]
            visibility = 2 if conf > 0.1 else 0
            keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
        return keypoints_flat

    def run_pose_estimation(self, confidence_threshold: float = 0.25) -> COCOManager:
        """
        Run pose estimation on images and update COCO dataset.

        Args:
            input_dir: Directory containing input images
            confidence_threshold: Minimum confidence threshold for detections

        Returns:
            Updated COCOManager object with new pose annotations
        """
        images = self.coco_manager.get_images()
        processed_count = 0

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

            processed_count += 1

            # Run pose estimation
            results = self.model(
                image, conf=confidence_threshold, verbose=False
            )

            # Process detections
            self._process_detections(results[0], img_id, category_id)

        return self.coco_manager

    def run_pose_estimation_on_video(
        self,
        video_path: str,
        output_path: str,
        confidence_threshold: float = 0.25,
        draw_predictions: bool = True,
    ) -> None:
        """
        Run pose estimation on video and save output video with predictions.

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            confidence_threshold: Minimum confidence threshold for detections
            draw_predictions: Whether to draw keypoints and skeleton on video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

            # Run pose estimation
            results = self.model(
                frame, conf=confidence_threshold, imgsz=1280, verbose=False
            )

            # TODO use a skeleton drawer
            # Draw predictions if requested
            if draw_predictions:
                frame = self._process_detections_for_video(results[0], frame)

            # Write frame to output video
            out.write(frame)

        # Cleanup
        cap.release()
        out.release()

        print(f"Video processing complete. Output saved to: {output_path}")

    def _process_detections(self, result, img_id: int, category_id: int) -> None:
        """Process YOLO detections and add annotations."""
        if result.keypoints is None or result.boxes is None:
            return

        keypoints_tensor = result.keypoints
        boxes_tensor = result.boxes

        # Process each detected person
        for person_idx in range(keypoints_tensor.shape[0]):
            kpts_xy = keypoints_tensor.xy[person_idx].cpu().numpy()
            kpts_conf = keypoints_tensor.conf[person_idx].cpu().numpy()

            # Initialize all keypoints with zero values first
            coco_kpts_person = {}
            for custom_name in self.kept_keypoint_names:
                coco_kpts_person[custom_name] = {"x": 0.0, "y": 0.0, "confidence": 0.0}

            # Convert YOLO keypoints to your custom format using correct mapping
            for custom_name in self.kept_keypoint_names:
                if custom_name in self.yolo_to_custom_mapping:
                    yolo_idx = self.yolo_to_custom_mapping[custom_name]

                    if yolo_idx >= 0:  # Real keypoint from YOLO
                        coco_kpts_person[custom_name] = {
                            "x": float(kpts_xy[yolo_idx, 0]),
                            "y": float(kpts_xy[yolo_idx, 1]),
                            "confidence": float(kpts_conf[yolo_idx]),
                        }

            # Calculate virtual keypoints
            self._calculate_virtual_keypoints(coco_kpts_person)
            keypoints_flat = self._keypoints_to_flat_array(
                coco_kpts_person, self.kept_keypoint_names
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
