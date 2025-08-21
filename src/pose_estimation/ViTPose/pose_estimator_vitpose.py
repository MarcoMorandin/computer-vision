"""ViTPose-based pose estimation module."""

from logging import Logger
import os
import cv2
import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation
from pathlib import Path
from utils.skeleton.pose_plotter_2d import SkeletonDrawer

from ...utils.dataset.coco_utils import COCOManager
from ...utils.keypoints import (
    calculate_virtual_keypoints,
    create_keypoint_mapping,
)
from ...utils.geometry import keypoints_to_flat_array


class ViTPoseEstimator:
    """ViTPose-based pose estimation for updating COCO dataset annotations."""

    def __init__(
        self,
        coco_manager: COCOManager,
        config: Optional[Any] = None,
        logger: Logger = None,
    ):
        """Initialize the ViTPoseEstimator.

        Args:
            coco_manager: Initialized COCOManager object
            config: Configuration object with ViTPose settings
            detector_weights_path: Path to YOLO model weights for person detection (overrides config)
            vit_model_name: HuggingFace ViTPose model repository name (overrides config)
            prune_patterns: Keypoint patterns to prune (overrides config)
        """
        self.coco_manager = coco_manager
        self.config = config
        self.logger = logger

        self.device = config.models.device
        detector_path = config.models.vit.detection_model_path
        model_name = config.models.vit.model_name
        
        # Initialize models
        self.detector = YOLO(os.path.abspath(detector_path))
        self.detector.to(self.device)
        self.detector.set_classes(["person"])

        
        # Configure processor settings
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.vit_model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self.device)
        self.vit_model.eval()
        
        
        # Determine prune patterns
        patterns = config.models.keypoints.prune_patterns    
        self.kept_keypoint_names = self.coco_manager.prune_keypoints(patterns)
        
        # Create keypoint mapping
        self.vit_to_custom_mapping = create_keypoint_mapping(self.kept_keypoint_names)

    def run_pose_estimation(self, confidence_threshold: Optional[float] = None) -> COCOManager:
        """Run pose estimation on images and update COCO dataset.

        Args:
            confidence_threshold: Minimum confidence threshold for detections (uses config if None)

        Returns:
            Updated COCOManager object with new pose annotations
        """
        # Determine confidence threshold
        confidence_threshold = self.config.models.vit.confidence_threshold
        images = self.coco_manager.get_images()
        
        # Clear all existing annotations
        self.coco_manager.clear_annotations()
        
        # Get person category for annotations
        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]
        
        for img in tqdm(images, desc="Running ViTPose Estimation", unit="image"):
            img_id = img["id"]
            file_name = img["file_name"]
            
           
            image = cv2.imread(file_name)
            
            # Run pose estimation
            self._process_frame(image, img_id, category_id, confidence_threshold)
        
        return self.coco_manager

    def run_pose_estimation_on_video(
        self,
        video_path: str,
        output_path: str,
    ) -> COCOManager:
        """Run pose estimation on video and create COCO dataset with predictions.

        Args:
            video_path: Path to input video
            output_path: Path to save output video with predictions
            confidence_threshold: Minimum confidence threshold for detections (uses config if None)

        Returns:
            Updated COCOManager with pose predictions
        """
        # Determine confidence threshold
        confidence_threshold = self.config.models.vit.confidence_threshold
        
        # Open video
        drawer = SkeletonDrawer(self.coco_manager)
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get person category for annotations
        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]
        
        # Clear existing data
        self.coco_manager.clear_annotations()
        self.coco_manager.clear_images()
        
        self.logger.info(f"Processing video: {video_path}")
        
        for frame_idx in tqdm(range(total_frames), desc="Running ViTPose Estimation", unit="frame"):
            _, frame = cap.read()

            filename = f"{Path(video_path).stem}_frame_{frame_idx:04d}.rf.jpg"

            # Add image to COCO dataset
            img_id = self.coco_manager.add_image(
                file_name=filename,
                height=frame.shape[0],
                width=frame.shape[1],
            )
            
            # Process frame and get keypoints
            keypoints = self._process_frame(frame, img_id, category_id, confidence_threshold)
            
            # Draw skeleton if keypoints detected
            if keypoints is not None:
                frame = drawer.draw_skeleton_on_image(frame, keypoints)
            
            # Write frame to output video
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save COCO annotations
        self.coco_manager.save(str(output_path).replace(".mp4", ".json"))

        return self.coco_manager

    def _process_frame(self, frame, img_id: int, category_id: int, confidence_threshold: float) -> Optional[List[float]]:
        """Process a single frame for pose estimation.

        Args:
            frame: Input frame in BGR format
            img_id: Image ID for annotation
            category_id: Person category ID
            confidence_threshold: Detection confidence threshold

        Returns:
            First person's keypoints as flat array, or None if no detections
        """
        # Detect persons with YOLO with configuration options
        det_results = self.detector.predict(frame, conf=confidence_threshold, verbose=False, device=self.device)
        det = det_results[0]
        
        if det.boxes is None or det.boxes.xyxy.shape[0] == 0:
            return None
        
        # Keep only person class (class id 0)
        boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
        classes = det.boxes.cls.detach().cpu().numpy().astype(int)
        person_mask = classes == 0
        boxes_xyxy = boxes_xyxy[person_mask]
        
        if boxes_xyxy.shape[0] == 0:
            return None
        
        # Convert to COCO xywh format
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]  # width
        boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]  # height
        
        # Prepare input for ViTPose (RGB/PIL)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        # Run ViTPose inference
        inputs = self.processor(pil_img, boxes=[boxes_xywh], return_tensors="pt").to(self.device)
        dataset_index = torch.tensor([0], device=self.device)
        
        with torch.no_grad():
            outputs = self.vit_model(**inputs, dataset_index=dataset_index)
        
        pose_results_list = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])
        pose_results = pose_results_list[0] if len(pose_results_list) > 0 else []
        
        if not pose_results:
            return None
        
        first_person_keypoints = None
        
        # Process each detected person
        for person_idx, pose_result in enumerate(pose_results):
            kpts_xy = pose_result['keypoints']
            kpts_conf = pose_result['scores']
            
            # Convert tensors to numpy if needed
            if torch.is_tensor(kpts_xy):
                kpts_xy = kpts_xy.detach().cpu().numpy()
            if torch.is_tensor(kpts_conf):
                kpts_conf = kpts_conf.detach().cpu().numpy()
            
            # Initialize custom keypoint dictionary
            custom_kpts = {name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in self.kept_keypoint_names}
            
            # Fill in available ViTPose keypoints
            for custom_name, vit_idx in self.vit_to_custom_mapping.items():
                if vit_idx >= 0:  # Not a virtual keypoint
                    custom_kpts[custom_name] = {
                        "x": float(kpts_xy[vit_idx, 0]),
                        "y": float(kpts_xy[vit_idx, 1]),
                        "confidence": float(kpts_conf[vit_idx]),
                    }
            
            # Calculate virtual keypoints
            calculate_virtual_keypoints(custom_kpts)
            
            # Convert to COCO format
            keypoints_flat = keypoints_to_flat_array(custom_kpts, self.kept_keypoint_names)
            
            # Get bounding box for this person
            x1, y1, x2, y2 = boxes_xyxy[person_idx]
            bbox = [
                float(x1),
                float(y1),
                float(x2 - x1),  # width
                float(y2 - y1),  # height
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
            
            # Store first person's keypoints for drawing
            if first_person_keypoints is None:
                first_person_keypoints = keypoints_flat
        
        return first_person_keypoints