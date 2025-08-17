import os
import cv2
import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', ".."))
from utils.dataset.coco_utils import COCOManager


class ViTPoseEstimator:
    """
    A class for running ViT Pose estimation using YOLO as the person detector and
    updating COCO dataset annotations via COCOManager.
    """

    def __init__(
        self,
        coco_manager: COCOManager,
        detector_yolo_weights_path: str = 'yolo11l.pt',
        vit_model_name: str = 'usyd-community/vitpose-plus-base',
    ):
        """
        Initialize the ViTPoseEstimator.

        Args:
            coco_manager: Initialized COCOManager object.
            detector_yolo_model_name: YOLO model name/path for person detection (detection-only).
            vit_model_name: HuggingFace ViT Pose model repo name.
        """
        self.coco_manager = coco_manager

        # Models
        self.detector = YOLO(os.path.abspath((detector_yolo_weights_path)))
        self.detector.to("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(vit_model_name, use_fast=False)
        self.vit_model = VitPoseForPoseEstimation.from_pretrained(vit_model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.vit_model.eval()

        # COCO-17 keypoint order used by ViTPose (lowercase)
        self.vit_coco_keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # Prune foot/toe keypoints in COCO via the manager
        self.kept_keypoint_names = self.coco_manager.prune_keypoints(["foot", "toe"])

        # Create mapping from your custom keypoints to ViTPose indices
        self.custom_to_vit_mapping = self._create_keypoint_mapping()

    def _create_keypoint_mapping(self) -> Dict[str, int]:
        """
        Create mapping from your custom keypoint names to ViTPose (COCO-17) indices.
        Virtual keypoints (Hips, Neck, Spine) get -1 to be computed later.
        """
        # ViTPose indices
        idx = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }

        # Map the most common custom names to ViTPose COCO indices
        mapping = {
            "LShoulder": idx["left_shoulder"],
            "RShoulder": idx["right_shoulder"],
            "LElbow": idx["left_elbow"],
            "RElbow": idx["right_elbow"],
            # Prefer hand naming used elsewhere
            "LHand": idx["left_wrist"],
            "RHand": idx["right_wrist"],
            "LHip": idx["left_hip"],
            "RHip": idx["right_hip"],
            "LKnee": idx["left_knee"],
            "RKnee": idx["right_knee"],
            "LAnkle": idx["left_ankle"],
            "RAnkle": idx["right_ankle"],
            "Head": idx["nose"],
            # Optional facial (only keep if present in dataset schema)
            "Nose": idx["nose"],
            "LEye": idx["left_eye"],
            "REye": idx["right_eye"],
            "LEar": idx["left_ear"],
            "REar": idx["right_ear"],
            "Hips": -1,
            "Neck": -1,
            "Spine": -1
        }

        # Only keep what's in custom_kept set
        custom_mapping = {}
        for custom_name in self.kept_keypoint_names:
            if custom_name in mapping:
                custom_mapping[custom_name] = mapping[custom_name]
            else:
                # Not in ViTPose or not relevant -> default to -2 (no source)
                custom_mapping[custom_name] = -2

        return custom_mapping

    def _calculate_virtual_keypoints(self, custom_kpts: Dict[str, Dict[str, float]]) -> None:
        """Compute virtual keypoints Hips, Neck, Spine if their dependencies exist with > 0 confidence."""
        # Hips midpoint
        if "Hips" in custom_kpts and "LHip" in custom_kpts and "RHip" in custom_kpts:
            l_hip, r_hip = custom_kpts["LHip"], custom_kpts["RHip"]
            if l_hip["confidence"] > 0 and r_hip["confidence"] > 0:
                custom_kpts["Hips"] = {
                    "x": (l_hip["x"] + r_hip["x"]) / 2,
                    "y": (l_hip["y"] + r_hip["y"]) / 2,
                    "confidence": min(l_hip["confidence"], r_hip["confidence"])
                }

        # Neck midpoint
        if "Neck" in custom_kpts and "LShoulder" in custom_kpts and "RShoulder" in custom_kpts:
            l_sh, r_sh = custom_kpts["LShoulder"], custom_kpts["RShoulder"]
            if l_sh["confidence"] > 0 and r_sh["confidence"] > 0:
                custom_kpts["Neck"] = {
                    "x": (l_sh["x"] + r_sh["x"]) / 2,
                    "y": (l_sh["y"] + r_sh["y"]) / 2,
                    "confidence": min(l_sh["confidence"], r_sh["confidence"])
                }

        # Spine midpoint
        if "Spine" in custom_kpts and "Hips" in custom_kpts and "Neck" in custom_kpts:
            hips, neck = custom_kpts["Hips"], custom_kpts["Neck"]
            if hips["confidence"] > 0 and neck["confidence"] > 0:
                custom_kpts["Spine"] = {
                    "x": (hips["x"] + neck["x"]) / 2,
                    "y": (hips["y"] + neck["y"]) / 2,
                    "confidence": min(hips["confidence"], neck["confidence"])
                }

    def _keypoints_to_flat_array(self, custom_kpts: Dict[str, Dict[str, float]]) -> List[float]:
        """Convert dictionary of keypoints to flat [x,y,v,...] array for COCO format."""
        keypoints_flat = []
        for name in self.kept_keypoint_names:
            kpt = custom_kpts[name]
            x, y, conf = kpt['x'], kpt['y'], kpt['confidence']
            visibility = 2 if conf > 0.1 else 0
            keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
        return keypoints_flat

    def run_pose_estimation(self, detection_conf: float = 0.5) -> COCOManager:
        """
        Run ViTPose pose estimation on all images and update the COCO dataset.

        Args:
            detection_conf: Minimum confidence threshold for YOLO person detections.

        Returns:
            Updated COCOManager object with new pose annotations.
        """
        images = self.coco_manager.get_images()
        self.coco_manager.clear_annotations()

        person_category = self.coco_manager.get_person_category()
        category_id = person_category["id"]

        for img in tqdm(images, desc="Running ViT Pose Estimation", unit="image"):
            img_id = img["id"]
            file_name = img["file_name"]

            if not os.path.exists(file_name):
                print(f"  Warning: Image not found at {file_name}")
                continue

            image_bgr = cv2.imread(file_name)
            if image_bgr is None:
                print(f"  Warning: Could not load image {file_name}")
                continue

            # Detect persons with YOLO (detection-only)
            det_results = self.detector(image_bgr, conf=detection_conf, imgsz=1280, verbose=False)
            det = det_results[0]
            if det.boxes is None or det.boxes.xyxy.shape[0] == 0:
                continue

            # Keep only person class (class id 0)
            boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
            classes = det.boxes.cls.detach().cpu().numpy().astype(int)
            person_mask = classes == 0
            boxes_xyxy = boxes_xyxy[person_mask]
            if boxes_xyxy.shape[0] == 0:
                continue

            # Convert to COCO xywh
            boxes_xywh = boxes_xyxy.copy()
            boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
            boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]

            # Prepare input for ViTPose (RGB/PIL)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)

            inputs = self.processor(pil_img, boxes=[boxes_xywh], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            dataset_index = torch.tensor([0], device="cuda" if torch.cuda.is_available() else "cpu")  # 0 = COCO for ViTPose

            with torch.no_grad():
                outputs = self.vit_model(**inputs, dataset_index=dataset_index)
            pose_results_list = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])
            pose_results = pose_results_list[0] if len(pose_results_list) > 0 else []

            # Add annotations for each detected person
            for person_idx, pose_result in enumerate(pose_results):
                kpts_xy = pose_result['keypoints']
                kpts_conf = pose_result['scores']
                if torch.is_tensor(kpts_xy):
                    kpts_xy = kpts_xy.detach().cpu().numpy()
                if torch.is_tensor(kpts_conf):
                    kpts_conf = kpts_conf.detach().cpu().numpy()

                # Initialize all keypoints with zero values
                custom_kpts = {
                    name: {"x": 0.0, "y": 0.0, "confidence": 0.0}
                    for name in self.kept_keypoint_names
                }

                # Fill from ViTPose predictions where mapping exists
                for custom_name, vit_idx in self.custom_to_vit_mapping.items():
                    if custom_name not in custom_kpts:
                        continue
                    if vit_idx >= 0:
                        custom_kpts[custom_name] = {
                            "x": float(kpts_xy[vit_idx, 0]),
                            "y": float(kpts_xy[vit_idx, 1]),
                            "confidence": float(kpts_conf[vit_idx])
                        }
                    # vit_idx == -1 => virtual, computed later
                    # vit_idx == -2 => no source; stays zero

                # Compute virtuals (Hips, Neck, Spine)
                self._calculate_virtual_keypoints(custom_kpts)

                # Flatten to COCO keypoints array
                keypoints_flat = self._keypoints_to_flat_array(custom_kpts)

                # Detection bbox for this person (use matching index after masking)
                x1, y1, x2, y2 = boxes_xyxy[person_idx]
                w = x2 - x1
                h = y2 - y1
                bbox = [float(x1), float(y1), float(w), float(h)]

                # Add annotation
                self.coco_manager.add_annotation(
                    image_id=img_id,
                    category_id=category_id,
                    keypoints=keypoints_flat,
                    bbox=bbox,
                    area=float(w * h),
                    num_keypoints=sum(1 for v in keypoints_flat[2::3] if v > 0)
                )

        return self.coco_manager