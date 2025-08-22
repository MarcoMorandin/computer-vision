"""ProbPose-based pose estimation pipeline (MMPose) with YOLO detections.

This script remaps COCO keypoints to the custom skeleton (with virtual joints),
then runs a ProbPose model to predict keypoints and writes a COCO JSON.
"""

from __future__ import annotations

import os
import json
import copy
import argparse
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# MMPose (ProbPose) APIs
from mmengine.config import Config
from mmpose.apis.inference import init_model, inference_topdown

device = "cuda" if torch.cuda.is_available() else "cpu"

#!! TEST SCRIPT WORKS BUT I CANNOT BE ABLE TO RE-PERFORM THE INSTALLATION (DONE ONE MONTH AGO, NOW THERE ARE DEPENDENCIES CONFLICTS)

# --------------------------- Mapping & Utilities ---------------------------


def remap_to_custom_skeleton(
    coco_kpts: Dict[str, Dict[str, float]], custom_keypoint_names: List[str]
):
    """Remap COCO keypoints to a custom skeleton and add virtual joints.

    Parameters
    ----------
    coco_kpts : dict[str, dict[str, float]]
        COCO keypoints map (e.g., nose, left_hip, ...).
    custom_keypoint_names : list[str]
        Target keypoint order for the custom skeleton.

    Returns
    -------
    dict[str, dict[str, float]]
        Keypoints in the custom schema, including Hips/Neck/Spine when available.
    """
    custom_kpts = {
        name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in custom_keypoint_names
    }

    def get_kpt(name):
        return coco_kpts.get(name, {"x": 0.0, "y": 0.0, "confidence": 0.0})

    # Direct mappings (subset of COCO 17)
    custom_kpts["RHip"] = get_kpt("right_hip")
    custom_kpts["RKnee"] = get_kpt("right_knee")
    custom_kpts["RAnkle"] = get_kpt("right_ankle")
    custom_kpts["LHip"] = get_kpt("left_hip")
    custom_kpts["LKnee"] = get_kpt("left_knee")
    custom_kpts["LAnkle"] = get_kpt("left_ankle")
    custom_kpts["RShoulder"] = get_kpt("right_shoulder")
    custom_kpts["RElbow"] = get_kpt("right_elbow")
    custom_kpts["LShoulder"] = get_kpt("left_shoulder")
    custom_kpts["LElbow"] = get_kpt("left_elbow")
    custom_kpts["Head"] = get_kpt("nose")
    custom_kpts["RHand"] = get_kpt("right_wrist")
    custom_kpts["LHand"] = get_kpt("left_wrist")

    # Virtual joints
    l_hip, r_hip = custom_kpts["LHip"], custom_kpts["RHip"]
    if l_hip["confidence"] > 0 and r_hip["confidence"] > 0:
        custom_kpts["Hips"] = {
            "x": (l_hip["x"] + r_hip["x"]) / 2,
            "y": (l_hip["y"] + r_hip["y"]) / 2,
            "confidence": min(l_hip["confidence"], r_hip["confidence"]),
        }

    l_shoulder, r_shoulder = custom_kpts["LShoulder"], custom_kpts["RShoulder"]
    if l_shoulder["confidence"] > 0 and r_shoulder["confidence"] > 0:
        custom_kpts["Neck"] = {
            "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
            "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
            "confidence": min(l_shoulder["confidence"], r_shoulder["confidence"]),
        }

    hips = custom_kpts.get("Hips", {"confidence": 0})
    neck = custom_kpts.get("Neck", {"confidence": 0})
    if hips.get("confidence", 0) > 0 and neck.get("confidence", 0) > 0:
        custom_kpts["Spine"] = {
            "x": (hips["x"] + neck["x"]) / 2,
            "y": (hips["y"] + neck["y"]) / 2,
            "confidence": min(hips["confidence"], neck["confidence"]),
        }

    return custom_kpts


def keypoints_to_flat_array(
    custom_kpts: Dict[str, Dict[str, float]], custom_keypoint_names: List[str]
):
    """Convert a custom keypoint dict to a flat COCO keypoint array.

    Parameters
    ----------
    custom_kpts : dict[str, dict[str, float]]
        Mapping from keypoint name to {x, y, confidence}.
    custom_keypoint_names : list[str]
        Ordered list of keypoints in the target schema.

    Returns
    -------
    list[float]
        Flattened keypoints in COCO format [x1, y1, v1, ...].
    """
    keypoints_flat = []
    for name in custom_keypoint_names:
        kpt = custom_kpts[name]
        x, y, conf = kpt["x"], kpt["y"], kpt["confidence"]
        visibility = 2 if conf > 0.1 else 0
        keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
    return keypoints_flat


def prune_category_feet(category: Dict):
    """Remove foot/toe keypoints (if any) and rebuild skeleton indices in-place.

    For standard COCO (17 keypoints) this often leaves the list unchanged, but
    the logic is kept for consistency with other scripts.

    Parameters
    ----------
    category : dict
        COCO person category definition to modify in-place.

    Returns
    -------
    list[str]
        Kept keypoint names.
    """
    keypoints = category.get("keypoints", [])

    def is_foot(name):
        n = name.lower()
        return ("foot" in n) or ("toe" in n)

    kept = [n for n in keypoints if not is_foot(n)]
    old_to_new = {}
    for idx, name in enumerate(keypoints, start=1):
        if name in kept:
            old_to_new[idx] = kept.index(name) + 1
        else:
            old_to_new[idx] = None
    if "skeleton" in category:
        new_skel = []
        for a, b in category["skeleton"]:
            na, nb = old_to_new.get(a), old_to_new.get(b)
            if na is not None and nb is not None:
                new_skel.append([na, nb])
        category["skeleton"] = new_skel
    category["keypoints"] = kept
    return kept


# --------------------------- Core Pipeline ---------------------------


def run_pose_estimation_probpose(
    input_dir: str,
    input_json: str,
    output_json: str,
    config_path: str,
    checkpoint_path: str,
    detector_model_path: str = "yolo11l.pt",
    detection_conf: float = 0.50,
):
    """Run ProbPose (MMPose) pose estimation on a COCO image set.

    Parameters
    ----------
    input_dir : str
        Directory containing images referenced by the input JSON.
    input_json : str
        Input COCO JSON path.
    output_json : str
        Output COCO JSON path to write predictions.
    config_path : str
        MMPose/ProbPose config file path.
    checkpoint_path : str
        MMPose/ProbPose checkpoint (.pth) path.
    detector_model_path : str, default="yolo11l.pt"
        YOLO weights for person detection.
    detection_conf : float, default=0.5
        YOLO person detection confidence threshold.
    """
    # Load COCO json and deep copy
    with open(input_json, "r") as f:
        coco_data = json.load(f)
    coco_pose = copy.deepcopy(coco_data)

    # Locate person category
    person_category = None
    for cat in coco_pose.get("categories", []):
        if cat.get("name") == "person":
            person_category = cat
            break
    if person_category is None:
        raise ValueError("No 'person' category found in COCO json.")

    kept_keypoint_names = prune_category_feet(person_category)

    # Standard COCO 17 keypoint ordering used by ProbPose / COCO meta
    coco_keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    # ---------------- Models ----------------
    # YOLO detector
    detector = YOLO(detector_model_path)
    try:
        detector.to(device)
    except Exception:
        pass

    # ProbPose (MMPose) model
    cfg = Config.fromfile(config_path)
    # Ensure no training-specific settings cause issues; rely on API for init
    model = init_model(cfg, checkpoint=checkpoint_path, device=device)

    processed = 0
    for img in coco_pose["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            print(f"[WARN] Missing image: {file_path}")
            continue

        # Remove existing annotations for this image
        coco_pose["annotations"] = [
            a for a in coco_pose["annotations"] if a["image_id"] != img_id
        ]

        # Person detection (YOLO) on PIL image
        pil_img = Image.open(file_path).convert("RGB")
        with torch.no_grad():
            det_results = detector(pil_img, conf=detection_conf, verbose=False)

        det = det_results[0]
        if det.boxes is None or det.boxes.xyxy.shape[0] == 0:
            print(f"No persons detected in {file_name}")
            processed += 1
            continue

        boxes_xyxy = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int)
        person_mask = classes == 0
        boxes_xyxy = boxes_xyxy[person_mask]
        if boxes_xyxy.shape[0] == 0:
            print(f"No person class boxes in {file_name}")
            processed += 1
            continue

        # Convert to COCO (x, y, w, h)
        person_boxes = boxes_xyxy.copy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

        # Pose estimation for all boxes at once
        # inference_topdown expects xyxy by default
        pose_samples = inference_topdown(
            model, file_path, bboxes=boxes_xyxy, bbox_format="xyxy"
        )

        # Build annotations
        max_ann_id = max([a["id"] for a in coco_pose["annotations"]], default=0)
        for person_idx, sample in enumerate(pose_samples):
            pred = sample.pred_instances
            if not hasattr(pred, "keypoints"):
                continue

            # keypoints: (1, K, 2)
            kpts_xy_raw = pred.keypoints
            # scores may be absent depending on test_cfg
            kpts_scores_raw = getattr(pred, "keypoint_scores", None)

            # Convert to numpy
            if torch.is_tensor(kpts_xy_raw):
                kpts_xy_np = kpts_xy_raw[0].detach().cpu().numpy()
            else:
                kpts_xy_np = kpts_xy_raw[0]

            if kpts_scores_raw is None:
                kpts_conf_np = np.zeros(kpts_xy_np.shape[0], dtype=np.float32)
            else:
                if torch.is_tensor(kpts_scores_raw):
                    kpts_conf_np = kpts_scores_raw[0].detach().cpu().numpy()
                else:
                    kpts_conf_np = kpts_scores_raw[0]

            coco_kpts_person = {}
            for i, name in enumerate(coco_keypoint_names):
                coco_kpts_person[name] = {
                    "x": float(kpts_xy_np[i, 0]),
                    "y": float(kpts_xy_np[i, 1]),
                    "confidence": float(kpts_conf_np[i]),
                }

            custom_keypoints = remap_to_custom_skeleton(
                coco_kpts_person, kept_keypoint_names
            )
            keypoints_flat = keypoints_to_flat_array(
                custom_keypoints, kept_keypoint_names
            )

            det_box_xyxy = boxes_xyxy[person_idx]
            x1, y1, x2, y2 = det_box_xyxy
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox = [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(bbox_w), 2),
                round(float(bbox_h), 2),
            ]

            max_ann_id += 1
            coco_pose["annotations"].append(
                {
                    "id": max_ann_id,
                    "image_id": img_id,
                    "category_id": person_category.get("id", 1),
                    "bbox": bbox,
                    "area": float(bbox_w * bbox_h),
                    "segmentation": [],
                    "iscrowd": 0,
                    "keypoints": keypoints_flat,
                    "num_keypoints": sum(1 for v in keypoints_flat[2::3] if v > 0),
                }
            )

        processed += 1
        if processed % 25 == 0:
            print(f"Processed {processed} images...")

    print(f"Finished. Total images processed: {processed}")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_pose, f, indent=4)


def parse_args():
    """Parse command-line arguments for the ProbPose script."""
    p = argparse.ArgumentParser(description="Dataset-wide ProbPose pose estimation")
    p.add_argument(
        "--input-image-dir",
        required=True,
        help="Directory containing images referenced in input COCO json",
    )
    p.add_argument(
        "--input-json",
        required=True,
        help="Input COCO dataset json (will be copied/pruned and filled with predictions)",
    )
    p.add_argument(
        "--output-json", required=True, help="Output COCO json path for predictions"
    )
    p.add_argument("--config", required=True, help="ProbPose config file path")
    p.add_argument(
        "--checkpoint", required=True, help="ProbPose checkpoint (.pth) path"
    )
    p.add_argument(
        "--detector-model",
        default="yolo11l.pt",
        help="YOLO weights for person detection",
    )
    p.add_argument(
        "--det-conf",
        type=float,
        default=0.5,
        help="YOLO person detection confidence threshold",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pose_estimation_probpose(
        input_dir=args.input_image_dir,
        input_json=args.input_json,
        output_json=args.output_json,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        detector_model_path=args.detector_model,
        detection_conf=args.det_conf,
    )
