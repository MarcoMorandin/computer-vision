"""Dataset-wide ViT Pose estimation pipeline.

Replicates the structure of run_yolo_pose.py but uses:
  1. A YOLO (detection only) model to find person bounding boxes.
  2. A ViT Pose (HuggingFace transformers) model to estimate COCO keypoints inside those boxes.

It loads an input COCO json (e.g. the rectified dataset), prunes feet/toe keypoints,
rebuilds annotations for every image, and writes a new predictions COCO json.
"""

import os
import json
import copy
import torch
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------- Mapping & Utilities ---------------------------
def remap_to_custom_skeleton(coco_kpts, custom_keypoint_names):
    """Remap COCO keypoints to custom skeleton, add virtual joints (Hips, Neck, Spine).

    This mirrors the logic in run_yolo_pose.py. Missing joints default to zero-confidence.
    """
    custom_kpts = {name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in custom_keypoint_names}

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
            "confidence": min(l_hip["confidence"], r_hip["confidence"])
        }

    l_shoulder, r_shoulder = custom_kpts["LShoulder"], custom_kpts["RShoulder"]
    if l_shoulder["confidence"] > 0 and r_shoulder["confidence"] > 0:
        custom_kpts["Neck"] = {
            "x": (l_shoulder["x"] + r_shoulder["x"]) / 2,
            "y": (l_shoulder["y"] + r_shoulder["y"]) / 2,
            "confidence": min(l_shoulder["confidence"], r_shoulder["confidence"])
        }

    hips = custom_kpts.get("Hips", {"confidence": 0})
    neck = custom_kpts.get("Neck", {"confidence": 0})
    if hips.get("confidence", 0) > 0 and neck.get("confidence", 0) > 0:
        custom_kpts["Spine"] = {
            "x": (hips["x"] + neck["x"]) / 2,
            "y": (hips["y"] + neck["y"]) / 2,
            "confidence": min(hips["confidence"], neck["confidence"])
        }

    return custom_kpts


def keypoints_to_flat_array(custom_kpts, custom_keypoint_names):
    keypoints_flat = []
    for name in custom_keypoint_names:
        kpt = custom_kpts[name]
        x, y, conf = kpt['x'], kpt['y'], kpt['confidence']
        visibility = 2 if conf > 0.1 else 0
        keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
    return keypoints_flat


def prune_category_feet(category):
    """Remove foot/toe keypoints and rebuild skeleton indices in-place."""
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
def run_pose_estimation_vit(
    input_dir: str,
    input_json: str,
    output_json: str,
    detector_model_path: str = "yolo11l.pt",
    detection_conf: float = 0.50,
    vit_model_name: str = "usyd-community/vitpose-plus-huge",
):
    # Load COCO json and deep copy
    with open(input_json, 'r') as f:
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

    # COCO 17 keypoint ordering used by ViTPose
    coco_keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Load models
    detector = YOLO(detector_model_path)
    try:
        detector.to(device)
    except Exception:
        pass
    image_processor = AutoProcessor.from_pretrained(vit_model_name, use_fast=False)
    vit_model = VitPoseForPoseEstimation.from_pretrained(vit_model_name).to(device)

    processed = 0
    for img in coco_pose["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            print(f"[WARN] Missing image: {file_path}")
            continue

        # Remove existing annotations for this image
        coco_pose["annotations"] = [a for a in coco_pose["annotations"] if a["image_id"] != img_id]

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

        # Prepare inputs for ViTPose
        inputs = image_processor(pil_img, boxes=[person_boxes], return_tensors="pt").to(device)
        dataset_index = torch.tensor([0], device=device)
        with torch.no_grad():
            outputs = vit_model(**inputs, dataset_index=dataset_index)
        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])[0]

        # Iterate persons
        max_ann_id = max([a["id"] for a in coco_pose["annotations"]], default=0)
        for person_idx, pose_result in enumerate(pose_results):
            kpts_xy = pose_result['keypoints']  # Tensor shape (17, 2)
            kpts_conf = pose_result['scores']   # Tensor shape (17,)
            if torch.is_tensor(kpts_xy):
                kpts_xy = kpts_xy.cpu().numpy()
            if torch.is_tensor(kpts_conf):
                kpts_conf = kpts_conf.cpu().numpy()

            coco_kpts_person = {}
            for i, name in enumerate(coco_keypoint_names):
                coco_kpts_person[name] = {
                    "x": float(kpts_xy[i, 0]),
                    "y": float(kpts_xy[i, 1]),
                    "confidence": float(kpts_conf[i])
                }

            custom_keypoints = remap_to_custom_skeleton(coco_kpts_person, kept_keypoint_names)
            keypoints_flat = keypoints_to_flat_array(custom_keypoints, kept_keypoint_names)

            # Bounding box for this person (use detection box that corresponds order after masking)
            det_box_xyxy = boxes_xyxy[person_idx]  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = det_box_xyxy
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox = [round(float(x1), 2), round(float(y1), 2), round(float(bbox_w), 2), round(float(bbox_h), 2)]

            max_ann_id += 1
            coco_pose["annotations"].append({
                "id": max_ann_id,
                "image_id": img_id,
                "category_id": person_category.get("id", 1),
                "bbox": bbox,
                "area": float(bbox_w * bbox_h),
                "segmentation": [],
                "iscrowd": 0,
                "keypoints": keypoints_flat,
                "num_keypoints": sum(1 for v in keypoints_flat[2::3] if v > 0)
            })

        processed += 1
        if processed % 25 == 0:
            print(f"Processed {processed} images...")

    print(f"Finished. Total images processed: {processed}")
    with open(output_json, 'w') as f:
        json.dump(coco_pose, f, indent=4)


if __name__ == "__main__":
    input_image_directory = '../rectification/output/dataset/train'
    input_json_file = '../rectification/output/dataset/train/_annotations.coco.json'
    output_json_file = 'output/predictions_coco_vit.json'

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    run_pose_estimation_vit(
        input_dir=input_image_directory,
        input_json=input_json_file,
        output_json=output_json_file,
        detector_model_path='yolo11l.pt',
        detection_conf=0.60,
        vit_model_name='usyd-community/vitpose-plus-base'
    )