import os
import cv2
import json
import copy
from ultralytics import YOLO

def remap_to_custom_skeleton(coco_kpts, custom_keypoint_names):
    """
    Remaps COCO keypoints to a custom skeleton format. This version is more robust
    and checks for the existence of keys before using them.
    """
    custom_kpts = {name: {"x": 0.0, "y": 0.0, "confidence": 0.0} for name in custom_keypoint_names}

    # Helper to safely get a keypoint, returning a zero-confidence dict if not found
    def get_kpt(name):
        return coco_kpts.get(name, {"x": 0.0, "y": 0.0, "confidence": 0.0})

    # Direct and proxy mappings
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

    # --- Calculate Virtual Keypoints ---
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

    hips, neck = custom_kpts["Hips"], custom_kpts["Neck"]
    if hips["confidence"] > 0 and neck["confidence"] > 0:
        custom_kpts["Spine"] = {
            "x": (hips["x"] + neck["x"]) / 2,
            "y": (hips["y"] + neck["y"]) / 2,
            "confidence": min(hips["confidence"], neck["confidence"])
        }

    return custom_kpts

def keypoints_to_flat_array(custom_kpts, custom_keypoint_names):
    """
    Converts dictionary of keypoints to a flat array for COCO format
    """
    keypoints_flat = []
    for name in custom_keypoint_names:
        kpt = custom_kpts[name]
        x, y, conf = kpt['x'], kpt['y'], kpt['confidence']
        visibility = 2 if conf > 0.1 else 0  # Use a small threshold for visibility
        keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
    return keypoints_flat

def update_bbox(keypoints_flat):
    """
    Recalculates the bounding box based on the keypoints
    """
    num_kpts = len(keypoints_flat) // 3
    xs = []
    ys = []
    for i in range(num_kpts):
        v = keypoints_flat[3 * i + 2]
        if v > 0:
            xs.append(keypoints_flat[3 * i])
            ys.append(keypoints_flat[3 * i + 1])
    
    if not xs:
        return None
    
    minx = min(xs)
    miny = min(ys)
    maxx = max(xs)
    maxy = max(ys)
    w = maxx - minx
    h = maxy - miny
    return [minx, miny, w, h]

def run_pose_estimation(input_dir, input_json, output_json, model_name='yolov11x-pose.pt', confidence_threshold=0.25):
    """
    Runs pose estimation on images and updates an existing COCO dataset with new keypoints.
    Similar structure to rectify_dataset.py
    """
    # Load input COCO data
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
        
    coco_pose = copy.deepcopy(coco_data)
    
    # Extract existing skeleton configuration from COCO data and prune feet/toes
    custom_keypoint_names = None
    person_category = None
    for category in coco_pose["categories"]:
        if category["name"] == "person":
            person_category = category
            custom_keypoint_names = category.get("keypoints", [])
            break

    # Remove keypoints that refer to feet/toes (case-insensitive)
    def is_foot_kpt(name: str) -> bool:
        n = name.lower()
        return ("foot" in n) or ("toe" in n)

    kept_keypoint_names = [n for n in custom_keypoint_names if not is_foot_kpt(n)]

    # If present, remap skeleton to exclude removed joints while preserving order
    if person_category is not None:
        # Build old index (1-based) -> new index (1-based) map
        old_to_new = {}
        for idx, name in enumerate(custom_keypoint_names, start=1):
            if name in kept_keypoint_names:
                new_idx = kept_keypoint_names.index(name) + 1
                old_to_new[idx] = new_idx
            else:
                old_to_new[idx] = None

        if "skeleton" in person_category:
            new_skeleton = []
            for bone in person_category["skeleton"]:
                if len(bone) >= 2:
                    a, b = bone[0], bone[1]
                    na = old_to_new.get(a)
                    nb = old_to_new.get(b)
                    if na is not None and nb is not None:
                        new_skeleton.append([na, nb])
            person_category["skeleton"] = new_skeleton

        # Update keypoint names in the category to match pruned list
        person_category["keypoints"] = kept_keypoint_names
    
    # Load the YOLO model
    model = YOLO(model_name)
    
    # COCO keypoint names for the model
    coco_keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Process each image in the dataset
    processed_images = set()
    processed_count = 0
    
    for img in coco_pose["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        file_path = os.path.join(input_dir, file_name)
        
        if file_name in processed_images:
            continue
            
        print(f"Processing image {processed_count + 1}: {file_name}")
        image = cv2.imread(file_path)
        processed_images.add(file_name)
        processed_count += 1

        # Run pose estimation on the image
        results = model(image, conf=confidence_threshold, imgsz=1280, verbose=False)

        # Process each detected person
        keypoints_tensor = results[0].keypoints
        boxes_tensor = results[0].boxes
        
        # Clear existing annotations for this image
        coco_pose["annotations"] = [ann for ann in coco_pose["annotations"] if ann["image_id"] != img_id]
        
        # Add new annotations based on pose estimation
        for person_idx in range(keypoints_tensor.shape[0]):
            kpts_xy = keypoints_tensor.xy[person_idx].cpu().numpy()
            kpts_conf = keypoints_tensor.conf[person_idx].cpu().numpy()
            
            coco_kpts_person = {}
            for i, name in enumerate(coco_keypoint_names):
                coco_kpts_person[name] = {
                    "x": float(kpts_xy[i, 0]),
                    "y": float(kpts_xy[i, 1]),
                    "confidence": float(kpts_conf[i])
                }
            
            # Build keypoints only for kept keypoint names (feet/toes removed)
            custom_keypoints = remap_to_custom_skeleton(coco_kpts_person, kept_keypoint_names)
            keypoints_flat = keypoints_to_flat_array(custom_keypoints, kept_keypoint_names)
            
            # Convert bbox from [center_x, center_y, w, h] to [top_left_x, top_left_y, w, h]
            bbox_xywh = boxes_tensor.xywh[person_idx].cpu().numpy()
            top_left_x = float(bbox_xywh[0] - bbox_xywh[2] / 2)
            top_left_y = float(bbox_xywh[1] - bbox_xywh[3] / 2)
            bbox_w = float(bbox_xywh[2])
            bbox_h = float(bbox_xywh[3])
            bbox = [round(top_left_x, 2), round(top_left_y, 2), round(bbox_w, 2), round(bbox_h, 2)]

            # Find the highest annotation ID to ensure unique IDs
            max_ann_id = 0
            for ann in coco_pose["annotations"]:
                max_ann_id = max(max_ann_id, ann["id"])
            
            # Add new annotation
            coco_pose["annotations"].append({
                "id": max_ann_id + 1,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox_w * bbox_h,
                "segmentation": [],
                "iscrowd": 0,
                "keypoints": keypoints_flat,
                "num_keypoints": sum(1 for v in keypoints_flat[2::3] if v > 0)
            })

    # Save the updated COCO JSON file
    print(f"\nProcessed {processed_count} images.")
    with open(output_json, 'w') as f:
        json.dump(coco_pose, f, indent=4)
        


if __name__ == '__main__':
    input_image_directory = '../rectification/output/dataset/train'
    input_json_file = '../rectification/output/dataset/train/_annotations.coco.json'
    output_json_file = 'output/predictions_coco.json'
    
    output_dir = os.path.dirname(output_json_file)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    model_name = 'yolo11l-pose.pt'
    confidence_threshold = 0.25

    run_pose_estimation(input_image_directory, input_json_file, output_json_file, model_name, confidence_threshold)