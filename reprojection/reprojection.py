import os
import json
import numpy as np
import cv2
import re
from datetime import datetime
from tqdm import tqdm
import copy
from pycocotools.coco import COCO

def prepare_coco_dataset(original_coco_file):
    coco = COCO(original_coco_file)
    coco_dataset = copy.deepcopy(coco.dataset)
    
    image_lookup = {img['id']: img for img in coco_dataset['images']}
    annotation_lookup = {}
    for ann in coco_dataset['annotations']:
        annotation_lookup.setdefault(ann['image_id'], []).append(ann)
    
    return coco_dataset, image_lookup, annotation_lookup

def get_rectified_images(rectified_dataset_dir):
    rectified_images = {}

    for filename in os.listdir(rectified_dataset_dir):
        if filename.endswith('.jpg'):
            camera_id, frame_num = parse_image_filename(filename)
            rectified_images.setdefault(camera_id, {})[frame_num] = filename
    
    return rectified_images

def parse_image_filename(filename):
    pattern = r'out(\d+)_frame_(\d+)_png\.rf\.[a-f0-9]+\.jpg'
    
    match = re.match(pattern, filename)
    if match:
        return match.group(1), int(match.group(2))
    
    return None, None

def load_camera_calibrations(camera_data_dir):
    cameras = {}

    for cam_dir in sorted(os.listdir(camera_data_dir)):
        cam_num = cam_dir.split('_')[1]
        calib_file = os.path.join(camera_data_dir, cam_dir, 'calib', 'camera_calib.json')
        
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        
        cameras[cam_num] = {
            'mtx': np.array(calib["mtx"], dtype=np.float32),
            'dist': np.zeros((5, 1), dtype=np.float32), # Force dist to zero for rectified images
            'rvec': np.array(calib["rvecs"], dtype=np.float32).flatten(),
            'tvec': np.array(calib["tvecs"], dtype=np.float32).flatten()
        }

    return cameras

def project_3d_to_2d(points_3d, camera_params):
    if points_3d.size == 0:
        return np.array([])
    
    points_2d, _ = cv2.projectPoints(
        points_3d,
        camera_params['rvec'],
        camera_params['tvec'],
        camera_params['mtx'],
        camera_params['dist']
    )
    return points_2d.reshape(-1, 2)

def calculate_bbox_from_keypoints(keypoints_2d, visibility):
    visible_points = keypoints_2d[np.array(visibility) > 0]
    if len(visible_points) == 0:
        return [0, 0, 0, 0]

    x_min, y_min = visible_points.min(axis=0)
    x_max, y_max = visible_points.max(axis=0)
    
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    width = x_max - x_min + 2 * padding
    height = y_max - y_min + 2 * padding
    
    return [float(x_min), float(y_min), float(width), float(height)]

def process_single_frame(image_id, camera_id, frame_num, poses_3d, cameras, image_lookup):
    frame_str = str(frame_num)

    pose_3d = np.array(poses_3d[frame_str], dtype=np.float32)
    camera_params = cameras[camera_id]
    
    image = image_lookup[image_id]
    image_width, image_height = image["width"], image["height"]
    
    keypoints_2d = project_3d_to_2d(pose_3d, camera_params)
    
    visibility = [
        2 if 0 <= p[0] <= image_width and 0 <= p[1] <= image_height else 0 
        for p in keypoints_2d
    ]
    
    if not any(v > 0 for v in visibility):
        return None # No visible points

    coco_keypoints = []
    for kp, v in zip(keypoints_2d, visibility):
        coco_keypoints.extend([float(kp[0]), float(kp[1]), v])
    
    bbox = calculate_bbox_from_keypoints(keypoints_2d, visibility)
    
    return coco_keypoints, bbox

def reproject_all_frames(coco_dataset, poses_3d, cameras, rectified_images, annotation_lookup, image_lookup):
    
    image_mapping = {}
    for image in coco_dataset["images"]:
        camera_id, frame_num = parse_image_filename(image["file_name"])
        if camera_id and frame_num:
            if camera_id in rectified_images and frame_num in rectified_images[camera_id]:
                image_mapping[image["id"]] = (camera_id, frame_num)
    
    updated_images_count = 0
    updated_annotations_count = 0
    
    for image_id, (camera_id, frame_num) in tqdm(image_mapping.items(), desc="Reprojecting frames"):
        result = process_single_frame(image_id, camera_id, frame_num, poses_3d, cameras, image_lookup)
        
        if result:
            coco_keypoints, bbox = result
            
            if image_id in annotation_lookup and annotation_lookup[image_id]:
                for ann in annotation_lookup[image_id]:
                    ann["keypoints"] = coco_keypoints
                    ann["bbox"] = bbox
                    ann["area"] = bbox[2] * bbox[3]
                    updated_annotations_count += 1
            else:
                annotation_id = max([ann["id"] for ann in coco_dataset["annotations"]], default=0) + 1
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming 'person' is category_id 1
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                    "iscrowd": 0,
                    "keypoints": coco_keypoints,
                }
                coco_dataset["annotations"].append(annotation)
                annotation_lookup.setdefault(image_id, []).append(annotation)
                updated_annotations_count += 1
            
            updated_images_count += 1

    print("Reprojection complete.")
    
    return coco_dataset

def save_coco_dataset(coco_dataset, output_dir, output_filename="reprojected_annotations.json"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(coco_dataset, f, indent=4)

def main():
    triangulation_file = os.path.join("..", "triangulation", "output", "player_3d_poses.json")
    camera_data_dir = os.path.join("..", "data")
    rectified_dataset_dir = os.path.join("..", "rectification", "output", "dataset", "train")
    original_coco_file = os.path.join("..", "data", "dataset", "train", "_annotations.coco.json")
    output_dir = os.path.join("output")
    camera_version = "v2"
    output_filename = "reprojected_annotations.json"
    camera_data_dir = os.path.join(camera_data_dir, f"camera_data_{camera_version}")
    
    with open(triangulation_file, 'r') as f:
        poses_3d = json.load(f)
    
    cameras = load_camera_calibrations(camera_data_dir)
    rectified_images = get_rectified_images(rectified_dataset_dir)
    
    coco_dataset, image_lookup, annotation_lookup = prepare_coco_dataset(original_coco_file)
    
    coco_dataset = reproject_all_frames(coco_dataset, poses_3d, cameras, rectified_images, annotation_lookup, image_lookup)
    
    save_coco_dataset(coco_dataset, output_dir, output_filename)

if __name__ == "__main__":
    main()
