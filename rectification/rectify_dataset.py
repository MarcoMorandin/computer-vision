import cv2
import numpy as np
import json
import os
import re
import copy

def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def rectify_keypoints(keypoints, mtx, dist):
    num_kpts = len(keypoints) // 3
    new_keypoints = keypoints.copy()
    
    pts_list = []
    indices = []
    for i in range(num_kpts):
        x = keypoints[3 * i]
        y = keypoints[3 * i + 1]
        v = keypoints[3 * i + 2]
        if v > 0:
            pts_list.append([[x, y]])
            indices.append(i)
    
    if not pts_list:
        return new_keypoints
    
    pts = np.array(pts_list, dtype=np.float32)  # Shape (N, 1, 2)
    undist = cv2.undistortPoints(pts, mtx, dist, P=mtx)  # Shape (N, 1, 2)
    
    for j, i in enumerate(indices):
        newx = undist[j, 0, 0]
        newy = undist[j, 0, 1]
        new_keypoints[3 * i] = float(newx)
        new_keypoints[3 * i + 1] = float(newy)
    
    return new_keypoints

def update_bbox(keypoints):
    num_kpts = len(keypoints) // 3
    xs = []
    ys = []
    for i in range(num_kpts):
        v = keypoints[3 * i + 2]
        if v > 0:
            xs.append(keypoints[3 * i])
            ys.append(keypoints[3 * i + 1])
    
    if not xs:
        return None
    
    minx = min(xs)
    miny = min(ys)
    maxx = max(xs)
    maxy = max(ys)
    w = maxx - minx
    h = maxy - miny
    return [minx, miny, w, h]

def get_cam_index(file_name):
    match = re.search(r'(?:out|cam)(\d+)', file_name)
    if match:
        return match.group(1)
    return None

def rectify_image(image_path, output_path, mtx, dist):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return False
    
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, img.shape[:2][::-1], cv2.CV_32FC1)

    rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    cv2.imwrite(output_path, rectified_img)
    return True

def main(input_json, output_json, calib_base_dir, input_images_dir, output_images_dir):
    with open(input_json, 'r') as f:
        coco = json.load(f)
    
    coco_rect = copy.deepcopy(coco)
    processed_images = set()
    
    for img in coco_rect["images"]:
        file_name = img["file_name"]
        cam_index = get_cam_index(file_name)
        if cam_index is None:
            print(f"Cannot extract camera index from file_name: {file_name}")
            continue
        
        calib_path = os.path.join(calib_base_dir, f"cam_{cam_index}", "calib", "camera_calib.json")
        if not os.path.exists(calib_path):
            print(f"Calibration file not found: {calib_path}")
            continue
        
        mtx, dist = load_calibration(calib_path)
        
        if file_name not in processed_images:
            input_image_path = os.path.join(input_images_dir, file_name)
            output_image_path = os.path.join(output_images_dir, file_name)
            
            if os.path.exists(input_image_path):
                if rectify_image(input_image_path, output_image_path, mtx, dist):
                    print(f"Rectified image: {file_name}")
                    processed_images.add(file_name)
                else:
                    print(f"Failed to rectify image: {file_name}")
            else:
                print(f"Input image not found: {input_image_path}")
        
        image_id = img["id"]
        anns = [ann for ann in coco_rect["annotations"] if ann["image_id"] == image_id]
        
        for ann in anns:
            if "keypoints" in ann:                
                ann["keypoints"] = rectify_keypoints(ann["keypoints"], mtx, dist)
                new_bbox = update_bbox(ann["keypoints"])
                if new_bbox:
                    ann["bbox"] = new_bbox
                    if "area" in ann:
                        ann["area"] = new_bbox[2] * new_bbox[3]
    
    with open(output_json, 'w') as f:
        json.dump(coco_rect, f)
    

if __name__ == "__main__":
    input_json = os.path.join("..", "data", "dataset", "train", "_annotations.coco.json")
    output_json = os.path.join("output", "dataset", "train", "_annotations.coco.json")
    
    input_images_dir = os.path.join("..", "data", "dataset", "train")
    output_images_dir = os.path.join("output", "dataset", "train")
    
    params_version = "v2"
    calib_base_dir = os.path.join("..", "data", f"camera_data_{params_version}")
    
    if not os.path.exists(os.path.dirname(output_json)):
        os.makedirs(os.path.dirname(output_json))
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    main(input_json, output_json, calib_base_dir, input_images_dir, output_images_dir)