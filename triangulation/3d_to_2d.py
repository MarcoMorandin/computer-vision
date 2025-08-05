import json
import numpy as np
import cv2
import re
from pycocotools.coco import COCO
import os

# Constants (adjust as needed)
NUM_VIEWS = 6
NUM_JOINTS = 18  # COCO standard
SINGLE_COCO_FILE = '../rectification/rectified/train/_annotations.coco.json'  # Replace with actual path
JSON_3D_POSES = '3d_poses.json'  # Input from triangulation script
OUTPUT_FILE = 'reprojection_errors.json'  # Summary output
ERROR_THRESHOLD = 5.0  # Pixels; % of reprojections below this is "good"

def load_camera_parameters(camera_data_folder):
    """
    Load camera intrinsics and extrinsics from camera_data folder.
    
    Args:
        camera_data_folder (str): Path to camera_data folder
        use_real_calib (bool): If True, use camera_calib_real.json, else camera_calib.json
    
    Returns:
        tuple: (projection_matrices, intrinsics, dist_coeffs)
    """
    projection_matrices = []
    intrinsics = []
    dist_coeffs = []
    
    # Camera numbers in your data (1, 2, 3, 4, 5, 7)
    camera_nums = [1, 2, 3, 4, 5, 7]  # Adjust based on your actual cameras
    
    for i, cam_num in enumerate(camera_nums):
        calib_file = 'camera_calib_real.json'
        calib_path = os.path.join(camera_data_folder, f'cam_{cam_num}', 'calib', calib_file)
        
        if not os.path.exists(calib_path):
            print(f"Warning: Calibration file not found: {calib_path}")
            # Use identity matrices as fallback
            K = np.eye(3)
            dist = np.zeros((1, 5))
            R = np.eye(3)
            t = np.zeros((3, 1))
        else:
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            
            # Extract intrinsic matrix
            K = np.array(calib_data['mtx'])
            
            # Extract distortion coefficients
            dist = np.array(calib_data['dist'])
            
            # Extract rotation and translation vectors
            rvec = np.array(calib_data['rvecs']).flatten()
            tvec = np.array(calib_data['tvecs']).flatten()
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
        
        # Create extrinsic matrix [R|t]
        RT = np.hstack([R, t])
        
        # Create projection matrix P = K * [R|t]
        P = K @ RT
        
        projection_matrices.append(P)
        intrinsics.append(K)
        dist_coeffs.append(dist)
    
    return projection_matrices, intrinsics, dist_coeffs

CAMERA_DATA_FOLDER = '../rectification/data/camera_data'
PROJECTION_MATRICES, INTRINSICS, DIST_COEFFS = load_camera_parameters(CAMERA_DATA_FOLDER)

# Function to parse view and frame from filename (same as before)
def parse_view_and_frame(filename):
    match = re.search(r'out(\d+)_frame_(\d+)_png\.rf\.', filename)
    if match:
        view_num = int(match.group(1))  # 1 to 7
        frame_id = match.group(2)       # e.g., '0001'
        view_id = view_num - 1          # Map to 0-6
        if 0 <= view_id < NUM_VIEWS:
            return view_id, frame_id
    return None, None

# Precompute mapping: {frame_id: [img_id_view0, ..., img_id_view6]} (same as before)
def build_frame_view_mapping(coco):
    mapping = {}
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        view_id, frame_id = parse_view_and_frame(filename)
        if view_id is not None and frame_id is not None:
            if frame_id not in mapping:
                mapping[frame_id] = [None] * NUM_VIEWS
            mapping[frame_id][view_id] = img_id
    return mapping

# Load 2D keypoints for a frame (same as before)
def load_keypoints_for_frame(coco, img_ids_per_view):
    all_keypoints = np.zeros((NUM_VIEWS, NUM_JOINTS, 3))  # x, y, v
    for view_idx, img_id in enumerate(img_ids_per_view):
        if img_id is None:
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if anns:
            keypoints = np.array(anns[0]['keypoints']).reshape(NUM_JOINTS, 3)
            all_keypoints[view_idx] = keypoints
    return all_keypoints

# Decompose P into R and t (assuming P = K @ [R | t])
def decompose_projection_matrix(P, K):
    Rt = np.linalg.inv(K) @ P
    R = Rt[:, :3]
    t = Rt[:, 3]
    return R, t

# Compute reprojection error for one frame
def compute_frame_reprojection_error(pose_3d, keypoints_2d):
    errors = []  # List of per-reprojection errors
    for joint_idx in range(NUM_JOINTS):
        point_3d = pose_3d[joint_idx]
        if point_3d is None:
            continue  # Skip if not triangulated
        
        for view_idx in range(NUM_VIEWS):
            x_orig, y_orig, v = keypoints_2d[view_idx, joint_idx]
            if v < 2:
                continue  # Skip invisible/not labeled
            
            # Decompose P for this view
            P = PROJECTION_MATRICES[view_idx]
            K = INTRINSICS[view_idx]
            dist = DIST_COEFFS[view_idx]
            R, t = decompose_projection_matrix(P, K)
            
            # Convert R to rvec (Rodrigues)
            rvec, _ = cv2.Rodrigues(R)
            
            # Reproject 3D to 2D
            point_3d_np = point_3d.reshape(1, 1, 3).astype(np.float32)
            projected, _ = cv2.projectPoints(point_3d_np, rvec, t, K, dist)
            projected = projected.reshape(2)
            
            # Compute Euclidean distance
            error = np.linalg.norm(projected - np.array([x_orig, y_orig]))
            errors.append(error)
    
    if not errors:
        return None, None  # No valid points
    
    avg_error = np.mean(errors)
    good_percentage = (np.sum(np.array(errors) < ERROR_THRESHOLD) / len(errors)) * 100
    return avg_error, good_percentage

def main():
    # Load 3D poses
    with open(JSON_3D_POSES, 'r') as f:
        poses_data = json.load(f)
    # Convert back to numpy
    for frame_id in poses_data:
        poses_data[frame_id] = [np.array(pt) if pt is not None else None for pt in poses_data[frame_id]]
    
    # Load COCO
    coco = COCO(SINGLE_COCO_FILE)
    frame_view_mapping = build_frame_view_mapping(coco)
    frame_ids = sorted(frame_view_mapping.keys())
    
    results = {}  # {frame_id: {'avg_error': float, 'good_percentage': float}}
    all_errors = []  # For overall stats
    
    for frame_id in frame_ids:
        img_ids_per_view = frame_view_mapping.get(frame_id, [None] * NUM_VIEWS)
        keypoints_2d = load_keypoints_for_frame(coco, img_ids_per_view)
        pose_3d = poses_data.get(frame_id, [None] * NUM_JOINTS)
        
        avg_error, good_percentage = compute_frame_reprojection_error(pose_3d, keypoints_2d)
        if avg_error is not None:
            results[frame_id] = {'avg_error': avg_error, 'good_percentage': good_percentage}
            all_errors.extend([avg_error])  # Collect for overall avg
            print(f"Frame {frame_id}: Avg Error = {avg_error:.2f} px, Good % = {good_percentage:.2f}% (< {ERROR_THRESHOLD}px)")
        else:
            print(f"Frame {frame_id}: No valid points for error computation")
    
    if all_errors:
        overall_avg = np.mean(all_errors)
        overall_good_pct = np.mean([r['good_percentage'] for r in results.values()])
        print(f"\nOverall: Avg Error = {overall_avg:.2f} px, Good % = {overall_good_pct:.2f}%")
        results['overall'] = {'avg_error': overall_avg, 'good_percentage': overall_good_pct}
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, default=float)
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()