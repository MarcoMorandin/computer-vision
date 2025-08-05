import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re

def load_calibration(calib_path):
    """Load camera calibration parameters from a JSON file."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32).flatten()
    
    # Extract rotation and translation vectors
    rvec = np.array(calib["rvecs"], dtype=np.float32).flatten()
    tvec = np.array(calib["tvecs"], dtype=np.float32).flatten()
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create the projection matrix P = K[R|t]
    P = np.zeros((3, 4), dtype=np.float32)
    P[:3, :3] = R
    P[:3, 3] = tvec
    P = mtx @ P
    
    return mtx, dist, rvec, tvec, P

def project_3d_point(point_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Project a 3D point back to a 2D camera view."""
    point_3d = np.array(point_3d, dtype=np.float32).reshape(1, 3)
    point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return point_2d.reshape(2)

def extract_frame_info(filename):
    """Extract camera number and frame number from the file name."""
    cam_match = re.search(r'out(\d+)_frame', filename)
    frame_match = re.search(r'frame_(\d+)', filename)
    
    cam_num = cam_match.group(1) if cam_match else None
    frame_num = int(frame_match.group(1)) if frame_match else None
    
    return cam_num, frame_num

def compute_reprojection_error(point_3d, point_2d, rvec, tvec, camera_matrix, dist_coeffs):
    """Compute reprojection error between projected 3D point and ground truth 2D point."""
    projected_point = project_3d_point(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    error = np.linalg.norm(projected_point - point_2d)
    return error

def compute_metrics(errors):
    """Compute various error metrics."""
    if len(errors) == 0:
        return {
            'mpjpe': 0.0,
            'rmse': 0.0
        }
        
    errors = np.array(errors)
    metrics = {
        'mpjpe': np.mean(errors),  # Mean Per Joint Position Error
        'rmse': np.sqrt(np.mean(errors**2)),  # Root Mean Squared Error
    }
    return metrics

def evaluate_reprojection(poses_3d_path, annotations_path, calib_base_dir):
    """Evaluate reprojection accuracy of 3D skeleton against ground truth 2D annotations."""
    # Load 3D poses
    with open(poses_3d_path, 'r') as f:
        poses_3d = json.load(f)
    
    # Load ground truth annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract images and annotations
    images = {img["id"]: img for img in coco_data["images"]}
    
    # Group annotations by frame number and camera
    frames_data = {}
    for ann in coco_data["annotations"]:
        if "keypoints" in ann:
            image_id = ann["image_id"]
            image_info = images[image_id]
            file_name = image_info["file_name"]
            
            cam_num, frame_num = extract_frame_info(file_name)
            
            if frame_num is not None and cam_num is not None:
                if frame_num not in frames_data:
                    frames_data[frame_num] = {}
                
                # Store keypoints for this frame and camera
                keypoints = ann["keypoints"]
                num_keypoints = len(keypoints) // 3
                
                # Reshape keypoints from [x1, y1, v1, x2, y2, v2, ...] to [(x1, y1, v1), (x2, y2, v2), ...]
                keypoints_reshaped = []
                for i in range(num_keypoints):
                    x = keypoints[i * 3]
                    y = keypoints[i * 3 + 1]
                    v = keypoints[i * 3 + 2]
                    keypoints_reshaped.append((x, y, v))
                
                frames_data[frame_num][cam_num] = keypoints_reshaped
    
    # Load camera calibration data
    camera_calib = {}
    for cam_num in [str(i) for i in range(1, 8)]:
        if cam_num == '6':  # Skip camera 6 if it doesn't exist
            continue
            
        calib_path = os.path.join(calib_base_dir, f"cam_{cam_num}", "calib", "camera_calib.json")
        if os.path.exists(calib_path):
            mtx, dist, rvec, tvec, P = load_calibration(calib_path)
            camera_calib[cam_num] = {
                "mtx": mtx,
                "dist": dist,
                "rvec": rvec,
                "tvec": tvec
            }
    
    # Initialize error metrics
    all_errors = []
    per_camera_errors = {cam_num: [] for cam_num in camera_calib.keys()}
    per_joint_errors = [[] for _ in range(18)]  # Assuming 18 joints
    
    # Process each frame
    for frame_num, frame_data in sorted(frames_data.items()):
        frame_str = str(frame_num)
        
        # Skip if we don't have 3D poses for this frame
        if frame_str not in poses_3d:
            continue
        
        # Get 3D poses for this frame
        keypoints_3d = poses_3d[frame_str]
        
        # Process each camera view
        for cam_num, gt_keypoints in frame_data.items():
            if cam_num not in camera_calib:
                continue
                
            mtx = camera_calib[cam_num]["mtx"]
            dist = camera_calib[cam_num]["dist"]
            rvec = camera_calib[cam_num]["rvec"]
            tvec = camera_calib[cam_num]["tvec"]
            
            # Project each 3D keypoint to 2D
            for kp_idx, (point_3d, gt_point) in enumerate(zip(keypoints_3d, gt_keypoints)):
                # Skip if the 3D point or ground truth is not available
                if point_3d is None or gt_point[2] <= 0:
                    continue
                    
                # Ground truth 2D point
                gt_point_2d = np.array([gt_point[0], gt_point[1]], dtype=np.float32)
                
                # Project 3D point to 2D
                projected_point = project_3d_point(point_3d, rvec, tvec, mtx, dist)
                
                # Compute error
                error = np.linalg.norm(projected_point - gt_point_2d)
                
                all_errors.append(error)
                per_camera_errors[cam_num].append(error)
                per_joint_errors[kp_idx].append(error)
    
    # Compute overall metrics
    overall_metrics = compute_metrics(all_errors)
    
    # Compute per-camera metrics
    camera_metrics = {}
    for cam_num, errors in per_camera_errors.items():
        camera_metrics[cam_num] = compute_metrics(errors)
    
    return overall_metrics, camera_metrics


    
    
def main():
    poses_3d_path = os.path.join('output', 'player_3d_poses.json')
    annotations_path = os.path.join('..', 'rectification', 'rectified', 'dataset', 'train', '_annotations.coco.json')
    calib_base_dir = os.path.join('..', 'data', 'camera_data_v2')
    
    # Evaluate reprojection
    print("Evaluating reprojection accuracy...")
    overall_metrics, camera_metrics = evaluate_reprojection(
        poses_3d_path, annotations_path, calib_base_dir
    )
    
    # Print results
    print("\nOverall metrics:")
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value:.2f} pixels")
    
    print("\nPer-camera metrics (MPJPE):")
    for cam_num, metrics in camera_metrics.items():
        print(f"  Camera {cam_num}: {metrics['mpjpe']:.2f} pixels")
    

if __name__ == "__main__":
    main()
