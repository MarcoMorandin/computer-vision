import os
import json
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

def load_calibration(calib_path, use_camera_center=False):
    """Load camera calibration with proper coordinate system handling."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32).flatten()
    
    # Force zero distortion for rectified images
    print(f"Original distortion coeffs: {dist}")
    dist = np.zeros_like(dist)
    
    rvec = np.array(calib["rvecs"], dtype=np.float32).flatten()
    tvec = np.array(calib["tvecs"], dtype=np.float32).flatten()
    
    R, _ = cv2.Rodrigues(rvec)
    
    if use_camera_center:
        # If tvec represents camera center in world coords
        # Then t_cam = -R @ C_world
        t_cam = -R @ tvec.reshape(3, 1)
        P = mtx @ np.hstack([R, t_cam])
    else:
        # Standard: tvec is world origin in camera coords
        P = mtx @ np.hstack([R, tvec.reshape(3, 1)])
    
    # Alternative: Try decomposing and reconstructing
    # This can help identify scale issues
    print(f"Projection matrix condition number: {np.linalg.cond(P)}")
    
    return mtx, dist, rvec, tvec, P

def get_frame_number(file_name):
    """Extract frame number from the file name."""
    match = re.search(r'frame_(\d+)', file_name)
    if match:
        return int(match.group(1))
    return None

def get_camera_number(file_name):
    """Extract camera number from the file name."""
    match = re.search(r'(?:out|cam)(\d+)', file_name)
    if match:
        return match.group(1)
    return None

def triangulate_point_dlt(points_2d, projection_matrices):
    """
    Triangulate a 3D point from multiple 2D points using Direct Linear Transform (DLT).
    
    Args:
        points_2d: List of 2D points from different views
        projection_matrices: List of camera projection matrices
    
    Returns:
        3D point coordinates (X, Y, Z)
    """
    num_views = len(points_2d)
    A = np.zeros((2 * num_views, 4))
    
    for i in range(num_views):
        x, y = points_2d[i]
        P = projection_matrices[i]
        
        A[2*i] = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]  # Normalize homogeneous coordinate
    
    return X[:3]

def triangulate_points(points_2d_by_camera, projection_matrices, min_views=2):
    """
    Triangulate a 3D point from multiple 2D observations across different cameras.
    
    Args:
        points_2d_by_camera: Dictionary mapping camera indices to 2D points
        projection_matrices: Dictionary mapping camera indices to projection matrices
        min_views: Minimum number of views required for triangulation
    
    Returns:
        3D point coordinates (X, Y, Z) or None if not enough views
    """
    valid_cameras = [cam for cam in points_2d_by_camera.keys() if cam in projection_matrices]
    
    if len(valid_cameras) < min_views:
        return None
    
    points_2d = [points_2d_by_camera[cam] for cam in valid_cameras]
    proj_matrices = [projection_matrices[cam] for cam in valid_cameras]
    
    # For more than 2 views, use DLT method
    return triangulate_point_dlt(points_2d, proj_matrices)

def bundle_adjust_point(X0, points_2d, proj_matrices_list, camera_indices):
    """
    Bundle adjustment for a single 3D point to refine the triangulation.
    
    Args:
        X0: Initial 3D point estimate
        points_2d: List of observed 2D points
        proj_matrices_list: List of projection matrices (ordered to match camera_indices)
        camera_indices: List of indices for the projection matrices
    
    Returns:
        Refined 3D point
    """
    def residuals(X):
        X_homogeneous = np.append(X, 1)
        errors = []
        
        for i, cam_idx in enumerate(camera_indices):
            P = proj_matrices_list[cam_idx]
            x_proj = P @ X_homogeneous
            x_proj = x_proj[:2] / x_proj[2]
            
            errors.extend(points_2d[i] - x_proj)
        
        return errors
    
    result = least_squares(residuals, X0, method='lm')
    return result.x

def triangulate_player_pose(annotations_file, calib_base_dir, output_dir):
    """
    Triangulate the 3D positions of player keypoints from multiple views.
    
    Args:
        annotations_file: Path to the COCO annotations file with keypoints
        calib_base_dir: Base directory for camera calibration files
        output_dir: Directory to save output files
    """
    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Extract images and annotations
    images = {img["id"]: img for img in coco_data["images"]}
    
    # Group annotations by frame number
    frames_data = {}
    for ann in coco_data["annotations"]:
        if "keypoints" in ann:
            image_id = ann["image_id"]
            image_info = images[image_id]
            file_name = image_info["file_name"]
            
            # Extract frame number and camera number
            frame_num = get_frame_number(file_name)
            cam_num = get_camera_number(file_name)
            
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
    projection_matrices = {}
    
    # Get available camera numbers from the data
    available_cameras = set()
    for frame_data in frames_data.values():
        available_cameras.update(frame_data.keys())
    
    # Load calibration for each camera
    for cam_num in available_cameras:
        calib_path = os.path.join(calib_base_dir, f"cam_{cam_num}", "calib", "camera_calib.json")
        if os.path.exists(calib_path):
            mtx, dist, rvec, tvec, P = load_calibration(calib_path)
            
            # Debug: Print calibration info
            print(f"Camera {cam_num} calibration:")
            print(f"  Focal lengths: fx={mtx[0,0]:.1f}, fy={mtx[1,1]:.1f}")
            print(f"  Translation magnitude: {np.linalg.norm(tvec):.1f}")
            print(f"  Translation: {tvec.flatten()}")
            
            camera_calib[cam_num] = {
                "mtx": mtx,
                "dist": dist,
                "rvec": rvec,
                "tvec": tvec
            }
            projection_matrices[cam_num] = P
    
    # Process each frame
    player_3d_poses = {}
    
    for frame_num, frame_data in sorted(frames_data.items()):
        print(f"Processing frame {frame_num}...")
        
        # Skip frames that don't have data from at least 2 cameras
        if len(frame_data) < 2:
            print(f"  Skipping frame {frame_num}: insufficient camera views")
            continue
        
        # Prepare 3D keypoints for this frame
        keypoints_3d = []
        
        # Get the number of keypoints from the first camera
        first_cam = list(frame_data.keys())[0]
        num_keypoints = len(frame_data[first_cam])
        
        # Process each keypoint
        for kp_idx in range(num_keypoints):
            # Collect 2D observations of this keypoint from all cameras
            points_2d_by_camera = {}
            
            for cam_num, keypoints in frame_data.items():
                x, y, v = keypoints[kp_idx]
                # Only use keypoints that are visible (v > 0)
                if v > 0:
                    points_2d_by_camera[cam_num] = (x, y)
            
            # Triangulate this keypoint if we have enough observations
            point_3d = None
            if len(points_2d_by_camera) >= 2:
                point_3d = triangulate_points(points_2d_by_camera, projection_matrices)
                
                # Refine the 3D point using bundle adjustment
                if point_3d is not None:
                    points_2d_list = [points_2d_by_camera[cam] for cam in points_2d_by_camera.keys()]
                    camera_indices = list(points_2d_by_camera.keys())
                    # Create a list of projection matrices matching the order of camera_indices
                    proj_matrices_list = [projection_matrices[cam] for cam in camera_indices]
                    point_3d = bundle_adjust_point(point_3d, points_2d_list, proj_matrices_list, range(len(camera_indices)))
            
            keypoints_3d.append(point_3d.tolist() if point_3d is not None else None)
        
        player_3d_poses[frame_num] = keypoints_3d
    
    # Save the 3D poses to a JSON file
    output_file = os.path.join(output_dir, "player_3d_poses.json")
    with open(output_file, 'w') as f:
        json.dump(player_3d_poses, f, indent=2)

    return player_3d_poses

if __name__ == "__main__":
    annotations_file = os.path.join("..", "rectification", "output", "dataset", "train", "_annotations.coco.json")
    calib_base_dir = os.path.join("..", "data", "camera_data_v2")
    output_dir = os.path.join("output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Triangulate player poses
    player_3d_poses = triangulate_player_pose(annotations_file, calib_base_dir, output_dir)
