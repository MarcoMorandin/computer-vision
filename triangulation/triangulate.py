import json
import numpy as np
import cv2
import re
from pycocotools.coco import COCO
import os
from pycalib.calib import triangulate_Npts

# Constants (adjust as needed)
NUM_VIEWS = 5
NUM_JOINTS = 18  # COCO standard (nose, eyes, ears, shoulders, etc.)
MIN_VISIBLE_VIEWS = 3  # Need at least 2 views for triangulation


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
    camera_nums = [1, 2, 3, 4]  # Adjust based on your actual cameras
    
    for i, cam_num in enumerate(camera_nums):
        calib_file = 'camera_calib_real.json'
        calib_path = os.path.join(camera_data_folder, f'cam_{cam_num}', 'calib', calib_file)
        

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

# Function to parse view and frame from filename (updated for your format!)
def parse_view_and_frame(filename):
    """
    Extract view_id (0-6) and frame_id from filename.
    (Corrected to handle Roboflow's hashed filenames)
    Returns: (view_id: int, frame_id: str) or (None, None) if parse fails.
    """
    # Pattern: 'out' followed by a number, '_frame_', and another number.
    # This is more robust and ignores the hash and final extension.
    match = re.search(r'out(\d+)_frame_(\d+)', filename)
    if match:
        view_num = int(match.group(1))  # Camera number (e.g., 1, 2, ..., 7)
        frame_id = match.group(2)       # Frame number (e.g., '0016')

        # --- IMPORTANT MAPPING LOGIC ---
        # Your script maps camera numbers [1, 2, 3, 4, 5, 7] to view_ids [0, 1, 2, 3, 4, 5]
        # We need to handle this mapping correctly.
        camera_map = {1: 0, 2: 1, 3: 2, 4: 3}
        
        if view_num in camera_map:
            view_id = camera_map[view_num]
            return view_id, frame_id
            
    return None, None

# Precompute mapping: {frame_id: [img_id_view0, img_id_view1, ..., img_id_view6]}
def build_frame_view_mapping(coco):
    mapping = {}
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        view_id, frame_id = parse_view_and_frame(filename)
        if view_id is not None and frame_id is not None and 0 <= view_id < NUM_VIEWS:
            if frame_id not in mapping:
                mapping[frame_id] = [None] * NUM_VIEWS
            mapping[frame_id][view_id] = img_id
    return mapping

# Function to implement multi-view triangulation using pycalib's triangulate_Npts
def triangulate_joints_batch(keypoints_2d, projection_matrices):
    """
    Triangulate all joints for a single frame using pycalib's triangulate_Npts.
    
    Args:
        keypoints_2d (np.array): Shape (NUM_VIEWS, NUM_JOINTS, 3) -> [x, y, v] per joint per view
        projection_matrices (list): List of 3x4 projection matrices
    
    Returns:
        list: List of 3D points for each joint (or None if insufficient views)
    """
    # Convert projection matrices to numpy array
    P_matrices = np.array(projection_matrices)  # Shape: (NUM_VIEWS, 3, 4)
    
    # Prepare data for triangulate_Npts
    # We need to collect valid 2D points for each joint
    joint_3d_points = []
    
    for joint_idx in range(NUM_JOINTS):
        # Extract 2D points for this joint across all views
        valid_views = []
        valid_points = []
        
        for view_idx in range(NUM_VIEWS):
            x, y, v = keypoints_2d[view_idx, joint_idx]
            if v >= 2:  # Visible and labeled
                valid_views.append(view_idx)
                valid_points.append([x, y])
        
        if len(valid_views) < MIN_VISIBLE_VIEWS:
            joint_3d_points.append(None)
            continue
        
        # Prepare data for triangulate_Npts
        # Shape should be (num_cameras, num_points, 2) but we have only 1 point per joint
        points_2d = np.array(valid_points).reshape(len(valid_views), 1, 2)
        cameras_P = P_matrices[valid_views]  # Select only valid cameras
        
        try:
            # triangulate_Npts expects (Nc, Np, 2) and (Nc, 3, 4)
            triangulated_points = triangulate_Npts(points_2d, cameras_P)
            # triangulated_points shape is (Np, 3), we only have 1 point
            joint_3d_points.append(triangulated_points[0])
        except Exception as e:
            print(f"Triangulation failed for joint {joint_idx}: {e}")
            joint_3d_points.append(None)
    
    return joint_3d_points

# Function to load COCO keypoints for a specific frame (updated for single COCO)
def load_keypoints_for_frame(coco, img_ids_per_view):
    """
    Load 2D keypoints for a given frame's img_ids per view.
    
    Args:
        coco (COCO): Single COCO object.
        img_ids_per_view (list): [img_id_view0, ..., img_id_view6] (None if missing).
    
    Returns:
        np.array: Shape (NUM_VIEWS, NUM_JOINTS, 3) -> [x, y, v] per joint per view.
    """
    all_keypoints = np.zeros((NUM_VIEWS, NUM_JOINTS, 3))  # x, y, v
    for view_idx, img_id in enumerate(img_ids_per_view):
        if img_id is None:
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if anns:
            # Assume one annotation per image (single person); take the first
            keypoints = np.array(anns[0]['keypoints']).reshape(NUM_JOINTS, 3)
            all_keypoints[view_idx] = keypoints
    return all_keypoints

# Main function to triangulate all frames (updated)
def triangulate_multiview_coco(single_coco_file, projection_matrices, output_file='3d_poses.json'):
    """
    Triangulate 2D COCO keypoints to 3D for all frames from a single COCO file.
    
    Args:
        single_coco_file (str): Path to the single COCO JSON file.
        projection_matrices (list of np.array): 3x4 P matrices per view.
        output_file (str): Path to save 3D poses as JSON.
    """
    # Load single COCO object
    coco = COCO(single_coco_file)
    
    # Build mapping of frame_id to img_ids per view
    frame_view_mapping = build_frame_view_mapping(coco)
    frame_ids = sorted(frame_view_mapping.keys())  # All unique frame_ids
    
    all_3d_poses = {}  # {frame_id: list of 3D points (NUM_JOINTS, 3)}
    
    for frame_id in frame_ids:
        img_ids_per_view = frame_view_mapping[frame_id]
        
        # Load keypoints for this frame across views (NUM_VIEWS, NUM_JOINTS, 3)
        keypoints_2d = load_keypoints_for_frame(coco, img_ids_per_view)
        
        # Use batch triangulation for all joints at once
        frame_3d = triangulate_joints_batch(keypoints_2d, projection_matrices)
        
        all_3d_poses[frame_id] = frame_3d
        print(f"Processed frame {frame_id}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_3d_poses, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"Saved 3D poses to {output_file}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual paths
    SINGLE_COCO_FILE = '../rectification/data/annotations/train/_annotations.coco.json'

    
    triangulate_multiview_coco(SINGLE_COCO_FILE, PROJECTION_MATRICES)