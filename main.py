# -*- coding: utf-8 -*-
"""
Python script for Multi-view Human Pose Estimation Project.

This script implements the core functionalities required for the project, including:
1. Loading multi-view 2D annotations and camera calibration parameters.
2. Triangulating 2D keypoints from multiple views to obtain 3D poses (Ground Truth).
3. Evaluating the accuracy of 3D triangulation via reprojection error.
4. Executing a pre-trained human pose estimation model (YOLOv8-Pose) on video frames.
5. Evaluating the 2D accuracy of the pose estimation model against ground truth.
6. Triangulating the 2D keypoints estimated by the model to obtain 3D poses.
7. Evaluating the 3D accuracy of the estimated poses using MPJPE.
8. Providing a framework for fine-tuning the pose estimation model (optional).
9. Outlining steps for exporting data for visualization (e.g., in Unreal Engine).

Prerequisites:
- Python 3.8+
- Libraries: opencv-python, numpy, ultralytics, matplotlib
  Install using: pip install opencv-python numpy ultralytics matplotlib
- Annotated dataset (e.g., from Roboflow) in a consistent format.
- Camera calibration parameters (intrinsic, distortion, extrinsic) for all views.
"""

import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from ultralytics import YOLO
from ultralytics.engine.results import Results, Keypoints # For type hinting YOLO results

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Constants and Configuration ---
# TODO: Update these paths and parameters accurately
ANNOTATION_FOLDER: str = "path/to/your/roboflow/export" # Folder with annotation files (e.g., JSON per frame/view)
VIDEO_FOLDER: str = "path/to/your/videos" # Folder containing input video files
CAMERA_PARAM_FILE: str = "path/to/camera_parameters.json" # File with intrinsic/extrinsic params
OUTPUT_FOLDER: str = "output" # Directory for saving results
YOLO_MODEL_PATH: str = 'yolov8n-pose.pt' # Path to the pre-trained YOLOv8-Pose model
FINETUNED_YOLO_PATH: str = 'path/to/your/finetuned/model/weights/best.pt' # Path after fine-tuning
NUM_CAMERAS: int = 6 # Number of camera views in the setup
NUM_KEYPOINTS: int = 17 # Number of keypoints per pose (e.g., COCO format)
TARGET_FRAME: int = 1 # Example frame number to process

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Type alias for camera parameters dictionary
CameraParams = Dict[str, Union[np.ndarray, str]] # K, dist, R, t, P matrices

# --- 1. Data Loading ---

def load_camera_parameters(filepath: str) -> Optional[Dict[str, CameraParams]]:
    """
    Loads camera intrinsic and extrinsic parameters from a specified file.

    Args:
        filepath (str): Path to the camera parameter file (expects JSON format).

    Returns:
        Optional[Dict[str, CameraParams]]: A dictionary where keys are camera IDs (e.g., 'cam1')
        and values are dictionaries containing camera parameters ('K', 'dist', 'R', 't', 'P').
        Returns None if the file cannot be loaded or parsed.
        'K': 3x3 intrinsic matrix (np.ndarray)
        'dist': 1xN distortion coefficients (np.ndarray)
        'R': 3x3 rotation matrix (world to camera, np.ndarray)
        't': 3x1 translation vector (world to camera, np.ndarray)
        'P': 3x4 projection matrix (K @ [R|t], np.ndarray)
    """
    logging.info(f"Loading camera parameters from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            params_data = json.load(f)

        camera_params: Dict[str, CameraParams] = {}
        for cam_id, data in params_data.items():
            # Convert lists from JSON to NumPy arrays
            K = np.array(data['K'], dtype=np.float64)
            dist = np.array(data['dist'], dtype=np.float64)
            R = np.array(data['R'], dtype=np.float64)
            t = np.array(data['t'], dtype=np.float64).reshape(3, 1) # Ensure t is 3x1

            # Validate shapes
            if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3, 1):
                logging.error(f"Invalid matrix dimensions for {cam_id} in {filepath}")
                return None
            # Dist coeffs can vary in length, basic check
            if dist.ndim != 1 and dist.ndim != 2:
                 logging.error(f"Invalid distortion coefficients shape for {cam_id} in {filepath}")
                 return None

            # Calculate projection matrix P = K @ [R | t]
            Rt = np.hstack((R, t))
            P = K @ Rt

            camera_params[cam_id] = {
                'K': K,
                'dist': dist,
                'R': R,
                't': t,
                'P': P
            }
        logging.info(f"Successfully loaded parameters for {len(camera_params)} cameras.")
        return camera_params

    except FileNotFoundError:
        logging.error(f"Camera parameter file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from camera parameter file: {filepath}")
        return None
    except KeyError as e:
        logging.error(f"Missing key {e} in camera parameter file for a camera.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading camera parameters: {e}")
        return None

def load_annotations(annotation_folder: str, frame_number: int, num_cameras: int, num_keypoints: int) -> Dict[str, Optional[np.ndarray]]:
    """
    Loads 2D ground truth annotations for a specific frame from all camera views.

    Assumes annotations are stored in JSON files named like 'frame_XXX_camY.json',
    where XXX is the zero-padded frame number and Y is the camera number.
    Each JSON file should contain a key (e.g., 'keypoints') with a list of
    [x, y] coordinates for each keypoint.

    Args:
        annotation_folder (str): Path to the folder containing annotation files.
        frame_number (int): The frame number to load annotations for.
        num_cameras (int): The total number of camera views.
        num_keypoints (int): The expected number of keypoints per pose.

    Returns:
        Dict[str, Optional[np.ndarray]]: A dictionary where keys are camera IDs (e.g., 'cam1')
        and values are NumPy arrays of shape (num_keypoints, 2) containing (x, y) coordinates.
        If annotations are missing or invalid for a camera, the value is None.
    """
    logging.info(f"Loading annotations for frame {frame_number} from: {annotation_folder}")
    annotations: Dict[str, Optional[np.ndarray]] = {}
    for i in range(1, num_cameras + 1):
        cam_id = f'cam{i}'
        # Adjust filename pattern if your Roboflow export structure differs
        annotation_file = os.path.join(annotation_folder, f"frame_{frame_number:03d}_{cam_id}.json")
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                # --- Adapt parsing based on your Roboflow JSON structure ---
                # Adjust the key 'keypoints' if necessary
                if 'keypoints' not in data:
                    logging.warning(f"Key 'keypoints' not found in {annotation_file}")
                    annotations[cam_id] = None
                    continue

                keypoints = np.array(data['keypoints'], dtype=np.float32)
                # --- End of adaptation ---

                # Validate shape
                if keypoints.shape == (num_keypoints, 2):
                    annotations[cam_id] = keypoints
                else:
                    logging.warning(f"Unexpected keypoint shape {keypoints.shape} in {annotation_file}. Expected ({num_keypoints}, 2).")
                    annotations[cam_id] = None

        except FileNotFoundError:
            # This might be expected if not all views have annotations for every frame
            logging.debug(f"Annotation file not found (optional): {annotation_file}")
            annotations[cam_id] = None
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from annotation file: {annotation_file}")
            annotations[cam_id] = None
        except Exception as e:
            logging.error(f"Error loading annotation {annotation_file}: {e}")
            annotations[cam_id] = None

    loaded_count = sum(1 for anno in annotations.values() if anno is not None)
    logging.info(f"Annotations loaded for frame {frame_number}. Found valid data for {loaded_count}/{num_cameras} views.")
    return annotations


# --- 2. Triangulation ---

def triangulate_points_linear(points_2d_dict: Dict[str, np.ndarray], camera_params: Dict[str, CameraParams]) -> Optional[np.ndarray]:
    """
    Triangulates 3D points from 2D points using a linear method (SVD-based).

    This method constructs a system of linear equations Ax = 0 based on the
    projection matrices and 2D points, then solves for the 3D point (x)
    using Singular Value Decomposition (SVD). Handles multiple views (>2).

    Args:
        points_2d_dict (Dict[str, np.ndarray]): Dictionary of 2D points
            {'cam1': pts_cam1, 'cam2': pts_cam2, ...} where pts_camX is a
            NumPy array (N_keypoints, 2). Only cameras with non-None points are used.
        camera_params (Dict[str, CameraParams]): Dictionary containing camera
            parameters (requires projection matrices 'P').

    Returns:
        Optional[np.ndarray]: Triangulated 3D points (N_keypoints, 3) in Euclidean coordinates.
                               Returns None if triangulation is not possible (e.g., < 2 views).
                               Individual keypoints that fail triangulation will have NaN values.
    """
    valid_cam_ids = [cam_id for cam_id, pts in points_2d_dict.items() if pts is not None]
    if len(valid_cam_ids) < 2:
        logging.error("Triangulation failed: Need at least two valid views.")
        return None

    # Use the shape from the first valid camera's points
    num_keypoints = points_2d_dict[valid_cam_ids[0]].shape[0]
    points_3d = np.full((num_keypoints, 3), np.nan, dtype=np.float64) # Initialize with NaN

    proj_matrices = {cam_id: camera_params[cam_id]['P'] for cam_id in valid_cam_ids}
    points_2d_valid = {cam_id: points_2d_dict[cam_id] for cam_id in valid_cam_ids}

    logging.info(f"Triangulating {num_keypoints} points using {len(valid_cam_ids)} views...")

    for i in range(num_keypoints):
        A = []
        num_valid_views_for_kp = 0
        for cam_id in valid_cam_ids:
            point_2d = points_2d_valid[cam_id][i]
            P = proj_matrices[cam_id]

            # Check if the keypoint coordinate is valid (e.g., not negative, or based on confidence if available)
            if point_2d is not None and np.all(point_2d >= 0): # Basic validity check
                x, y = point_2d
                # Construct rows for the linear system Ax = 0
                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])
                num_valid_views_for_kp += 1

        # Need at least 2 views (4 equations) to solve for 3D point (4 homogeneous coords)
        if num_valid_views_for_kp >= 2:
            A = np.array(A)
            # Solve Ax = 0 using SVD
            _, _, Vh = np.linalg.svd(A)
            # The solution is the last column of V (last row of Vh)
            point_4d_hom = Vh[-1, :]
            # Convert from homogeneous to Euclidean coordinates
            if point_4d_hom[3] != 0: # Avoid division by zero
                 points_3d[i, :] = point_4d_hom[:3] / point_4d_hom[3]
            else:
                 logging.warning(f"Triangulation resulted in zero homogeneous coordinate w for keypoint {i}.")
                 # points_3d[i, :] remains NaN

        else:
            logging.warning(f"Not enough valid views ({num_valid_views_for_kp}) for keypoint {i}. Skipping triangulation.")
            # points_3d[i, :] remains NaN

    num_failed = np.isnan(points_3d[:, 0]).sum()
    if num_failed > 0:
        logging.warning(f"Triangulation failed for {num_failed}/{num_keypoints} keypoints.")
    logging.info("Triangulation complete.")
    return points_3d

# --- 2a. Evaluation: Reprojection Error ---

def calculate_reprojection_error(points_3d: np.ndarray,
                                 points_2d_gt_dict: Dict[str, Optional[np.ndarray]],
                                 camera_params: Dict[str, CameraParams]) -> Dict[str, float]:
    """
    Calculates the mean reprojection error for triangulated 3D points onto each camera view.

    Args:
        points_3d (np.ndarray): The triangulated 3D points (N_keypoints, 3).
        points_2d_gt_dict (Dict[str, Optional[np.ndarray]]): Ground truth 2D points
            {'cam1': pts_cam1, ...}. Points can be None if GT is missing.
        camera_params (Dict[str, CameraParams]): Camera parameters ('K', 'R', 't', 'dist').

    Returns:
        Dict[str, float]: Dictionary of mean reprojection errors (in pixels) per camera
                          {'cam1': error1, ...}. Value is np.nan if error cannot be calculated.
    """
    reprojection_errors: Dict[str, float] = {}
    num_keypoints = points_3d.shape[0]

    logging.info("Calculating reprojection error...")
    for cam_id, params in camera_params.items():
        points_2d_gt = points_2d_gt_dict.get(cam_id)
        # Skip if no ground truth 2D points for this camera view
        if points_2d_gt is None:
            reprojection_errors[cam_id] = np.nan
            continue

        K, R, t, dist = params['K'], params['R'], params['t'], params['dist']

        # --- Input Validation ---
        if points_3d.shape[0] != points_2d_gt.shape[0]:
             logging.warning(f"[{cam_id}] Mismatch between 3D points ({points_3d.shape[0]}) and 2D GT points ({points_2d_gt.shape[0]}). Skipping reprojection.")
             reprojection_errors[cam_id] = np.nan
             continue
        # --- End Validation ---


        # Filter out pairs where 3D point is NaN or 2D GT is invalid (e.g., < 0)
        valid_3d_mask = ~np.isnan(points_3d).any(axis=1)
        valid_2d_gt_mask = np.all(points_2d_gt >= 0, axis=1) # Basic validity check for GT
        valid_mask = valid_3d_mask & valid_2d_gt_mask

        valid_points_3d = points_3d[valid_mask]
        valid_points_2d_gt = points_2d_gt[valid_mask]

        if valid_points_3d.shape[0] == 0:
            logging.warning(f"[{cam_id}] No valid point pairs found for reprojection.")
            reprojection_errors[cam_id] = np.nan
            continue

        # Project valid 3D points back to 2D using camera parameters
        try:
            # cv2.projectPoints requires objectPoints (N, 1, 3) or (N, 3)
            # rvec = rotation vector (from R), tvec = translation vector
            rvec, _ = cv2.Rodrigues(R) # Convert rotation matrix to rotation vector
            projected_points_2d, _ = cv2.projectPoints(valid_points_3d, rvec, t, K, dist)

            # Reshape projected points from (N, 1, 2) to (N, 2)
            projected_points_2d = projected_points_2d.reshape(-1, 2)

            # Calculate Euclidean distance between projected and ground truth points
            errors = np.linalg.norm(projected_points_2d - valid_points_2d_gt, axis=1)
            mean_error = np.mean(errors)
            reprojection_errors[cam_id] = mean_error
            logging.info(f"  {cam_id}: Mean Reprojection Error = {mean_error:.4f} pixels ({valid_points_3d.shape[0]} points)")

        except cv2.error as e:
            logging.error(f"[{cam_id}] OpenCV error during projection: {e}")
            reprojection_errors[cam_id] = np.nan
        except Exception as e:
            logging.error(f"[{cam_id}] Unexpected error during projection: {e}")
            reprojection_errors[cam_id] = np.nan

    logging.info("Reprojection error calculation complete.")
    return reprojection_errors

# --- 3. Run Human Pose Estimation (YOLOv8-Pose) ---

def run_pose_estimation(model: YOLO, image_path: str, num_keypoints: int) -> Optional[np.ndarray]:
    """
    Runs YOLOv8-Pose on a single image and extracts keypoints for the most confident person.

    Args:
        model (YOLO): An initialized YOLO model object.
        image_path (str): Path to the input image file.
        num_keypoints (int): Expected number of keypoints.

    Returns:
        Optional[np.ndarray]: A NumPy array of shape (num_keypoints, 2) containing (x, y)
                               coordinates for the detected pose, or None if no pose is
                               detected or an error occurs.
                               Returns the keypoints for the detection with the highest
                               bounding box confidence if multiple people are detected.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file not found for pose estimation: {image_path}")
        return None

    logging.debug(f"Running pose estimation on: {image_path}")
    try:
        results: List[Results] = model(image_path, verbose=False) # Suppress Ultralytics logs

        if not results or results[0].keypoints is None:
            logging.warning(f"No keypoints detected in image: {image_path}")
            return None

        keypoints_data: Keypoints = results[0].keypoints
        xy_coordinates = keypoints_data.xy.cpu().numpy() # Shape: (N_persons, N_kpts, 2)
        confidences = keypoints_data.conf # Shape: (N_persons, N_kpts) - keypoint confidences
        box_confidences = results[0].boxes.conf.cpu().numpy() # Shape: (N_persons,) - bbox confidences

        if xy_coordinates.shape[0] == 0:
             logging.warning(f"Keypoints detected, but array is empty in: {image_path}")
             return None

        # Select the person with the highest bounding box confidence
        best_person_idx = np.argmax(box_confidences)
        selected_keypoints = xy_coordinates[best_person_idx] # Shape: (N_kpts, 2)

        if selected_keypoints.shape == (num_keypoints, 2):
            logging.debug(f"  Detected pose with {selected_keypoints.shape[0]} keypoints in {image_path}.")
            # Optional: You could also return the keypoint confidences if needed later
            # kpt_confs = confidences[best_person_idx].cpu().numpy()
            return selected_keypoints
        else:
            logging.warning(f"Detected keypoints shape mismatch in {image_path}. Got {selected_keypoints.shape}, expected ({num_keypoints}, 2).")
            return None

    except Exception as e:
        logging.error(f"Error during pose estimation on {image_path}: {e}")
        return None

def evaluate_2d_pose(pose_estimated: Optional[np.ndarray], pose_gt: Optional[np.ndarray]) -> float:
    """
    Evaluates the estimated 2D pose against the ground truth using Mean Per Joint Position Error (MPJPE).

    Args:
        pose_estimated (Optional[np.ndarray]): Estimated keypoints (N_kpts, 2).
        pose_gt (Optional[np.ndarray]): Ground truth keypoints (N_kpts, 2).

    Returns:
        float: Mean Per Joint Position Error (MPJPE) in 2D pixels.
               Returns np.nan if inputs are invalid, shapes mismatch, or no valid joints found.
    """
    if pose_estimated is None or pose_gt is None:
        logging.debug("Cannot evaluate 2D pose: Missing estimated or GT pose.")
        return np.nan
    if pose_estimated.shape != pose_gt.shape:
        logging.warning(f"Cannot evaluate 2D pose: Shape mismatch. Est: {pose_estimated.shape}, GT: {pose_gt.shape}")
        return np.nan

    # Consider only keypoints that are valid in BOTH estimated and GT
    # Validity check: coordinates >= 0 (adjust if using confidence scores)
    valid_mask_gt = np.all(pose_gt >= 0, axis=1)
    valid_mask_est = np.all(pose_estimated >= 0, axis=1)
    valid_mask = valid_mask_gt & valid_mask_est

    if not np.any(valid_mask):
        logging.warning("Cannot evaluate 2D pose: No commonly valid keypoints found.")
        return np.nan

    # Calculate Euclidean distance for valid keypoints
    errors = np.linalg.norm(pose_estimated[valid_mask] - pose_gt[valid_mask], axis=1)
    mean_error = np.mean(errors)
    logging.debug(f"Calculated 2D MPJPE: {mean_error:.4f} pixels ({valid_mask.sum()} valid points)")
    return mean_error


# --- 3a. Triangulate YOLO Poses & Evaluate 3D ---

def evaluate_3d_pose_mpjpe(pose_3d_estimated: Optional[np.ndarray], pose_3d_gt: Optional[np.ndarray]) -> float:
    """
    Evaluates the estimated 3D pose against the ground truth 3D pose using MPJPE.

    Args:
        pose_3d_estimated (Optional[np.ndarray]): Estimated 3D keypoints (N_kpts, 3).
        pose_3d_gt (Optional[np.ndarray]): Ground truth 3D keypoints (N_kpts, 3)
                                         (typically from triangulating GT annotations).

    Returns:
        float: Mean Per Joint Position Error (MPJPE) in 3D units (e.g., mm, depending
               on the scale of the world coordinates used in calibration).
               Returns np.nan if inputs are invalid, shapes mismatch, or no valid joints found.
    """
    if pose_3d_estimated is None or pose_3d_gt is None:
        logging.debug("Cannot evaluate 3D pose: Missing estimated or GT 3D pose.")
        return np.nan
    if pose_3d_estimated.shape != pose_3d_gt.shape:
        logging.warning(f"Cannot evaluate 3D pose: Shape mismatch. Est: {pose_3d_estimated.shape}, GT: {pose_3d_gt.shape}")
        return np.nan

    # Filter out pairs where either estimated or GT 3D keypoint is NaN
    valid_mask_est = ~np.isnan(pose_3d_estimated).any(axis=1)
    valid_mask_gt = ~np.isnan(pose_3d_gt).any(axis=1)
    valid_mask = valid_mask_est & valid_mask_gt

    if not np.any(valid_mask):
        logging.warning("Cannot evaluate 3D pose: No commonly valid keypoints found after triangulation.")
        return np.nan

    # Calculate Euclidean distance for valid keypoints
    errors = np.linalg.norm(pose_3d_estimated[valid_mask] - pose_3d_gt[valid_mask], axis=1)
    mpjpe = np.mean(errors)
    logging.info(f"Calculated 3D MPJPE: {mpjpe:.4f} (world units) ({valid_mask.sum()} valid points)")
    return mpjpe

# --- 4. Fine-tuning YOLO-Pose ---

def finetune_yolo_pose(data_yaml_path: str,
                       base_model_path: str,
                       epochs: int = 50,
                       batch_size: int = 16,
                       project_name: str = 'hpe_finetune_project',
                       run_name: str = 'run1',
                       device: Optional[Union[str, int]] = None) -> Optional[str]:
    """
    Initiates the fine-tuning process for a YOLOv8-Pose model using Ultralytics API.

    Args:
        data_yaml_path (str): Path to the dataset YAML file (e.g., generated by Roboflow).
                              This file defines paths to train/val images and label structure.
        base_model_path (str): Path to the pre-trained model weights (.pt file) to start from.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size. Adjust based on available memory.
        project_name (str): Name for the parent directory where training runs will be saved.
        run_name (str): Specific name for this training run within the project directory.
        device (Optional[Union[str, int]]): Device to run training on ('cpu', 0, '0,1', etc.).
                                             Defaults to None (auto-select).

    Returns:
        Optional[str]: Path to the best trained model weights ('best.pt') if training completes,
                       otherwise None.
    """
    if not os.path.exists(data_yaml_path):
        logging.error(f"Fine-tuning failed: Dataset YAML file not found at {data_yaml_path}")
        return None
    if not os.path.exists(base_model_path):
        logging.error(f"Fine-tuning failed: Base model file not found at {base_model_path}")
        return None

    logging.info(f"Starting fine-tuning from model: {base_model_path}")
    logging.info(f"Using dataset config: {data_yaml_path}")
    logging.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, device={device or 'auto'}")

    try:
        model = YOLO(base_model_path)

        # Note: GPU is highly recommended. Training on CPU will be extremely slow.
        # The project guide mentioned no GPUs provided by default.
        if device is None and cv2.cuda.getCudaEnabledDeviceCount() == 0:
             logging.warning("No GPU detected or specified. Fine-tuning will proceed on CPU and may be very slow.")

        # Start training
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            project=project_name,
            name=run_name,
            device=device,
            exist_ok=False # Prevent overwriting previous runs with the same name
            # Add other relevant ultralytics train() arguments as needed
        )

        # Training completion is implicit if no exception occurs
        logging.info("Fine-tuning process completed.")
        # Construct the expected path to the best model weights
        best_model_path = os.path.join(project_name, run_name, 'weights', 'best.pt')

        if os.path.exists(best_model_path):
            logging.info(f"Best model weights saved to: {best_model_path}")
            return best_model_path
        else:
            logging.error(f"Fine-tuning finished, but best model weights not found at expected location: {best_model_path}")
            return None

    except Exception as e:
        logging.error(f"An error occurred during the fine-tuning process: {e}", exc_info=True)
        return None

# --- Utility Function for Frame Extraction ---
def extract_frame(video_path: str, frame_number: int, output_path: str) -> bool:
    """
    Extracts a specific frame from a video file and saves it as an image.

    Args:
        video_path (str): Path to the input video file.
        frame_number (int): The 0-based index of the frame to extract.
        output_path (str): Path to save the extracted frame image (e.g., .png).

    Returns:
        bool: True if the frame was successfully extracted and saved, False otherwise.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return False

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        logging.error(f"Frame number {frame_number} is out of bounds for video {video_path} (Total frames: {total_frames})")
        cap.release()
        return False

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            logging.debug(f"Successfully extracted frame {frame_number} from {os.path.basename(video_path)} to {output_path}")
            cap.release()
            return True
        except Exception as e:
            logging.error(f"Failed to write frame image to {output_path}: {e}")
            cap.release()
            return False
    else:
        logging.error(f"Failed to read frame {frame_number} from video: {video_path}")
        cap.release()
        return False


# --- Main Execution Logic ---
def main():
    """Main function to execute the pose estimation pipeline."""
    logging.info("--- Starting Multi-view Pose Estimation Pipeline ---")

    # 1. Load Camera Parameters
    logging.info("--- Step 1: Loading Camera Parameters ---")
    camera_params = load_camera_parameters(CAMERA_PARAM_FILE)
    if not camera_params:
        logging.error("Exiting: Failed to load camera parameters.")
        return # Exit script

    # --- Define Frame to Process ---
    # Note: Frame numbers in annotations/filenames might be 1-based or 0-based.
    # Ensure consistency with how `load_annotations` and `extract_frame` expect them.
    # The current `load_annotations` assumes 1-based frame numbers in filenames.
    # `extract_frame` uses 0-based indexing for cv2.CAP_PROP_POS_FRAMES.
    target_frame_for_files = TARGET_FRAME # Use this for filenames (assumed 1-based)
    target_frame_for_extraction = TARGET_FRAME - 1 # Use this for cv2 (0-based)

    # --- Step 2: Triangulation from Ground Truth ---
    logging.info(f"\n--- Step 2: Triangulation (Ground Truth Frame {target_frame_for_files}) ---")
    gt_annotations_frame = load_annotations(ANNOTATION_FOLDER, target_frame_for_files, NUM_CAMERAS, NUM_KEYPOINTS)

    pose_3d_gt: Optional[np.ndarray] = None # Initialize
    if not any(gt_annotations_frame.values()):
         logging.warning(f"No Ground Truth annotations found for frame {target_frame_for_files}. Cannot perform GT triangulation or evaluation.")
    else:
        pose_3d_gt = triangulate_points_linear(gt_annotations_frame, camera_params)
        if pose_3d_gt is not None:
            valid_gt_kpts = (~np.isnan(pose_3d_gt)).any(axis=1).sum()
            logging.info(f"Ground Truth 3D Pose calculated ({valid_gt_kpts}/{NUM_KEYPOINTS} valid keypoints).")
            # logging.debug("Ground Truth 3D Pose (first 5 keypoints):\n%s", pose_3d_gt[:5])

            # --- Step 2a: Evaluate Triangulation Accuracy ---
            logging.info("\n--- Step 2a: Reprojection Error (Ground Truth) ---")
            reprojection_errors_gt = calculate_reprojection_error(pose_3d_gt, gt_annotations_frame, camera_params)
            logging.info("Mean Reprojection Errors (pixels): %s", reprojection_errors_gt)
        else:
            logging.error("Failed to triangulate ground truth points.")


    # --- Step 3: Run Pre-trained YOLO-Pose ---
    logging.info(f"\n--- Step 3: Run Pre-trained YOLOv8-Pose (Frame {target_frame_for_files}) ---")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load YOLO model from {YOLO_MODEL_PATH}: {e}")
        return # Exit script

    yolo_poses_2d: Dict[str, Optional[np.ndarray]] = {}
    avg_2d_errors_yolo: List[float] = []
    temp_frame_dir = os.path.join(OUTPUT_FOLDER, "temp_frames") # Dir to store extracted frames

    for i in range(1, NUM_CAMERAS + 1):
        cam_id = f'cam{i}'
        logging.info(f"Processing {cam_id}...")

        # Construct paths for video and extracted frame
        # TODO: Adapt video filename pattern if needed (e.g., video_cam1.avi, cam1_action.mp4)
        video_file = os.path.join(VIDEO_FOLDER, f"video_{cam_id}.mp4")
        frame_output_path = os.path.join(temp_frame_dir, f"frame_{target_frame_for_files:03d}_{cam_id}.png")

        # Extract the frame
        if not extract_frame(video_file, target_frame_for_extraction, frame_output_path):
            logging.warning(f"Could not extract frame {target_frame_for_extraction} for {cam_id}. Skipping pose estimation.")
            yolo_poses_2d[cam_id] = None
            continue

        # Run pose estimation on the extracted frame
        estimated_pose_2d = run_pose_estimation(yolo_model, frame_output_path, NUM_KEYPOINTS)
        yolo_poses_2d[cam_id] = estimated_pose_2d

        # Evaluate 2D Pose Estimation (if GT available for this view)
        gt_pose_2d_cam = gt_annotations_frame.get(cam_id)
        if gt_pose_2d_cam is not None and estimated_pose_2d is not None:
            error_2d = evaluate_2d_pose(estimated_pose_2d, gt_pose_2d_cam)
            if not np.isnan(error_2d):
                logging.info(f"  {cam_id}: YOLO 2D Pose Error (vs GT): {error_2d:.4f} pixels")
                avg_2d_errors_yolo.append(error_2d)
            else:
                logging.info(f"  {cam_id}: Could not calculate 2D pose error (likely no valid points).")
        elif gt_pose_2d_cam is None:
             logging.debug(f"  {cam_id}: No GT annotation available for 2D evaluation.")
        else: # estimated_pose_2d is None
             logging.info(f"  {cam_id}: No pose estimated by YOLO for 2D evaluation.")

    if avg_2d_errors_yolo:
        logging.info(f"Average Pre-trained YOLO 2D Pose Error across views: {np.mean(avg_2d_errors_yolo):.4f} pixels")


    # --- Step 3a: Triangulate YOLO Poses & Evaluate 3D ---
    logging.info("\n--- Step 3a: Triangulation & Evaluation (YOLO Pose) ---")
    pose_3d_yolo: Optional[np.ndarray] = None # Initialize
    if not any(yolo_poses_2d.values()):
        logging.warning("Cannot triangulate YOLO poses: No valid poses detected in at least two views.")
    else:
        pose_3d_yolo = triangulate_points_linear(yolo_poses_2d, camera_params)
        if pose_3d_yolo is not None:
            valid_yolo_kpts = (~np.isnan(pose_3d_yolo)).any(axis=1).sum()
            logging.info(f"YOLO Estimated 3D Pose calculated ({valid_yolo_kpts}/{NUM_KEYPOINTS} valid keypoints).")
            # logging.debug("YOLO Estimated 3D Pose (first 5 keypoints):\n%s", pose_3d_yolo[:5])

            # Evaluate 3D MPJPE (requires GT 3D pose from Step 2)
            if pose_3d_gt is not None:
                mpjpe_yolo = evaluate_3d_pose_mpjpe(pose_3d_yolo, pose_3d_gt)
                if not np.isnan(mpjpe_yolo):
                    logging.info(f"YOLO 3D Pose MPJPE (vs GT Triangulation): {mpjpe_yolo:.4f} (world units)")
                else:
                    logging.warning("Could not calculate 3D MPJPE for YOLO pose.")
            else:
                logging.warning("Cannot calculate 3D MPJPE: Ground Truth 3D pose was not successfully generated.")

            # Optional: Calculate reprojection error for YOLO's triangulated points
            # logging.info("\n--- Reprojection Error (YOLO Triangulation) ---")
            # reprojection_errors_yolo = calculate_reprojection_error(pose_3d_yolo, yolo_poses_2d, camera_params)
            # logging.info("YOLO Mean Reprojection Errors (pixels): %s", reprojection_errors_yolo)

        else:
            logging.error("Failed to triangulate YOLO poses.")


    # --- Step 4: Fine-tuning (Example Call - Uncomment to Run) ---
    # logging.info("\n--- Step 4: Fine-tuning YOLOv8-Pose (Optional) ---")
    # # NOTE: This requires preparing your Roboflow dataset export and its data.yaml file.
    # data_yaml = 'path/to/your/roboflow_dataset/data.yaml' # IMPORTANT: Set this path
    # if os.path.exists(data_yaml):
    #     logging.info("Initiating fine-tuning process (this may take a long time)...")
    #     # Use fewer epochs for testing, significantly more for actual training (e.g., 50-100+)
    #     # Specify device=0 for GPU 0, device='cpu', or leave as None for auto-detect
    #     best_model_path = finetune_yolo_pose(data_yaml,
    #                                          YOLO_MODEL_PATH,
    #                                          epochs=5, # Low epoch count for quick test
    #                                          batch_size=8, # Adjust based on memory
    #                                          project_name='hpe_finetuning_output',
    #                                          run_name='initial_test_run',
    #                                          device=None) # Or set to 0, 'cpu', etc.
    #     if best_model_path:
    #         logging.info(f"Fine-tuning completed. Best model saved at: {best_model_path}")
    #         # TODO: Add evaluation logic here using the fine-tuned model
    #         # 1. Load the fine-tuned model: fine_tuned_model = YOLO(best_model_path)
    #         # 2. Re-run Step 3 and 3a using fine_tuned_model
    #         # 3. Compare 2D/3D errors with the pre-trained model's results
    #         FINETUNED_YOLO_PATH = best_model_path # Update constant for potential later use
    #     else:
    #         logging.error("Fine-tuning process failed or did not produce a model.")
    # else:
    #      logging.warning(f"Skipping fine-tuning: Dataset YAML not found at {data_yaml}")


    # --- Step 4a: Unreal Engine Data Export ---
    logging.info("\n--- Step 4a: Data Export for Visualization ---")
    # Export the 3D pose data (e.g., from YOLO triangulation) in a format
    # suitable for external visualization tools like Unreal Engine, Blender, etc.
    # Common formats include CSV or JSON.
    if pose_3d_yolo is not None:
        output_csv_path = os.path.join(OUTPUT_FOLDER, f'pose3d_yolo_frame_{target_frame_for_files}.csv')
        try:
            # Save only valid (non-NaN) keypoints if desired, or save all including NaNs
            # np.savetxt(output_csv_path, pose_3d_yolo[~np.isnan(pose_3d_yolo).any(axis=1)], delimiter=',')
            np.savetxt(output_csv_path, pose_3d_yolo, delimiter=',', fmt='%.6f', header='X,Y,Z', comments='')
            logging.info(f"Saved YOLO 3D pose for frame {target_frame_for_files} to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save 3D pose data to CSV: {e}")
    else:
        logging.warning("Skipping 3D pose export: YOLO 3D pose data is not available.")

    # Clean up temporary frame directory if it was created
    # if os.path.exists(temp_frame_dir):
    #     try:
    #         import shutil
    #         shutil.rmtree(temp_frame_dir)
    #         logging.info(f"Removed temporary frame directory: {temp_frame_dir}")
    #     except Exception as e:
    #         logging.warning(f"Could not remove temporary frame directory {temp_frame_dir}: {e}")


    logging.info("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
