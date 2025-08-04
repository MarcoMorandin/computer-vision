#!/usr/bin/env python3
"""
Complete COCO Dataset Rectification Script

This script rectifies both training images and their corresponding COCO annotations
using camera calibration data, producing a complete rectified COCO format dataset.
"""

import json
import os
import glob
import re
from typing import List, Tuple
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_calibration(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera calibration parameters from a JSON file.
    
    Args:
        calib_path: Path to the calibration JSON file
        
    Returns:
        Tuple of (camera_matrix, distortion_coefficients)
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def extract_camera_index(filename: str) -> str:
    """
    Extract camera index from filename (e.g., 'out1_frame_001.jpg' -> '1')
    
    Args:
        filename: Image filename
        
    Returns:
        Camera index as string
    """
    match = re.search(r'out(\d+)_', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract camera index from filename: {filename}")

def rectify_keypoints(keypoints: List[float], mtx: np.ndarray, dist: np.ndarray) -> List[float]:
    """
    Rectify keypoints using camera calibration parameters.
    
    Args:
        keypoints: List of keypoints in COCO format [x1, y1, v1, x2, y2, v2, ...]
        mtx: Camera matrix
        dist: Distortion coefficients
        
    Returns:
        Rectified keypoints in the same format
    """
    if not keypoints or len(keypoints) == 0:
        return keypoints
    
    # Reshape keypoints to extract coordinates and visibility
    keypoints_array = np.array(keypoints).reshape(-1, 3)
    
    # Extract x, y coordinates (skip visibility values)
    points_2d = keypoints_array[:, :2]
    visibility = keypoints_array[:, 2]
    
    # Filter out invalid keypoints (visibility = 0)
    valid_mask = visibility > 0
    if not np.any(valid_mask):
        return keypoints
    
    valid_points = points_2d[valid_mask]
    
    # Reshape for cv2.undistortPoints (needs shape (N, 1, 2))
    points_cv = valid_points.reshape(-1, 1, 2).astype(np.float32)
    
    # Apply undistortion
    undistorted_points = cv2.undistortPoints(points_cv, mtx, dist, P=mtx)
    undistorted_points = undistorted_points.reshape(-1, 2)
    
    # Update the valid points with rectified coordinates
    rectified_keypoints = keypoints_array.copy()
    rectified_keypoints[valid_mask, :2] = undistorted_points
    
    # Flatten back to original format and convert to regular Python floats
    return [float(x) for x in rectified_keypoints.flatten().tolist()]

def rectify_bbox(bbox: List[float], mtx: np.ndarray, dist: np.ndarray) -> List[float]:
    """
    Rectify bounding box using camera calibration parameters.
    
    Args:
        bbox: Bounding box in COCO format [x, y, width, height]
        mtx: Camera matrix
        dist: Distortion coefficients
        
    Returns:
        Rectified bounding box in the same format
    """
    x, y, width, height = bbox
    
    # Define the four corners of the bounding box
    corners = np.array([
        [x, y],                    # top-left
        [x + width, y],            # top-right
        [x, y + height],           # bottom-left
        [x + width, y + height]    # bottom-right
    ], dtype=np.float32)
    
    # Reshape for cv2.undistortPoints
    corners_cv = corners.reshape(-1, 1, 2)
    
    # Apply undistortion
    undistorted_corners = cv2.undistortPoints(corners_cv, mtx, dist, P=mtx)
    undistorted_corners = undistorted_corners.reshape(-1, 2)
    
    # Calculate new bounding box from rectified corners
    min_x = np.min(undistorted_corners[:, 0])
    max_x = np.max(undistorted_corners[:, 0])
    min_y = np.min(undistorted_corners[:, 1])
    max_y = np.max(undistorted_corners[:, 1])
    
    new_width = max_x - min_x
    new_height = max_y - min_y
    
    return [float(min_x), float(min_y), float(new_width), float(new_height)]

def rectify_image_with_map(image_path: str, map_x: np.ndarray, map_y: np.ndarray, output_path: str) -> bool:
    """
    Rectify a single image using pre-computed distortion maps.
    
    Args:
        image_path: Path to the input image
        map_x: Pre-computed x distortion map
        map_y: Pre-computed y distortion map
        output_path: Path to save the rectified image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return False
        
        # Apply the undistortion map to the image (same as rectified_videos.py)
        rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the rectified image
        success = cv2.imwrite(output_path, rectified_img)
        if not success:
            print(f"Error: Could not save rectified image to {output_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error rectifying image {image_path}: {e}")
        return False

def compute_distortion_map(mtx: np.ndarray, dist: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distortion maps using the same procedure as rectified_videos.py
    
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (map_x, map_y)
    """
    # Create a grid of all pixel coordinates (same as rectified_videos.py)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    
    # Apply undistortion to get the mapping
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]
    
    return map_x, map_y

def process_coco_dataset(dataset_dir: str, camera_data_dir: str, output_dir: str) -> bool:
    """
    Process a complete COCO dataset (images + annotations) with rectification.
    
    Args:
        dataset_dir: Directory containing the COCO dataset
        camera_data_dir: Directory containing camera calibration data
        output_dir: Directory to save the rectified dataset
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Processing COCO dataset: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all camera calibrations
    calibrations = {}
    print("\nLoading camera calibrations...")
    
    for cam_dir in glob.glob(os.path.join(camera_data_dir, "cam_*")):
        cam_name = os.path.basename(cam_dir)
        cam_index = cam_name.split('_')[1]
        calib_path = os.path.join(cam_dir, "calib", "camera_calib.json")
        
        if os.path.exists(calib_path):
            try:
                mtx, dist = load_calibration(calib_path)
                calibrations[cam_index] = (mtx, dist)
                print(f"  ✓ Loaded calibration for camera {cam_index}")
            except Exception as e:
                print(f"  ✗ Error loading calibration for camera {cam_index}: {e}")
        else:
            print(f"  ⚠ Warning: Calibration file not found for camera {cam_index}: {calib_path}")
    
    if not calibrations:
        print("Error: No camera calibrations loaded!")
        return False
    
    # Find all COCO annotation files
    annotation_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json') and 'coco' in file.lower():
                annotation_files.append(os.path.join(root, file))
    
    if not annotation_files:
        print("No COCO annotation files found!")
        return False
    
    print(f"\nFound {len(annotation_files)} COCO annotation files:")
    for file in annotation_files:
        rel_path = os.path.relpath(file, dataset_dir)
        print(f"  {rel_path}")
    
    # Process each annotation file and its corresponding images
    total_images_processed = 0
    total_annotations_processed = 0
    
    for annotations_path in annotation_files:
        print(f"\n{'='*80}")
        rel_path = os.path.relpath(annotations_path, dataset_dir)
        print(f"Processing: {rel_path}")
        print(f"{'='*80}")
        
        # Load COCO annotations
        try:
            with open(annotations_path, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            continue
        
        # Get the directory containing the images for this annotation file
        images_dir = os.path.dirname(annotations_path)
        
        # Find all images referenced in the annotations
        image_filenames = {img['file_name'] for img in coco_data['images']}
        
        # Find actual image files in the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        actual_images = []
        for ext in image_extensions:
            actual_images.extend(glob.glob(os.path.join(images_dir, ext)))
        
        # Filter to only process images that are referenced in annotations
        images_to_process = []
        for image_path in actual_images:
            filename = os.path.basename(image_path)
            if filename in image_filenames:
                images_to_process.append(image_path)
        
        if not images_to_process:
            print(f"No matching images found for annotations in {images_dir}")
            continue
        
        print(f"Found {len(images_to_process)} images to process")
        
        # Pre-compute distortion maps for each camera
        distortion_maps = {}
        print("Pre-computing distortion maps...")
        
        # Get image dimensions from the first image of each camera
        for image_path in images_to_process[:10]:  # Check first 10 images to find different cameras
            try:
                filename = os.path.basename(image_path)
                cam_index = extract_camera_index(filename)
                
                if cam_index in calibrations and cam_index not in distortion_maps:
                    img = cv2.imread(image_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        mtx, dist = calibrations[cam_index]
                        map_x, map_y = compute_distortion_map(mtx, dist, width, height)
                        distortion_maps[cam_index] = (map_x, map_y, width, height)
                        print(f"  ✓ Pre-computed distortion map for camera {cam_index} ({width}x{height})")
            except (ValueError, Exception):
                continue
        
        # Process images
        print("Rectifying images...")
        images_successful = 0
        images_failed = 0
        
        for i, image_path in enumerate(images_to_process):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(images_to_process)} images processed")
            
            try:
                filename = os.path.basename(image_path)
                cam_index = extract_camera_index(filename)
                
                if cam_index not in calibrations:
                    print(f"Warning: No calibration data for camera {cam_index} (file: {filename})")
                    images_failed += 1
                    continue
                
                # Determine output path
                rel_image_path = os.path.relpath(image_path, dataset_dir)
                output_image_path = os.path.join(output_dir, rel_image_path)
                
                # Rectify the image
                if cam_index in distortion_maps:
                    map_x, map_y, expected_width, expected_height = distortion_maps[cam_index]
                    
                    # Verify image dimensions
                    img = cv2.imread(image_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        if width == expected_width and height == expected_height:
                            # Use pre-computed map
                            if rectify_image_with_map(image_path, map_x, map_y, output_image_path):
                                images_successful += 1
                            else:
                                images_failed += 1
                        else:
                            # Different dimensions, compute new map
                            mtx, dist = calibrations[cam_index]
                            map_x, map_y = compute_distortion_map(mtx, dist, width, height)
                            if rectify_image_with_map(image_path, map_x, map_y, output_image_path):
                                images_successful += 1
                            else:
                                images_failed += 1
                    else:
                        images_failed += 1
                else:
                    # No pre-computed map, compute on-the-fly
                    img = cv2.imread(image_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        mtx, dist = calibrations[cam_index]
                        map_x, map_y = compute_distortion_map(mtx, dist, width, height)
                        if rectify_image_with_map(image_path, map_x, map_y, output_image_path):
                            images_successful += 1
                        else:
                            images_failed += 1
                    else:
                        images_failed += 1
                        
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                images_failed += 1
                continue
        
        print(f"Images: {images_successful} successful, {images_failed} failed")
        total_images_processed += images_successful
        
        # Process annotations
        print("Rectifying annotations...")
        rectified_annotations = []
        annotation_errors = 0
        
        for i, annotation in enumerate(coco_data['annotations']):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(coco_data['annotations'])} annotations processed")
            
            try:
                # Get the corresponding image
                image_id = annotation['image_id']
                image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
                
                if image_info is None:
                    annotation_errors += 1
                    continue
                
                filename = image_info['file_name']
                cam_index = extract_camera_index(filename)
                
                if cam_index not in calibrations:
                    annotation_errors += 1
                    continue
                
                mtx, dist = calibrations[cam_index]
                
                # Create a copy of the annotation
                rectified_annotation = annotation.copy()
                
                # Rectify keypoints if they exist
                if 'keypoints' in annotation and annotation['keypoints']:
                    rectified_keypoints = rectify_keypoints(annotation['keypoints'], mtx, dist)
                    rectified_annotation['keypoints'] = rectified_keypoints
                
                # Rectify bounding box
                if 'bbox' in annotation:
                    rectified_bbox = rectify_bbox(annotation['bbox'], mtx, dist)
                    rectified_annotation['bbox'] = rectified_bbox
                    
                    # Recalculate area
                    rectified_annotation['area'] = float(rectified_bbox[2] * rectified_bbox[3])
                
                rectified_annotations.append(rectified_annotation)
                
            except Exception as e:
                print(f"Error processing annotation {annotation.get('id', 'unknown')}: {e}")
                annotation_errors += 1
                continue
        
        # Create the rectified COCO dataset
        rectified_coco = coco_data.copy()
        rectified_coco['annotations'] = rectified_annotations
        
        # Save rectified annotations
        output_annotations_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_annotations_path), exist_ok=True)
        
        try:
            with open(output_annotations_path, 'w') as f:
                json.dump(rectified_coco, f, indent=2)
            
            print(f"Annotations: {len(rectified_annotations)} successful, {annotation_errors} failed")
            total_annotations_processed += len(rectified_annotations)

            
        except Exception as e:
            print(f"Error saving rectified annotations: {e}")
            continue
        
        # Copy other files (non-image, non-annotation files)
        print("Copying other files...")
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if not (file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.json'))):
                    source_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(source_path, dataset_dir)
                    dest_path = os.path.join(output_dir, rel_file_path)
                    
                    try:
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(source_path, dest_path)
                    except Exception as e:
                        print(f"  Warning: Could not copy {rel_file_path}: {e}")
    
    print(f"\n{'='*80}")
    print("COCO Dataset Rectification Complete!")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total annotations processed: {total_annotations_processed}")
    print(f"Rectified dataset saved to: {output_dir}")
    print(f"{'='*80}")
    
    return total_images_processed > 0 and total_annotations_processed > 0

def main():
    """
    Main function to rectify complete COCO datasets.
    """
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    annotations_dir = os.path.join(data_dir, "annotations")
    camera_data_dir = os.path.join(data_dir, "camera_data")
    rectified_dir = os.path.join(base_dir, "rectified")
    
    print("="*80)
    print("COCO Dataset Rectification Tool")
    print("="*80)
    print(f"Input dataset directory: {annotations_dir}")
    print(f"Camera calibration directory: {camera_data_dir}")
    print(f"Output directory: {rectified_dir}")
    
    # Check if directories exist
    if not os.path.exists(annotations_dir):
        print(f"Error: Dataset directory not found: {annotations_dir}")
        return
    
    if not os.path.exists(camera_data_dir):
        print(f"Error: Camera calibration directory not found: {camera_data_dir}")
        return
    
    # Create output directory
    os.makedirs(rectified_dir, exist_ok=True)
    
    # Process the dataset
    try:
        success = process_coco_dataset(annotations_dir, camera_data_dir, rectified_dir)
        if success:
            print("\n✓ Dataset rectification completed successfully!")
        else:
            print("\n✗ Dataset rectification failed!")
    except Exception as e:
        print(f"\n✗ Error during dataset rectification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
