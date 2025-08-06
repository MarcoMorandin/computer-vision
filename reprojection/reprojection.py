import os
import json
import numpy as np
import cv2
import re
from datetime import datetime
from tqdm import tqdm
import argparse
import copy
from pycocotools.coco import COCO

class SkeletonReprojector:

    def __init__(self, triangulation_file: str, camera_data_dir: str, rectified_dataset_dir: str,
                 original_coco_file: str, output_dir: str, camera_version: str = "v3"):
        """
        Initialize the reprojector.

        Args:
            triangulation_file: Path to the 3D triangulated poses JSON file.
            camera_data_dir: Directory containing camera calibration data.
            rectified_dataset_dir: Directory containing rectified images.
            original_coco_file: Path to original COCO annotations for metadata.
            output_dir: Directory to save the reprojected COCO dataset.
            camera_version: Version of camera data to use (v1, v2, v3).
        """
        self.triangulation_file = triangulation_file
        self.camera_data_dir = os.path.join(camera_data_dir, f"camera_data_{camera_version}")
        self.rectified_dataset_dir = rectified_dataset_dir
        self.original_coco_file = original_coco_file
        self.output_dir = output_dir
        self.camera_version = camera_version

        # Load data
        self.poses_3d = self._load_3d_poses()
        self.cameras = self._load_camera_calibrations()
        self.rectified_images = self._get_rectified_images()
        
        # Load original COCO dataset and prepare for modifications
        self._prepare_coco_dataset()

    def _prepare_coco_dataset(self):
        """Load the original COCO dataset and create a copy to be modified."""
        print(f"Loading original COCO structure from: {self.original_coco_file}")
        coco = COCO(self.original_coco_file)
        self.coco_dataset = copy.deepcopy(coco.dataset)
        
        # Keep the original structure but prepare to modify it
        # Create lookup dictionaries for fast access
        self.image_lookup = {img['id']: img for img in self.coco_dataset['images']}
        self.annotation_lookup = {}
        for ann in self.coco_dataset['annotations']:
            if ann['image_id'] not in self.annotation_lookup:
                self.annotation_lookup[ann['image_id']] = []
            self.annotation_lookup[ann['image_id']].append(ann)

        # Update info
        self.coco_dataset['info']['description'] = "Reprojected 3D skeleton data to rectified 2D camera views"
        self.coco_dataset['info']['year'] = str(datetime.now().year)
        self.coco_dataset['info']['date_created'] = datetime.now().isoformat()
        
        person_category = coco.loadCats(coco.getCatIds(catNms=['person']))[0]
        self.keypoint_names = person_category['keypoints']
        self.skeleton_connections = person_category['skeleton']
        
        print(f"Loaded {len(self.keypoint_names)} keypoints and {len(self.skeleton_connections)} skeleton connections")

    def _get_rectified_images(self) -> dict:
        """Get list of available rectified images."""
        rectified_images = {}
        if not os.path.exists(self.rectified_dataset_dir):
            raise FileNotFoundError(f"Rectified dataset directory not found: {self.rectified_dataset_dir}")

        for filename in os.listdir(self.rectified_dataset_dir):
            if filename.endswith('.jpg'):
                camera_id, frame_num = self._parse_image_filename(filename)
                if camera_id is not None and frame_num is not None:
                    if camera_id not in rectified_images:
                        rectified_images[camera_id] = {}
                    rectified_images[camera_id][frame_num] = filename
        
        print(f"Found rectified images for cameras: {sorted(rectified_images.keys())}")
        return rectified_images

    def _parse_image_filename(self, filename: str) -> tuple:
        """Parse image filename to extract camera ID and frame number."""
        # Try the standard format first
        match = re.match(r'out(\d+)_frame_(\d+)_png\.rf\.[a-f0-9]+\.jpg', filename)
        if match:
            return match.group(1), int(match.group(2))
            
        # Try alternative formats
        cam_match = re.search(r'(?:out|cam)(\d+)', filename)
        frame_match = re.search(r'frame_(\d+)', filename)
        
        if cam_match and frame_match:
            return cam_match.group(1), int(frame_match.group(1))
            
        return None, None

    def _load_3d_poses(self) -> dict:
        """Load 3D triangulated poses from JSON file."""
        print(f"Loading 3D poses from: {self.triangulation_file}")
        with open(self.triangulation_file, 'r') as f:
            return json.load(f)

    def _load_camera_calibrations(self) -> dict:
        """Load camera calibration parameters."""
        cameras = {}
        print(f"Loading camera calibrations from: {self.camera_data_dir}")
        for cam_dir in sorted(os.listdir(self.camera_data_dir)):
            if os.path.isdir(os.path.join(self.camera_data_dir, cam_dir)) and cam_dir.startswith('cam_'):
                cam_num = cam_dir.split('_')[1]
                calib_file = os.path.join(self.camera_data_dir, cam_dir, 'calib', 'camera_calib.json')
                if os.path.exists(calib_file):
                    with open(calib_file, 'r') as f:
                        calib = json.load(f)
                    cameras[cam_num] = {
                        'mtx': np.array(calib["mtx"], dtype=np.float32),
                        'dist': np.array(calib["dist"], dtype=np.float32).flatten(),
                        'rvec': np.array(calib["rvecs"], dtype=np.float32).flatten(),
                        'tvec': np.array(calib["tvecs"], dtype=np.float32).flatten()
                    }
                    print(f"Loaded calibration for camera {cam_num}")
                else:
                    print(f"Warning: Calibration file not found for camera {cam_num}")
        return cameras

    def _project_3d_to_2d(self, points_3d: np.ndarray, camera_params: dict) -> np.ndarray:
        """Project 3D points to 2D using camera parameters."""
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

    def _calculate_bbox_from_keypoints(self, keypoints_2d: np.ndarray, visibility: list) -> list:
        """Calculate bounding box from visible keypoints."""
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

    def reproject_all_frames(self):
        """Reproject all 3D poses to 2D for all available images while preserving the original dataset structure."""
        print("Starting reprojection process...")
        
        # Initialize the mapping between original images and rectified files
        # We'll use file_name to identify the camera and frame
        image_mapping = {}
        updated_images_count = 0
        updated_annotations_count = 0
        
        # Find matching rectified images for each original image
        for image in self.coco_dataset["images"]:
            file_name = image["file_name"]
            camera_id = self._parse_image_filename(file_name)[0]
            frame_num = self._parse_image_filename(file_name)[1]
            
            if camera_id and frame_num:
                if camera_id in self.rectified_images and frame_num in self.rectified_images[camera_id]:
                    image_mapping[image["id"]] = (camera_id, frame_num)
        
        # Process each image that has a match
        for image_id, (camera_id, frame_num) in tqdm(image_mapping.items(), desc="Reprojecting frames"):
            # Get the 3D pose data for this frame
            frame_str = str(frame_num)
            if frame_str not in self.poses_3d:
                continue
                
            pose_3d = np.array(self.poses_3d[frame_str], dtype=np.float32)
            camera_params = self.cameras[camera_id]
            
            # Get original image dimensions
            image = self.image_lookup[image_id]
            image_width, image_height = image["width"], image["height"]
            
            # Project 3D points to 2D
            keypoints_2d = self._project_3d_to_2d(pose_3d, camera_params)
            visibility = [2 if 0 <= p[0] <= image_width and 0 <= p[1] <= image_height else 0 for p in keypoints_2d]
            
            # Skip if no points are visible
            if not any(v > 0 for v in visibility):
                continue
                
            # Convert to COCO format
            coco_keypoints = []
            for i, (kp, v) in enumerate(zip(keypoints_2d, visibility)):
                coco_keypoints.extend([float(kp[0]), float(kp[1]), v])
            
            # Update existing annotations or create a new one if needed
            if image_id in self.annotation_lookup and self.annotation_lookup[image_id]:
                for ann in self.annotation_lookup[image_id]:
                    ann["keypoints"] = coco_keypoints
                    # Update bounding box based on keypoints
                    bbox = self._calculate_bbox_from_keypoints(keypoints_2d, visibility)
                    ann["bbox"] = bbox
                    ann["area"] = bbox[2] * bbox[3]
                    updated_annotations_count += 1
            else:
                # Create a new annotation if none exists
                self.annotation_id = max([ann["id"] for ann in self.coco_dataset["annotations"]], default=0) + 1
                bbox = self._calculate_bbox_from_keypoints(keypoints_2d, visibility)
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming 'person' is category_id 1
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                    "iscrowd": 0,
                    "keypoints": coco_keypoints,
                }
                self.coco_dataset["annotations"].append(annotation)
                
                if image_id not in self.annotation_lookup:
                    self.annotation_lookup[image_id] = []
                self.annotation_lookup[image_id].append(annotation)
                updated_annotations_count += 1
            
            updated_images_count += 1

        print("Reprojection complete.")
        print(f"  Total images processed: {updated_images_count}")
        print(f"  Total annotations updated: {updated_annotations_count}")

    def save_coco_dataset(self, output_filename: str = "reprojected_annotations.json"):
        """Save the modified COCO dataset to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Clean up temporary lookup dictionaries before saving
        if hasattr(self, 'image_lookup'):
            del self.image_lookup
        if hasattr(self, 'annotation_lookup'):
            del self.annotation_lookup
            
        with open(output_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=4)
        
        print(f"Reprojected COCO dataset saved to: {output_path}")

def main():
    reprojector = SkeletonReprojector(
        triangulation_file="../triangulation/output/player_3d_poses.json",
        camera_data_dir="../data",
        rectified_dataset_dir="../rectification/rectified/dataset/train",
        original_coco_file="../data/dataset/train/_annotations.coco.json",
        output_dir="./output",
        camera_version="v2"
    )
    
    reprojector.reproject_all_frames()
    reprojector.save_coco_dataset("reprojected_annotations.json")
    
    print("\nReprojection completed successfully!")

if __name__ == "__main__":
    main()