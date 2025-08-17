import cv2
import numpy as np
import os
import re
import glob
from typing import List, Dict, Any, Optional
import sys

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .utils.camera.camera_manager import CameraManager
from .utils.dataset.coco_utils import COCOManager


class DatasetRectifier:
    """
    A class for rectifying COCO datasets or videos by undistorting images/frames
    using camera calibration parameters.
    """
    
    def __init__(self, camera_manager: CameraManager, coco_dataset: Optional[COCOManager] = None, 
                 mode: str = "coco"):
        """
        Initialize the DatasetRectifier.
        
        Args:
            camera_manager: Initialized CameraManager object
            coco_dataset: Initialized COCODataset object (required if mode="coco")
            mode: Either "coco" for dataset rectification or "video" for video rectification
        """
        self.camera_manager = camera_manager
        self.mode = mode.lower()
        
        if self.mode == "coco":
            if coco_dataset is None:
                raise ValueError("coco_dataset is required when mode='coco'")
            self.coco_dataset = coco_dataset
        elif self.mode == "video":
            self.coco_dataset = None
        else:
            raise ValueError("mode must be either 'coco' or 'video'")
        
    def _extract_camera_index(self, file_name: str) -> str:
        """Extract camera index from filename."""
        match = re.search(r'(?:out|cam)(\d+)', file_name)
        if not match:
            raise ValueError(f"Cannot extract camera index from: {file_name}")
        return match.group(1)
    
    def _rectify_keypoints(self, keypoints: List[float], mtx: np.ndarray, dist: np.ndarray) -> List[float]:
        """Rectify keypoints."""
        num_kpts = len(keypoints) // 3
        new_keypoints = keypoints.copy()
        
        # Collect visible keypoints
        visible_points = []
        indices = []
        for i in range(num_kpts):
            x, y, v = keypoints[3*i], keypoints[3*i+1], keypoints[3*i+2]
            if v > 0:
                visible_points.append([[x, y]])
                indices.append(i)
        
        if not visible_points:
            return new_keypoints
        
        # Rectify points
        pts = np.array(visible_points, dtype=np.float32)
        undist = cv2.undistortPoints(pts, mtx, dist, P=mtx)
        
        # Update keypoints
        for j, i in enumerate(indices):
            new_keypoints[3*i] = float(undist[j, 0, 0])
            new_keypoints[3*i+1] = float(undist[j, 0, 1])
        
        return new_keypoints
    
    def _calculate_bbox_from_keypoints(self, keypoints: List[float]) -> List[float]:
        """Calculate bounding box from visible keypoints."""
        visible_coords = []
        for i in range(len(keypoints) // 3):
            if keypoints[3*i+2] > 0:  # visible
                visible_coords.extend([keypoints[3*i], keypoints[3*i+1]])
        
        if not visible_coords:
            raise ValueError("No visible keypoints found")
        
        xs = visible_coords[::2]
        ys = visible_coords[1::2]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
        return [minx, miny, maxx - minx, maxy - miny]
    
    def rectify_dataset(self, input_images_dir: str, output_images_dir: str) -> COCOManager:
        """
        Rectify entire COCO dataset and update image paths.
        
        Args:
            input_images_dir: Directory containing input images
            output_images_dir: Directory to save rectified images
            project_root: Root directory of the project for relative paths
            
        Returns:
            Modified COCODataset object with updated paths and rectified annotations
        """
        if self.mode != "coco":
            raise ValueError("rectify_dataset() can only be used in 'coco' mode")
            
        os.makedirs(output_images_dir, exist_ok=True)
        
        # Get images from COCO dataset
        images = self.coco_dataset.get_images()
        
        # Process each image
        for img_info in tqdm(images, desc="Rectifying images", unit="image"):
            self._process_single_image(img_info, input_images_dir, output_images_dir)
        
        return self.coco_dataset
    
    def rectify_videos(self, input_videos_dir: str, output_videos_dir: str, 
                      video_pattern: str = "out*.mp4") -> None:
        """
        Rectify videos by undistorting each frame.
        
        Args:
            input_videos_dir: Directory containing input videos
            output_videos_dir: Directory to save rectified videos
            video_pattern: Glob pattern to match video files
        """
        if self.mode != "video":
            raise ValueError("rectify_videos() can only be used in 'video' mode")
            
        os.makedirs(output_videos_dir, exist_ok=True)
        
        # Find video files
        video_files = glob.glob(os.path.join(input_videos_dir, video_pattern))
        
        for video_path in video_files:
            self._process_single_video(video_path, output_videos_dir)
    
    def _process_single_video(self, video_path: str, output_dir: str) -> None:
        """Process single video file."""
        basename = os.path.basename(video_path)
        cam_index = self._extract_camera_index(basename)
        
        # Get calibration
        camera = self.camera_manager.get_camera(cam_index)
        mtx, dist = camera.get_calibration_matrices()

        # Setup video capture and writer
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = os.path.join(output_dir, basename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Pre-compute undistortion maps for efficiency
        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, mtx, (width, height), cv2.CV_32FC1
        )
        
        frame_count = 0
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply undistortion
            rectified_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            out.write(rectified_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames for {basename}")
        
        cap.release()
        out.release()
        print(f"Finished processing video: {basename}")
    
    def _process_single_image(self, img_info: Dict[str, Any], input_dir: str, 
                             output_dir: str) -> None:
        """Process single image and its annotations."""
        file_name = img_info["file_name"]
        image_id = img_info["id"]
        cam_index = self._extract_camera_index(file_name)
        
        # Get calibration
        camera = self.camera_manager.get_camera(int(cam_index))
        mtx, dist = camera.get_calibration_matrices()

        # Rectify image
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        img = cv2.imread(input_path)
        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, mtx, img.shape[:2][::-1], cv2.CV_32FC1
        )
        rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path, rectified_img)
        
        # Update image path to be relative to project root
        self.coco_dataset.update_image_path(image_id, output_path)
        
        # Get and rectify annotations for this image
        annotations = self.coco_dataset.get_annotations_by_image_id(image_id)
        for ann in annotations:
            if "keypoints" in ann:
                rectified_keypoints = self._rectify_keypoints(ann["keypoints"], mtx, dist)
                new_bbox = self._calculate_bbox_from_keypoints(rectified_keypoints)
                new_area = new_bbox[2] * new_bbox[3]
                
                # Update annotation using COCODataset methods
                self.coco_dataset.update_annotation(
                    ann["id"], 
                    keypoints=rectified_keypoints, 
                    bbox=new_bbox, 
                    area=new_area
                )

    def __str__(self) -> str:
        return f"DatasetRectifier(camera_manager={self.camera_manager}, coco_dataset={self.coco_dataset}, mode={self.mode})"

def main():
    """Main function to run dataset or video rectification."""
    # Configuration
    calib_base_dir = os.path.join("..", "data", "camera_data_v2")
    
    # Choose mode: "coco" or "video"
    mode = "video"  # Change this to "coco" for dataset rectification
    
    camera_manager = CameraManager(calib_base_dir)
    
    if mode == "coco":
        # COCO dataset rectification
        input_json = os.path.join("..", "data", "dataset", "train", "_annotations.coco.json")
        input_images_dir = os.path.join("..", "data", "dataset", "train")
        output_images_dir = os.path.join("output", "dataset", "train")
    
        
        coco_dataset = COCOManager(input_json)
        rectifier = DatasetRectifier(camera_manager, coco_dataset, mode="coco")
        
        rectified_coco = rectifier.rectify_dataset(input_images_dir, output_images_dir)
        print(f"Rectified {len(rectified_coco.get_images())} images")
        print("COCO dataset has been modified in place with rectified data and updated paths")
        
    elif mode == "video":
        # Video rectification
        input_videos_dir = os.path.join("..", "data", "videos")
        output_videos_dir = os.path.join("output", "videos")
        
        rectifier = DatasetRectifier(camera_manager, mode="video")
        rectifier.rectify_videos(input_videos_dir, output_videos_dir)
        print("Video rectification completed")


if __name__ == "__main__":
    main()