from pathlib import Path
import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
import sys

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset.coco_utils import COCOManager


class SkeletonDrawer:
    """
    A class for drawing skeletons on images with red keypoints and green connections.
    """
    
    def __init__(self, coco_manager: COCOManager):
        """
        Initialize the SkeletonDrawer.
        
        Args:
            coco_manager: Optional COCOManager to extract keypoint info from
            keypoint_names: List of keypoint names (if no coco_manager provided)
            skeleton_connections: List of skeleton connections as (start, end) tuples
        """
        
        self.keypoint_names = coco_manager.get_keypoint_names()
        self.skeleton_connections = self._convert_skeleton_indices_to_names(
            coco_manager.get_skeleton(), self.keypoint_names
        )
        
        # Fixed colors: red keypoints, green connections
        self.keypoint_color = (0, 0, 255)    # Red in BGR
        self.connection_color = (0, 255, 0)  # Green in BGR
    
    def _convert_skeleton_indices_to_names(self, skeleton_indices: List[List[int]], 
                                          keypoint_names: List[str]) -> List[Tuple[str, str]]:
        """Convert skeleton connections from indices to keypoint names."""
        connections = []
        for connection in skeleton_indices:
            if len(connection) >= 2:
                start_idx, end_idx = connection[0] - 1, connection[1] - 1  # COCO is 1-indexed
                if 0 <= start_idx < len(keypoint_names) and 0 <= end_idx < len(keypoint_names):
                    connections.append((keypoint_names[start_idx], keypoint_names[end_idx]))
        return connections
    
    def draw_skeleton_on_image(self, frame: np.ndarray, keypoints: List[float]) -> np.ndarray:
        """
        Draw skeleton on a frame with red keypoints and green connections.
        
        Args:
            frame: Input frame (BGR format) - will be modified in place
            keypoints: Flat list of keypoints [x1, y1, v1, x2, y2, v2, ...]
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines
            
        Returns:
            Frame with skeleton drawn (same object as input, modified in place)
        """
        num_keypoints = len(keypoints) // 3
        
        # Parse keypoints into dictionary
        keypoint_dict = {}
        for i in range(min(num_keypoints, len(self.keypoint_names))):
            x, y, v = keypoints[i*3], keypoints[i*3+1], keypoints[i*3+2]
            if v > 0:  # visible
                keypoint_dict[self.keypoint_names[i]] = (int(x), int(y))
        
        # Draw skeleton connections first (so they appear behind keypoints)
        for start_name, end_name in self.skeleton_connections:
            if start_name in keypoint_dict and end_name in keypoint_dict:
                start_pt = keypoint_dict[start_name]
                end_pt = keypoint_dict[end_name]
                cv2.line(frame, start_pt, end_pt, self.connection_color, 2)
        
        # Draw keypoints on top
        for (x, y) in keypoint_dict.values():
            cv2.circle(frame, (x, y), 5, self.keypoint_color, -1)
        
        return frame
    
    
    def draw_skeleton_on_coco(self, coco_manager: COCOManager, output_dir: str) -> None:
        """
        Process entire COCO dataset and save images with drawn skeletons.
        
        Args:
            coco_manager: COCOManager object with dataset
            images_dir: Directory containing input images
            output_dir: Directory to save output images
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines
        """
        os.makedirs(output_dir, exist_ok=True)
        
        images = coco_manager.get_images()
        total_images = len(images)

        for idx, img_info in tqdm(enumerate(images), total=total_images, desc="Drawing skeletons", unit="image"):
                        
            # Load image
            image = cv2.imread(img_info['file_name'])
            
            # Get annotations for this image
            annotations = coco_manager.get_annotations_by_image_id(img_info['id'])

            if(len(annotations) > 0):
                # Draw skeletons
                image = self.draw_skeleton_on_image(image, annotations[0]['keypoints'])
            
            path = Path(img_info['file_name'])
            out_path = Path(output_dir) / path.name
            # Save result
            cv2.imwrite(str(out_path), image)