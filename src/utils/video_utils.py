"""Video utilities for creating COCO datasets from video frames."""

import os
import cv2
from typing import List, Tuple
from pathlib import Path

from .dataset.coco_utils import COCOManager


def create_video_coco_dataset(
    video_path: str,
    output_frames_dir: str,
    base_coco_path: str,
    output_coco_path: str,
    frame_interval: int = 1
) -> COCOManager:
    """Create a COCO dataset from video frames.
    
    Args:
        video_path: Path to input video file
        output_frames_dir: Directory to save extracted frames
        base_coco_path: Path to base COCO dataset to copy structure from
        output_coco_path: Path to save the new COCO dataset
        frame_interval: Interval between frames to extract (1 = every frame)
        
    Returns:
        COCOManager with video frames as images
    """
    # Load base COCO dataset and create a copy
    base_coco = COCOManager(base_coco_path)
    video_coco = base_coco.copy()
    
    # Clear existing images and annotations
    video_coco.clear_annotations()
    video_coco.coco.dataset["images"] = []
    video_coco.coco.imgs = {}
    
    # Create output directory
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # Open video and extract frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = 0
    image_id = 1
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip frames based on interval
        if frame_count % frame_interval != 0:
            continue
            
        # Save frame as image
        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_path = os.path.join(output_frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # Add image info to COCO dataset
        height, width = frame.shape[:2]
        image_info = {
            "id": image_id,
            "file_name": frame_path,
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        
        video_coco.coco.dataset["images"].append(image_info)
        video_coco.coco.imgs[image_id] = image_info
        
        extracted_frames.append((frame_count, frame_path))
        image_id += 1
    
    cap.release()
    
    # Save the COCO dataset
    video_coco.save(output_coco_path)
    
    print(f"Extracted {len(extracted_frames)} frames from video")
    print(f"Video COCO dataset saved to: {output_coco_path}")
    
    return video_coco


def get_frame_paths_from_video_coco(video_coco_manager: COCOManager) -> List[Tuple[int, str]]:
    """Get frame paths and numbers from a video COCO dataset.
    
    Args:
        video_coco_manager: COCOManager created from video
        
    Returns:
        List of tuples (frame_number, frame_path)
    """
    frame_info = []
    
    for img in video_coco_manager.get_images():
        frame_file = img["file_name"]
        # Extract frame number from filename
        import re
        match = re.search(r'frame_(\d+)', frame_file)
        if match:
            frame_num = int(match.group(1))
            frame_info.append((frame_num, frame_file))
    
    # Sort by frame number
    frame_info.sort(key=lambda x: x[0])
    return frame_info
