"""Data source implementations for the CV pipeline"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional
import logging

import cv2
import numpy as np

# Optional imports - will raise error only if used without installation
try:
    from roboflow import Roboflow
    HAS_ROBOFLOW = True
except ImportError:
    HAS_ROBOFLOW = False

from ..core.base import DataSource, DataSourceType, FrameData


class RoboflowDataSource(DataSource):
    """Data source for Roboflow annotated datasets"""
    
    def __init__(self, **kwargs):
        # Ignore extra kwargs passed by Hydra (like 'type', etc.)
        self.dataset = None
        self.dataset_path = None
        self.annotations_data = None
        self.images_data = None
        self.camera_ids = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_type(self) -> DataSourceType:
        return DataSourceType.ROBOFLOW_DATASET
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Roboflow dataset
        
        Config should contain:
        - api_key: Roboflow API key
        - workspace: Workspace name
        - project: Project name  
        - version: Dataset version
        - format: Download format (default: 'coco')
        - download_path: Path to download dataset (default: 'roboflow_data')
        """
        if not HAS_ROBOFLOW:
            self.logger.error("Roboflow package not installed. Install with: pip install roboflow")
            return False
            
        try:
            api_key = config.get('api_key')
            workspace = config.get('workspace')
            project = config.get('project')
            version = config.get('version')
            download_path = config.get('download_path', 'roboflow_data')
            
            # Initialize Roboflow
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            version_obj = project_obj.version(int(version))
                        
            # Download dataset
            self.dataset_path = Path(download_path)

            self.logger.info(f"Downloading dataset from Roboflow to {self.dataset_path}")
            self.logger.info("This may take a few minutes...")
            
            # Download dataset synchronously
            version_obj.download("coco", location=str(self.dataset_path))
        
            
            # Check if download was successful by looking for expected files
            train_dir = self.dataset_path / "train"
            
            # Look for annotations file in different possible locations
            annotations_file = None
            possible_annotations = [
                self.dataset_path / "train" / "_annotations.coco.json",
                self.dataset_path / "_annotations.coco.json",
                self.dataset_path / "annotations.json",
                train_dir / "annotations.json"
            ]
            
            for ann_file in possible_annotations:
                if ann_file.exists():
                    annotations_file = ann_file
                    break
            
            if not annotations_file:
                # List available files for debugging
                available_files = []
                for root, dirs, files in os.walk(self.dataset_path):
                    for file in files:
                        if file.endswith('.json'):
                            available_files.append(os.path.join(root, file))
                
                self.logger.error(f"No COCO annotations found in {self.dataset_path}")
                self.logger.error(f"Available JSON files: {available_files}")
                return False
            
            # Load annotations
            self.logger.info(f"Loading annotations from {annotations_file}")
            with open(annotations_file, 'r') as f:
                self.annotations_data = json.load(f)
            
            # Create image lookup
            self.images_data = {img['id']: img for img in self.annotations_data['images']}
            
            # Extract camera IDs from image filenames
            self.camera_ids = self._extract_camera_ids()
            
            self.logger.info(f"Loaded Roboflow dataset with {len(self.images_data)} images")
            self.logger.info(f"Found {len(self.annotations_data.get('annotations', []))} annotations")
            self.logger.info(f"Camera IDs: {self.camera_ids}")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Roboflow dataset: {e}")
            return False
    
    def _extract_camera_ids(self) -> List[str]:
        """Extract camera IDs from image filenames"""
        camera_ids = set()
        for img_info in self.images_data.values():
            filename = img_info['file_name']
            # Assume filename format contains camera identifier
            # e.g., "out1_frame_0001.jpg" -> camera_id = "1"
            import re
            match = re.search(r'out(\d+)', filename)
            if match:
                camera_ids.add(match.group(1))
        return sorted(list(camera_ids))
    
    def get_frames(self) -> Iterator[FrameData]:
        """Get frames with annotations from the dataset"""
        if not self.annotations_data:
            return
        
        # Try to find images directory - could be in train, valid, test, or root
        possible_image_dirs = [
            self.dataset_path / "train",
            self.dataset_path / "valid", 
            self.dataset_path / "test",
            self.dataset_path
        ]
        
        images_dir = None
        for img_dir in possible_image_dirs:
            if img_dir.exists() and any(img_dir.glob("*.jpg")) or any(img_dir.glob("*.png")):
                images_dir = img_dir
                break
        
        if not images_dir:
            self.logger.warning("No images directory found, returning annotation-only data")
            
        annotations_by_image = {}
        
        # Group annotations by image ID
        for ann in self.annotations_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Yield frames
        for img_id, img_info in self.images_data.items():
            filename = img_info['file_name']
            frame = None
            
            if images_dir:
                image_path = images_dir / filename
                if image_path.exists():
                    # Load image
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        self.logger.warning(f"Could not load image: {image_path}")
            
            # Extract frame ID and camera ID
            frame_id, camera_id = self._parse_filename(filename)
            
            # Get annotations for this image
            annotations = {
                'image_info': img_info,
                'annotations': annotations_by_image.get(img_id, []),
                'categories': self.annotations_data.get('categories', [])
            }
            
            yield FrameData(
                frame=frame,
                frame_id=frame_id,
                camera_id=camera_id,
                annotations=annotations
            )
    
    def _parse_filename(self, filename: str) -> tuple[int, str]:
        """Parse frame ID and camera ID from filename"""
        import re
        # Assume format: "out{camera_id}_frame_{frame_id:04d}.jpg"
        match = re.search(r'out(\d+)_frame_(\d+)', filename)
        if match:
            camera_id = match.group(1)
            frame_id = int(match.group(2))
            return frame_id, camera_id
        else:
            # Fallback: use image ID as frame ID, extract camera from filename
            camera_match = re.search(r'(\d+)', filename)
            camera_id = camera_match.group(1) if camera_match else "1"
            return 0, camera_id
    
    def has_annotations(self) -> bool:
        return True
    
    def get_camera_ids(self) -> List[str]:
        return self.camera_ids
    
    def cleanup(self) -> None:
        pass


class VideoFileDataSource(DataSource):
    """Data source for video files"""
    
    def __init__(self, **kwargs):
        # Ignore extra kwargs passed by Hydra (like 'type', etc.)
        self.video_files = []
        self.camera_ids = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_type(self) -> DataSourceType:
        return DataSourceType.VIDEO_FILE
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize video file data source
        
        Config should contain:
        - path: Path to video file or directory containing video files
        - camera_ids: List of camera IDs (optional, will be auto-detected)
        """
        try:
            path = Path(config['path'])
            
            if path.is_file():
                # Single video file
                self.video_files = [path]
                # Extract camera ID from filename
                camera_id = self._extract_camera_id_from_filename(path.name)
                self.camera_ids = [camera_id] if camera_id else ["1"]
                
            elif path.is_dir():
                # Directory with multiple video files
                self.video_files = list(path.glob("*.mp4")) + list(path.glob("*.avi"))
                self.camera_ids = []
                
                for video_file in self.video_files:
                    camera_id = self._extract_camera_id_from_filename(video_file.name)
                    if camera_id:
                        self.camera_ids.append(camera_id)
                
                if not self.camera_ids:
                    # Fallback: use sequential IDs
                    self.camera_ids = [str(i+1) for i in range(len(self.video_files))]
            else:
                self.logger.error(f"Video path does not exist: {path}")
                return False
            
            self.logger.info(f"Initialized with {len(self.video_files)} video files")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize video file data source: {e}")
            return False
    
    def _extract_camera_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract camera ID from filename"""
        import re
        # Assume format: "out{camera_id}.mp4" or similar
        match = re.search(r'out(\d+)', filename)
        return match.group(1) if match else None
    
    def get_frames(self) -> Iterator[FrameData]:
        """Get frames from video files"""
        for i, video_file in enumerate(self.video_files):
            camera_id = self.camera_ids[i]
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                self.logger.warning(f"Could not open video file: {video_file}")
                continue
            
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                yield FrameData(
                    frame=frame,
                    frame_id=frame_id,
                    camera_id=camera_id,
                    timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )
                
                frame_id += 1
            
            cap.release()
    
    def has_annotations(self) -> bool:
        return False
    
    def get_camera_ids(self) -> List[str]:
        return self.camera_ids
    
    def cleanup(self) -> None:
        pass


class StreamingVideoDataSource(DataSource):
    """Data source for streaming video (simulated with OpenCV)"""
    
    def __init__(self, **kwargs):
        # Ignore extra kwargs passed by Hydra (like 'type', etc.)
        self.stream_url = None
        self.camera_id = "stream"
        self.cap = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_type(self) -> DataSourceType:
        return DataSourceType.STREAMING_VIDEO
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize streaming video data source
        
        Config should contain:
        - url: Stream URL or webcam index (0 for default webcam)
        - camera_id: Camera identifier (optional, default: 'stream')
        """
        try:
            self.stream_url = config.get('url', 0)  # Default to webcam
            self.camera_id = config.get('camera_id', 'stream')
            
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                self.logger.error(f"Could not open stream: {self.stream_url}")
                return False
            
            self.logger.info(f"Initialized streaming video from: {self.stream_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize streaming video: {e}")
            return False
    
    def get_frames(self) -> Iterator[FrameData]:
        """Get frames from video stream"""
        if not self.cap or not self.cap.isOpened():
            return
        
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield FrameData(
                frame=frame,
                frame_id=frame_id,
                camera_id=self.camera_id,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency()
            )
            
            frame_id += 1
    
    def has_annotations(self) -> bool:
        return False
    
    def get_camera_ids(self) -> List[str]:
        return [self.camera_id]
    
    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()


class ExistingDataSource(DataSource):
    """Data source for existing COCO-format datasets with annotations"""
    
    def __init__(self, **kwargs):
        # Ignore extra kwargs passed by Hydra (like 'type', etc.)
        self.annotations_data = None
        self.images_data = None
        self.camera_ids = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_type(self) -> DataSourceType:
        return DataSourceType.EXISTING_DATASET
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize existing COCO dataset
        
        Config should contain:
        - coco_file: Path to COCO annotations file
        - images_dir: Optional path to images directory 
        - use_existing_poses: Whether to use existing pose annotations
        """
        try:
            coco_file = config.get('coco_file')
            if not coco_file:
                self.logger.error("COCO file path not specified")
                return False
            
            coco_path = Path(coco_file)
            if not coco_path.exists():
                self.logger.error(f"COCO file not found: {coco_path}")
                return False
            
            # Load annotations
            with open(coco_path, 'r') as f:
                self.annotations_data = json.load(f)
            
            # Extract camera IDs from images
            self.camera_ids = []
            for image in self.annotations_data.get('images', []):
                # Extract camera ID from filename or use default
                filename = image.get('file_name', '')
                if 'cam_' in filename:
                    cam_id = filename.split('cam_')[1].split('_')[0]
                    cam_id = f"cam_{cam_id}"
                else:
                    cam_id = "cam_1"  # Default
                
                if cam_id not in self.camera_ids:
                    self.camera_ids.append(cam_id)
            
            if not self.camera_ids:
                self.camera_ids = ["cam_1"]  # Default
            
            self.logger.info(f"Loaded COCO dataset from {coco_path}")
            self.logger.info(f"Found {len(self.annotations_data.get('images', []))} images")
            self.logger.info(f"Found {len(self.annotations_data.get('annotations', []))} annotations")
            self.logger.info(f"Camera IDs: {self.camera_ids}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize existing dataset: {e}")
            return False
    
    def get_frames(self) -> Iterator[FrameData]:
        """Get frames from existing dataset"""
        if not self.annotations_data:
            return
        
        # Group annotations by image
        image_annotations = {}
        for ann in self.annotations_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Yield frame data for each image
        for image in self.annotations_data.get('images', []):
            image_id = image['id']
            filename = image.get('file_name', '')
            
            # Extract camera ID
            if 'cam_' in filename:
                cam_id = filename.split('cam_')[1].split('_')[0]
                camera_id = f"cam_{cam_id}"
            else:
                camera_id = "cam_1"
            
            # Get annotations for this image
            annotations = image_annotations.get(image_id, [])
            
            yield FrameData(
                frame=None,  # No actual image data
                frame_id=image_id,
                camera_id=camera_id,
                annotations=annotations,
                metadata={
                    'width': image.get('width'),
                    'height': image.get('height'),
                    'file_name': filename
                }
            )
    
    def has_annotations(self) -> bool:
        return True
    
    def get_camera_ids(self) -> List[str]:
        return self.camera_ids
    
    def cleanup(self) -> None:
        pass
