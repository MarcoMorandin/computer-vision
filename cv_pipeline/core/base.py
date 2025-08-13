"""Core abstract base classes for the computer vision pipeline"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, Tuple, List
from dataclasses import dataclass
import numpy as np
from enum import Enum


class DataSourceType(Enum):
    """Types of data sources supported by the pipeline"""
    ROBOFLOW_DATASET = "roboflow_dataset"
    VIDEO_FILE = "video_file"
    STREAMING_VIDEO = "streaming_video"


class PoseEstimationModel(Enum):
    """Available human pose estimation models"""
    YOLO = "yolo"
    VIT_POSE = "vit_pose"
    PROB_POSE = "prob_pose"
    MEDIAPIPE = "mediapipe"


@dataclass
class CameraParameters:
    """Camera calibration parameters"""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray  # distortion coefficients
    rotation_vector: np.ndarray  # rotation vector (rvec)
    translation_vector: np.ndarray  # translation vector (tvec)
    camera_id: str  # camera identifier
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from rotation vector"""
        import cv2
        R, _ = cv2.Rodrigues(self.rotation_vector)
        return R
    
    @property 
    def projection_matrix(self) -> np.ndarray:
        """Get 3x4 projection matrix P = K[R|t]"""
        R = self.rotation_matrix
        t = self.translation_vector.reshape(-1, 1)
        return self.camera_matrix @ np.hstack([R, t])


@dataclass
class FrameData:
    """Container for frame data and metadata"""
    frame: np.ndarray  # image frame
    frame_id: int  # frame identifier
    camera_id: str  # camera identifier
    timestamp: Optional[float] = None  # timestamp if available
    annotations: Optional[Dict[str, Any]] = None  # COCO-style annotations if available


@dataclass
class PipelineConfig:
    """Configuration for the computer vision pipeline"""
    data_source_type: DataSourceType
    data_source_path: str
    camera_params_version: str  # e.g., "v1", "v2", "v3"
    pose_estimation_model: Optional[PoseEstimationModel] = None
    confidence_threshold: float = 0.25
    output_dir: str = "output"
    enable_rectification: bool = True
    enable_triangulation: bool = True
    enable_reprojection: bool = True
    min_cameras_for_triangulation: int = 2


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def get_type(self) -> DataSourceType:
        """Get the type of this data source"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the data source with configuration"""
        pass
    
    @abstractmethod
    def get_frames(self) -> Iterator[FrameData]:
        """Get frames from the data source"""
        pass
    
    @abstractmethod
    def has_annotations(self) -> bool:
        """Check if this data source provides annotations"""
        pass
    
    @abstractmethod
    def get_camera_ids(self) -> List[str]:
        """Get list of camera IDs available in this data source"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass


class PoseEstimator(ABC):
    """Abstract base class for pose estimation models"""
    
    @abstractmethod
    def get_model_type(self) -> PoseEstimationModel:
        """Get the type of this pose estimation model"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the pose estimation model"""
        pass
    
    @abstractmethod
    def estimate_poses(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate poses in the given frame, return COCO-style annotations"""
        pass
    
    @abstractmethod
    def get_keypoint_names(self) -> List[str]:
        """Get the names of keypoints used by this model"""
        pass


class Rectifier(ABC):
    """Abstract base class for image rectification"""
    
    @abstractmethod
    def initialize(self, camera_params: CameraParameters) -> bool:
        """Initialize rectifier with camera parameters"""
        pass
    
    @abstractmethod
    def rectify_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rectify a single frame"""
        pass
    
    @abstractmethod
    def rectify_annotations(self, annotations: Dict[str, Any], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Rectify annotations (keypoints, bounding boxes) to match rectified frame"""
        pass


class Triangulator(ABC):
    """Abstract base class for 3D triangulation"""
    
    @abstractmethod
    def initialize(self, camera_params: Dict[str, CameraParameters]) -> bool:
        """Initialize triangulator with camera parameters"""
        pass
    
    @abstractmethod
    def triangulate_poses(self, multi_view_annotations: Dict[str, Dict[str, Any]]) -> Dict[str, List[Optional[List[float]]]]:
        """Triangulate 3D poses from multi-view annotations
        
        Args:
            multi_view_annotations: Dict[frame_id][camera_id] -> annotations
            
        Returns:
            Dict[frame_id] -> List of 3D keypoints (or None for missing keypoints)
        """
        pass


class ReprojectionEvaluator(ABC):
    """Abstract base class for reprojection evaluation"""
    
    @abstractmethod
    def initialize(self, camera_params: Dict[str, CameraParameters]) -> bool:
        """Initialize evaluator with camera parameters"""
        pass
    
    @abstractmethod
    def reproject_3d_poses(self, poses_3d: Dict[str, List[Optional[List[float]]]]) -> Dict[str, Dict[str, List[Optional[List[float]]]]]:
        """Reproject 3D poses back to 2D camera views
        
        Args:
            poses_3d: Dict[frame_id] -> List of 3D keypoints
            
        Returns:
            Dict[frame_id][camera_id] -> List of 2D keypoints
        """
        pass
    
    @abstractmethod
    def evaluate_reprojection_error(self, original_2d: Dict[str, Dict[str, Any]], 
                                   reprojected_2d: Dict[str, Dict[str, List[Optional[List[float]]]]]) -> Dict[str, Any]:
        """Evaluate reprojection error between original and reprojected 2D poses"""
        pass
