"""Configuration management for the CV pipeline"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .base import PipelineConfig, DataSourceType, PoseEstimationModel


@dataclass 
class DataSourceConfig:
    """Configuration for data sources"""
    # Roboflow specific
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    version: int = 1
    format_type: str = 'coco'
    download_path: str = 'roboflow_data'
    
    # Video specific
    video_extensions: list = None
    frame_rate: Optional[int] = None
    
    # Stream specific
    stream_url: Optional[str] = None
    buffer_size: int = 10

    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv']


@dataclass
class ModelConfig:
    """Configuration for pose estimation models"""
    # YOLO
    yolo_model_name: str = 'yolo11l-pose.pt'
    
    # ViT Pose
    vit_model_name: str = 'microsoft/vitpose-base-simple'
    vit_detector_model: str = 'yolo11l.pt'
    
    # ProbPose
    probpose_config_path: str = 'pose_estimation_probpose/ProbPose/configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py'
    probpose_checkpoint_path: str = 'pose_estimation_probpose/ProbPose.pth'
    probpose_detector_model: str = 'yolo11l.pt'
    
    # MediaPipe
    mediapipe_model_complexity: int = 2
    mediapipe_enable_segmentation: bool = False
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5


class ConfigManager:
    """Configuration manager for the CV pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "pipeline_config.json"
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        
    def save_config(self, config: PipelineConfig, 
                   data_source_config: Optional[DataSourceConfig] = None,
                   model_config: Optional[ModelConfig] = None) -> bool:
        """Save pipeline configuration to file"""
        try:
            config_dict = {
                'pipeline': asdict(config),
                'data_source': asdict(data_source_config) if data_source_config else {},
                'models': asdict(model_config) if model_config else {}
            }
            
            # Convert enums to strings
            config_dict['pipeline']['data_source_type'] = config.data_source_type.value
            if config.pose_estimation_model:
                config_dict['pipeline']['pose_estimation_model'] = config.pose_estimation_model.value
            
            config_path = self.config_dir / self.config_file
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self) -> tuple[Optional[PipelineConfig], Optional[DataSourceConfig], Optional[ModelConfig]]:
        """Load pipeline configuration from file"""
        config_path = self.config_dir / self.config_file
        
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            return None, None, None
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Load pipeline config
            pipeline_dict = config_dict.get('pipeline', {})
            
            # Convert string enums back to enums
            if 'data_source_type' in pipeline_dict:
                pipeline_dict['data_source_type'] = DataSourceType(pipeline_dict['data_source_type'])
            if 'pose_estimation_model' in pipeline_dict and pipeline_dict['pose_estimation_model']:
                pipeline_dict['pose_estimation_model'] = PoseEstimationModel(pipeline_dict['pose_estimation_model'])
            
            pipeline_config = PipelineConfig(**pipeline_dict)
            
            # Load data source config
            data_source_dict = config_dict.get('data_source', {})
            data_source_config = DataSourceConfig(**data_source_dict) if data_source_dict else None
            
            # Load model config
            model_dict = config_dict.get('models', {})
            model_config = ModelConfig(**model_dict) if model_dict else None
            
            print(f"Configuration loaded from {config_path}")
            return pipeline_config, data_source_config, model_config
            
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return None, None, None
    
    def create_default_config(self, data_source_type: DataSourceType) -> tuple[PipelineConfig, DataSourceConfig, ModelConfig]:
        """Create default configuration based on data source type"""
        
        # Default pipeline config
        pipeline_config = PipelineConfig(
            data_source_type=data_source_type,
            data_source_path="",
            camera_params_version="v2",
            pose_estimation_model=PoseEstimationModel.YOLO if data_source_type != DataSourceType.ROBOFLOW_DATASET else None,
            confidence_threshold=0.25,
            output_dir="output",
            enable_rectification=True,
            enable_triangulation=True,
            enable_reprojection=True,
            min_cameras_for_triangulation=2
        )
        
        # Data source specific defaults
        data_source_config = DataSourceConfig()
        
        if data_source_type == DataSourceType.ROBOFLOW_DATASET:
            pipeline_config.data_source_path = "roboflow_data"
            pipeline_config.pose_estimation_model = None  # Has annotations
            
        elif data_source_type == DataSourceType.VIDEO_FILE:
            pipeline_config.data_source_path = "old_files/data/videos"
            pipeline_config.pose_estimation_model = PoseEstimationModel.YOLO
            
        elif data_source_type == DataSourceType.STREAMING_VIDEO:
            pipeline_config.data_source_path = "0"  # Default webcam
            pipeline_config.pose_estimation_model = PoseEstimationModel.MEDIAPIPE
            pipeline_config.enable_triangulation = False  # Single camera
            pipeline_config.enable_reprojection = False
            data_source_config.stream_url = "0"
        
        model_config = ModelConfig()
        
        return pipeline_config, data_source_config, model_config
    
    def list_configs(self) -> list[str]:
        """List available configuration files"""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.name for f in config_files]
    
    def delete_config(self, config_name: str) -> bool:
        """Delete a configuration file"""
        try:
            config_path = self.config_dir / config_name
            if config_path.exists():
                config_path.unlink()
                print(f"Configuration {config_name} deleted")
                return True
            else:
                print(f"Configuration {config_name} not found")
                return False
        except Exception as e:
            print(f"Failed to delete configuration: {e}")
            return False
