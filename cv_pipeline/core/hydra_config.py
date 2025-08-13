"""Hydra-based configuration management for the CV pipeline"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra


@dataclass 
class DataSourceConfig:
    """Configuration for data sources"""
    _target_: str = ""
    type: str = ""
    path: Optional[str] = None
    # Video specific
    frame_rate: Optional[int] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    extensions: list = field(default_factory=lambda: [".mp4", ".avi", ".mov"])
    # Roboflow specific
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    version: int = 1
    format_type: str = "coco"
    download_path: str = "roboflow_data"
    # Streaming specific
    stream_url: Optional[str] = None
    buffer_size: int = 10
    reconnect_attempts: int = 3
    timeout: int = 30
    # Existing data specific
    coco_file: Optional[str] = None
    images_dir: Optional[str] = None
    use_existing_poses: bool = False


@dataclass
class PoseModelConfig:
    """Configuration for pose estimation models"""
    _target_: str = ""
    type: str = ""
    confidence_threshold: float = 0.25
    device: str = "auto"
    # YOLO specific
    model_name: Optional[str] = None
    # ViT Pose specific
    detector_model: Optional[str] = None
    # MediaPipe specific
    model_complexity: int = 2
    enable_segmentation: bool = False
    static_image_mode: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # ProbPose specific
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    enable_rectification: bool = True
    enable_triangulation: bool = True
    enable_reprojection: bool = True
    min_cameras_for_triangulation: int = 2
    confidence_threshold: float = 0.25


@dataclass
class CameraConfig:
    """Camera configuration"""
    params_version: str = "v2"
    calibration_base_dir: str = "data/camera_data_v2"


@dataclass
class OutputConfig:
    """Output configuration"""
    base_dir: str = "output"
    save_intermediate: bool = True
    save_visualizations: bool = True
    save_3d_poses: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    save_logs: bool = True
    log_file: str = "output/pipeline.log"


@dataclass
class Config:
    """Main configuration class"""
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    pose_model: PoseModelConfig = field(default_factory=PoseModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def register_configs():
    """Register configuration schemas with Hydra"""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="data_source", name="video_file_schema", node=DataSourceConfig)
    cs.store(group="data_source", name="roboflow_schema", node=DataSourceConfig) 
    cs.store(group="data_source", name="streaming_schema", node=DataSourceConfig)
    cs.store(group="data_source", name="existing_coco_schema", node=DataSourceConfig)
    cs.store(group="pose_model", name="yolo_schema", node=PoseModelConfig)
    cs.store(group="pose_model", name="vit_pose_schema", node=PoseModelConfig)
    cs.store(group="pose_model", name="mediapipe_schema", node=PoseModelConfig)
    cs.store(group="pose_model", name="prob_pose_schema", node=PoseModelConfig)


class HydraConfigManager:
    """Hydra-based configuration manager"""
    
    def __init__(self, config_path: str = "configs"):
        self.config_path = Path(config_path)
        self.config: Optional[DictConfig] = None
        
    def setup_logging(self, config: DictConfig):
        """Setup logging based on configuration"""
        level = getattr(logging, config.logging.level.upper())
        
        # Create output directory
        log_file = Path(config.logging.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if config.logging.save_logs else logging.NullHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {level} level")
        if config.logging.save_logs:
            logger.info(f"Logs will be saved to: {log_file}")
    
    def create_output_directory(self, config: DictConfig):
        """Create output directory structure"""
        output_dir = Path(config.output.base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["poses", "visualizations", "intermediate", "logs"]
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        logging.getLogger(__name__).info(f"Output directory created: {output_dir}")
        
        return output_dir
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate configuration"""
        logger = logging.getLogger(__name__)
        
        # Validate data source
        if not config.data_source._target_:
            logger.error("Data source _target_ not specified")
            return False
            
        # Validate pose model if pose estimation is enabled
        if not config.pose_model._target_:
            logger.warning("Pose model _target_ not specified - pose estimation will be disabled")
        
        # Validate camera configuration
        calib_dir = Path(config.camera.calibration_base_dir)
        if not calib_dir.exists():
            logger.warning(f"Camera calibration directory not found: {calib_dir}")
        
        # Validate output directory can be created
        try:
            output_dir = Path(config.output.base_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create output directory: {e}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_instantiate_config(self, config: DictConfig, key: str) -> Dict[str, Any]:
        """Get configuration for instantiating objects with hydra.utils.instantiate"""
        obj_config = config[key]
        
        # Convert OmegaConf to dict and remove _target_
        config_dict = OmegaConf.to_container(obj_config, resolve=True)
        if isinstance(config_dict, dict):
            config_dict.pop('_target_', None)
            config_dict.pop('type', None)  # Remove type field as well
        
        return config_dict
    
    def print_config(self, config: DictConfig):
        """Print configuration in a readable format"""
        logger = logging.getLogger(__name__)
        
        logger.info("=== Pipeline Configuration ===")
        logger.info(f"Data Source: {config.data_source.type}")
        logger.info(f"Pose Model: {config.pose_model.type}")
        logger.info(f"Camera Version: {config.camera.params_version}")
        logger.info(f"Output Directory: {config.output.base_dir}")
        logger.info(f"Rectification: {config.pipeline.enable_rectification}")
        logger.info(f"Triangulation: {config.pipeline.enable_triangulation}")
        logger.info(f"Reprojection: {config.pipeline.enable_reprojection}")
        
        # Print detailed config if debug level
        if logging.getLogger().level <= logging.DEBUG:
            logger.debug("Full configuration:")
            logger.debug(OmegaConf.to_yaml(config))


def load_config_with_hydra(config_path: str = "../../configs", config_name: str = "config") -> DictConfig:
    """Load configuration using Hydra (for standalone use)"""
    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra with relative path
    with hydra.initialize(config_path=config_path, version_base=None):
        config = hydra.compose(config_name=config_name)
    
    return config


# Example usage functions
def create_example_configs():
    """Create example configuration files for different scenarios"""
    
    # Multi-camera triangulation config
    multi_camera_config = """
# Multi-camera 3D pose reconstruction
defaults:
  - data_source: existing_coco
  - pose_model: yolo
  - _self_

pipeline:
  enable_rectification: true
  enable_triangulation: true
  enable_reprojection: true
  min_cameras_for_triangulation: 2

camera:
  params_version: "v2"

output:
  base_dir: "output/multi_camera_3d"
  save_3d_poses: true
  save_visualizations: true
"""

    # Real-time processing config
    realtime_config = """
# Real-time video processing
defaults:
  - data_source: streaming
  - pose_model: mediapipe
  - _self_

pipeline:
  enable_rectification: false  # Skip for speed
  enable_triangulation: false
  enable_reprojection: false

pose_model:
  model_complexity: 1  # Faster model

output:
  base_dir: "output/realtime"
  save_intermediate: false
"""

    # High accuracy config
    accuracy_config = """
# High accuracy pose estimation
defaults:
  - data_source: video_file
  - pose_model: vit_pose
  - _self_

pipeline:
  enable_rectification: true
  enable_triangulation: true
  enable_reprojection: true
  confidence_threshold: 0.5  # Higher threshold

pose_model:
  confidence_threshold: 0.5

output:
  base_dir: "output/high_accuracy"
  save_intermediate: true
  save_visualizations: true
"""

    # Save example configs
    examples_dir = Path("configs/examples")
    examples_dir.mkdir(exist_ok=True)
    
    (examples_dir / "multi_camera.yaml").write_text(multi_camera_config)
    (examples_dir / "realtime.yaml").write_text(realtime_config)
    (examples_dir / "high_accuracy.yaml").write_text(accuracy_config)
    
    print("✅ Created example configurations in configs/examples/")


if __name__ == "__main__":
    # Register configurations
    register_configs()
    
    # Create example configs
    create_example_configs()
    
    # Test configuration loading
    try:
        config = load_config_with_hydra()
        manager = HydraConfigManager()
        manager.print_config(config)
        manager.validate_config(config)
        print("✅ Configuration system working correctly!")
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
