"""Main computer vision pipeline orchestrator"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import (
    DataSource, PoseEstimator, Rectifier, Triangulator, ReprojectionEvaluator,
    PipelineConfig, FrameData, CameraParameters, DataSourceType, PoseEstimationModel
)


class CVPipeline:
    """Main computer vision pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Pipeline components
        self.data_source: Optional[DataSource] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.rectifier_dict: Dict[str, Rectifier] = {}
        self.triangulator: Optional[Triangulator] = None
        self.reprojection_evaluator: Optional[ReprojectionEvaluator] = None
        
        # Camera parameters
        self.camera_params: Dict[str, CameraParameters] = {}
        
        # Results storage
        self.rectified_annotations: Dict[str, Dict[str, Any]] = {}
        self.pose_annotations: Dict[str, Dict[str, Any]] = {}
        self.poses_3d: Dict[str, List[Optional[List[float]]]] = {}
        self.reprojected_poses: Dict[str, Dict[str, List[Optional[List[float]]]]] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        logger = logging.getLogger('CVPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize(self) -> bool:
        """Initialize the pipeline components"""
        self.logger.info("Initializing CV Pipeline...")
        
        try:
            # Load camera parameters
            if not self._load_camera_parameters():
                return False
            
            # Initialize data source
            if not self._initialize_data_source():
                return False
            
            # Initialize rectifiers if rectification is enabled
            if self.config.enable_rectification:
                if not self._initialize_rectifiers():
                    return False
            
            # Initialize pose estimator if needed (not needed for annotated datasets)
            if not self.data_source.has_annotations() and self.config.pose_estimation_model:
                if not self._initialize_pose_estimator():
                    return False
            
            # Initialize triangulator if enabled
            if self.config.enable_triangulation:
                if not self._initialize_triangulator():
                    return False
            
            # Initialize reprojection evaluator if enabled
            if self.config.enable_reprojection:
                if not self._initialize_reprojection_evaluator():
                    return False
            
            self.logger.info("Pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        self.logger.info("Running CV Pipeline...")
        
        try:
            results = {}
            
            # Step 1: Process frames through data source and rectification
            self._process_frames()
            results['frames_processed'] = len(self.rectified_annotations)
            
            # Step 2: Run pose estimation if needed
            if not self.data_source.has_annotations() and self.pose_estimator:
                self._run_pose_estimation()
                results['pose_estimation_completed'] = True
            
            # Step 3: Triangulation
            if self.config.enable_triangulation and self.triangulator:
                self._run_triangulation()
                results['triangulation_completed'] = True
                results['frames_triangulated'] = len(self.poses_3d)
            
            # Step 4: Reprojection evaluation
            if self.config.enable_reprojection and self.reprojection_evaluator:
                evaluation_results = self._run_reprojection_evaluation()
                results['reprojection_evaluation'] = evaluation_results
            
            # Step 5: Save results
            self._save_results()
            results['results_saved'] = True
            
            self.logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {'error': str(e)}
        finally:
            self._cleanup()
    
    def _load_camera_parameters(self) -> bool:
        """Load camera parameters from files"""
        self.logger.info("Loading camera parameters...")
        
        base_dir = Path("data") / f"camera_data_{self.config.camera_params_version}"
        
        if not base_dir.exists():
            self.logger.error(f"Camera data directory not found: {base_dir}")
            return False
        
        camera_dirs = list(base_dir.glob("cam_*"))
        if not camera_dirs:
            self.logger.error(f"No camera directories found in {base_dir}")
            return False
        
        for cam_dir in camera_dirs:
            cam_id = cam_dir.name.split('_')[-1]
            calib_file = cam_dir / "calib" / "camera_calib.json"
            
            if calib_file.exists():
                try:
                    with open(calib_file, 'r') as f:
                        calib_data = json.load(f)
                    
                    import numpy as np
                    
                    self.camera_params[cam_id] = CameraParameters(
                        camera_matrix=np.array(calib_data['mtx']),
                        distortion_coeffs=np.array(calib_data['dist']).flatten(),
                        rotation_vector=np.array(calib_data.get('rvecs', [[0], [0], [0]])).flatten(),
                        translation_vector=np.array(calib_data.get('tvecs', [[0], [0], [0]])).flatten(),
                        camera_id=cam_id
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to load calibration for camera {cam_id}: {e}")
                    return False
        
        self.logger.info(f"Loaded parameters for {len(self.camera_params)} cameras")
        return True
    
    def _initialize_data_source(self) -> bool:
        """Initialize the appropriate data source"""
        # Import data sources here to avoid circular imports
        from ..data_sources import (
            RoboflowDataSource, VideoFileDataSource, StreamingVideoDataSource
        )
        
        config = {
            'path': self.config.data_source_path,
            'camera_ids': list(self.camera_params.keys())
        }
        
        if self.config.data_source_type == DataSourceType.ROBOFLOW_DATASET:
            self.data_source = RoboflowDataSource()
        elif self.config.data_source_type == DataSourceType.VIDEO_FILE:
            self.data_source = VideoFileDataSource()
        elif self.config.data_source_type == DataSourceType.STREAMING_VIDEO:
            self.data_source = StreamingVideoDataSource()
        else:
            self.logger.error(f"Unsupported data source type: {self.config.data_source_type}")
            return False
        
        return self.data_source.initialize(config)
    
    def _initialize_rectifiers(self) -> bool:
        """Initialize rectifiers for each camera"""
        from ..rectification import CameraRectifier
        
        for cam_id, cam_params in self.camera_params.items():
            rectifier = CameraRectifier()
            if rectifier.initialize(cam_params):
                self.rectifier_dict[cam_id] = rectifier
            else:
                self.logger.error(f"Failed to initialize rectifier for camera {cam_id}")
                return False
        
        return True
    
    def _initialize_pose_estimator(self) -> bool:
        """Initialize the pose estimation model"""
        from ..pose_estimation import (
            YOLOPoseEstimator, ViTPoseEstimator, ProbPoseEstimator, MediaPipePoseEstimator
        )
        
        config = {
            'confidence_threshold': self.config.confidence_threshold
        }
        
        if self.config.pose_estimation_model == PoseEstimationModel.YOLO:
            self.pose_estimator = YOLOPoseEstimator()
        elif self.config.pose_estimation_model == PoseEstimationModel.VIT_POSE:
            self.pose_estimator = ViTPoseEstimator()
        elif self.config.pose_estimation_model == PoseEstimationModel.PROB_POSE:
            self.pose_estimator = ProbPoseEstimator()
        elif self.config.pose_estimation_model == PoseEstimationModel.MEDIAPIPE:
            self.pose_estimator = MediaPipePoseEstimator()
        else:
            self.logger.error(f"Unsupported pose estimation model: {self.config.pose_estimation_model}")
            return False
        
        return self.pose_estimator.initialize(config)
    
    def _initialize_triangulator(self) -> bool:
        """Initialize the triangulator"""
        from ..triangulation import MultiViewTriangulator
        
        self.triangulator = MultiViewTriangulator()
        config = {
            'min_cameras': self.config.min_cameras_for_triangulation
        }
        return self.triangulator.initialize(self.camera_params, config)
    
    def _initialize_reprojection_evaluator(self) -> bool:
        """Initialize the reprojection evaluator"""
        from ..reprojection import ReprojectionEvaluator
        
        self.reprojection_evaluator = ReprojectionEvaluator()
        return self.reprojection_evaluator.initialize(self.camera_params)
    
    def _process_frames(self) -> None:
        """Process frames through data source and rectification"""
        self.logger.info("Processing frames...")
        
        frame_count = 0
        for frame_data in self.data_source.get_frames():
            frame_id = str(frame_data.frame_id)
            cam_id = frame_data.camera_id
            
            # Initialize frame storage
            if frame_id not in self.rectified_annotations:
                self.rectified_annotations[frame_id] = {}
            
            # Rectify frame if rectification is enabled
            rectified_frame = frame_data.frame
            rectified_annotations = frame_data.annotations
            
            if self.config.enable_rectification and cam_id in self.rectifier_dict:
                rectifier = self.rectifier_dict[cam_id]
                rectified_frame = rectifier.rectify_frame(frame_data.frame)
                
                if frame_data.annotations:
                    rectified_annotations = rectifier.rectify_annotations(
                        frame_data.annotations, frame_data.frame.shape[:2]
                    )
            
            # Store rectified data
            self.rectified_annotations[frame_id][cam_id] = {
                'frame': rectified_frame,
                'annotations': rectified_annotations
            }
            
            frame_count += 1
            if frame_count % 50 == 0:
                self.logger.info(f"Processed {frame_count} frames")
    
    def _run_pose_estimation(self) -> None:
        """Run pose estimation on rectified frames"""
        self.logger.info("Running pose estimation...")
        
        frame_count = 0
        for frame_id, cameras in self.rectified_annotations.items():
            self.pose_annotations[frame_id] = {}
            
            for cam_id, data in cameras.items():
                frame = data['frame']
                annotations = self.pose_estimator.estimate_poses(frame)
                self.pose_annotations[frame_id][cam_id] = annotations
                
                frame_count += 1
                if frame_count % 25 == 0:
                    self.logger.info(f"Estimated poses for {frame_count} frames")
    
    def _run_triangulation(self) -> None:
        """Run 3D triangulation"""
        self.logger.info("Running triangulation...")
        
        # Use pose annotations if available, otherwise use original annotations
        source_annotations = self.pose_annotations if self.pose_annotations else self.rectified_annotations
        
        # Convert to the format expected by triangulator
        multi_view_data = {}
        for frame_id, cameras in source_annotations.items():
            multi_view_data[frame_id] = {}
            for cam_id, data in cameras.items():
                if isinstance(data, dict) and 'annotations' in data:
                    multi_view_data[frame_id][cam_id] = data['annotations']
                else:
                    multi_view_data[frame_id][cam_id] = data
        
        self.poses_3d = self.triangulator.triangulate_poses(multi_view_data)
        self.logger.info(f"Triangulated {len(self.poses_3d)} frames")
    
    def _run_reprojection_evaluation(self) -> Dict[str, Any]:
        """Run reprojection evaluation"""
        self.logger.info("Running reprojection evaluation...")
        
        # Reproject 3D poses back to 2D
        self.reprojected_poses = self.reprojection_evaluator.reproject_3d_poses(self.poses_3d)
        
        # Get original 2D annotations for comparison
        original_2d = {}
        source_annotations = self.pose_annotations if self.pose_annotations else self.rectified_annotations
        
        for frame_id, cameras in source_annotations.items():
            original_2d[frame_id] = {}
            for cam_id, data in cameras.items():
                if isinstance(data, dict) and 'annotations' in data:
                    original_2d[frame_id][cam_id] = data['annotations']
                else:
                    original_2d[frame_id][cam_id] = data
        
        # Evaluate reprojection error
        evaluation_results = self.reprojection_evaluator.evaluate_reprojection_error(
            original_2d, self.reprojected_poses
        )
        
        return evaluation_results
    
    def _save_results(self) -> None:
        """Save pipeline results"""
        self.logger.info("Saving results...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save 3D poses if available
        if self.poses_3d:
            poses_3d_file = output_dir / "poses_3d.json"
            with open(poses_3d_file, 'w') as f:
                json.dump(self.poses_3d, f, indent=2)
            self.logger.info(f"Saved 3D poses to {poses_3d_file}")
        
        # Save reprojected poses if available
        if self.reprojected_poses:
            reprojected_file = output_dir / "reprojected_poses.json"
            with open(reprojected_file, 'w') as f:
                json.dump(self.reprojected_poses, f, indent=2)
            self.logger.info(f"Saved reprojected poses to {reprojected_file}")
        
        # Save pose annotations if we ran pose estimation
        if self.pose_annotations:
            pose_annotations_file = output_dir / "pose_annotations.json"
            # Convert numpy arrays to lists for JSON serialization
            serializable_annotations = self._make_json_serializable(self.pose_annotations)
            with open(pose_annotations_file, 'w') as f:
                json.dump(serializable_annotations, f, indent=2)
            self.logger.info(f"Saved pose annotations to {pose_annotations_file}")
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to JSON-serializable format"""
        import numpy as np
        
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        else:
            return data
    
    def _cleanup(self) -> None:
        """Cleanup pipeline resources"""
        if self.data_source:
            self.data_source.cleanup()
        
        self.logger.info("Pipeline cleanup completed")
