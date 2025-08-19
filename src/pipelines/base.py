"""Base pipeline class with common functionality."""

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..utils.camera.camera_manager import CameraManager
from ..utils.dataset.coco_utils import COCOManager
from ..rectifier import Rectifier
from ..triangulator import PlayerTriangulator
from ..reprojector import SkeletonReprojector
from ..utils.skeleton.pose_plotter_2d import SkeletonDrawer
from ..utils.skeleton.pose_plotter_3d import PosePlotter3D
from ..utils.evaluation.pose_evaluator import PoseEvaluator
from ..utils.config_utils import log_section


class BasePipeline:
    """Base pipeline with common functionality."""
    
    def __init__(self, config: DictConfig, logger: logging.Logger):
        """Initialize base pipeline.
        
        Args:
            config: Hydra configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.evaluator = PoseEvaluator()
        
        # Load camera manager
        self.camera_manager = CameraManager()
        self.camera_manager.load_cameras(config.paths.data.camera_calib_dir)
        self.logger.info(f"Loaded cameras from: {config.paths.data.camera_calib_dir}")
        
        self.rectifier = Rectifier(
            camera_manager=self.camera_manager,
        )
        
        # Load COCO dataset
        self.coco_dataset = COCOManager(config.paths.data.dataset_json)
        self.logger.info(f"Loaded COCO dataset: {config.paths.data.dataset_json}")
    
    
    def triangulate_and_visualize(self, coco_manager: COCOManager, output_root: str, output_visuals: str) -> Any:
        """Triangulate 3D poses and create visualizations."""
        triangulator = PlayerTriangulator(self.camera_manager, coco_manager)
        skeleton_manager = triangulator.triangulate(
            use_bundle_adjustment=self.config.models.triangulation.use_bundle_adjustment
        )
        
        skeleton_json_path = Path(output_root) / "player_3d_poses.json"
        skeleton_manager.save(str(skeleton_json_path))
        self.logger.info(f"Saved 3D skeletons: {skeleton_json_path}")
        
        if self.config.pipeline.stages.triangulation.save_visualizations:
            animation_path = Path(output_visuals) / "3d_pose_animation.mp4"
            PosePlotter3D(coco_manager, skeleton_manager).animate_frames(
                save=str(animation_path)
            )
            self.logger.info(f"Saved 3D pose animation: {animation_path}")

        return skeleton_manager
    
    def reproject_draw_and_eval(self, gt_coco: COCOManager, skeleton_manager: Any,
                               output_root: str, output_visuals: str, 
                               eval_output_dir: str) -> COCOManager:
        """Reproject 3D poses to 2D, create visualizations, and evaluate."""
        reprojector = SkeletonReprojector(
            camera_manager=self.camera_manager,
            coco_manager=gt_coco.copy(),
            skeleton_manager=skeleton_manager,
        )
        reprojected_coco = reprojector.reproject()

        reprojected_json_path = Path(output_root) / "reprojected.coco.json"
        reprojected_coco.save(str(reprojected_json_path))
        self.logger.info(f"Saved reprojected annotations: {reprojected_json_path}")

        if self.config.pipeline.stages.reprojection.save_visualizations:
            SkeletonDrawer(reprojected_coco).draw_skeleton_on_coco(
                reprojected_coco, output_visuals
            )
            self.logger.info(f"Saved reprojected visualizations to: {output_visuals}")

        if self.config.pipeline.stages.reprojection.evaluation.enabled:
            self.evaluator.evaluate(
                gt_manager=gt_coco,
                pred_manager=reprojected_coco,
                output_dir=eval_output_dir,
            )
            self.logger.info(f"Saved evaluation to: {eval_output_dir}")

        return reprojected_coco
    
    
    def process_triangulation(
        self, model_name, predicted_coco, output_paths
    ):
        """Process triangulation and reprojection for a model."""
        log_section(
            self.logger, f"{model_name}: Triangulate predicted 2D and visualize"
        )
        model_skeletons = self.triangulate_and_visualize(
            coco_manager=predicted_coco,
            output_root=output_paths.triangulations.root,
            output_visuals=output_paths.triangulations.visualizations,
        )

        log_section(
            self.logger, f"{model_name}: Reproject 3D -> 2D, visualize, and evaluate"
        )
        self.reproject_draw_and_eval(
            gt_coco=predicted_coco,
            skeleton_manager=model_skeletons,
            output_root=output_paths.reprojections.root,
            output_visuals=output_paths.reprojections.visualizations,
            eval_output_dir=output_paths.evaluations.reprojections,
        )