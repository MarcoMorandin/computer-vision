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

        Parameters
        ----------
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

    def triangulate_and_visualize(
        self, coco_manager: COCOManager, output_root: str, output_visuals: str
    ) -> Any:
        """Triangulate 3D poses and optionally save a 3D animation.

        Parameters
        ----------
        coco_manager : COCOManager
            Source of 2D annotations to triangulate.
        output_root : str
            Directory where serialized 3D skeletons are written.
        output_visuals : str
            Directory where animations are saved when enabled.

        Returns
        -------
        Any
            The SkeletonManager3D holding triangulated poses.
        """
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

    def reproject_draw_and_eval(
        self,
        predicted_coco: COCOManager,
        skeleton_manager: Any,
        output_root: str,
        output_visuals: str,
        eval_output_dir: str,
        gt_coco: COCOManager = None,
    ) -> COCOManager:
        """Reproject 3D poses to 2D, draw them, and optionally evaluate.

        Parameters
        ----------
        predicted_coco : COCOManager
            Original 2D predictions to be reprojected.
        skeleton_manager : Any
            Triangulated 3D poses.
        output_root : str
            Directory to save the reprojected COCO annotations.
        output_visuals : str
            Directory to save 2D drawings.
        eval_output_dir : str
            Directory to write evaluation results when enabled.
        gt_coco : COCOManager | None
            Ground truth for evaluation; if None, predicted_coco is used as reference.
        """
        reprojector = SkeletonReprojector(
            camera_manager=self.camera_manager,
            coco_manager=predicted_coco.copy(),
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
                gt_manager=gt_coco if gt_coco else predicted_coco,
                pred_manager=reprojected_coco,
                output_dir=eval_output_dir,
            )
            self.logger.info(f"Saved evaluation to: {eval_output_dir}")

        return reprojected_coco

    def process_triangulation(
        self,
        model_name: str,
        predicted_coco: COCOManager,
        output_paths: str,
        gt_coco: COCOManager = None,
    ):
        """Run triangulation, reprojection, and evaluation for a given model output.

        Parameters
        ----------
        model_name : str
            Name of the pose estimation method (for logging).
        predicted_coco : COCOManager
            2D predictions to triangulate.
        output_paths : Any
            Output directory structure (from config.paths.output.structure).
        gt_coco : COCOManager | None
            Ground-truth used for evaluation.
        """
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
            predicted_coco=predicted_coco,
            skeleton_manager=model_skeletons,
            output_root=output_paths.reprojections.root,
            output_visuals=output_paths.reprojections.visualizations,
            eval_output_dir=output_paths.evaluations.reprojections,
            gt_coco=gt_coco,
        )
