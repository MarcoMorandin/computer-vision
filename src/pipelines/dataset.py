"""Dataset processing pipeline."""

from pathlib import Path

from ..rectifier import Rectifier

from .base import BasePipeline
from ..pose_estimation.ViTPose.pose_estimator_vitpose import ViTPoseEstimator
from ..pose_estimation.yolo.pose_estimator_yolo import YOLOPoseEstimator
from ..utils.skeleton.pose_plotter_2d import SkeletonDrawer
from ..utils.config_utils import log_section


class DatasetPipeline(BasePipeline):
    """Pipeline for processing datasets with ground truth annotations."""

    def run(self) -> None:
        """Execute the complete dataset processing pipeline."""
        pipeline_stages = self.config.pipeline.stages

        ##################################################
        # Rectify dataset
        ##################################################

        log_section(self.logger, "Rectify dataset (images + annotations)")

        rectifier = Rectifier(
            camera_manager=self.camera_manager,
        )

        rectified_output = self.config.paths.output.structure.rectified
        rectified_coco = rectifier.rectify_dataset(
            coco_dataset=self.coco_dataset,
            input_images_dir=self.config.paths.data.input_images_dir,
            output_images_dir=rectified_output.root,
        )

        rectified_json_path = Path(rectified_output.root) / "rectified.coco.json"
        rectified_coco.save(str(rectified_json_path))

        self.logger.info(f"Saved rectified annotations: {rectified_json_path}")

        if pipeline_stages.rectification.save_visualizations:
            SkeletonDrawer(rectified_coco).draw_skeleton_on_coco(
                rectified_coco,
                rectified_output.visualizations,
            )
            self.logger.info(
                f"Saved rectified visualizations to: {rectified_output.visualizations}"
            )

        ##################################################
        # Triangulate dataset
        ##################################################
        if pipeline_stages.triangulation.enabled:
            self.process_triangulation(
                model_name="Ground Truth",
                predicted_coco=rectified_coco,
                output_paths=self.config.paths.output.structure.ground_truth
            )

        ##################################################
        # Human Pose Estimation
        ##################################################
        if pipeline_stages.pose_estimation.enabled:
            models = pipeline_stages.pose_estimation.models
            for model_name in models:
                self._process_hpe(model_name=model_name, rectified_coco=rectified_coco)

    def _process_hpe(self, model_name, rectified_coco):
        """Process with HPE pose estimator."""
        pipeline_stages = self.config.pipeline.stages
        log_section(self.logger, f"{model_name}: Pose estimation, visualization, and evaluation")
        out_folder_structure = self.config.paths.output.structure
        if "yolo" in model_name.lower():
            pose_estimator = YOLOPoseEstimator(
                coco_manager=rectified_coco.copy(),
                config=self.config,
                logger=self.logger,
            )
            output_predictions = out_folder_structure.yolo
        else:
            pose_estimator = ViTPoseEstimator(
                coco_manager=rectified_coco.copy(),
                config=self.config,
                logger=self.logger,
            )
            output_predictions = out_folder_structure.vit

        # Run pose estimation (confidence threshold now handled by estimator from config)
        predicted = pose_estimator.run_pose_estimation()

        output_coco_path = Path(output_predictions.predictions.root) / "predicted.coco.json"
        predicted.save(str(output_coco_path))

        self.logger.info(f"Saved {model_name} predictions: {output_coco_path}")

       
        if pipeline_stages.pose_estimation.save_visualizations:
            SkeletonDrawer(predicted).draw_skeleton_on_coco(
                predicted, output_predictions.predictions.visualizations
            )
            self.logger.info(
                f"Saved {model_name} prediction visualizations to: {output_predictions.predictions.visualizations}"
            )

        rectified_coco_pruned = rectified_coco.copy()
        rectified_coco_pruned.prune_keypoints(
            remove_patterns=["Foot"]
        )

        ##################################################
        # Evaluate Human Pose Estimation Results
        ##################################################
        if pipeline_stages.pose_estimation.evaluation.enabled:
            self.evaluator.evaluate(
                gt_manager=rectified_coco_pruned,
                pred_manager=predicted,
                output_dir=output_predictions.evaluations.predictions
            )
            self.logger.info(
                f"Saved {model_name} predictions evaluation to: {output_predictions.evaluations.predictions}"
            )
    
        ##################################################
        # Triangulate Human Pose Estimation Results
        ##################################################
        if pipeline_stages.pose_estimation.triangulation.enabled:
            self.process_triangulation(
                model_name=model_name,
                predicted_coco=predicted,
                output_paths=output_predictions,
                gt_coco=rectified_coco_pruned
            )

    
