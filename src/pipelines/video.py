"""Video processing pipeline."""

import logging
import cv2
import re
from pathlib import Path
from typing import Dict, List

from omegaconf import DictConfig

from utils.dataset.coco_utils import COCOManager

from .base import BasePipeline
from ..pose_estimation.ViTPose.pose_estimator_vitpose import ViTPoseEstimator
from ..pose_estimation.yolo.pose_estimator_yolo import YOLOPoseEstimator
from ..utils.skeleton.pose_plotter_2d import SkeletonDrawer
from ..utils.config_utils import log_section
from ..rectifier import Rectifier

class VideoPipeline(BasePipeline):
    """Pipeline for processing videos without ground truth."""
    
    def run(self) -> None:
        """Execute the complete video processing pipeline."""

        video_files = [str(f) for f in self.config.paths.data.video_files if Path(f).exists()]        
        self.logger.info(f"Processing {len(video_files)} video files")
        
        ##################################################
        # Rectify Videos
        ##################################################
        rectified_output = Path(self.config.paths.output.structure.rectified.root)
        
        for video_path in video_files:            
            self.logger.info(f"Rectifying video: {video_path}")
            self.rectifier.rectify_video(video_path, rectified_output)
            self.logger.info(f"Rectified video saved to: {rectified_output}")

        video_files_rectified = [f for f in rectified_output.iterdir() if f.is_file()]

        ##################################################
        # Estimate human pose on videos
        ##################################################
        models = self.config.pipeline.stages.pose_estimation.models
        for model_name in models:
            log_section(self.logger, f"{model_name}: Processing videos")

            pose_estimator = None
            if "yolo" in model_name.lower():
                pose_estimator = YOLOPoseEstimator(
                    coco_manager=self.coco_dataset,
                    model_weights_path=self.config.paths.weights.yolo_pose,
                )
                output_prediction = Path(self.config.paths.output.structure.yolo.predictions.root)
                output_triangulation = Path(self.config.paths.output.structure.yolo.triangulations.root)
            else:
                pose_estimator = ViTPoseEstimator(
                    coco_manager=self.coco_dataset,
                    detector_weights_path=self.config.paths.weights.yolo_det,
                    vit_model_name=self.config.models.vit.model_name,
                )
                output_prediction = Path(self.config.paths.output.structure.vit.predictions.root)
                output_triangulation = Path(self.config.paths.output.structure.vit.triangulations.root)

            for video_path in video_files_rectified:            
                video_name = Path(video_path).stem
                self.logger.info(f"Processing {video_path}")
                output_path = output_prediction / Path(f"{video_name}_{model_name}.mp4")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                pose_estimator.run_pose_estimation_on_video(
                    video_path=video_path,
                    output_path=output_path,
                    confidence_threshold=self.config.models.yolo.confidence_threshold,
                )
                self.logger.info(f"Saved {model_name} video output: {output_path}")
            
            ##################################################
            # Triangulate human pose estimation from videos
            ##################################################
            log_section(self.logger, f"{model_name}: Triangulate predicted 2D and visualize")
            cocos_json = list(Path(output_prediction).glob("*.json"))
            coco_manager = COCOManager(str(cocos_json[0]))
            for coco_json in cocos_json[1:]:
                coco_manager.merge([COCOManager(str(coco_json))])

            self.triangulate_and_visualize(
                coco_manager=coco_manager,
                output_root=output_triangulation,
                output_visuals=output_triangulation
            )
