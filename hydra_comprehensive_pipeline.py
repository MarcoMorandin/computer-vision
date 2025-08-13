#!/usr/bin/env python3
"""
Hydra-based Comprehensive Pipeline Runner
=========================================

This script uses the existing Hydra pipeline infrastructure to run
the comprehensive computer vision workflow with all the requested steps.
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path

from cv_pipeline.core.pipeline import CVPipeline
from cv_pipeline.core.base import PipelineConfig, DataSourceType, PoseEstimationModel
from cv_pipeline.visualization.pipeline_visualizer import PipelineVisualizer


def convert_hydra_config_to_pipeline_config(cfg: DictConfig) -> PipelineConfig:
    """Convert Hydra DictConfig to PipelineConfig"""
    
    # Extract data source type from config
    data_source_type = DataSourceType.ROBOFLOW_DATASET  # Default
    if hasattr(cfg, 'data_source') and hasattr(cfg.data_source, '_target_'):
        target = cfg.data_source._target_.lower()
        if 'roboflow' in target:
            data_source_type = DataSourceType.ROBOFLOW_DATASET
        elif 'video' in target:
            data_source_type = DataSourceType.VIDEO_FILE
        elif 'stream' in target:
            data_source_type = DataSourceType.STREAMING_VIDEO
    
    # Extract pose model type from config
    pose_model = PoseEstimationModel.YOLO  # Default
    if hasattr(cfg, 'pose_model') and hasattr(cfg.pose_model, '_target_'):
        target = cfg.pose_model._target_.lower()
        if 'yolo' in target:
            pose_model = PoseEstimationModel.YOLO
        elif 'vit' in target:
            pose_model = PoseEstimationModel.VIT_POSE
        elif 'prob' in target:
            pose_model = PoseEstimationModel.PROB_POSE
        elif 'mediapipe' in target:
            pose_model = PoseEstimationModel.MEDIAPIPE
    
    return PipelineConfig(
        data_source_type=data_source_type,
        data_source_path=cfg.data_source.get('dataset_path', 'dataset'),
        camera_params_version=cfg.get('camera', {}).get('params_version', 'v3'),
        pose_estimation_model=pose_model,
        confidence_threshold=cfg.pipeline.get('confidence_threshold', 0.25),
        output_dir=cfg.output.get('base_dir', 'output/comprehensive_pipeline'),
        enable_rectification=cfg.pipeline.get('enable_rectification', True),
        enable_triangulation=cfg.pipeline.get('enable_triangulation', True),
        enable_reprojection=cfg.pipeline.get('enable_reprojection', True),
        min_cameras_for_triangulation=cfg.pipeline.get('min_cameras_for_triangulation', 2)
    )


class ComprehensiveHydraPipeline(CVPipeline):
    """Extended pipeline with comprehensive workflow capabilities"""
    
    def __init__(self, cfg: DictConfig):
        # Convert Hydra config to PipelineConfig
        pipeline_config = convert_hydra_config_to_pipeline_config(cfg)
        super().__init__(pipeline_config)
        
        self.hydra_cfg = cfg  # Keep reference to original Hydra config
        self.visualizer = None
        
    def initialize_data_source_with_hydra_config(self) -> bool:
        """Initialize data source with full Hydra configuration"""
        from cv_pipeline.data_sources.data_sources import RoboflowDataSource
        
        # Create data source
        self.data_source = RoboflowDataSource()
        
        # Use the full hydra config for data source initialization
        data_source_config = dict(self.hydra_cfg.data_source)
        
        # Initialize with full config
        success = self.data_source.initialize(data_source_config)
        if success:
            self.logger.info("Data source initialized successfully with Hydra config")
        else:
            self.logger.error("Failed to initialize data source")
            
        return success
    
    def load_roboflow_data(self):
        """Load images and annotations from RoboflowDataSource"""
        images = {}
        annotations = {}
        
        # Get frames from data source
        frame_iterator = self.data_source.get_frames()
        
        # Process frames
        for frame_data in frame_iterator:
            if frame_data and hasattr(frame_data, 'frame') and hasattr(frame_data, 'camera_id'):
                images[frame_data.camera_id] = frame_data.frame
                
                if hasattr(frame_data, 'annotations') and frame_data.annotations:
                    annotations[frame_data.camera_id] = frame_data.annotations
                    
        return {'images': images, 'annotations': annotations}
        """Initialize the pipeline visualizer"""
        vis_config = {
            'output_dir': str(Path(self.config.output_dir) / "visualizations")
        }
        self.visualizer = PipelineVisualizer(**vis_config)
        
        # Set visualization settings after initialization
        if hasattr(self.visualizer, 'figure_size'):
            figure_size = self.hydra_cfg.get('visualization', {}).get('figure_size', [12, 8])
            self.visualizer.figure_size = tuple(figure_size)
        
        if hasattr(self.visualizer, 'dpi'):
            self.visualizer.dpi = self.hydra_cfg.get('visualization', {}).get('dpi', 300)
        
        self.logger.info("Pipeline visualizer initialized")
    
    def run_comprehensive_workflow(self):
        """
        Run the complete comprehensive workflow:
        1. Load data using roboflow data source
        2. Rectify images and save results
        3. Triangulate ground truth and save animation
        4. Run YOLO pose estimation and save predictions + visualizations
        5. Triangulate YOLO predictions and save animation
        6. Reproject both triangulations and compute evaluation metrics
        7. Save all reprojections and results
        """
        self.logger.info("Starting Comprehensive Computer Vision Workflow")
        self.logger.info("=" * 60)
        
        try:
            # Initialize visualizer
            self.initialize_visualizer()
            
            # Initialize data source with full Hydra config instead of base pipeline init
            if not self.initialize_data_source_with_hydra_config():
                self.logger.error("Failed to initialize data source")
                return False
            
            # Step 1: Load data and save raw images
            self.logger.info("=== Step 1: Loading Data ===")
            frame_data = self.load_roboflow_data()
            if not frame_data:
                self.logger.error("Failed to load frame data")
                return False
                
            raw_images = frame_data.get('images', {})
            annotations = frame_data.get('annotations', {})
            
            self.logger.info(f"Loaded {len(raw_images)} images")
            self.logger.info(f"Loaded {len(annotations)} annotations")
            
            # Save raw images
            self.visualizer.save_raw_images(raw_images)
            
            # Step 2: Rectify images
            self.logger.info("=== Step 2: Rectifying Images ===")
            rectified_images = {}
            for camera_id, image in raw_images.items():
                rectified = self.rectifier.rectify_image(image, camera_id)
                if rectified is not None:
                    rectified_images[camera_id] = rectified
                    self.logger.info(f"Rectified image from {camera_id}")
            
            # Save rectified images
            if rectified_images:
                self.visualizer.save_rectified_images(rectified_images)
                self.logger.info(f"Saved {len(rectified_images)} rectified images")
            
            # Step 3: Triangulate ground truth
            self.logger.info("=== Step 3: Triangulating Ground Truth ===")
            ground_truth_poses = {}
            for camera_id, annotation in annotations.items():
                if 'keypoints' in annotation:
                    ground_truth_poses[camera_id] = {
                        'keypoints': annotation['keypoints'],
                        'scores': annotation.get('scores', [1.0] * len(annotation['keypoints'])),
                        'annotations': annotation
                    }
            
            triangulated_gt = None
            if ground_truth_poses:
                # Prepare keypoints for triangulation
                keypoints_2d = {}
                for camera_id, pose_data in ground_truth_poses.items():
                    keypoints_2d[camera_id] = pose_data['keypoints']
                
                triangulation_result = self.triangulator.triangulate(keypoints_2d)
                if triangulation_result and 'skeleton_3d' in triangulation_result:
                    triangulated_gt = triangulation_result['skeleton_3d']
                    
                    # Save triangulation animation
                    self.visualizer.save_triangulated_skeleton(
                        triangulation_result, suffix="ground_truth"
                    )
                    self.logger.info("Ground truth triangulation completed")
            
            # Step 4: YOLO Pose Estimation
            self.logger.info("=== Step 4: YOLO Pose Estimation ===")
            predicted_poses = {}
            
            # Use rectified images if available, otherwise raw images
            images_for_prediction = rectified_images if rectified_images else raw_images
            
            for camera_id, image in images_for_prediction.items():
                pose_result = self.pose_estimator.estimate_poses(image)
                if pose_result and 'keypoints' in pose_result:
                    predicted_poses[camera_id] = pose_result
                    self.logger.info(f"Pose estimation completed for {camera_id}")
            
            # Save pose estimation results with visualizations
            if predicted_poses:
                self.visualizer.save_pose_estimation_results(images_for_prediction, predicted_poses)
                
                # Save predictions in JSON format
                import json
                predictions_dir = Path(self.config.output_dir) / "predictions"
                predictions_dir.mkdir(parents=True, exist_ok=True)
                
                predictions_json = {}
                for camera_id, pose_data in predicted_poses.items():
                    predictions_json[camera_id] = {
                        'keypoints': pose_data['keypoints'].tolist() if hasattr(pose_data['keypoints'], 'tolist') else pose_data['keypoints'],
                        'scores': pose_data.get('scores', []).tolist() if hasattr(pose_data.get('scores', []), 'tolist') else pose_data.get('scores', []),
                        'bbox': pose_data.get('bbox', []).tolist() if hasattr(pose_data.get('bbox', []), 'tolist') else pose_data.get('bbox', [])
                    }
                
                with open(predictions_dir / "yolo_predictions.json", 'w') as f:
                    json.dump(predictions_json, f, indent=2)
                
                self.logger.info(f"Saved predictions for {len(predicted_poses)} images")
            
            # Step 5: Triangulate predictions
            self.logger.info("=== Step 5: Triangulating Predictions ===")
            triangulated_pred = None
            if predicted_poses:
                # Prepare keypoints for triangulation
                keypoints_2d = {}
                for camera_id, pose_data in predicted_poses.items():
                    keypoints_2d[camera_id] = pose_data['keypoints']
                
                triangulation_result = self.triangulator.triangulate(keypoints_2d)
                if triangulation_result and 'skeleton_3d' in triangulation_result:
                    triangulated_pred = triangulation_result['skeleton_3d']
                    
                    # Save triangulation animation
                    self.visualizer.save_triangulated_skeleton(
                        triangulation_result, suffix="predictions"
                    )
                    self.logger.info("Prediction triangulation completed")
            
            # Step 6: Reproject and evaluate
            self.logger.info("=== Step 6: Reprojection and Evaluation ===")
            reprojected_gt = {}
            reprojected_pred = {}
            
            # Reproject ground truth triangulation
            if triangulated_gt is not None:
                for camera_id in images_for_prediction.keys():
                    reprojection_result = self.evaluator.reproject_3d_to_2d(
                        triangulated_gt, camera_id
                    )
                    if reprojection_result:
                        reprojected_gt[camera_id] = reprojection_result
            
            # Reproject prediction triangulation
            if triangulated_pred is not None:
                for camera_id in images_for_prediction.keys():
                    reprojection_result = self.evaluator.reproject_3d_to_2d(
                        triangulated_pred, camera_id
                    )
                    if reprojection_result:
                        reprojected_pred[camera_id] = reprojection_result
            
            # Compute evaluation metrics
            evaluation_metrics = {}
            if ground_truth_poses and predicted_poses:
                evaluation_metrics = self.evaluator.compute_metrics(
                    ground_truth_poses, predicted_poses
                )
                
                # Save evaluation metrics
                metrics_dir = Path(self.config.output_dir) / "evaluation"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                
                with open(metrics_dir / "evaluation_metrics.json", 'w') as f:
                    json.dump(evaluation_metrics, f, indent=2, default=str)
            
            # Save reprojections
            if reprojected_gt or reprojected_pred:
                self.visualizer.save_reprojected_poses(
                    images_for_prediction, reprojected_pred, reprojected_gt
                )
            
            # Step 7: Save comprehensive summary
            self.logger.info("=== Step 7: Saving Final Results ===")
            results_summary = {
                'pipeline_info': {
                    'total_images': len(raw_images),
                    'rectified_images': len(rectified_images),
                    'ground_truth_poses': len(ground_truth_poses),
                    'predicted_poses': len(predicted_poses),
                    'has_gt_triangulation': triangulated_gt is not None,
                    'has_pred_triangulation': triangulated_pred is not None,
                    'gt_reprojections': len(reprojected_gt),
                    'pred_reprojections': len(reprojected_pred)
                },
                'evaluation_metrics': evaluation_metrics,
                'configuration': dict(self.hydra_cfg)
            }
            
            summary_file = Path(self.config.output_dir) / "comprehensive_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            # Log final summary
            self.logger.info("=== Pipeline Execution Summary ===")
            self.logger.info(f"Total images processed: {len(raw_images)}")
            self.logger.info(f"Rectified images: {len(rectified_images)}")
            self.logger.info(f"Ground truth poses: {len(ground_truth_poses)}")
            self.logger.info(f"Predicted poses: {len(predicted_poses)}")
            self.logger.info(f"Ground truth triangulation: {'✓' if triangulated_gt is not None else '✗'}")
            self.logger.info(f"Prediction triangulation: {'✓' if triangulated_pred is not None else '✗'}")
            self.logger.info(f"Reprojections computed: {'✓' if reprojected_gt or reprojected_pred else '✗'}")
            
            if evaluation_metrics:
                self.logger.info("Evaluation metrics:")
                for metric, value in evaluation_metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {metric}: {value:.4f}")
            
            self.logger.info(f"All results saved to: {self.config.output_dir}")
            self.logger.info("🎉 Comprehensive Pipeline Completed Successfully!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Comprehensive workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False


@hydra.main(version_base=None, config_path="configs", config_name="comprehensive_pipeline")
def main(cfg: DictConfig) -> None:
    """Main function to run the comprehensive pipeline"""
    
    print("🚀 Starting Comprehensive Computer Vision Pipeline")
    print("=" * 60)
    
    try:
        # Create and initialize pipeline
        pipeline = ComprehensiveHydraPipeline(cfg)
        
        # Run comprehensive workflow (initialization happens inside)
        success = pipeline.run_comprehensive_workflow()
        
        if success:
            print("\n🎉 Comprehensive pipeline completed successfully!")
            print(f"📁 Check results in: {cfg.pipeline.output_dir}")
        else:
            print("\n❌ Pipeline failed. Check logs for details.")
            
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
