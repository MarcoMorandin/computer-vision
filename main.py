import os

from src.pose_estimation.ViTPose.pose_estimator_vitpose import ViTPoseEstimator
from src.pose_estimation.yolo.pose_estimator_yolo import YOLOPoseEstimator
from src.triangulator import PlayerTriangulator
from src.utils.camera.camera_manager import CameraManager
from src.utils.dataset.coco_utils import COCOManager
from src.rectifier import DatasetRectifier
from src.reprojector import SkeletonReprojector
from src.utils.skeleton.pose_plotter_2d import SkeletonDrawer
from src.utils.skeleton.pose_plotter_3d import PosePlotter3D
from src.utils.evaluation.pose_evaluator import PoseEvaluator


def main():
    calib_base_dir = os.path.join("data", "camera_data_v2")
    coco_dataset_json = os.path.join("data", "dataset", "_annotations.coco.json")

    evaluator = PoseEvaluator()

    camera_manager = CameraManager()
    camera_manager.load_cameras(calib_base_dir)
    print(f"Initialized: {camera_manager}")

    coco_dataset = COCOManager(coco_dataset_json)
    print(f"Initialized: {coco_dataset}")

    dataset_rectifier = DatasetRectifier(
        camera_manager=camera_manager, coco_dataset=coco_dataset, mode="coco"
    )
    print(f"Initialized: {dataset_rectifier}")

    rectified_coco_dataset = dataset_rectifier.rectify_dataset(
        input_images_dir=os.path.join("data", "dataset"),
        output_images_dir=os.path.join("output", "rectified"),
    )

    rectified_coco_dataset.save(
        os.path.join("output", "rectified", "rectified.coco.json")
    )

    SkeletonDrawer(rectified_coco_dataset).draw_skeleton_on_coco(
        rectified_coco_dataset, os.path.join("output", "rectified", "visualizations")
    )

    triangulator = PlayerTriangulator(camera_manager, rectified_coco_dataset)
    print(f"Initialized: {triangulator}")
    skeleton_manager = triangulator.triangulate(use_bundle_adjustment=True)

    skeleton_manager.save(
        os.path.join("output", "triangulated", "player_3d_poses.json")
    )

    PosePlotter3D(rectified_coco_dataset, skeleton_manager).animate_frames(
        save=os.path.join(
            "output", "triangulated", "visualizations", "3d_pose_animation.mp4"
        ),
    )

    reprojector = SkeletonReprojector(
        camera_manager=camera_manager,
        coco_manager=rectified_coco_dataset.copy(),
        skeleton_manager=skeleton_manager,
    )

    reprojected_coco = reprojector.reproject()
    reprojected_coco.save(
        os.path.join("output", "reprojected", "reprojected.coco.json")
    )

    SkeletonDrawer(reprojected_coco).draw_skeleton_on_coco(
        reprojected_coco, os.path.join("output", "reprojected", "visualizations")
    )

    evaluator.evaluate(
        gt_manager=rectified_coco_dataset,
        pred_manager=reprojected_coco,
        output_dir="output/reprojected",
    )

    pose_estimator_yolo = YOLOPoseEstimator(
        coco_manager=rectified_coco_dataset.copy(),
        model_weights_path="weights/yolo11s-pose.pt",
    )
    yolo_predicted_coco = pose_estimator_yolo.run_pose_estimation(
        confidence_threshold=0.25
    )

    yolo_predicted_coco.save(
        os.path.join("output", "yolo_predictions", "yolo_predictions.coco.json")
    )

    SkeletonDrawer(yolo_predicted_coco).draw_skeleton_on_coco(
        yolo_predicted_coco,
        os.path.join("output", "yolo_predictions", "visualizations"),
    )

    rectified_coco_dataset_pruned = rectified_coco_dataset.copy()
    rectified_coco_dataset_pruned.prune_keypoints(["foot"])

    evaluator.evaluate(
        gt_manager=rectified_coco_dataset_pruned,
        pred_manager=yolo_predicted_coco,
        output_dir="output/yolo_predictions",
    )

    pose_estimator_vit = ViTPoseEstimator(
        coco_manager=rectified_coco_dataset.copy(),
        detector_yolo_weights_path="weights/yolo11s.pt",
        vit_model_name="usyd-community/vitpose-plus-base",
    )
    vit_predicted_dataset = pose_estimator_vit.run_pose_estimation()

    vit_predicted_dataset.save(
        os.path.join("output", "vit_predictions", "vit_predictions.coco.json")
    )

    SkeletonDrawer(vit_predicted_dataset).draw_skeleton_on_coco(
        vit_predicted_dataset,
        os.path.join("output", "vit_predictions", "visualizations"),
    )

    evaluator.evaluate(
        gt_manager=rectified_coco_dataset_pruned,
        pred_manager=vit_predicted_dataset,
        output_dir="output/vit_predictions",
    )


if __name__ == "__main__":
    main()
