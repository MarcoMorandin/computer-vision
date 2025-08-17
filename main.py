import os
import sys
import logging
from pathlib import Path

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


# =========================
# 1) Configuration & Setup
# =========================

# Input/config paths
CALIB_BASE_DIR = Path("data/camera_data_v2")
COCO_DATASET_JSON = Path("data/dataset/_annotations.coco.json")
INPUT_IMAGES_DIR = Path("data/dataset")

WEIGHTS_YOLO_POSE = Path("weights/yolo11s-pose.pt")
WEIGHTS_YOLO_DET = Path("weights/yolo11s.pt")
VIT_MODEL_NAME = "usyd-community/vitpose-plus-base"

# Validation: a simple list of required paths
REQUIRED_INPUT_PATHS = [
    CALIB_BASE_DIR,
    COCO_DATASET_JSON,
    INPUT_IMAGES_DIR,
    WEIGHTS_YOLO_POSE,
    WEIGHTS_YOLO_DET,
]

# Output directory structure (dict for readability)
OUTPUT_BASE = Path("output")

OUT = {
    "rectified": {
        "root": OUTPUT_BASE / "rectified",
        "visualizations": OUTPUT_BASE / "rectified" / "visualizations",
    },
    "ground_truth": {
        "triangulated": {
            "root": OUTPUT_BASE / "ground_truth" / "triangulated",
            "visualizations": OUTPUT_BASE / "ground_truth" / "triangulated" / "visualizations",
        },
        "reprojected": {
            "root": OUTPUT_BASE / "ground_truth" / "reprojected",
            "visualizations": OUTPUT_BASE / "ground_truth" / "reprojected" / "visualizations",
        },
    },
    "yolo": {
        "predictions": {
            "root": OUTPUT_BASE / "yolo" / "predictions",
            "visualizations": OUTPUT_BASE / "yolo" / "predictions" / "visualizations",
        },
        "triangulations": {
            "root": OUTPUT_BASE / "yolo" / "triangulations",
            "visualizations": OUTPUT_BASE / "yolo" / "triangulations" / "visualizations",
        },
        "reprojections": {
            "root": OUTPUT_BASE / "yolo" / "reprojections",
            "visualizations": OUTPUT_BASE / "yolo" / "reprojections" / "visualizations",
        },
        "evaluations": {
            "predictions": OUTPUT_BASE / "yolo" / "evaluations" / "predictions",
            "reprojections": OUTPUT_BASE / "yolo" / "evaluations" / "reprojections",
        },
    },
    "vit": {
        "predictions": {
            "root": OUTPUT_BASE / "vit" / "predictions",
            "visualizations": OUTPUT_BASE / "vit" / "predictions" / "visualizations",
        },
        "triangulations": {
            "root": OUTPUT_BASE / "vit" / "triangulations",
            "visualizations": OUTPUT_BASE / "vit" / "triangulations" / "visualizations",
        },
        "reprojections": {
            "root": OUTPUT_BASE / "vit" / "reprojections",
            "visualizations": OUTPUT_BASE / "vit" / "reprojections" / "visualizations",
        },
        "evaluations": {
            "predictions": OUTPUT_BASE / "vit" / "evaluations" / "predictions",
            "reprojections": OUTPUT_BASE / "vit" / "evaluations" / "reprojections",
        },
    },
}

# Flat list of output dirs for creation
ALL_OUTPUT_DIRS = [
    OUT["rectified"]["root"],
    OUT["rectified"]["visualizations"],

    OUT["ground_truth"]["triangulated"]["root"],
    OUT["ground_truth"]["triangulated"]["visualizations"],
    OUT["ground_truth"]["reprojected"]["root"],
    OUT["ground_truth"]["reprojected"]["visualizations"],

    OUT["yolo"]["predictions"]["root"],
    OUT["yolo"]["predictions"]["visualizations"],
    OUT["yolo"]["triangulations"]["root"],
    OUT["yolo"]["triangulations"]["visualizations"],
    OUT["yolo"]["reprojections"]["root"],
    OUT["yolo"]["reprojections"]["visualizations"],
    OUT["yolo"]["evaluations"]["predictions"],
    OUT["yolo"]["evaluations"]["reprojections"],

    OUT["vit"]["predictions"]["root"],
    OUT["vit"]["predictions"]["visualizations"],
    OUT["vit"]["triangulations"]["root"],
    OUT["vit"]["triangulations"]["visualizations"],
    OUT["vit"]["reprojections"]["root"],
    OUT["vit"]["reprojections"]["visualizations"],
    OUT["vit"]["evaluations"]["predictions"],
    OUT["vit"]["evaluations"]["reprojections"],
]


def setup_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("pose_pipeline")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_dirs(paths_list) -> None:
    for p in paths_list:
        p.mkdir(parents=True, exist_ok=True)


def validate_inputs(required_paths, logger: logging.Logger) -> None:
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        logger.error("Missing required paths:\n" + "\n".join(f"- {m}" for m in missing))
        sys.exit(1)


def log_section(logger: logging.Logger, title: str) -> None:
    sep = "=" * 80
    logger.info(sep)
    logger.info(title)
    logger.info(sep)


# =========================
# 2) Helper steps
# =========================

def rectify_dataset(camera_manager: CameraManager,
                    coco_dataset: COCOManager,
                    input_images_dir: Path,
                    out_rectified_root: Path,
                    out_rectified_visuals: Path,
                    logger: logging.Logger) -> COCOManager:
    dataset_rectifier = DatasetRectifier(
        camera_manager=camera_manager, coco_dataset=coco_dataset, mode="coco"
    )

    rectified_coco = dataset_rectifier.rectify_dataset(
        input_images_dir=str(input_images_dir),
        output_images_dir=str(out_rectified_root),
    )

    rectified_json_path = out_rectified_root / "rectified.coco.json"
    rectified_coco.save(str(rectified_json_path))
    logger.info(f"Saved rectified annotations: {rectified_json_path}")

    SkeletonDrawer(rectified_coco).draw_skeleton_on_coco(
        rectified_coco, str(out_rectified_visuals)
    )
    logger.info(f"Saved rectified visualizations to: {out_rectified_visuals}")

    return rectified_coco


def triangulate_and_visualize(camera_manager: CameraManager,
                              coco_manager: COCOManager,
                              out_dir_root: Path,
                              out_visuals: Path,
                              logger: logging.Logger):
    triangulator = PlayerTriangulator(camera_manager, coco_manager)
    skeleton_manager = triangulator.triangulate(use_bundle_adjustment=True)

    skeleton_json_path = out_dir_root / "player_3d_poses.json"
    skeleton_manager.save(str(skeleton_json_path))
    logger.info(f"Saved 3D skeletons: {skeleton_json_path}")

    animation_path = out_visuals / "3d_pose_animation.mp4"
    PosePlotter3D(coco_manager, skeleton_manager).animate_frames(save=str(animation_path))
    logger.info(f"Saved 3D pose animation: {animation_path}")

    return skeleton_manager


def reproject_draw_and_eval(camera_manager: CameraManager,
                            coco_for_reprojection: COCOManager,
                            skeleton_manager,
                            out_dir_root: Path,
                            out_visuals: Path,
                            evaluator: PoseEvaluator,
                            gt_for_eval: COCOManager,
                            eval_out_dir: Path,
                            logger: logging.Logger) -> COCOManager:
    reprojector = SkeletonReprojector(
        camera_manager=camera_manager,
        coco_manager=coco_for_reprojection.copy(),
        skeleton_manager=skeleton_manager,
    )
    reprojected_coco = reprojector.reproject()

    reprojected_json_path = out_dir_root / "reprojected.coco.json"
    reprojected_coco.save(str(reprojected_json_path))
    logger.info(f"Saved reprojected annotations: {reprojected_json_path}")

    SkeletonDrawer(reprojected_coco).draw_skeleton_on_coco(
        reprojected_coco, str(out_visuals)
    )
    logger.info(f"Saved reprojected visualizations to: {out_visuals}")

    evaluator.evaluate(
        gt_manager=gt_for_eval,
        pred_manager=reprojected_coco,
        output_dir=str(eval_out_dir),
    )
    logger.info(f"Saved evaluation to: {eval_out_dir}")

    return reprojected_coco


# =========================
# 3) Pipeline
# =========================

def main():
    logger = setup_logging()
    validate_inputs(REQUIRED_INPUT_PATHS, logger)
    ensure_dirs(ALL_OUTPUT_DIRS)

    evaluator = PoseEvaluator()

    # Load Cameras & COCO Data
    log_section(logger, "Load cameras and COCO dataset")
    camera_manager = CameraManager()
    camera_manager.load_cameras(str(CALIB_BASE_DIR))
    logger.info(f"Loaded cameras from: {CALIB_BASE_DIR}")

    coco_dataset = COCOManager(str(COCO_DATASET_JSON))
    logger.info(f"Loaded COCO dataset: {COCO_DATASET_JSON}")

    # Rectify Dataset (images + annotations)
    log_section(logger, "Rectify dataset (images + annotations)")
    rectified_coco = rectify_dataset(
        camera_manager=camera_manager,
        coco_dataset=coco_dataset,
        input_images_dir=INPUT_IMAGES_DIR,
        out_rectified_root=OUT["rectified"]["root"],
        out_rectified_visuals=OUT["rectified"]["visualizations"],
        logger=logger,
    )

    # Ground Truth: Triangulate and Reproject
    log_section(logger, "Ground truth: Triangulate 3D poses and visualize")
    gt_skeletons = triangulate_and_visualize(
        camera_manager=camera_manager,
        coco_manager=rectified_coco,
        out_dir_root=OUT["ground_truth"]["triangulated"]["root"],
        out_visuals=OUT["ground_truth"]["triangulated"]["visualizations"],
        logger=logger,
    )

    log_section(logger, "Ground truth: Reproject 3D -> 2D and evaluate")
    _ = reproject_draw_and_eval(
        camera_manager=camera_manager,
        coco_for_reprojection=rectified_coco,
        skeleton_manager=gt_skeletons,
        out_dir_root=OUT["ground_truth"]["reprojected"]["root"],
        out_visuals=OUT["ground_truth"]["reprojected"]["visualizations"],
        evaluator=evaluator,
        gt_for_eval=rectified_coco,
        eval_out_dir=OUT["ground_truth"]["reprojected"]["root"],
        logger=logger,
    )

    # Prepare a pruned GT for model evaluations
    rectified_coco_pruned = rectified_coco.copy()
    rectified_coco_pruned.prune_keypoints(["foot"])
    logger.info("Pruned GT keypoints for evaluations: ['foot']")

    # YOLO: Predictions, Triangulations, Reprojections, Evaluations
    log_section(logger, "YOLO: Pose estimation, visualization, and evaluation")
    pose_estimator_yolo = YOLOPoseEstimator(
        coco_manager=rectified_coco.copy(),
        model_weights_path=str(WEIGHTS_YOLO_POSE),
    )
    yolo_predicted = pose_estimator_yolo.run_pose_estimation(confidence_threshold=0.25)

    yolo_json_path = OUT["yolo"]["predictions"]["root"] / "yolo_predictions.coco.json"
    yolo_predicted.save(str(yolo_json_path))
    logger.info(f"Saved YOLO predictions: {yolo_json_path}")

    SkeletonDrawer(yolo_predicted).draw_skeleton_on_coco(
        yolo_predicted, str(OUT["yolo"]["predictions"]["visualizations"])
    )
    logger.info(f"Saved YOLO prediction visualizations to: {OUT['yolo']['predictions']['visualizations']}")

    evaluator.evaluate(
        gt_manager=rectified_coco_pruned,
        pred_manager=yolo_predicted,
        output_dir=str(OUT["yolo"]["evaluations"]["predictions"]),
    )
    logger.info(f"Saved YOLO predictions evaluation to: {OUT['yolo']['evaluations']['predictions']}")

    log_section(logger, "YOLO: Triangulate predicted 2D and visualize")
    yolo_skeletons = triangulate_and_visualize(
        camera_manager=camera_manager,
        coco_manager=yolo_predicted,
        out_dir_root=OUT["yolo"]["triangulations"]["root"],
        out_visuals=OUT["yolo"]["triangulations"]["visualizations"],
        logger=logger,
    )

    log_section(logger, "YOLO: Reproject 3D -> 2D, visualize, and evaluate")
    _ = reproject_draw_and_eval(
        camera_manager=camera_manager,
        coco_for_reprojection=rectified_coco_pruned,
        skeleton_manager=yolo_skeletons,
        out_dir_root=OUT["yolo"]["reprojections"]["root"],
        out_visuals=OUT["yolo"]["reprojections"]["visualizations"],
        evaluator=evaluator,
        gt_for_eval=rectified_coco_pruned,
        eval_out_dir=OUT["yolo"]["evaluations"]["reprojections"],
        logger=logger,
    )

    # ViT: Predictions, Triangulations, Reprojections, Evaluations
    log_section(logger, "ViT: Pose estimation, visualization, and evaluation")
    pose_estimator_vit = ViTPoseEstimator(
        coco_manager=rectified_coco.copy(),
        detector_yolo_weights_path=str(WEIGHTS_YOLO_DET),
        vit_model_name=VIT_MODEL_NAME,
    )
    vit_predicted = pose_estimator_vit.run_pose_estimation()

    vit_json_path = OUT["vit"]["predictions"]["root"] / "vit_predictions.coco.json"
    vit_predicted.save(str(vit_json_path))
    logger.info(f"Saved ViT predictions: {vit_json_path}")

    SkeletonDrawer(vit_predicted).draw_skeleton_on_coco(
        vit_predicted, str(OUT["vit"]["predictions"]["visualizations"])
    )
    logger.info(f"Saved ViT prediction visualizations to: {OUT['vit']['predictions']['visualizations']}")

    evaluator.evaluate(
        gt_manager=rectified_coco_pruned,
        pred_manager=vit_predicted,
        output_dir=str(OUT["vit"]["evaluations"]["predictions"]),
    )
    logger.info(f"Saved ViT predictions evaluation to: {OUT['vit']['evaluations']['predictions']}")

    log_section(logger, "ViT: Triangulate predicted 2D and visualize")
    vit_skeletons = triangulate_and_visualize(
        camera_manager=camera_manager,
        coco_manager=vit_predicted,
        out_dir_root=OUT["vit"]["triangulations"]["root"],
        out_visuals=OUT["vit"]["triangulations"]["visualizations"],
        logger=logger,
    )

    log_section(logger, "ViT: Reproject 3D -> 2D, visualize, and evaluate")
    _ = reproject_draw_and_eval(
        camera_manager=camera_manager,
        coco_for_reprojection=rectified_coco_pruned,
        skeleton_manager=vit_skeletons,
        out_dir_root=OUT["vit"]["reprojections"]["root"],
        out_visuals=OUT["vit"]["reprojections"]["visualizations"],
        evaluator=evaluator,
        gt_for_eval=rectified_coco_pruned,
        eval_out_dir=OUT["vit"]["evaluations"]["reprojections"],
        logger=logger,
    )


if __name__ == "__main__":
    main()