import os
import json
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from src.utils.dataset.coco_utils import COCOManager


class PoseEvaluator:
    """
    Pose evaluation using COCOManager.
    Computes PCK, PCKh, MPJPE, PA-MPJPE, MPJAE between a GT COCO dataset and a prediction/reprojection COCO dataset.
    """

    def __init__(self):
        pass

    def get_matching_annotations(
        self, gt_manager: COCOManager, pred_manager: COCOManager
    ) -> List[Dict]:
        matches = []

        gt_cat_id = gt_manager.get_person_category().get("id")
        pred_cat_id = pred_manager.get_person_category().get("id")

        # Build filename -> image_id map for GT
        gt_lookup = {img["file_name"]: img["id"] for img in gt_manager.get_images()}
        

        # Iterate prediction images
        for pred_img in pred_manager.get_images():
            filename = pred_img["file_name"]
            if filename not in gt_lookup:
                continue

            gt_img_id = gt_lookup[filename]
            pred_img_id = pred_img["id"]

            # Annotations
            gt_anns = [
                ann
                for ann in gt_manager.get_annotations_by_image_id(gt_img_id)
                if ann["category_id"] == gt_cat_id
            ]
            pred_anns = [
                ann
                for ann in pred_manager.get_annotations_by_image_id(pred_img_id)
                if ann["category_id"] == pred_cat_id
            ]
            if not gt_anns or not pred_anns:
                continue

            gt_ann = gt_anns[0]
            pred_ann = pred_anns[0]

            cam, frame = self.extract_camera_info(filename)

            matches.append(
                {
                    "filename": filename,
                    "camera": cam,
                    "frame": frame,
                    "gt_keypoints": np.array(gt_ann["keypoints"]).reshape(-1, 3),
                    "pred_keypoints": np.array(pred_ann["keypoints"]).reshape(-1, 3),
                }
            )
            
        return matches

    @staticmethod
    def extract_camera_info(filename: str) -> Tuple[str, int]:
        parts = filename.split("_")
        if len(parts) >= 3:
            camera = parts[0].replace("out", "")
            try:
                frame = int(parts[2])
            except ValueError:
                frame = -1
            return camera, frame
        return "", -1

    # ------------ Metric Helpers ------------

    @staticmethod
    def compute_head_size(keypoints: np.ndarray) -> float:
        if keypoints.shape[0] > 11 and keypoints[11, 2] > 0 and keypoints[10, 2] > 0:
            d = np.linalg.norm(keypoints[11, :2] - keypoints[10, :2])
            return max(d, 1.0)
        return 60.0

    def compute_pck(self, gt_kpts, pred_kpts, threshold, normalize="bbox") -> float:
        valid = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
        if not np.any(valid):
            return 0.0
        dists = np.linalg.norm(gt_kpts[:, :2] - pred_kpts[:, :2], axis=1)

        if normalize == "head":
            norm_factor = self.compute_head_size(gt_kpts)
        else:
            if gt_kpts[0, 2] > 0 and gt_kpts.shape[0] > 11 and gt_kpts[11, 2] > 0:
                norm_factor = np.linalg.norm(gt_kpts[11, :2] - gt_kpts[0, :2])
            else:
                norm_factor = 100.0
        thr_pixels = threshold * norm_factor
        correct = (dists <= thr_pixels) & valid
        return float(np.sum(correct) / np.sum(valid))

    @staticmethod
    def compute_mpjpe(gt_kpts, pred_kpts) -> float:
        valid = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
        if not np.any(valid):
            return float("inf")
        dists = np.linalg.norm(gt_kpts[:, :2] - pred_kpts[:, :2], axis=1)
        return float(np.mean(dists[valid]))

    @staticmethod
    def procrustes_alignment(gt_kpts, pred_kpts):
        valid = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
        if np.sum(valid) < 3:
            return pred_kpts[:, :2]
        gt_valid = gt_kpts[valid, :2]
        pred_valid = pred_kpts[valid, :2]
        gt_c = gt_valid - np.mean(gt_valid, axis=0)
        pred_c = pred_valid - np.mean(pred_valid, axis=0)
        H = pred_c.T @ gt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        s = np.linalg.norm(gt_c) / (np.linalg.norm(pred_c) + 1e-8)
        pred_all_c = pred_kpts[:, :2] - np.mean(pred_valid, axis=0)
        return s * (pred_all_c @ R.T) + np.mean(gt_valid, axis=0)

    def compute_angular_error(self, gt_kpts, pred_kpts, skeleton) -> float:
        errors = []
        for a, b in skeleton:
            i, j = a - 1, b - 1
            if (
                gt_kpts[i, 2] > 0
                and gt_kpts[j, 2] > 0
                and pred_kpts[i, 2] > 0
                and pred_kpts[j, 2] > 0
            ):
                gt_vec = gt_kpts[j, :2] - gt_kpts[i, :2]
                pr_vec = pred_kpts[j, :2] - pred_kpts[i, :2]
                gt_vec /= (np.linalg.norm(gt_vec) + 1e-8)
                pr_vec /= (np.linalg.norm(pr_vec) + 1e-8)
                cos_a = np.clip(np.dot(gt_vec, pr_vec), -1.0, 1.0)
                errors.append(np.degrees(np.arccos(cos_a)))
        return float(np.mean(errors)) if errors else float("inf")

    # ------------ Core Evaluation ------------

    def evaluate(
        self,
        gt_manager: COCOManager,
        pred_manager: COCOManager,
        output_dir: str,
    ) -> Dict:
        """
        Evaluate prediction (or reprojection) keypoints against ground truth.

        Args:
            gt_manager: COCOManager for ground truth annotations
            pred_manager: COCOManager for predicted/reprojected annotations
            output_dir: directory to store results JSON
            output_filename: name of the results file

        Returns:
            Dict with aggregated metrics.
        """
        os.makedirs(output_dir, exist_ok=True)

        skeleton = gt_manager.get_skeleton()
        matches = self.get_matching_annotations(gt_manager, pred_manager)


        metrics_store = {
            "pck_0.1": [],
            "pck_0.2": [],
            "pck_0.5": [],
            "pckh_0.1": [],
            "pckh_0.2": [],
            "pckh_0.5": [],
            "mpjpe": [],
            "pa_mpjpe": [],
            "mpjae": [],
        }

        for m in tqdm(matches, desc="Evaluating"):
            gt = m["gt_keypoints"]
            pred = m["pred_keypoints"]

            # PCK / PCKh
            metrics_store["pck_0.1"].append(self.compute_pck(gt, pred, 0.1, "bbox"))
            metrics_store["pck_0.2"].append(self.compute_pck(gt, pred, 0.2, "bbox"))
            metrics_store["pck_0.5"].append(self.compute_pck(gt, pred, 0.5, "bbox"))
            metrics_store["pckh_0.1"].append(self.compute_pck(gt, pred, 0.1, "head"))
            metrics_store["pckh_0.2"].append(self.compute_pck(gt, pred, 0.2, "head"))
            metrics_store["pckh_0.5"].append(self.compute_pck(gt, pred, 0.5, "head"))

            # MPJPE / PA-MPJPE
            mpjpe = self.compute_mpjpe(gt, pred)
            aligned = self.procrustes_alignment(gt, pred)
            pred_aligned = pred.copy()
            pred_aligned[:, :2] = aligned
            pa_mpjpe = self.compute_mpjpe(gt, pred_aligned)

            # Angular
            mpjae = self.compute_angular_error(gt, pred, skeleton)

            metrics_store["mpjpe"].append(mpjpe)
            metrics_store["pa_mpjpe"].append(pa_mpjpe)
            metrics_store["mpjae"].append(mpjae)

        results = {
            "overall": {
                "PCK@0.1": float(np.mean(metrics_store["pck_0.1"])) if matches else 0.0,
                "PCK@0.2": float(np.mean(metrics_store["pck_0.2"])) if matches else 0.0,
                "PCK@0.5": float(np.mean(metrics_store["pck_0.5"])) if matches else 0.0,
                "PCKh@0.1": float(np.mean(metrics_store["pckh_0.1"])) if matches else 0.0,
                "PCKh@0.2": float(np.mean(metrics_store["pckh_0.2"])) if matches else 0.0,
                "PCKh@0.5": float(np.mean(metrics_store["pckh_0.5"])) if matches else 0.0,
                "MPJPE": float(np.mean(metrics_store["mpjpe"])) if matches else 0.0,
                "PA-MPJPE": float(np.mean(metrics_store["pa_mpjpe"])) if matches else 0.0,
                "MPJAE": float(np.mean(metrics_store["mpjae"])) if matches else 0.0,
                "num_samples": len(matches),
            }
        }

        out_path = os.path.join(output_dir, "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
    
