import os
import json
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from src.utils.dataset.coco_utils import COCOManager


class PoseEvaluator:
    """
    Pose evaluation using COCOManager.
    Computes PCKh, MPJPE, PA-MPJPE between a GT COCO dataset and a prediction/reprojection COCO dataset.
    """

    def __init__(self):
        pass

    def get_matching_annotations(
        self, gt_manager: COCOManager, pred_manager: COCOManager
    ) -> List[Dict]:
        matches = []

        gt_cat_id = gt_manager.get_person_category().get("id")
        pred_cat_id = pred_manager.get_person_category().get("id")

        # Build filename -> image_id map for both GT and predictions
        pred_lookup = {img["file_name"]: img["id"] for img in pred_manager.get_images()}
        
        # Process all GT images to include missing detections
        for gt_img in gt_manager.get_images():
            filename = gt_img["file_name"]
            gt_img_id = gt_img["id"]

            # Get GT annotations
            gt_anns = [
                ann
                for ann in gt_manager.get_annotations_by_image_id(gt_img_id)
                if ann["category_id"] == gt_cat_id
            ]
            
            if not gt_anns:
                continue  # Skip if no ground truth person in this image

            gt_ann = gt_anns[0]  # Take first person annotation
            cam, frame = self.extract_camera_info(filename)

            # Check if we have a prediction for this image
            if filename in pred_lookup:
                pred_img_id = pred_lookup[filename]
                pred_anns = [
                    ann
                    for ann in pred_manager.get_annotations_by_image_id(pred_img_id)
                    if ann["category_id"] == pred_cat_id
                ]
                
                if pred_anns:
                    # Normal case: both GT and prediction exist
                    pred_ann = pred_anns[0]
                    matches.append(
                        {
                            "filename": filename,
                            "camera": cam,
                            "frame": frame,
                            "gt_keypoints": np.array(gt_ann["keypoints"]).reshape(-1, 3),
                            "pred_keypoints": np.array(pred_ann["keypoints"]).reshape(-1, 3),
                            "missing_detection": False,
                        }
                    )
                else:
                    # Prediction image exists but no person detected
                    matches.append(
                        {
                            "filename": filename,
                            "camera": cam,
                            "frame": frame,
                            "gt_keypoints": np.array(gt_ann["keypoints"]).reshape(-1, 3),
                            "pred_keypoints": None,  # No prediction available
                            "missing_detection": True,
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

    def compute_pck(self, gt_kpts, pred_kpts, threshold) -> float:
        valid = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
        if not np.any(valid):
            return 0.0
        dists = np.linalg.norm(gt_kpts[:, :2] - pred_kpts[:, :2], axis=1)
       
        norm_factor = self.compute_head_size(gt_kpts)
       
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

        matches = self.get_matching_annotations(gt_manager, pred_manager)

        metrics_store = {
            "pckh_0.1": [],
            "pckh_0.2": [],
            "pckh_0.5": [],
            "mpjpe": [],
            "pa_mpjpe": [],
        }

        # Count missing detections for statistics
        missing_detections = sum(1 for m in matches if m["missing_detection"])
        successful_detections = len(matches) - missing_detections
        detection_rate = successful_detections / len(matches) if matches else 0.0
        counter = 0
        
        for m in tqdm(matches, desc="Evaluating"):
            gt = m["gt_keypoints"]
            pred = m["pred_keypoints"]
            if not m["missing_detection"]:
                # PCKh
                metrics_store["pckh_0.1"].append(self.compute_pck(gt, pred, 0.1))
                metrics_store["pckh_0.2"].append(self.compute_pck(gt, pred, 0.2))
                metrics_store["pckh_0.5"].append(self.compute_pck(gt, pred, 0.5))
                
                # MPJPE / PA-MPJPE
                mpjpe = self.compute_mpjpe(gt, pred)
                
                # Handle PA-MPJPE for missing detections    
                aligned = self.procrustes_alignment(gt, pred)
                pred_aligned = pred.copy()
                pred_aligned[:, :2] = aligned
                pa_mpjpe = self.compute_mpjpe(gt, pred_aligned)
                
                metrics_store["mpjpe"].append(mpjpe)
                metrics_store["pa_mpjpe"].append(pa_mpjpe)
                counter += 1

        results = {
            "overall": {
                "PCKh@0.1": float(np.mean(metrics_store["pckh_0.1"])),
                "PCKh@0.2": float(np.mean(metrics_store["pckh_0.2"])),
                "PCKh@0.5": float(np.mean(metrics_store["pckh_0.5"])),
                "MPJPE": float(np.mean(metrics_store["mpjpe"])),
                "PA-MPJPE": float(np.mean(metrics_store["pa_mpjpe"])),
                "num_samples": counter,
            },
            "detection_stats": {
                "total_ground_truth_samples": len(matches),
                "detection_rate": detection_rate,
                "missing_detection_rate": 1.0 - detection_rate,
            }
        }

        out_path = os.path.join(output_dir, "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
    
