#!/usr/bin/env python3
import argparse
import json
import os
from typing import Tuple, Dict

import numpy as np
import cv2


def _pick(calib: Dict, keys, default=None):
    for k in keys:
        if k in calib:
            return calib[k]
    return default


def load_calibration(calib_path: str, ignore_distortion: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(calib_path, "r") as f:
        calib = json.load(f)

    K_list = _pick(calib, ["K", "mtx", "camera_matrix", "new_mtx", "new_camera_mtx"])
    if K_list is None:
        raise ValueError(f"Camera matrix not found in {calib_path}")
    K = np.array(K_list, dtype=np.float64).reshape(3, 3)

    if ignore_distortion:
        dist = np.zeros(5, dtype=np.float64)
    else:
        dist_list = _pick(calib, ["dist", "distCoeffs", "dist_coeffs"], default=[0, 0, 0, 0, 0])
        dist = np.array(dist_list, dtype=np.float64).reshape(-1)

    rvec_list = _pick(calib, ["rvec", "rvecs"])
    tvec_list = _pick(calib, ["tvec", "tvecs"])
    if rvec_list is None or tvec_list is None:
        raise ValueError(f"rvec/tvec not found in {calib_path}")

    rvec = np.array(rvec_list, dtype=np.float64).flatten()
    tvec = np.array(tvec_list, dtype=np.float64).flatten()
    if rvec.size > 3:
        rvec = rvec[:3]
    if tvec.size > 3:
        tvec = tvec[:3]

    return K, dist, rvec, tvec


def load_correspondences(corresp_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(corresp_path, "r") as f:
        data = json.load(f)
    pts3d = np.array(data.get("points3d") or data.get("points3d_world"), dtype=np.float64)
    pts2d = np.array(data.get("points2d"), dtype=np.float64)

    if pts3d.ndim != 2 or pts3d.shape[1] != 3:
        raise ValueError("points3d must be an array of shape (N,3)")
    if pts2d.ndim != 2 or pts2d.shape[1] != 2 or pts2d.shape[0] != pts3d.shape[0]:
        raise ValueError("points2d must be an array of shape (N,2) and match points3d length")
    return pts3d, pts2d


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def rmat_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def evaluate_assumption(K, dist, rvec_file, tvec_file, pts3d, pts2d,
                        treat_t_as="opencv", invert_r=False) -> Dict:
    """
    treat_t_as: "opencv" (tvec is t), or "camera_center" (tvec is C_world)
    invert_r: if True, use R^T instead of R
    Returns: dict with metrics
    """
    R_file = rodrigues(rvec_file)
    R_use = R_file.T if invert_r else R_file
    rvec_use = rmat_to_rvec(R_use)

    if treat_t_as == "opencv":
        t_use = tvec_file.copy().reshape(3, 1)
        C_world = -R_use.T @ t_use  # derived camera center (world)
        C_world = C_world.flatten()
    elif treat_t_as == "camera_center":
        C_world = tvec_file.copy().reshape(3, 1)
        t_use = (-R_use @ C_world)  # convert to OpenCV t
        C_world = C_world.flatten()
    else:
        raise ValueError("treat_t_as must be 'opencv' or 'camera_center'")

    # Project points
    img_pts, _ = cv2.projectPoints(pts3d, rvec_use, t_use, K, dist)
    img_pts = img_pts.reshape(-1, 2)

    # Depths
    Xc = (R_use @ pts3d.T).T + t_use.reshape(1, 3)
    depths = Xc[:, 2]
    pos_mask = depths > 0

    # Reprojection error (only for points in front of camera)
    valid_idx = np.where(pos_mask)[0]
    if valid_idx.size > 0:
        diffs = img_pts[valid_idx] - pts2d[valid_idx]
        errs = np.linalg.norm(diffs, axis=1)
        mean_err = float(np.mean(errs))
        median_err = float(np.median(errs))
    else:
        errs = np.array([])
        mean_err = float("inf")
        median_err = float("inf")

    behind_ratio = float(np.mean(~pos_mask))
    pos_ratio = float(np.mean(pos_mask))

    return {
        "assumption": f"t={treat_t_as}, R_inverted={invert_r}",
        "mean_err_px": mean_err,
        "median_err_px": median_err,
        "num_points": int(len(pts3d)),
        "num_valid": int(valid_idx.size),
        "pos_depth_ratio": pos_ratio,
        "behind_ratio": behind_ratio,
        "camera_center_world": C_world.tolist(),
    }


def decide_best(results):
    # Prefer higher pos_depth_ratio; break ties with lower mean_err_px
    results_sorted = sorted(
        results,
        key=lambda r: (-r["pos_depth_ratio"], r["mean_err_px"])
    )
    return results_sorted[0], results_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Detect whether tvec in your calibration is (A) world origin in camera coords (OpenCV) "
                    "or (B) camera center in world coords."
    )
    parser.add_argument("--calib", required=True, nargs="+",
                        help="Calibration JSON file(s). Will test each; conventions should be consistent.")
    parser.add_argument("--corresp", required=True,
                        help="JSON with 3D world points and their 2D image points: "
                             "{'points3d':[[x,y,z],...], 'points2d':[[u,v],...]} for the SAME camera.")
    parser.add_argument("--ignore-distortion", action="store_true",
                        help="Ignore distortion (assume undistorted/rectified images).")
    parser.add_argument("--also-test-rinv", action="store_true",
                        help="Also test with R inverted (in case rvec is camera->world).")
    args = parser.parse_args()

    pts3d, pts2d = load_correspondences(args.corresp)

    for calib_path in args.calib:
        print(f"\n=== Testing calibration: {calib_path} ===")
        K, dist, rvec, tvec = load_calibration(calib_path, ignore_distortion=args.ignore_distortion)

        tests = []
        for treat_t in ["opencv", "camera_center"]:
            tests.append(evaluate_assumption(K, dist, rvec, tvec, pts3d, pts2d,
                                             treat_t_as=treat_t, invert_r=False))
            if args.also_test_rinv:
                tests.append(evaluate_assumption(K, dist, rvec, tvec, pts3d, pts2d,
                                                 treat_t_as=treat_t, invert_r=True))

        best, ranked = decide_best(tests)

        # Print a small table
        for r in ranked:
            print(f"- {r['assumption']}: "
                  f"mean={r['mean_err_px']:.3f}px, median={r['median_err_px']:.3f}px, "
                  f"pos_depth={r['pos_depth_ratio']*100:.1f}%, valid={r['num_valid']}/{r['num_points']}, "
                  f"C_world={np.round(r['camera_center_world'], 3).tolist()}")

        print("\n>> Decision:")
        if "t=opencv" in best["assumption"]:
            print("   Likely convention: tvec is WORLD ORIGIN in CAMERA coords (OpenCV standard).")
            print("   Use P = K [R | t] and project with cv2.projectPoints(rvec, tvec).")
        else:
            print("   Likely convention: tvec is CAMERA CENTER in WORLD coords.")
            print("   Convert to OpenCV t as: t = -R @ C and then use P = K [R | t].")

        if "R_inverted=True" in best["assumption"]:
            print("   Note: Inverting R gave better results, suggesting rvec in file is camera->world. "
                  "Use R = Rodrigues(rvec).T for world->camera.")

    print("\nDone.")


if __name__ == "__main__":
    main()