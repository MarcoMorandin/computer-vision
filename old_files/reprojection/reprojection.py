"""Minimal script to reproject 3D skeleton keypoints onto each camera
and export a COCO-style annotations file.

Simplifications vs earlier version:
    * Hardcoded input / output paths (no CLI)
    * Removed fallback logic & optional branches
    * Always rebuild images list synthetically
    * Single category (id=1, name='person') always created
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_FILENAME_RE = re.compile(r"out(\d+)_frame_(\d+)")

@dataclass(frozen=True)
class CameraCalibration:
    """Container for intrinsic / extrinsic camera parameters.

    Attributes
    -----------
    K: (3,3) intrinsic matrix
    dist: (n,) distortion coefficients as stored (unused in current reprojection path)
    rvec: (3,) Rodrigues rotation vector
    tvec: (3,) translation vector
    R: (3,3) rotation matrix
    P: (3,4) projection matrix  K [R|t]
    """

    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    P: np.ndarray

    @staticmethod
    def from_json(path: Path) -> "CameraCalibration":
        with path.open("r") as f:
            calib = json.load(f)
        K = np.asarray(calib["mtx"], dtype=np.float32)
        dist = np.asarray(calib["dist"], dtype=np.float32).reshape(-1)
        rvec = np.asarray(calib["rvecs"], dtype=np.float32).reshape(-1)
        tvec = np.asarray(calib["tvecs"], dtype=np.float32).reshape(-1)
        R, _ = cv2.Rodrigues(rvec)
        P = K @ np.hstack([R, tvec[:, None]])
        return CameraCalibration(K, dist, rvec, tvec, R, P)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points (N,3) to pixel coords (N,2) using precomputed P.

        Points with NaNs propagate NaNs in output.
        """
        if points_3d.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        # Homogenize
        X_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)], axis=1)
        x = (self.P @ X_h.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = x[:, :2] / x[:, 2:3]
        return uv.astype(np.float32)

def load_all_cameras(calib_dir: Path) -> Dict[str, CameraCalibration]:
    cameras: Dict[str, CameraCalibration] = {}
    for cam_path in calib_dir.glob("cam_*"):
        calib_path = cam_path / "calib" / "camera_calib.json"
        if not calib_path.exists():
            continue
        cam_id = cam_path.name.split("_")[-1]
        cameras[cam_id] = CameraCalibration.from_json(calib_path)
    if not cameras:
        raise FileNotFoundError(f"No calibration files found in {calib_dir}")
    return cameras

def project_points_with_P(points_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    X_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)], axis=1)
    x = (P @ X_h.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = x[:, :2] / x[:, 2:3]
    return uv

def depth_in_camera(points_3d: np.ndarray, R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    X_cam = (R @ points_3d.T + tvec.reshape(3, 1)).T
    return X_cam[:, 2]

def _to_np_keypoints_3d(kps: Sequence[Optional[Sequence[float]]]) -> np.ndarray:
    """Convert a list of optional [x,y,z] to an (N,3) array with NaNs for missing."""
    arr = [kp if kp is not None else (np.nan, np.nan, np.nan) for kp in kps]
    return np.asarray(arr, dtype=np.float32)


def reproject_3d_skeletons_to_2d(skeleton_3d_path: Path, calib_dir: Path) -> Dict[str, Dict[str, List[Optional[List[float]]]]]:
    with skeleton_3d_path.open("r") as f:
        skel3d = json.load(f)
    skel3d = {str(k): v for k, v in skel3d.items()}
    cameras = load_all_cameras(calib_dir)

    out: Dict[str, Dict[str, List[Optional[List[float]]]]] = {}
    for frame_id, keypoints in sorted(skel3d.items(), key=lambda x: int(x[0])):
        pts3d = _to_np_keypoints_3d(keypoints)
        frame_result: Dict[str, List[Optional[List[float]]]] = {}
        for cam_id, calib in cameras.items():
            uv = calib.project(pts3d)
            valid = np.isfinite(uv).all(axis=1)
            frame_result[cam_id] = [ [float(u), float(v)] if valid[i] else None for i, (u, v) in enumerate(uv) ]
        out[frame_id] = frame_result
    return out

def build_coco(reprojected: Dict[str, Dict[str, List[Optional[List[float]]]]], out_path: Path) -> Dict[str, object]:
    """Build a COCO dictionary copying the preamble (info/licenses/categories)
    from TEMPLATE_COCO_PATH if it exists, otherwise falling back to a minimal
    single 'person' category. Images list is always generated freshly.
    """
    # Copy preamble from template if available
    with TEMPLATE_COCO_PATH.open("r") as f:
        template = json.load(f)
    info = template.get("info", {})
    licenses = template.get("licenses", [])
    categories = template.get("categories", [])
    
    # Build image list trying to preserve template file names for matching
    template_images = []
    template_images = template.get("images", [])
    
    # Locate / ensure 'person' category and its id
    person_cat = next((c for c in categories if c.get("name") == "person"), None)
    person_cat_id = person_cat["id"]

    images: List[dict] = []
    mapping: Dict[Tuple[str, str], int] = {}  # (frame_id, cam_id) -> image_id
    
    img_id_counter = 0
    # First pass: reuse template image names where possible
    for img in template_images:
        file_name = img.get("file_name", "")
        m = _FILENAME_RE.search(file_name)
        if not m:
            continue
        cam_id = m.group(1)
        frame_id = str(int(m.group(2)))
        if frame_id in reprojected and cam_id in reprojected[frame_id]:
            images.append({
                "id": img_id_counter,
                "file_name": file_name,
                "width": img.get("width", 3840),
                "height": img.get("height", 2160),
            })
            mapping[(frame_id, cam_id)] = img_id_counter
            img_id_counter += 1

    # Second pass: add any missing frame/cam combos with synthetic names
    for frame_id, cams in reprojected.items():
        for cam_id in cams.keys():
            if (frame_id, cam_id) in mapping:
                continue
            file_name = f"out{cam_id}_frame_{int(frame_id):04d}.png"
            images.append({
                "id": img_id_counter,
                "file_name": file_name,
                "width": 3840,
                "height": 2160,
            })
            mapping[(frame_id, cam_id)] = img_id_counter
            img_id_counter += 1

    # Build annotations using mapping
    annotations: List[dict] = []
    ann_id = 0
    for frame_id, cams in reprojected.items():
        for cam_id, kps in cams.items():
            image_id = mapping[(frame_id, cam_id)]
            flat: List[float] = []
            xs: List[float] = []
            ys: List[float] = []
            for kp in kps:
                if kp is None:
                    flat.extend([0, 0, 0])
                else:
                    x, y = kp
                    flat.extend([float(x), float(y), 2])
                    xs.append(x)
                    ys.append(y)
            if not xs:
                continue
            x_min, y_min = float(np.min(xs)), float(np.min(ys))
            x_max, y_max = float(np.max(xs)), float(np.max(ys))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = float((x_max - x_min) * (y_max - y_min))
            num_kp = int(sum(1 for v in flat[2::3] if v > 0))
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": person_cat_id,
                "keypoints": flat,
                "num_keypoints": num_kp,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved COCO file to {out_path} (images={len(images)}, annotations={len(annotations)})")
    return coco



SKELETON_3D_PATH = Path("../triangulation/output/player_3d_poses.json")
CALIB_DIR = Path("../data/camera_data_v2")
TEMPLATE_COCO_PATH = Path("../rectification/output/dataset/train/_annotations.coco.json")
COCO_OUT = Path("output/reprojected_annotations.json")


def main():
    reproj = reproject_3d_skeletons_to_2d(SKELETON_3D_PATH, CALIB_DIR)
    build_coco(reproj, COCO_OUT)


if __name__ == "__main__":
    main()