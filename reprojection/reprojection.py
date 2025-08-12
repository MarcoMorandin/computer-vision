import os
import json
import glob
import argparse
from copy import deepcopy
import numpy as np
import cv2

def load_calibration(calib_path, use_camera_center=False, force_zero_dist=True):
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    K = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32).flatten()
    if force_zero_dist:
        dist = np.zeros_like(dist)

    rvec = np.array(calib["rvecs"], dtype=np.float32).flatten()
    tvec = np.array(calib["tvecs"], dtype=np.float32).flatten()
    R, _ = cv2.Rodrigues(rvec)

    if use_camera_center:
        # If tvec is camera center C_world -> t_cam = -R @ C_world
        t_cam = -R @ tvec.reshape(3, 1)
        P = K @ np.hstack([R, t_cam])
    else:
        # Standard OpenCV: tvec is world origin in camera coordinates
        P = K @ np.hstack([R, tvec.reshape(3, 1)])

    return {
        "K": K,
        "dist": dist,
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "P": P
    }

def load_all_cameras(calib_base_dir, use_camera_center=False, force_zero_dist=True):
    cams = {}
    for cam_dir in glob.glob(os.path.join(calib_base_dir, "cam_*")):
        calib_path = os.path.join(cam_dir, "calib", "camera_calib.json")
        if not os.path.exists(calib_path):
            continue
        cam_name = os.path.basename(cam_dir)  # e.g., "cam_1"
        cam_id = cam_name.split("_")[-1]      # "1"
        cams[cam_id] = load_calibration(calib_path, use_camera_center, force_zero_dist)
    return cams

def project_points_with_P(points_3d, P):
    """
    points_3d: (N,3)
    P: 3x4
    Returns (N,2) pixel coordinates
    """
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)])
    x = (P @ X_h.T).T
    uv = x[:, :2] / x[:, 2:3]
    return uv

def depth_in_camera(points_3d, R, tvec):
    """
    Compute Z in camera coords: Z_cam = (R X + t)_z
    points_3d: (N,3)
    """
    X_cam = (R @ points_3d.T + tvec.reshape(3,1)).T
    return X_cam[:, 2]

def reproject_3d_skeletons_to_2d(skeleton_3d_path, calib_base_dir, out_path=None,
                                 use_camera_center=False, force_zero_dist=True,
                                 drop_behind_camera=True):
    # Load 3D skeletons {frame_id(str or int): [[x,y,z], ...]}
    with open(skeleton_3d_path, "r") as f:
        skeletons_3d = json.load(f)

    # Normalize frame keys to strings for consistent save
    skeletons_3d = {str(k): v for k, v in skeletons_3d.items()}

    # Load cameras
    cams = load_all_cameras(calib_base_dir, use_camera_center, force_zero_dist)
    if not cams:
        raise RuntimeError(f"No camera calibrations found in {calib_base_dir}/cam_*/calib/camera_calib.json")

    # Prepare output: {frame_id: {cam_id: [[u,v] or None, ...]}}
    reprojected = {}

    for frame_id, keypoints_3d in sorted(skeletons_3d.items(), key=lambda x: int(x[0])):
        pts3d = []
        # In your data, each keypoint is [x, y, z]; they all seem valid floats
        for kp in keypoints_3d:
            if kp is None:
                pts3d.append(None)
            else:
                pts3d.append([float(kp[0]), float(kp[1]), float(kp[2])])
        pts3d = np.array([p if p is not None else [np.nan, np.nan, np.nan] for p in pts3d], dtype=np.float64)

        reprojected[frame_id] = {}
        for cam_id, C in cams.items():
            # Compute 2D via P
            uv = project_points_with_P(pts3d, C["P"])

            # Optionally mark points behind camera as invalid
            if drop_behind_camera:
                z_cam = depth_in_camera(pts3d, C["R"], C["tvec"])
                valid = np.isfinite(uv).all(axis=1) & (z_cam > 1e-6)
            else:
                valid = np.isfinite(uv).all(axis=1)

            # Build per-kp list with None for invalid
            pts2d_list = []
            for i in range(uv.shape[0]):
                if valid[i]:
                    pts2d_list.append([float(uv[i,0]), float(uv[i,1])])
                else:
                    pts2d_list.append(None)

            reprojected[frame_id][cam_id] = pts2d_list

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(reprojected, f, indent=2)
        print(f"Saved reprojected 2D keypoints to: {out_path}")

    return reprojected, cams


def _infer_keypoint_names(n_kp, existing_names=None):
    if existing_names and len(existing_names) == n_kp:
        return existing_names
    return [f"kp_{i}" for i in range(n_kp)]


def _parse_cam_and_frame_from_name(name):
    # Expect pattern like out<cam_id>_frame_<frame:04d>.png
    try:
        base = os.path.splitext(os.path.basename(name))[0]
        # Remove possible hashed / roboflow added parts after original pattern
        # Find the segment containing 'out' and 'frame'
        # Example original: out5_frame_0004_png.rf.xxxxxx  -> extra.name has out5_frame_0004
        if "_png" in base:
            base = base.split("_png")[0]
        parts = base.split("_frame_")
        cam_part = parts[0]  # out5
        frame_part = parts[1]
        cam_id = cam_part.replace("out", "")
        frame_id = str(int(frame_part))  # strip zeros
        return cam_id, frame_id
    except Exception:
        return None, None


def build_coco_from_reprojected(reprojected, template_coco_path, output_coco_path,
                                image_width=3840, image_height=2160,
                                overwrite_keypoint_names=False,
                                category_name="person"):
    """
    reprojected: {frame_id: {cam_id: [[u,v] or None, ...]}}
    template_coco_path: path to existing COCO json to copy structure (info, licenses, categories)
    output_coco_path: where to write new coco json with annotations built from reprojected data
    """
    with open(template_coco_path, 'r') as f:
        template = json.load(f)

    # Deep copy base for output
    coco = {
        "info": template.get("info", {}),
        "licenses": template.get("licenses", []),
        "categories": deepcopy(template.get("categories", [])),
        "images": [],
        "annotations": []
    }

    # Determine number of keypoints from first available entry
    first_frame = next(iter(reprojected.values()))
    first_cam = next(iter(first_frame.values()))
    n_kp = len(first_cam)

    # Locate category to update / use
    cat_idx = None
    for i, c in enumerate(coco["categories"]):
        if c.get("name") == category_name:
            cat_idx = i
            break
    if cat_idx is None and coco["categories"]:
        cat_idx = 0  # fallback to first
    if cat_idx is None:
        # Create a category if none
        coco["categories"].append({
            "id": 1,
            "name": category_name,
            "supercategory": "",
            "keypoints": [],
            "skeleton": []
        })
        cat_idx = 0

    category = coco["categories"][cat_idx]
    existing_names = category.get("keypoints", [])
    if overwrite_keypoint_names or len(existing_names) != n_kp:
        category["keypoints"] = _infer_keypoint_names(n_kp, existing_names if not overwrite_keypoint_names else None)
    if "skeleton" not in category:
        category["skeleton"] = []
    category_id = category.get("id", 1)
    # Ensure unique category ids (if template had weird 0/1 ids keep them)
    if category_id is None:
        category_id = 1
        category["id"] = 1

    # Build mapping (cam_id, frame_id) -> reprojected keypoints
    # We will try to leverage images in template if they have an "extra" name field.
    template_images = template.get("images", [])
    used_pairs = set()
    image_id_counter = 0
    image_entries = []

    for img in template_images:
        name_field = img.get("extra", {}).get("name") or img.get("file_name")
        cam_id, frame_id = _parse_cam_and_frame_from_name(name_field)
        if cam_id is None or frame_id is None:
            continue
        if frame_id in reprojected and cam_id in reprojected[frame_id]:
            image_id = img.get("id")
            if image_id is None:
                image_id = image_id_counter
            image_id_counter = max(image_id_counter, image_id + 1)
            new_img = {
                "id": image_id,
                "file_name": img.get("file_name", name_field),
                "width": img.get("width", image_width),
                "height": img.get("height", image_height),
                "date_captured": img.get("date_captured", ""),
                "license": img.get("license", 1),
                "extra": img.get("extra", {"name": name_field})
            }
            image_entries.append(new_img)
            used_pairs.add((cam_id, frame_id, image_id))

    # If template lacked images for some projections, create them
    if not image_entries:
        # build synthetic names
        for frame_id, cams_dict in reprojected.items():
            for cam_id in cams_dict.keys():
                image_id = image_id_counter
                image_id_counter += 1
                file_name = f"out{cam_id}_frame_{int(frame_id):04d}.png"
                image_entries.append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": image_width,
                    "height": image_height,
                    "date_captured": "",
                    "license": 1,
                    "extra": {"name": file_name}
                })
                used_pairs.add((cam_id, frame_id, image_id))

    coco["images"] = image_entries

    # Build annotations
    ann_id = 0
    for img in coco["images"]:
        name_field = img.get("extra", {}).get("name") or img.get("file_name")
        cam_id, frame_id = _parse_cam_and_frame_from_name(name_field)
        if cam_id is None or frame_id is None:
            continue
        if frame_id not in reprojected or cam_id not in reprojected[frame_id]:
            continue
        kp_list = reprojected[frame_id][cam_id]
        # Build flattened keypoints list
        flat = []
        xs, ys = [], []
        for kp in kp_list:
            if kp is None:
                flat.extend([0, 0, 0])
            else:
                x, y = kp
                flat.extend([float(x), float(y), 2])  # visibility=2 (visible)
                xs.append(x)
                ys.append(y)
        if xs:
            x_min, y_min = float(np.min(xs)), float(np.min(ys))
            x_max, y_max = float(np.max(xs)), float(np.max(ys))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = float((x_max - x_min) * (y_max - y_min))
        else:
            # Skip annotation if no valid keypoints
            continue
        num_kp = int(sum(1 for v in flat[2::3] if v > 0))
        ann = {
            "id": ann_id,
            "image_id": img["id"],
            "category_id": category_id,
            "keypoints": flat,
            "num_keypoints": num_kp,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
        coco["annotations"].append(ann)
        ann_id += 1

    os.makedirs(os.path.dirname(output_coco_path), exist_ok=True)
    with open(output_coco_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations saved to: {output_coco_path} (images={len(coco['images'])}, annotations={len(coco['annotations'])})")

    return coco

def parse_args():
    p = argparse.ArgumentParser(description="Reproject 3D skeletons and export COCO annotations.")
    p.add_argument('--skeleton_3d', default="../triangulation/output/player_3d_poses.json", help='Path to 3D skeleton JSON')
    p.add_argument('--calib_base', default=os.path.join('..', 'data', 'camera_data_v2'), help='Base directory containing cam_*/calib/camera_calib.json')

    p.add_argument('--template_coco', default=os.path.join('..', 'rectification', 'output', 'dataset', 'train', '_annotations.coco.json'), help='Template COCO json to copy info/licenses/categories/images')
    p.add_argument('--coco_out', default='output/reprojected_annotations.json', help='Output COCO annotations json file')
    p.add_argument('--no_zero_dist', action='store_true', help='Use original distortion coefficients')
    p.add_argument('--use_camera_center', action='store_true', help='Interpret tvec as camera center')
    p.add_argument('--keep_keypoint_names', action='store_true', help='Keep template keypoint names even if count mismatches (will pad/truncate)')
    p.add_argument('--no_drop_behind_camera', action='store_true', help='Do NOT invalidate points behind camera')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reprojected, cams = reproject_3d_skeletons_to_2d(
        args.skeleton_3d,
        args.calib_base,
        use_camera_center=False,
        force_zero_dist=not args.no_zero_dist,
        drop_behind_camera=not args.no_drop_behind_camera
    )

    if args.template_coco and args.coco_out:
        build_coco_from_reprojected(
            reprojected,
            args.template_coco,
            args.coco_out,
            overwrite_keypoint_names=not args.keep_keypoint_names
        )