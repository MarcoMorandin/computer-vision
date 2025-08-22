"""Geometry utilities for coordinate transformations and bounding box calculations."""

from typing import Dict, List


def calculate_bbox_from_keypoints(keypoints: List[float]) -> List[float]:
    """Calculate a tight bounding box around visible keypoints.

    Parameters
    ----------
    keypoints : list[float]
        Flattened COCO format [x1, y1, v1, x2, y2, v2, ...].

    Returns
    -------
    list[float]
        Bounding box in COCO format [x, y, width, height].

    Raises
    ------
    ValueError
        If no visible keypoints are found.
    """
    visible_coords = []
    for i in range(len(keypoints) // 3):
        if keypoints[3 * i + 2] > 0:  # visible
            visible_coords.extend([keypoints[3 * i], keypoints[3 * i + 1]])

    if not visible_coords:
        raise ValueError("No visible keypoints found")

    xs = visible_coords[::2]
    ys = visible_coords[1::2]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    return [minx, miny, maxx - minx, maxy - miny]


def keypoints_to_flat_array(
    keypoints_dict: Dict[str, Dict[str, float]],
    keypoint_names: List[str],
    confidence_threshold: float = 0.1,
) -> List[float]:
    """Convert a name-keyed dict to a flat COCO keypoint array.

    Parameters
    ----------
    keypoints_dict : dict[str, dict[str, float]]
        Mapping of keypoint name to {x, y, confidence}.
    keypoint_names : list[str]
        Ordered list of keypoint names.
    confidence_threshold : float, default=0.1
        Minimum confidence required to mark a keypoint visible (v=2).

    Returns
    -------
    list[float]
        Flattened keypoints in COCO format [x1, y1, v1, ...].
    """
    keypoints_flat = []
    for name in keypoint_names:
        kpt = keypoints_dict.get(name, {"x": 0, "y": 0, "confidence": 0})
        x, y, conf = kpt["x"], kpt["y"], kpt["confidence"]
        visibility = 2 if conf > confidence_threshold else 0
        keypoints_flat.extend([round(x, 2), round(y, 2), visibility])
    return keypoints_flat
