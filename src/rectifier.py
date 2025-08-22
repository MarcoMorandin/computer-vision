"""Rectification utilities for datasets and videos.

Rectify images/frames and 2D keypoints using per-camera calibration
(intrinsics + distortion) provided by the CameraManager.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any
from tqdm import tqdm

from .utils.camera.camera_manager import CameraManager
from .utils.dataset.coco_utils import COCOManager
from .utils.file_utils import extract_camera_number
from .utils.geometry import calculate_bbox_from_keypoints


class Rectifier:
    """
    A class for rectifying COCO datasets or videos by undistorting images/frames
    using camera calibration parameters.
    """

    def __init__(self, camera_manager: CameraManager):
        """Initialize the rectifier.

        Parameters
        ----------
        camera_manager : CameraManager
            Provides access to per-camera calibration (intrinsics + distortion).
        """
        self.camera_manager = camera_manager

    def _rectify_keypoints(
        self, keypoints: List[float], mtx: np.ndarray, dist: np.ndarray
    ) -> List[float]:
        """Undistort visible keypoints in COCO triplet format.

        Parameters
        ----------
        keypoints : list[float]
            Flattened COCO keypoints [x1, y1, v1, x2, y2, v2, ...]. Only entries
            with v > 0 are rectified.
        mtx : np.ndarray
            3x3 intrinsic camera matrix.
        dist : np.ndarray
            Distortion coefficient vector.

        Returns
        -------
        list[float]
            New flattened keypoints with undistorted coordinates for visible points.
        """
        num_kpts = len(keypoints) // 3
        new_keypoints = keypoints.copy()

        # Collect visible keypoints
        visible_points = []
        indices = []
        for i in range(num_kpts):
            x, y, v = keypoints[3 * i], keypoints[3 * i + 1], keypoints[3 * i + 2]
            if v > 0:
                visible_points.append([[x, y]])
                indices.append(i)

        if not visible_points:
            return new_keypoints

        # Rectify points
        pts = np.array(visible_points, dtype=np.float32)
        undist = cv2.undistortPoints(pts, mtx, dist, P=mtx)

        # Update keypoints
        for j, i in enumerate(indices):
            new_keypoints[3 * i] = float(undist[j, 0, 0])
            new_keypoints[3 * i + 1] = float(undist[j, 0, 1])

        return new_keypoints

    def rectify_dataset(
        self, coco_dataset: COCOManager, input_images_dir: str, output_images_dir: str
    ) -> COCOManager:
        """Rectify an entire COCO dataset and update image paths and annotations.

        Parameters
        ----------
        coco_dataset : COCOManager
            Dataset to modify in place (images/annotations updated).
        input_images_dir : str
            Directory containing original input images.
        output_images_dir : str
            Directory to save undistorted images.

        Returns
        -------
        COCOManager
            The same manager instance with updated file_name paths and rectified annotations.
        """

        os.makedirs(output_images_dir, exist_ok=True)

        # Get images from COCO dataset
        images = coco_dataset.get_images()

        # Process each image
        for img_info in tqdm(images, desc="Rectifying images", unit="image"):
            self._process_single_image(
                img_info, input_images_dir, output_images_dir, coco_dataset
            )

        return coco_dataset

    def rectify_video(self, video_path: str, output_dir: str) -> None:
        """Undistort a single video using the camera calibration inferred from its filename.

        Parameters
        ----------
        video_path : str
            Path to the input video. The filename must contain the camera index
            (e.g., out3_frame_0001.mp4) so it can be parsed.
        output_dir : str
            Directory where the rectified video will be written with the same basename.
        """
        basename = os.path.basename(video_path)
        cam_index = extract_camera_number(basename)
        if cam_index is None:
            raise ValueError(f"Cannot extract camera index from: {basename}")

        # Get calibration
        camera = self.camera_manager.get_camera(int(cam_index))
        mtx, dist = camera.get_calibration_matrices()

        # Setup video capture and writer
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(output_dir, basename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Pre-compute undistortion maps for efficiency
        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, mtx, (width, height), cv2.CV_32FC1
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply undistortion
            rectified_frame = cv2.remap(
                frame, map_x, map_y, interpolation=cv2.INTER_LINEAR
            )
            out.write(rectified_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def _process_single_image(
        self,
        img_info: Dict[str, Any],
        input_dir: str,
        output_dir: str,
        coco_dataset: COCOManager,
    ) -> None:
        """Rectify a single image and its associated annotations.

        Parameters
        ----------
        img_info : dict
            COCO image dictionary (must include id and file_name).
        input_dir : str
            Directory containing the original image file.
        output_dir : str
            Directory where the rectified image will be saved.
        coco_dataset : COCOManager
            Manager used to update image paths and annotation fields.
        """
        file_name = img_info["file_name"]
        image_id = img_info["id"]
        cam_index = extract_camera_number(file_name)
        if cam_index is None:
            raise ValueError(f"Cannot extract camera index from: {file_name}")

        # Get calibration
        camera = self.camera_manager.get_camera(int(cam_index))
        mtx, dist = camera.get_calibration_matrices()

        # Rectify image
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        img = cv2.imread(input_path)
        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, mtx, img.shape[:2][::-1], cv2.CV_32FC1
        )
        rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path, rectified_img)

        # Update image path to the rectified image (absolute path)
        coco_dataset.update_image_path(image_id, output_path)

        # Get and rectify annotations for this image
        annotations = coco_dataset.get_annotations_by_image_id(image_id)
        for ann in annotations:
            if "keypoints" in ann:
                rectified_keypoints = self._rectify_keypoints(
                    ann["keypoints"], mtx, dist
                )
                new_bbox = calculate_bbox_from_keypoints(rectified_keypoints)
                new_area = new_bbox[2] * new_bbox[3]

                # Update annotation using COCODataset methods
                coco_dataset.update_annotation(
                    ann["id"],
                    keypoints=rectified_keypoints,
                    bbox=new_bbox,
                    area=new_area,
                )

    def __str__(self) -> str:
        return f"DatasetRectifier(camera_manager={self.camera_manager})"
