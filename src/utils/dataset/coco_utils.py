"""
COCO dataset utilities and keypoint mapping using pycocotools and supervision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
import copy


class COCOManager:
    def __init__(self, annotation_path: str):
        """Initialize a COCOManager.

        Parameters
        ----------
        annotation_path : str
            Path to COCO annotation JSON file.
        """
        self.coco = COCO(annotation_path)
        # Store original path for reference
        self.annotation_path = annotation_path

        # Extract and store keypoint info from dataset
        self._extract_keypoint_info()

    def _extract_keypoint_info(self) -> None:
        """Extract keypoint names and skeleton information from the COCO dataset."""
        person_cat = self.get_person_category()
        self.keypoint_names = person_cat.get("keypoints", [])
        self.skeleton = person_cat.get("skeleton", [])

    def save(self, path: str = None) -> None:
        """Save the in-memory dataset to disk.

        Parameters
        ----------
        path : str | None
            Destination path; defaults to the original annotation path.
        """
        if path is None:
            path = self.annotation_path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.coco.dataset, f, indent=4)

    def get_person_category(self) -> Dict[str, Any]:
        """Return the 'person' category dictionary.

        Returns
        -------
        dict[str, Any]
            Category entry for person.

        Raises
        ------
        ValueError
            If no 'person' category exists.
        """
        cat_ids = self.coco.getCatIds(catNms=["person"])
        if not cat_ids:
            raise ValueError("No 'person' category found in the dataset")
        return self.coco.cats[cat_ids[0]]

    def get_keypoint_names(self) -> List[str]:
        """Get keypoint names from the dataset.

        Returns
        -------
        list[str]
            List of keypoint names.
        """
        return self.keypoint_names

    def get_skeleton(self) -> List[List[int]]:
        """Get skeleton connections from the dataset.

        Returns
        -------
        list[list[int]]
            List of (start, end) pairs representing skeleton connections.
        """
        return self.skeleton

    def get_images(self) -> List[Dict[str, Any]]:
        """Get all images from the dataset.

        Returns
        -------
        list[dict[str, Any]]
            List of COCO image dictionaries.
        """
        return list(self.coco.imgs.values())

    def get_annotations_by_image_id(self, image_id: int) -> List[Dict[str, Any]]:
        """Get all annotations for a specific image.

        Parameters
        ----------
        image_id : int
            COCO image identifier.

        Returns
        -------
        list[dict[str, Any]]
            Annotations belonging to the given image.
        """
        return [
            ann
            for ann in self.coco.dataset["annotations"]
            if ann["image_id"] == image_id
        ]

    def update_image_path(self, image_id: int, new_path: str) -> None:
        """Update the file path for a specific image.

        Parameters
        ----------
        image_id : int
            Target image identifier.
        new_path : str
            New file_name value (path) to set.
        """
        for img in self.coco.dataset["images"]:
            if img["id"] == image_id:
                img["file_name"] = new_path
                break
        # Also update the internal imgs dict
        if image_id in self.coco.imgs:
            self.coco.imgs[image_id]["file_name"] = new_path

    def update_annotation(self, annotation_id: int, **kwargs) -> None:
        """Update specific fields of an annotation.

        Parameters
        ----------
        annotation_id : int
            Annotation identifier.
        **kwargs
            Fields to update on the annotation (e.g., keypoints, bbox, area).
        """
        for ann in self.coco.dataset["annotations"]:
            if ann["id"] == annotation_id:
                for key, value in kwargs.items():
                    ann[key] = value
                break
        # Also update the internal anns dict
        if annotation_id in self.coco.anns:
            for key, value in kwargs.items():
                self.coco.anns[annotation_id][key] = value

    def prune_keypoints(self, remove_patterns: List[str]) -> List[str]:
        """Prune keypoints by name and update annotations accordingly.

        Removes keypoints whose names match any provided substring (case-insensitive),
        updates the skeleton, and rebuilds each person annotation's keypoints array,
        num_keypoints, bbox, and area.

        Parameters
        ----------
        remove_patterns : list[str]
            Substrings to match for removal (e.g., ["foot", "toe"]).

        Returns
        -------
        list[str]
            Ordered list of kept keypoint names.
        """
        person_cat = self.get_person_category()
        original_kpts = person_cat.get("keypoints", [])

        # Decide which keypoint names to keep
        kept_kpts = [
            name
            for name in original_kpts
            if not any(p.lower() in name.lower() for p in remove_patterns)
        ]

        # Fast exit if nothing changes
        if len(kept_kpts) == len(original_kpts):
            return kept_kpts

        # Indices (0-based) of keypoints we keep
        kept_indices = [i for i, name in enumerate(original_kpts) if name in kept_kpts]

        # Mapping old 1-based index -> new 1-based index (COCO format)
        old_to_new = {
            old_idx + 1: new_pos + 1 for new_pos, old_idx in enumerate(kept_indices)
        }

        # Update skeleton keeping only edges whose endpoints are kept, remapping indices
        new_skeleton = [
            [old_to_new[a], old_to_new[b]]
            for a, b in person_cat.get("skeleton", [])
            if a in old_to_new and b in old_to_new
        ]

        # Update category definition in dataset
        person_cat["keypoints"] = kept_kpts
        person_cat["skeleton"] = new_skeleton

        # Update local cached attributes
        self.keypoint_names = kept_kpts
        self.skeleton = new_skeleton

        # Person category id
        person_cat_id = person_cat["id"]

        # Rebuild each annotation's keypoints, bbox, num_keypoints, area
        for ann in self.coco.dataset.get("annotations", []):
            if ann.get("category_id") != person_cat_id:
                continue

            kp = ann.get("keypoints", [])
            if not kp:
                # Nothing to update
                ann["keypoints"] = []
                ann["num_keypoints"] = 0
                ann["bbox"] = [0, 0, 0, 0]
                ann["area"] = 0.0
                continue

            # kp is [x1,y1,v1,x2,y2,v2,...]; group into triplets
            new_kp = []
            for old_idx in kept_indices:
                base = old_idx * 3
                triplet = kp[base : base + 3]
                if len(triplet) == 3:
                    new_kp.extend(triplet)
                else:
                    # Safety: pad if malformed
                    new_kp.extend([0, 0, 0])

            # Recompute num_keypoints (visibility >0)
            num_keypoints = sum(1 for i in range(2, len(new_kp), 3) if new_kp[i] > 0)

            # Recompute bbox from visible keypoints
            visible_pts = [
                (new_kp[i], new_kp[i + 1])
                for i in range(0, len(new_kp), 3)
                if new_kp[i + 2] > 0
            ]
            if visible_pts:
                pts = np.array(visible_pts, dtype=float)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                bbox = [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                ]
                area = bbox[2] * bbox[3]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
                area = 0.0

            # Round bbox and keypoints (consistent with add_annotation)
            ann["keypoints"] = [round(float(x), 2) for x in new_kp]
            ann["num_keypoints"] = num_keypoints
            ann["bbox"] = [round(float(x), 2) for x in bbox]
            ann["area"] = float(area)

            # Internal dict self.coco.anns references same object; no extra sync needed.

        return kept_kpts

    def add_image(
        self,
        file_name: str,
        height: int,
        width: int,
        image_id: Optional[int] = None,
    ) -> int:
        """Add a new image to the dataset.

        Parameters
        ----------
        file_name : str
            Image file name.
        height : int
            Image height.
        width : int
            Image width.
        image_id : int | None, optional
            Specific id to assign; if None, a new id is generated.

        Returns
        -------
        int
            The id of the added image.

        Raises
        ------
        ValueError
            If the provided image_id already exists in the dataset.
        """
        if image_id is None:
            # Generate a new ID by finding the current maximum and adding 1
            image_id = max(self.coco.imgs.keys(), default=0) + 1
        elif image_id in self.coco.imgs:
            raise ValueError(f"Image ID {image_id} already exists in the dataset.")

        # Create the image dictionary following COCO format
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "2024-01-01T00:00:00.000000",
        }

        # Add to the dataset list and the internal pycocotools map
        self.coco.dataset["images"].append(image_info)
        self.coco.imgs[image_id] = image_info

        return image_id

    def add_annotation(
        self, image_id: int, category_id: int, keypoints: List[float], **kwargs
    ) -> int:
        """Add a new person annotation to the dataset.

        Parameters
        ----------
        image_id : int
            Target image id.
        category_id : int
            Person category id.
        keypoints : list[float]
            Flat list [x1, y1, v1, x2, y2, v2, ...].
        **kwargs
            Additional annotation attributes (e.g., iscrowd, segmentation).

        Returns
        -------
        int
            Id of the added annotation.
        """
        visible_pts = [
            (keypoints[i], keypoints[i + 1])
            for i in range(0, len(keypoints), 3)
            if keypoints[i + 2] > 0
        ]

        if not visible_pts:
            bbox = [0, 0, 0, 0]  # Default empty bbox
        else:
            pts = np.array(visible_pts)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        ann_id = max(self.coco.anns.keys(), default=0) + 1
        num_keypoints = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)

        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(float(x), 2) for x in bbox],
            "area": float(bbox[2] * bbox[3]),
            "keypoints": [round(float(x), 2) for x in keypoints],
            "num_keypoints": num_keypoints,
            "iscrowd": kwargs.get("iscrowd", 0),
            "segmentation": kwargs.get("segmentation", []),
        }

        self.coco.dataset["annotations"].append(ann)
        self.coco.anns[ann_id] = ann

        return ann_id

    def clear_annotations(self) -> None:
        """Clear all annotations from the dataset."""
        self.coco.dataset["annotations"] = []
        self.coco.anns = {}

    def clear_images(self) -> None:
        """Clear all images from the dataset."""
        self.coco.dataset["images"] = []
        self.coco.imgs = {}

    def copy(self) -> "COCOManager":
        """Create a completely independent copy of the COCO dataset."""
        # Create a new instance without loading from file
        new_manager = object.__new__(COCOManager)

        # Deep copy the dataset to ensure complete independence
        new_manager.coco = COCO()
        new_manager.coco.dataset = copy.deepcopy(self.coco.dataset)

        # Rebuild internal indices (imgs, anns, etc.)
        new_manager.coco.createIndex()

        # Copy other attributes
        new_manager.annotation_path = self.annotation_path
        new_manager.keypoint_names = copy.deepcopy(self.keypoint_names)
        new_manager.skeleton = copy.deepcopy(self.skeleton)

        return new_manager

    def merge(self, others: List["COCOManager"]) -> None:
        """Merge images and annotations from other datasets into this one.

        Parameters
        ----------
        others : list[COCOManager]
            Datasets to merge into the current dataset.
        """

        for other in others:
            # --- Merge images ---
            for img in other.get_images():
                added_image_id = self.add_image(
                    file_name=img["file_name"],
                    height=img["height"],
                    width=img["width"],
                )
                ann = other.get_annotations_by_image_id(img["id"])
                if len(ann) > 0:
                    self.add_annotation(
                        image_id=added_image_id,
                        category_id=ann[0]["category_id"],
                        keypoints=ann[0]["keypoints"],
                    )

        # Rebuild internal indices
        self.coco.createIndex()

    def __str__(self) -> str:
        return f"COCODataset(images={len(self.coco.dataset['images'])}, annotations={len(self.coco.dataset['annotations'])})"
