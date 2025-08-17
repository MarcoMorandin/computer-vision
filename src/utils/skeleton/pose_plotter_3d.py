import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.dataset.coco_utils import COCOManager
from utils.dataset.skeleton_3d import SkeletonManager3D


class PosePlotter3D:
    """A class for plotting 3D triangulated poses using COCO skeleton structure."""

    def __init__(self, coco_manager: COCOManager, skeleton_manager: SkeletonManager3D):
        """Initialize the PosePlotter3D."""
        self.coco_manager = coco_manager
        self.skeleton_manager = skeleton_manager
        self.keypoint_names = coco_manager.get_keypoint_names()
        self.skeleton_connections = [
            (s - 1, e - 1) for s, e in coco_manager.get_skeleton()
        ]

        self.keypoint_colors = {
            "all": "red"  # Red for all keypoints
        }
        self.connection_color = "green"  # Green for all connections
        self.side_indices = {
            side: {i for i, n in enumerate(self.keypoint_names) if n.lower().startswith(side)}
            for side in ["left", "right"]
        }
        self.side_indices["center"] = set(range(len(self.keypoint_names))) - self.side_indices["left"] - self.side_indices["right"]
        # Sort frame IDs numerically instead of lexicographically
        self.frame_ids = sorted(self.skeleton_manager.get_frame_ids(), key=lambda x: int(x))

        # Get poses data from skeleton manager
        self.poses_data = self._load_poses_from_skeleton_manager()
        

    def _load_poses_from_skeleton_manager(self):
        """Load 3D poses from the skeleton manager and convert to a NumPy array."""
        # Use the public methods from SkeletonManager3D
        num_kps = len(self.keypoint_names)

        poses_data = np.full((len(self.frame_ids), num_kps, 3), np.nan)

        for i, frame_id in enumerate(self.frame_ids):
            keypoints_3d = self.skeleton_manager.get_frame(frame_id)
            if keypoints_3d is not None:
                # Convert to numpy array and handle None values
                keypoints_array = []
                for kp in keypoints_3d:
                    if kp is not None and len(kp) == 3:
                        keypoints_array.append(kp)
                
                keypoints_array = np.array(keypoints_array)
                
                # Fill poses_data with available keypoints
                if keypoints_array.shape[0] <= num_kps:
                    poses_data[i, :keypoints_array.shape[0]] = keypoints_array
                else:
                    poses_data[i] = keypoints_array[:num_kps]
        
        return poses_data

    def _compute_bounds(self):
        """Computes the global bounds for all poses to set axis limits."""
        valid_points = self.poses_data[~np.isnan(self.poses_data)].reshape(-1, 3)
        if valid_points.size == 0:
            return [(-1, 1)] * 3
        
        min_vals, max_vals = valid_points.min(axis=0), valid_points.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # Avoid division by zero
        margins = ranges * 0.05
        
        return [(mi - m, ma + m) for mi, ma, m in zip(min_vals, max_vals, margins)]

    def _setup_plot(self, title: str):
        """Initializes the 3D plot figure, axes, and artists."""
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        xlim, ylim, zlim = self._compute_bounds()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        # Calculate ranges from the limits to be used in set_box_aspect
        ranges = (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
        ax.set_box_aspect(ranges)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        
        artists = {}
        # Single scatter plot for all keypoints in red
        artists["scatter"] = ax.scatter([], [], [], color=self.keypoint_colors["all"], s=40, label="Keypoints")
        ax.legend(loc="upper right")

        # All connections in green
        artists["lines"] = [
            ax.plot([], [], [], color=self.connection_color, linewidth=2)[0]
            for i, j in self.skeleton_connections
        ]
        return fig, ax, artists

    def _update_plot(self, frame_idx: int, artists: Dict, ax: Axes3D):
        """Updates the plot artists for a given frame."""
        pose = self.poses_data[frame_idx]
        
        # Update single scatter plot with all valid keypoints
        valid_points = pose[~np.isnan(pose)].reshape(-1, 3)
        if valid_points.size > 0:
            artists["scatter"]._offsets3d = (valid_points[:, 0], valid_points[:, 1], valid_points[:, 2])
        else:
            artists["scatter"]._offsets3d = ([], [], [])

        # Update skeleton connections
        for line, (i, j) in zip(artists["lines"], self.skeleton_connections):
            p1, p2 = pose[i], pose[j]
            if not np.isnan(p1).any() and not np.isnan(p2).any():
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        if "texts" in artists:
            for text in artists["texts"]:
                text.remove()
            artists["texts"].clear()
        
        all_artists = [artists["scatter"]] + artists["lines"] + artists.get("texts", [])
        return all_artists

    def plot_single_frame(self, frame_id: int, save: str):
        """Plots a single frame of the 3D pose."""
        if frame_id not in self.frame_ids:
            raise ValueError(f"Frame {frame_id} not found.")
        
        frame_idx = self.frame_ids.index(frame_id)
        title = f"Frame {frame_id}"
        fig, ax, artists = self._setup_plot(title)

        self._update_plot(frame_idx, artists, ax)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=150)

    def animate_frames(self, save: str, interval: int = 100):
        """Animates the sequence of 3D poses."""
        if self.poses_data.size == 0:
            raise ValueError("No frames available for animation.")
        
        title = "3D Pose Animation"
        fig, ax, artists = self._setup_plot(title)

        def update(frame_idx):
            ax.set_title(f"{title} | Frame {self.frame_ids[frame_idx]}")
            return self._update_plot(frame_idx, artists, ax)

        anim = FuncAnimation(fig, update, frames=len(self.frame_ids), interval=interval, blit=False)
        
        fps = max(1, int(1000 / interval))
        writer = "ffmpeg" if save.lower().endswith(".mp4") else "imagemagick"
        os.makedirs(os.path.dirname(save), exist_ok=True)
        anim.save(save, writer=writer, fps=fps)
        print(f"Saved animation to {save}")