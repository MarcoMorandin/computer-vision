#!/usr/bin/env python3
"""
Plot 3D triangulated poses (x, y, z) using your custom 18-keypoint skeleton.

Input JSON format (from your triangulation script):
{
  "<frame_number>": [
    [x, y, z] or null,  # keypoint 0
    ...
  ],
  ...
}

Usage:
  # Plot a single frame
  python plot_3d_poses_custom.py --input output/player_3d_poses.json --frame 42 --labels

  # Animate all frames
  python plot_3d_poses_custom.py --input output/player_3d_poses.json --animate

  # Save animation to MP4 (requires ffmpeg)
  python plot_3d_poses_custom.py --input output/player_3d_poses.json --animate --save out.mp4
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

# =========================
# Your skeleton definition
# =========================
KEYPOINT_NAMES = [
    "Hips",
    "RHip",
    "RKnee",
    "RAnkle",
    "RFoot",
    "LHip",
    "LKnee",
    "LAnkle",
    "LFoot",
    "Spine",
    "Neck",
    "Head",
    "RShoulder",
    "RElbow",
    "RHand",
    "LShoulder",
    "LElbow",
    "LHand"
]

# Edges provided as 1-based indices; convert to 0-based for Python
SKELETON_EDGES_1BASED = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [1, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [1, 10],
    [10, 11],
    [11, 12],
    [11, 13],
    [13, 14],
    [14, 15],
    [11, 16],
    [16, 17],
    [17, 18]
]
SKELETON_EDGES = [(i - 1, j - 1) for i, j in SKELETON_EDGES_1BASED]

# Auto-derive left/right/center sets from names
LEFT_SET = {i for i, n in enumerate(KEYPOINT_NAMES) if n.startswith("L")}
RIGHT_SET = {i for i, n in enumerate(KEYPOINT_NAMES) if n.startswith("R")}
CENTER_SET = set(range(len(KEYPOINT_NAMES))) - LEFT_SET - RIGHT_SET

def load_poses(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    poses = {int(k): v for k, v in data.items()}
    frames = sorted(poses.keys())
    return poses, frames

def points_for_frame(frame_points):
    # Returns arrays (x, y, z, idx) for visible points
    xs, ys, zs, idxs = [], [], [], []
    for i, p in enumerate(frame_points):
        if p is None:
            continue
        x, y, z = p
        xs.append(x); ys.append(y); zs.append(z); idxs.append(i)
    if not xs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return np.array(xs), np.array(ys), np.array(zs), np.array(idxs)

def split_by_side(idxs, xs, ys, zs):
    def sel(indices_set):
        mask = np.array([i in indices_set for i in idxs], dtype=bool)
        return xs[mask], ys[mask], zs[mask]
    xL, yL, zL = sel(LEFT_SET)
    xR, yR, zR = sel(RIGHT_SET)
    xC, yC, zC = sel(CENTER_SET)
    return (xL, yL, zL), (xR, yR, zR), (xC, yC, zC)

def compute_global_bounds(poses):
    xs, ys, zs = [], [], []
    for frame in poses.values():
        for p in frame:
            if p is None: continue
            x, y, z = p
            xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return (-1, 1), (-1, 1), (-1, 1)
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    z_min, z_max = np.min(zs), np.max(zs)
    # Add small margin
    margin = 0.05
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    x_min -= dx * margin; x_max += dx * margin
    y_min -= dy * margin; y_max += dy * margin
    z_min -= dz * margin; z_max += dz * margin
    # Ensure non-zero ranges
    if dx == 0: x_min -= 0.5; x_max += 0.5
    if dy == 0: y_min -= 0.5; y_max += 0.5
    if dz == 0: z_min -= 0.5; z_max += 0.5
    return (x_min, x_max), (y_min, y_max), (z_min, z_max)

def edge_color(i, j):
    if i in LEFT_SET and j in LEFT_SET:
        return "tab:blue"
    if i in RIGHT_SET and j in RIGHT_SET:
        return "tab:red"
    return "tab:gray"

def init_axes(ax, xlim, ylim, zlim, elev=None, azim=None, title=None):
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    try:
        ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
    except Exception:
        pass
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else ax.elev,
                     azim=azim if azim is not None else ax.azim)
    if title:
        ax.set_title(title)

def add_skeleton_lines(ax, num_kp):
    lines = []
    for i, j in SKELETON_EDGES:
        if i >= num_kp or j >= num_kp:
            continue
        color = edge_color(i, j)
        line, = ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                        color=color, linewidth=2, alpha=0.9)
        lines.append(((i, j), line))
    return lines

def update_skeleton_lines(lines, pos_by_idx):
    for (i, j), line in lines:
        if i in pos_by_idx and j in pos_by_idx:
            xi, yi, zi = pos_by_idx[i]
            xj, yj, zj = pos_by_idx[j]
            line.set_data([xi, xj], [yi, yj])
            line.set_3d_properties([zi, zj])
        else:
            line.set_data([np.nan, np.nan], [np.nan, np.nan])
            line.set_3d_properties([np.nan, np.nan])

def build_trajectories(poses):
    frames = sorted(poses.keys())
    max_kp = max(len(v) for v in poses.values())
    traj = {}
    for k in range(max_kp):
        xs, ys, zs = [], [], []
        for f in frames:
            p = poses[f][k] if k < len(poses[f]) else None
            if p is None:
                xs.append(np.nan); ys.append(np.nan); zs.append(np.nan)
            else:
                x, y, z = p
                xs.append(x); ys.append(y); zs.append(z)
        traj[k] = (np.array(xs), np.array(ys), np.array(zs))
    return frames, traj

def plot_single_frame(poses, frame_id, show_skeleton=True, show_labels=False,
                      elev=None, azim=None, title=None, show=True, save=None,
                      show_traj=False):
    if frame_id not in poses:
        raise ValueError(f"Frame {frame_id} not found in JSON.")
    frame_points = poses[frame_id]
    xs, ys, zs, idxs = points_for_frame(frame_points)

    xlim, ylim, zlim = compute_global_bounds(poses)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    init_axes(ax, xlim, ylim, zlim, elev=elev, azim=azim,
              title=title or f"Frame {frame_id}")

    if show_traj:
        frames_sorted, traj = build_trajectories(poses)
        for k, (tx, ty, tz) in traj.items():
            ax.plot(tx, ty, tz, color="lightgray", linewidth=1, alpha=0.7)

    # Split visible points into left/right/center
    (xL, yL, zL), (xR, yR, zR), (xC, yC, zC) = split_by_side(idxs, xs, ys, zs)
    scat_L = ax.scatter(xL, yL, zL, color="tab:blue", s=40, depthshade=True, label="Left")
    scat_R = ax.scatter(xR, yR, zR, color="tab:red", s=40, depthshade=True, label="Right")
    scat_C = ax.scatter(xC, yC, zC, color="tab:gray", s=40, depthshade=True, label="Center")
    if len(xL) + len(xR) + len(xC) > 0:
        ax.legend(loc="upper right")

    lines = []
    if show_skeleton:
        lines = add_skeleton_lines(ax, num_kp=len(frame_points))
        pos_by_idx = {int(i): (x, y, z) for i, x, y, z in zip(idxs, xs, ys, zs)}
        update_skeleton_lines(lines, pos_by_idx)

    texts = []
    if show_labels:
        for i, x, y, z in zip(idxs, xs, ys, zs):
            name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
            texts.append(ax.text(x, y, z, name, color="black", fontsize=9))

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"Saved figure to {save}")
    if show:
        plt.show()
    return fig, ax

def animate_frames(poses, interval=100, show_skeleton=True, show_labels=False,
                   elev=None, azim=None, title="3D Pose Animation", save=None):
    frames = sorted(poses.keys())
    if not frames:
        raise ValueError("No frames found in JSON.")

    xlim, ylim, zlim = compute_global_bounds(poses)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    init_axes(ax, xlim, ylim, zlim, elev=elev, azim=azim, title=title)

    f0 = frames[0]
    xs, ys, zs, idxs = points_for_frame(poses[f0])

    # Initial scatters split by side
    (xL, yL, zL), (xR, yR, zR), (xC, yC, zC) = split_by_side(idxs, xs, ys, zs)
    scat_L = ax.scatter(xL, yL, zL, color="tab:blue", s=40, depthshade=True, label="Left")
    scat_R = ax.scatter(xR, yR, zR, color="tab:red", s=40, depthshade=True, label="Right")
    scat_C = ax.scatter(xC, yC, zC, color="tab:gray", s=40, depthshade=True, label="Center")
    ax.legend(loc="upper right")

    lines = []
    if show_skeleton:
        max_kp = max(len(v) for v in poses.values())
        lines = add_skeleton_lines(ax, num_kp=max_kp)

    texts = []
    if show_labels:
        for i, x, y, z in zip(idxs, xs, ys, zs):
            name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
            texts.append(ax.text(x, y, z, name, color="black", fontsize=9))

    def set_offsets3d(scat, X, Y, Z):
        if len(X) == 0:
            scat._offsets3d = ([], [], [])
        else:
            scat._offsets3d = (X, Y, Z)

    def update(fi):
        frame_id = frames[fi]
        frame_points = poses[frame_id]
        xs, ys, zs, idxs = points_for_frame(frame_points)

        (xL, yL, zL), (xR, yR, zR), (xC, yC, zC) = split_by_side(idxs, xs, ys, zs)
        set_offsets3d(scat_L, xL, yL, zL)
        set_offsets3d(scat_R, xR, yR, zR)
        set_offsets3d(scat_C, xC, yC, zC)

        if show_skeleton:
            pos_by_idx = {int(i): (x, y, z) for i, x, y, z in zip(idxs, xs, ys, zs)}
            update_skeleton_lines(lines, pos_by_idx)

        if show_labels:
            for t in texts:
                t.remove()
            texts.clear()
            for i, x, y, z in zip(idxs, xs, ys, zs):
                name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i)
                texts.append(ax.text(x, y, z, name, color="black", fontsize=9))

        ax.set_title(f"{title} | Frame {frame_id}")
        return [scat_L, scat_R, scat_C] + [ln for (_, ln) in lines] + texts

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    plt.tight_layout()

    if save:
        try:
            if save.lower().endswith(".mp4"):
                anim.save(save, writer="ffmpeg", fps=max(1, int(1000/interval)))
            else:
                anim.save(save, writer="imagemagick", fps=max(1, int(1000/interval)))
            print(f"Saved animation to {save}")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    plt.show()
    return anim

def main():
    parser = argparse.ArgumentParser(description="Plot 3D triangulated poses with custom skeleton.")
    parser.add_argument("--input", required=True, help="Path to player_3d_poses.json")
    parser.add_argument("--frame", type=int, default=None, help="Frame ID to plot")
    parser.add_argument("--animate", action="store_true", help="Animate all frames")
    parser.add_argument("--interval", type=int, default=100, help="Animation interval (ms)")
    parser.add_argument("--no-skeleton", action="store_true", help="Disable skeleton connections")
    parser.add_argument("--labels", action="store_true", help="Show keypoint name labels")
    parser.add_argument("--traj", action="store_true", help="Show joint trajectories (single-frame view)")
    parser.add_argument("--elev", type=float, default=None, help="Camera elevation angle")
    parser.add_argument("--azim", type=float, default=None, help="Camera azimuth angle")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure/animation")
    args = parser.parse_args()

    poses, frames = load_poses(args.input)
    if not frames:
        raise SystemExit("No frames found in JSON.")

    if args.animate:
        animate_frames(
            poses,
            interval=args.interval,
            show_skeleton=not args.no_skeleton,
            show_labels=args.labels,
            elev=args.elev,
            azim=args.azim,
            title="3D Pose Animation",
            save=args.save,
        )
    else:
        frame_id = args.frame if args.frame is not None else frames[0]
        plot_single_frame(
            poses,
            frame_id=frame_id,
            show_skeleton=not args.no_skeleton,
            show_labels=args.labels,
            show_traj=args.traj,
            elev=args.elev,
            azim=args.azim,
            title=None,
            show=True,
            save=args.save,
        )

if __name__ == "__main__":
    main()