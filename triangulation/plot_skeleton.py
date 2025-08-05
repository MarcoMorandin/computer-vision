import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import os

def plot_and_save_skeleton_3d(json_path, output_folder='output_frames'):
    # --- 0. Create Output Directory ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # --- 1. Load the JSON Data ---
    with open(json_path, 'r') as f:
        data = json.load(f)

    # --- 2. Skeleton Definition ---
    skeleton_connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],    # Right Leg
        [0, 5], [5, 6], [6, 7], [7, 8],    # Left Leg
        [0, 9], [9, 10], [10, 11],         # Spine
        [10, 12], [12, 13], [13, 14],      # Right Arm
        [10, 15], [15, 16], [16, 17]       # Left Arm
    ]

    # --- 3. Set up the 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    frame_keys = sorted(data.keys(), key=int)
    num_frames = len(frame_keys)

    # Determine axis limits once to prevent the plot from resizing
    all_keypoints = np.array([kp for frame in data.values() for kp in frame])
    x_min, x_max = all_keypoints[:,0].min(), all_keypoints[:,0].max()
    y_min, y_max = all_keypoints[:,1].min(), all_keypoints[:,1].max()
    z_min, z_max = all_keypoints[:,2].min(), all_keypoints[:,2].max()

    # --- 4. Animation Function ---
    def update(frame_num):
        ax.cla() # Clear the previous frame's plot

        frame_key = frame_keys[frame_num]
        keypoints = np.array(data[frame_key])

        x_coords, y_coords, z_coords = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]

        # Plot the keypoints (joints)
        ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o', s=50) # s for size

        # Plot the bones (connections)
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                ax.plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]], 'b-', linewidth=2)

        ax.set_title(f'3D Skeleton Animation (Frame {frame_key})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.view_init(elev=20, azim=-60)

        # --- SAVE THE FRAMES ---
        save_path = os.path.join(output_folder, f'frame_{frame_key.zfill(2)}.jpg')
        plt.savefig(save_path, dpi=120)
        print(f"Saved frame: {save_path}")

    # --- 5. Create the Animation ---
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    ani.save('output/skeleton_animation.mp4', writer='ffmpeg', fps=5)



if __name__ == '__main__':
    json_file_path = os.path.join('output', 'player_3d_poses.json')
    output_folder = os.path.join('output', 'frames')

    os.makedirs(output_folder, exist_ok=True)

    plot_and_save_skeleton_3d(json_file_path, output_folder=output_folder)