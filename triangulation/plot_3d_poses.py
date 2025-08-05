import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    """Load 3D poses and COCO annotations"""
    # Load 3D poses
    poses_file = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/triangulation/3d_poses.json'
    with open(poses_file, 'r') as f:
        poses_3d = json.load(f)
    
    # Load COCO annotations to get skeleton connections
    coco_file = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/rectification/rectified/train/_annotations.coco.json'
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    return poses_3d, coco_data

def get_skeleton_connections(coco_data):
    """Extract skeleton connections from COCO data"""
    # Find the person category
    person_category = None
    for category in coco_data['categories']:
        if category['name'] == 'person':
            person_category = category
            break
    
    if person_category is None:
        raise ValueError("Person category not found in COCO data")
    
    keypoints = person_category['keypoints']
    skeleton = person_category['skeleton']
    
    print(f"Keypoints ({len(keypoints)}):")
    for i, kp in enumerate(keypoints):
        print(f"  {i+1}: {kp}")
    
    print(f"\nSkeleton connections ({len(skeleton)}):")
    for connection in skeleton:
        start_idx, end_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-based indexing
        print(f"  {keypoints[start_idx]} -> {keypoints[end_idx]}")
    
    return keypoints, skeleton

def plot_3d_skeleton(pose_3d, skeleton, keypoints, frame_id, save_plot=False):
    """Plot a single 3D skeleton"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert pose to numpy array
    pose_array = np.array(pose_3d)
    
    # Plot keypoints
    x_coords = pose_array[:, 0]
    y_coords = pose_array[:, 1]
    z_coords = pose_array[:, 2]
    
    # Plot points
    ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, alpha=0.8)
    
    # Add keypoint labels
    for i, (x, y, z) in enumerate(pose_array):
        ax.text(x, y, z, f'{i+1}:{keypoints[i]}', fontsize=8)
    
    # Plot skeleton connections
    for connection in skeleton:
        start_idx = connection[0] - 1  # Convert to 0-based indexing
        end_idx = connection[1] - 1
        
        if start_idx < len(pose_array) and end_idx < len(pose_array):
            start_point = pose_array[start_idx]
            end_point = pose_array[end_idx]
            
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   [start_point[2], end_point[2]], 
                   'b-', linewidth=2, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'3D Skeleton - Frame {frame_id}')
    
    # Set equal aspect ratio
    max_range = np.array([x_coords.max()-x_coords.min(), 
                         y_coords.max()-y_coords.min(),
                         z_coords.max()-z_coords.min()]).max() / 2.0
    
    mid_x = (x_coords.max()+x_coords.min()) * 0.5
    mid_y = (y_coords.max()+y_coords.min()) * 0.5
    mid_z = (z_coords.max()+z_coords.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    if save_plot:
        output_dir = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/triangulation/3d_plots'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/skeleton_3d_{frame_id}.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot for frame {frame_id}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def plot_multiple_skeletons(poses_3d, skeleton, keypoints, max_frames=5, save_plots=False):
    """Plot multiple skeletons in a grid"""
    frame_ids = list(poses_3d.keys())[:max_frames]
    
    fig = plt.figure(figsize=(20, 12))
    
    for i, frame_id in enumerate(frame_ids):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        pose_array = np.array(poses_3d[frame_id])
        
        # Plot keypoints
        x_coords = pose_array[:, 0]
        y_coords = pose_array[:, 1]
        z_coords = pose_array[:, 2]
        
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=30, alpha=0.8)
        
        # Plot skeleton connections
        for connection in skeleton:
            start_idx = connection[0] - 1
            end_idx = connection[1] - 1
            
            if start_idx < len(pose_array) and end_idx < len(pose_array):
                start_point = pose_array[start_idx]
                end_point = pose_array[end_idx]
                
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       'b-', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Frame {frame_id}')
        
        # Set equal aspect ratio
        max_range = np.array([x_coords.max()-x_coords.min(), 
                             y_coords.max()-y_coords.min(),
                             z_coords.max()-z_coords.min()]).max() / 2.0
        
        mid_x = (x_coords.max()+x_coords.min()) * 0.5
        mid_y = (y_coords.max()+y_coords.min()) * 0.5
        mid_z = (z_coords.max()+z_coords.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/triangulation/3d_plots'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/multiple_skeletons_{max_frames}_frames.png', dpi=300, bbox_inches='tight')
        print(f"Saved multiple skeletons plot ({max_frames} frames)")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def animate_skeleton(poses_3d, skeleton, keypoints, save_animation=False):
    """Create an animated visualization of the skeleton over time"""
    from matplotlib.animation import FuncAnimation
    
    frame_ids = list(poses_3d.keys())
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame_idx):
        ax.clear()
        
        frame_id = frame_ids[frame_idx]
        pose_array = np.array(poses_3d[frame_id])
        
        # Plot keypoints
        x_coords = pose_array[:, 0]
        y_coords = pose_array[:, 1]
        z_coords = pose_array[:, 2]
        
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, alpha=0.8)
        
        # Plot skeleton connections
        for connection in skeleton:
            start_idx = connection[0] - 1
            end_idx = connection[1] - 1
            
            if start_idx < len(pose_array) and end_idx < len(pose_array):
                start_point = pose_array[start_idx]
                end_point = pose_array[end_idx]
                
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       'b-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Skeleton Animation - Frame {frame_id}')
        
        # Set consistent axis limits
        all_poses = np.array(list(poses_3d.values()))
        x_min, x_max = all_poses[:,:,0].min(), all_poses[:,:,0].max()
        y_min, y_max = all_poses[:,:,1].min(), all_poses[:,:,1].max()
        z_min, z_max = all_poses[:,:,2].min(), all_poses[:,:,2].max()
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        ax.view_init(elev=20, azim=45)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(frame_ids), interval=200, repeat=True)
    
    if save_animation:
        output_dir = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/triangulation/3d_plots'
        os.makedirs(output_dir, exist_ok=True)
        anim.save(f'{output_dir}/skeleton_animation.gif', writer='pillow', fps=5)
        print("Animation saved as skeleton_animation.gif")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()
    return anim

def save_all_skeletons(poses_3d, skeleton, keypoints):
    """Save individual plots for all skeleton frames"""
    output_dir = '/Users/marcomorandin/Desktop/ComputerVision/computer-vision/triangulation/3d_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving individual skeleton plots for {len(poses_3d)} frames...")
    
    for i, (frame_id, pose_3d) in enumerate(poses_3d.items()):
        # Create figure for each frame
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert pose to numpy array
        pose_array = np.array(pose_3d)
        
        # Plot keypoints
        x_coords = pose_array[:, 0]
        y_coords = pose_array[:, 1]
        z_coords = pose_array[:, 2]
        
        # Plot points
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, alpha=0.8)
        
        # Add keypoint labels (optional, can be removed for cleaner plots)
        for j, (x, y, z) in enumerate(pose_array):
            ax.text(x, y, z, f'{j+1}', fontsize=8, alpha=0.7)
        
        # Plot skeleton connections
        for connection in skeleton:
            start_idx = connection[0] - 1  # Convert to 0-based indexing
            end_idx = connection[1] - 1
            
            if start_idx < len(pose_array) and end_idx < len(pose_array):
                start_point = pose_array[start_idx]
                end_point = pose_array[end_idx]
                
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       'b-', linewidth=2, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Skeleton - Frame {frame_id}')
        
        # Set equal aspect ratio
        max_range = np.array([x_coords.max()-x_coords.min(), 
                             y_coords.max()-y_coords.min(),
                             z_coords.max()-z_coords.min()]).max() / 2.0
        
        mid_x = (x_coords.max()+x_coords.min()) * 0.5
        mid_y = (y_coords.max()+y_coords.min()) * 0.5
        mid_z = (z_coords.max()+z_coords.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Save the plot
        plt.savefig(f'{output_dir}/skeleton_3d_{frame_id}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(poses_3d):
            print(f"  Saved {i + 1}/{len(poses_3d)} frames")
    
    print(f"All individual skeleton plots saved to: {output_dir}")

def print_statistics(poses_3d):
    """Print some statistics about the 3D poses"""
    print("\n=== 3D Poses Statistics ===")
    print(f"Total number of frames: {len(poses_3d)}")
    
    # Get frame IDs
    frame_ids = list(poses_3d.keys())
    print(f"Frame IDs: {frame_ids[:10]}{'...' if len(frame_ids) > 10 else ''}")
    
    # Check number of keypoints per frame
    keypoint_counts = [len(pose) for pose in poses_3d.values()]
    print(f"Keypoints per frame: {set(keypoint_counts)}")
    
    # Calculate coordinate ranges
    all_poses = np.array(list(poses_3d.values()))
    print("Coordinate ranges:")
    print(f"  X: {all_poses[:,:,0].min():.2f} to {all_poses[:,:,0].max():.2f} mm")
    print(f"  Y: {all_poses[:,:,1].min():.2f} to {all_poses[:,:,1].max():.2f} mm")
    print(f"  Z: {all_poses[:,:,2].min():.2f} to {all_poses[:,:,2].max():.2f} mm")

def main():
    """Main function to demonstrate the 3D skeleton plotting"""
    print("Loading 3D poses and COCO annotations...")
    poses_3d, coco_data = load_data()
    
    print("Extracting skeleton information...")
    keypoints, skeleton = get_skeleton_connections(coco_data)
    
    print_statistics(poses_3d)
    
    # Save all individual skeleton frames
    save_all_skeletons(poses_3d, skeleton, keypoints)
    
    # Save a grid plot of multiple skeletons
    print("\nSaving multiple skeletons grid plot...")
    plot_multiple_skeletons(poses_3d, skeleton, keypoints, max_frames=6, save_plots=True)
    
if __name__ == "__main__":
    main()
