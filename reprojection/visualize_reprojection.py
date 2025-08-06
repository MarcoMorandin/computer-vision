import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse

class ReprojectionVisualizer:
    """
    Visualize reprojected keypoints to validate the reprojection quality.
    """
    
    def __init__(self, coco_file, rectified_images_dir="../rectification/rectified/dataset/train"):
        """
        Initialize visualizer.
        
        Args:
            coco_file: Path to the reprojected COCO annotations
            rectified_images_dir: Directory containing rectified images
        """
        self.coco_file = coco_file
        self.rectified_images_dir = rectified_images_dir
        
        # Load COCO data
        with open(coco_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create lookup dictionaries
        self.images_by_id = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Keypoint names for labeling
        person_category = None
        for cat in self.coco_data['categories']:
            if cat['name'] == 'person':
                person_category = cat
                break
        
        if person_category:
            self.keypoint_names = person_category['keypoints']
        else:
            # Fallback keypoint names
            self.keypoint_names = [
                "Hips", "RHip", "RKnee", "RAnkle", "RFoot",
                "LHip", "LKnee", "LAnkle", "LFoot", "Spine",
                "Neck", "Head", "RShoulder", "RElbow", "RHand",
                "LShoulder", "LElbow", "LHand"
            ]
        
        # Colors for different keypoints (different body parts)
        self.colors = [
            (255, 0, 0),    # Hips - Red
            (255, 100, 0),  # RHip - Orange-Red  
            (255, 150, 0),  # RKnee - Orange
            (255, 200, 0),  # RAnkle - Yellow-Orange
            (255, 255, 0),  # RFoot - Yellow
            (100, 255, 0),  # LHip - Green-Yellow
            (0, 255, 0),    # LKnee - Green
            (0, 255, 100),  # LAnkle - Green-Cyan
            (0, 255, 255),  # LFoot - Cyan
            (0, 100, 255),  # Spine - Blue-Cyan
            (0, 0, 255),    # Neck - Blue
            (100, 0, 255),  # Head - Blue-Purple
            (200, 0, 255),  # RShoulder - Purple
            (255, 0, 200),  # RElbow - Purple-Pink
            (255, 0, 100),  # RHand - Pink
            (150, 0, 255),  # LShoulder - Purple
            (100, 0, 200),  # LElbow - Dark Purple
            (50, 0, 150)    # LHand - Dark Purple
        ]
    
    def _get_rectified_image(self, filename):
        """Load a rectified image by filename."""
        image_path = os.path.join(self.rectified_images_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Rectified image not found: {image_path}")
            return None
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not load image: {image_path}")
            return None
        
        return frame
    
    def visualize_sample_reprojections(self, num_samples=5, save_dir="./visualizations"):
        """
        Visualize a sample of reprojections overlaid on rectified images.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get a sample of images
        sample_images = list(self.images_by_id.values())[:num_samples]
        
        for i, img_info in enumerate(sample_images):
            img_id = img_info['id']
            filename = img_info['file_name']
            
            print(f"Processing sample {i+1}/{num_samples}: {filename}")
            
            # Get rectified image
            frame = self._get_rectified_image(filename)
            if frame is None:
                continue
            
            # Get annotations for this image
            annotations = self.annotations_by_image.get(img_id, [])
            
            # Draw keypoints and skeleton
            for ann in annotations:
                self._draw_keypoints_on_frame(frame, ann)
            
            # Save visualization
            output_filename = f"reprojection_{filename}"
            output_path = os.path.join(save_dir, output_filename)
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
    
    def _draw_keypoints_on_frame(self, frame, annotation):
        """Draw keypoints and skeleton on frame."""
        keypoints = annotation['keypoints']
        
        # Extract keypoints (x, y, visibility)
        points = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            points.append((int(x), int(y), int(v)))
        
        # Draw keypoints
        for i, (x, y, v) in enumerate(points):
            if v > 0:  # Visible
                color = self.colors[i % len(self.colors)]
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)  # White border
                
                # Add keypoint label
                cv2.putText(frame, str(i), (x-5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw skeleton connections
        person_category = None
        for cat in self.coco_data['categories']:
            if cat['name'] == 'person':
                person_category = cat
                break
        
        if person_category and 'skeleton' in person_category:
            skeleton = person_category['skeleton']
            for connection in skeleton:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-indexed
                
                if (pt1_idx < len(points) and pt2_idx < len(points) and 
                    points[pt1_idx][2] > 0 and points[pt2_idx][2] > 0):
                    
                    pt1 = (points[pt1_idx][0], points[pt1_idx][1])
                    pt2 = (points[pt2_idx][0], points[pt2_idx][1])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw bounding box
        bbox = annotation['bbox']
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    def create_keypoint_legend(self, save_path="./keypoint_legend.png"):
        """Create a legend showing keypoint colors and names."""
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create legend
        y_positions = np.linspace(0.9, 0.1, len(self.keypoint_names))
        
        for i, (name, y_pos) in enumerate(zip(self.keypoint_names, y_positions)):
            color = np.array(self.colors[i]) / 255.0  # Normalize for matplotlib
            color = color[::-1]  # BGR to RGB
            
            # Draw colored circle
            circle = Circle((0.1, y_pos), 0.02, color=color, transform=ax.transAxes)
            ax.add_patch(circle)
            
            # Add text
            ax.text(0.2, y_pos, f"{i}: {name}", transform=ax.transAxes, 
                   fontsize=12, verticalalignment='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Skeleton Keypoint Legend", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Legend saved to: {save_path}")
    
    def print_statistics(self):
        """Print statistics about the reprojected dataset."""
        total_images = len(self.coco_data['images'])
        total_annotations = len(self.coco_data['annotations'])
        
        # Count annotations per camera
        camera_counts = {}
        frame_counts = {}
        
        for img in self.coco_data['images']:
            camera_id = img['extra']['camera_id']
            frame_num = img['extra']['frame_number']
            
            camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
            frame_counts[frame_num] = frame_counts.get(frame_num, 0) + 1
        
        # Visibility statistics
        visibility_stats = {0: 0, 1: 0, 2: 0}
        total_keypoints = 0
        
        for ann in self.coco_data['annotations']:
            keypoints = ann['keypoints']
            for i in range(2, len(keypoints), 3):  # Every 3rd element is visibility
                v = keypoints[i]
                visibility_stats[v] = visibility_stats.get(v, 0) + 1
                total_keypoints += 1
        
        print("\nReprojection Dataset Statistics")
        print("=" * 40)
        print(f"Total images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"Keypoints per annotation: {len(self.keypoint_names)}")
        print(f"Total keypoints: {total_keypoints}")
        
        print("\nImages per camera:")
        for camera_id in sorted(camera_counts.keys()):
            print(f"  Camera {camera_id}: {camera_counts[camera_id]} images")
        
        print(f"\nFrame range: {min(frame_counts.keys())} - {max(frame_counts.keys())}")
        print(f"Total frames: {len(frame_counts)}")
        
        print("\nKeypoint visibility:")
        print(f"  Not visible (0): {visibility_stats[0]} ({100*visibility_stats[0]/total_keypoints:.1f}%)")
        print(f"  Occluded (1): {visibility_stats[1]} ({100*visibility_stats[1]/total_keypoints:.1f}%)")
        print(f"  Visible (2): {visibility_stats[2]} ({100*visibility_stats[2]/total_keypoints:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize reprojected skeleton keypoints on rectified images")
    parser.add_argument("--coco_file", type=str, 
                       default="./output/reprojected_annotations.json",
                       help="Path to reprojected COCO annotations file")
    parser.add_argument("--rectified_images_dir", type=str,
                       default="../rectification/rectified/dataset/train",
                       help="Directory containing rectified images")
    parser.add_argument("--output_dir", type=str,
                       default="./visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of sample frames to visualize")
    parser.add_argument("--stats_only", action="store_true",
                       help="Only print statistics, don't create visualizations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.coco_file):
        print(f"Error: COCO file not found: {args.coco_file}")
        print("Please run the reprojection first.")
        return
    
    print("Reprojection Visualization Tool")
    print("=" * 30)
    
    # Create visualizer
    visualizer = ReprojectionVisualizer(args.coco_file, args.rectified_images_dir)
    
    # Print statistics
    visualizer.print_statistics()
    
    if not args.stats_only:
        # Create keypoint legend
        legend_path = os.path.join(args.output_dir, "keypoint_legend.png")
        os.makedirs(args.output_dir, exist_ok=True)
        visualizer.create_keypoint_legend(legend_path)
        
        # Create sample visualizations
        print(f"\nCreating {args.num_samples} sample visualizations...")
        visualizer.visualize_sample_reprojections(args.num_samples, args.output_dir)
        
        print(f"\nVisualizations saved to: {args.output_dir}")
        print("Check the output images to validate reprojection quality.")


if __name__ == "__main__":
    main()
