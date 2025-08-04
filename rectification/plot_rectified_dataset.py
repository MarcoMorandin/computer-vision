import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
from pathlib import Path

def plot_skeleton_on_image(coco_json_path, images_dir, image_id, show_keypoints=True, show_skeleton=True, output_path=None):
    """
    Plots human pose skeletons on a sample image from a custom COCO-format dataset (18 keypoints).
    """
    # Load COCO annotations
    coco = COCO(coco_json_path)
    print(f"Loaded COCO annotations from {coco_json_path}")
    
    # Get image info and load the image
    img_info = coco.loadImgs(image_id)[0]
    image_path = f"{images_dir}/{img_info['file_name']}"
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
    print(f"Loaded image {image_path} with shape {img.shape}")
    
    # Get annotations for this image (filter for person category)
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=coco.getCatIds(catNms=['person']), iscrowd=None)
    annotations = coco.loadAnns(ann_ids)
    print(f"Found {len(annotations)} person annotations for image ID {image_id}")
    
    if not annotations:
        print("No annotations found - nothing to plot. Try a different image_id.")
    
    # Your custom keypoints names (for reference/debug)
    keypoint_names = [
        "Hips", "RHip", "RKnee", "RAnkle", "RFoot", "LHip", "LKnee", "LAnkle", "LFoot",
        "Spine", "Neck", "Head", "RShoulder", "RElbow", "RHand", "LShoulder", "LElbow", "LHand"
    ]
    print(f"Keypoint names: {keypoint_names}")
    
    # Your custom skeleton connections (adjusted to 0-based indexing)
    skeleton = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # Right leg/foot
        [0, 5], [5, 6], [6, 7], [7, 8],  # Left leg/foot
        [0, 9], [9, 10], [10, 11],       # Spine to head
        [10, 12], [12, 13], [13, 14],    # Right arm
        [10, 15], [15, 16], [16, 17]     # Left arm
    ]
    print(f"Skeleton connections: {skeleton}")
    
    # Draw for each person in the image
    for idx, ann in enumerate(annotations):
        if 'keypoints' not in ann or len(ann['keypoints']) != 54:
            print(f"Skipping invalid annotation {idx}: missing or incorrect keypoints (expected 54 values)")
            continue
        keypoints = np.array(ann['keypoints']).reshape(18, 3)  # Reshape to [18, (x, y, v)]
        
        visible_count = np.sum(keypoints[:, 2] > 0)
        print(f"Annotation {idx}: {visible_count} visible keypoints (out of 18)")
        if visible_count == 0:
            print("  - No visible keypoints, skipping drawing for this person")
            continue
        
        # Print sample keypoints for debugging
        print(f"  Sample keypoints (first 3): {keypoints[:3]}")  # First 3 for brevity
        
        # Draw keypoints (circles)
        if show_keypoints:
            for kp_idx, (x, y, v) in enumerate(keypoints):
                if v > 0:  # Only draw visible keypoints
                    x, y = int(x), int(y)
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Red circle
                        # Optional: Label with name for debug
                        # cv2.putText(img, keypoint_names[kp_idx], (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        print(f"  - Skipping keypoint {keypoint_names[kp_idx]} ({x}, {y}) - out of image bounds")
        
        # Draw skeleton lines
        if show_skeleton:
            for i, j in skeleton:
                x1, y1, v1 = keypoints[i]
                x2, y2, v2 = keypoints[j]
                if v1 > 0 and v2 > 0:  # Only draw if both keypoints are visible
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]) and \
                       (0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line
                    else:
                        print(f"  - Skipping line between {keypoint_names[i]} ({x1}, {y1}) and {keypoint_names[j]} ({x2}, {y2}) - out of bounds")
    
    # Display or save the image
    if output_path:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(output_path, img_bgr)
        print(f"Image saved to {output_path}")
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image ID: {image_id}")
        plt.show()
        print("Displayed image - check if keypoints/lines appear.")

def visualize_all_rectified_images(coco_json_path, images_dir, output_dir, show_keypoints=True, show_skeleton=True):
    """
    Visualizes human pose skeletons for all images in the rectified dataset and saves them to the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    coco = COCO(coco_json_path)
    print(f"Loaded COCO annotations from {coco_json_path}")
    
    # Get all image IDs
    image_ids = coco.getImgIds()
    print(f"Found {len(image_ids)} images to process")
    
    successful_count = 0
    error_count = 0
    
    for i, image_id in enumerate(image_ids):
        try:
            # Get image info
            img_info = coco.loadImgs(image_id)[0]
            input_image_path = os.path.join(images_dir, img_info['file_name'])
            
            # Create output filename (remove extension and add _skeleton.jpg)
            base_name = os.path.splitext(img_info['file_name'])[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_skeleton.jpg")
            
            print(f"Processing {i+1}/{len(image_ids)}: {img_info['file_name']}")
            
            # Load the image
            img = cv2.imread(input_image_path)
            if img is None:
                print(f"  Warning: Could not load image {input_image_path}")
                error_count += 1
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get annotations for this image (filter for person category)
            ann_ids = coco.getAnnIds(imgIds=image_id, catIds=coco.getCatIds(catNms=['person']), iscrowd=None)
            annotations = coco.loadAnns(ann_ids)
            
            if not annotations:
                print(f"  No person annotations found for {img_info['file_name']}")
                # Still save the image without annotations
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_image_path, img_bgr)
                successful_count += 1
                continue
            
            # Your custom skeleton connections (adjusted to 0-based indexing)
            skeleton = [
                [0, 1], [1, 2], [2, 3], [3, 4],  # Right leg/foot
                [0, 5], [5, 6], [6, 7], [7, 8],  # Left leg/foot
                [0, 9], [9, 10], [10, 11],       # Spine to head
                [10, 12], [12, 13], [13, 14],    # Right arm
                [10, 15], [15, 16], [16, 17]     # Left arm
            ]
            
            # Draw for each person in the image
            persons_drawn = 0
            for idx, ann in enumerate(annotations):
                if 'keypoints' not in ann or len(ann['keypoints']) != 54:
                    continue
                    
                keypoints = np.array(ann['keypoints']).reshape(18, 3)
                visible_count = np.sum(keypoints[:, 2] > 0)
                
                if visible_count == 0:
                    continue
                
                persons_drawn += 1
                
                # Draw keypoints (circles)
                if show_keypoints:
                    for kp_idx, (x, y, v) in enumerate(keypoints):
                        if v > 0:  # Only draw visible keypoints
                            x, y = int(x), int(y)
                            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Red circle
                
                # Draw skeleton lines
                if show_skeleton:
                    for i, j in skeleton:
                        x1, y1, v1 = keypoints[i]
                        x2, y2, v2 = keypoints[j]
                        if v1 > 0 and v2 > 0:  # Only draw if both keypoints are visible
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]) and \
                               (0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line
            
            # Save the image with skeleton overlay
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, img_bgr)
            
            if persons_drawn > 0:
                print(f"  Saved visualization with {persons_drawn} person(s) to {output_image_path}")
            else:
                print(f"  Saved image without valid keypoints to {output_image_path}")
                
            successful_count += 1
            
        except Exception as e:
            print(f"  Error processing image {image_id}: {str(e)}")
            error_count += 1
            continue
    
    print("\nProcessing complete!")
    print(f"Successfully processed: {successful_count} images")
    print(f"Errors: {error_count} images")
    print(f"Output directory: {output_dir}")

# Example usage for single image (replace with your paths and image ID)
# plot_skeleton_on_image('rectified/train/_annotations.coco.json', 'rectified/train', image_id=30)

# Process all rectified images and save visualizations
if __name__ == "__main__":
    coco_json_path = 'rectified/train/_annotations.coco.json'
    images_dir = 'rectified/train'
    output_dir = 'rectified/visualizations'
    
    visualize_all_rectified_images(coco_json_path, images_dir, output_dir)