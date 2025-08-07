import os
import json
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from datetime import datetime

def load_coco_datasets(gt_coco_path, pred_coco_path):
    gt_coco = COCO(gt_coco_path)
    pred_coco = COCO(pred_coco_path)
    
    keypoint_names = gt_coco.cats[1]['keypoints']
    skeleton = gt_coco.cats[1]['skeleton']
    
    return gt_coco, pred_coco, keypoint_names, skeleton

def get_matching_annotations(gt_coco, pred_coco):
    matches = []
    
    # Create lookup for ground truth by image filename
    gt_lookup = {}
    for img_id in gt_coco.imgs.keys():
        img_info = gt_coco.imgs[img_id]
        filename = img_info['file_name']
        gt_lookup[filename] = img_id
    
    # Find matches in predictions
    for img_id in pred_coco.imgs.keys():
        img_info = pred_coco.imgs[img_id]
        pred_filename = img_info['file_name']
        
        if pred_filename in gt_lookup:
            gt_img_id = gt_lookup[pred_filename]
            
            # Get annotations for both
            gt_anns = gt_coco.getAnnIds(imgIds=[gt_img_id], catIds=[1])
            pred_anns = pred_coco.getAnnIds(imgIds=[img_id], catIds=[1])
            
            if gt_anns and pred_anns:
                # For simplicity, take the first annotation if multiple exist
                gt_ann = gt_coco.anns[gt_anns[0]]
                pred_ann = pred_coco.anns[pred_anns[0]]
                
                camera_id, frame_num = extract_camera_info(pred_filename)
                
                matches.append({
                    'filename': pred_filename,
                    'camera': camera_id,
                    'frame': frame_num,
                    'gt_keypoints': np.array(gt_ann['keypoints']).reshape(-1, 3),
                    'pred_keypoints': np.array(pred_ann['keypoints']).reshape(-1, 3)
                })
    
    return matches

def extract_camera_info(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        camera = parts[0].replace('out', '')
        frame = int(parts[2])
        return camera, frame
    return None, None

def compute_head_size(keypoints):
    if keypoints[11, 2] > 0 and keypoints[10, 2] > 0:  # Both visible
        head_size = np.linalg.norm(keypoints[11, :2] - keypoints[10, :2])
        return max(head_size, 1.0)
    return 60.0  # Default head size in pixels

def compute_pck(gt_kpts, pred_kpts, threshold, normalize='bbox'):
    valid_joints = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
    
    if not np.any(valid_joints):
        return 0.0
    
    # Compute distances
    distances = np.linalg.norm(gt_kpts[:, :2] - pred_kpts[:, :2], axis=1)
    
    # Determine threshold based on normalization
    if normalize == 'head':
        norm_factor = compute_head_size(gt_kpts)
    else:  # bbox
        if gt_kpts[0, 2] > 0 and gt_kpts[11, 2] > 0:  # Hips and Head visible
            norm_factor = np.linalg.norm(gt_kpts[11, :2] - gt_kpts[0, :2])
        else:
            norm_factor = 100.0  # Default torso size
    
    threshold_pixels = threshold * norm_factor
    
    # Compute PCK
    correct = (distances <= threshold_pixels) & valid_joints
    pck_score = np.sum(correct) / np.sum(valid_joints) if np.sum(valid_joints) > 0 else 0.0
    
    return pck_score

def compute_mpjpe(gt_kpts, pred_kpts):
    valid_joints = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
    
    if not np.any(valid_joints):
        return np.inf
    
    distances = np.linalg.norm(gt_kpts[:, :2] - pred_kpts[:, :2], axis=1)
    mpjpe = np.mean(distances[valid_joints])
    
    return mpjpe

def procrustes_alignment(gt_kpts, pred_kpts):
    valid_joints = (gt_kpts[:, 2] > 0) & (pred_kpts[:, 2] > 0)
    
    if np.sum(valid_joints) < 3:  # Need at least 3 points for alignment
        return pred_kpts[:, :2]
    
    gt_valid = gt_kpts[valid_joints, :2]
    pred_valid = pred_kpts[valid_joints, :2]
    
    # Center the points
    gt_centered = gt_valid - np.mean(gt_valid, axis=0)
    pred_centered = pred_valid - np.mean(pred_valid, axis=0)
    
    # Compute optimal rotation using SVD
    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    pred_norm = np.linalg.norm(pred_centered)
    gt_norm = np.linalg.norm(gt_centered)
    s = gt_norm / pred_norm if pred_norm > 0 else 1.0
    
    # Apply transformation to all predicted points
    pred_all_centered = pred_kpts[:, :2] - np.mean(pred_valid, axis=0)
    aligned_pred = s * (pred_all_centered @ R.T) + np.mean(gt_valid, axis=0)
    
    return aligned_pred

def compute_angular_error(gt_kpts, pred_kpts, skeleton):
    bone_errors = []
    
    for bone in skeleton:
        joint1_idx = bone[0] - 1  # COCO is 1-indexed
        joint2_idx = bone[1] - 1
        
        # Check if both joints are valid in both predictions
        if (gt_kpts[joint1_idx, 2] > 0 and gt_kpts[joint2_idx, 2] > 0 and
            pred_kpts[joint1_idx, 2] > 0 and pred_kpts[joint2_idx, 2] > 0):
            
            # Compute bone vectors
            gt_bone = gt_kpts[joint2_idx, :2] - gt_kpts[joint1_idx, :2]
            pred_bone = pred_kpts[joint2_idx, :2] - pred_kpts[joint1_idx, :2]
            
            # Normalize vectors
            gt_bone_norm = gt_bone / (np.linalg.norm(gt_bone) + 1e-8)
            pred_bone_norm = pred_bone / (np.linalg.norm(pred_bone) + 1e-8)
            
            # Compute angle between vectors
            cos_angle = np.clip(np.dot(gt_bone_norm, pred_bone_norm), -1.0, 1.0)
            angle_error = np.arccos(cos_angle) * 180.0 / np.pi
            
            bone_errors.append(angle_error)
    
    mpjae = np.mean(bone_errors) if bone_errors else np.inf
    return mpjae

def evaluate(gt_coco_path, pred_coco_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    gt_coco, pred_coco, keypoint_names, skeleton = load_coco_datasets(gt_coco_path, pred_coco_path)
    
    # Get matching annotations
    matches = get_matching_annotations(gt_coco, pred_coco)
    
    # Initialize metric storage
    metrics = {
        'pck_0.1': [], 'pck_0.2': [], 'pck_0.5': [],
        'pckh_0.1': [], 'pckh_0.2': [], 'pckh_0.5': [],
        'mpjpe': [], 'pa_mpjpe': [], 'mpjae': []
    }
    
    # Compute metrics for each annotation
    for match in tqdm(matches):
        gt_kpts = match['gt_keypoints']
        pred_kpts = match['pred_keypoints']
        
        # PCK metrics
        pck_01 = compute_pck(gt_kpts, pred_kpts, 0.1, 'bbox')
        pck_02 = compute_pck(gt_kpts, pred_kpts, 0.2, 'bbox')
        pck_05 = compute_pck(gt_kpts, pred_kpts, 0.5, 'bbox')
        
        # PCKh metrics
        pckh_01 = compute_pck(gt_kpts, pred_kpts, 0.1, 'head')
        pckh_02 = compute_pck(gt_kpts, pred_kpts, 0.2, 'head')
        pckh_05 = compute_pck(gt_kpts, pred_kpts, 0.5, 'head')
        
        # MPJPE
        mpjpe = compute_mpjpe(gt_kpts, pred_kpts)
        
        # PA-MPJPE
        aligned_pred = procrustes_alignment(gt_kpts, pred_kpts)
        pred_kpts_aligned = pred_kpts.copy()
        pred_kpts_aligned[:, :2] = aligned_pred
        pa_mpjpe = compute_mpjpe(gt_kpts, pred_kpts_aligned)
        
        # MPJAE
        mpjae = compute_angular_error(gt_kpts, pred_kpts, skeleton)
        
        # Store metrics
        metrics['pck_0.1'].append(pck_01)
        metrics['pck_0.2'].append(pck_02)
        metrics['pck_0.5'].append(pck_05)
        metrics['pckh_0.1'].append(pckh_01)
        metrics['pckh_0.2'].append(pckh_02)
        metrics['pckh_0.5'].append(pckh_05)
        metrics['mpjpe'].append(mpjpe)
        metrics['pa_mpjpe'].append(pa_mpjpe)
        metrics['mpjae'].append(mpjae)
    
    # Compile results - only overall metrics without AUC
    results = {
        'overall': {
            'PCK@0.1': np.mean(metrics['pck_0.1']),
            'PCK@0.2': np.mean(metrics['pck_0.2']),
            'PCK@0.5': np.mean(metrics['pck_0.5']),
            'PCKh@0.1': np.mean(metrics['pckh_0.1']),
            'PCKh@0.2': np.mean(metrics['pckh_0.2']),
            'PCKh@0.5': np.mean(metrics['pckh_0.5']),
            'MPJPE': np.mean(metrics['mpjpe']),
            'PA-MPJPE': np.mean(metrics['pa_mpjpe']),
            'MPJAE': np.mean(metrics['mpjae']),
            'num_samples': len(matches)
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 20)
    for metric, value in results['overall'].items():
        if metric != 'num_samples':
            print(f"{metric:12}: {value:.4f}")
    
    return results

def main():
    # Hardcoded paths
    gt_coco = os.path.join("..", "rectification", "output", "dataset", "train", "_annotations.coco.json")
    pred_coco = os.path.join("output", "predictions_coco.json")
    output_dir = os.path.join("output", 'evaluation_results')
    
    # Run evaluation
    evaluate(gt_coco, pred_coco, output_dir)

if __name__ == "__main__":
    main()
