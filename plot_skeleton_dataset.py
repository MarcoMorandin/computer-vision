import cv2
import numpy as np
from pycocotools.coco import COCO
import os

def load_coco_data(coco_json_path):
    coco = COCO(coco_json_path)
    person_cat_ids = coco.getCatIds(catNms=['person'])
    
    skeleton_connections = None
    if person_cat_ids:
        person_cats = coco.loadCats(person_cat_ids)
        if person_cats and 'skeleton' in person_cats[0]:
            skeleton_connections = [[conn[0]-1, conn[1]-1] for conn in person_cats[0]['skeleton']]
    
    return coco, person_cat_ids, skeleton_connections

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_keypoints_and_skeleton(img, keypoints, skeleton_connections, color=(0, 255, 0)):
    for x, y, v in keypoints:
        if v > 0 and 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
            cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
    
    for i, j in skeleton_connections:
        x1, y1, v1 = keypoints[i]
        x2, y2, v2 = keypoints[j]
        if v1 > 0 and v2 > 0:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and 
                0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

def save_image(img, output_path=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(output_path, img_bgr):
        raise RuntimeError(f"Failed to save image to {output_path}")


def main(coco_json_path, images_dir, output_dir, rectified_json_path=None):
    coco, person_cat_ids, skeleton_connections = load_coco_data(coco_json_path)
    if rectified_json_path:
        rec_coco, rec_person_cat_ids, rec_skeleton_connections = load_coco_data(rectified_json_path)
        
    image_ids = coco.getImgIds()
        
    for i, image_id in enumerate(image_ids):

        img_info = coco.loadImgs(image_id)[0]
        input_image_path = os.path.join(images_dir, img_info['file_name'])
        
        base_name = os.path.splitext(img_info['file_name'])[0]
        output_image_path = os.path.join(output_dir, f"{base_name}_skeleton.jpg")
        
        if (i + 1) % 50 == 0:  # Progress update
            print(f"Processing {i+1}/{len(image_ids)}")
        
        img = load_image(input_image_path)
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=person_cat_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)
        
        if rectified_json_path:
            rec_ann_ids = rec_coco.getAnnIds(imgIds=image_id, catIds=rec_person_cat_ids, iscrowd=None)
            rec_annotations = rec_coco.loadAnns(rec_ann_ids)
        
       
        for ann in annotations:        
            keypoints = np.array(ann['keypoints']).reshape(18, 3)
            
            draw_keypoints_and_skeleton(img, keypoints, skeleton_connections)
            
        if rectified_json_path:
            for rec_ann in rec_annotations:
                rec_keypoints = np.array(rec_ann['keypoints']).reshape(18, 3)
                draw_keypoints_and_skeleton(img, rec_keypoints, rec_skeleton_connections, (0, 0, 255))
                    
        save_image(img, output_image_path)

if __name__ == "__main__":
    # MODE = 'rectified'
    # MODE = 'standard'
    MODE = 'reprojection'
    
    if MODE == 'standard':
        coco_json_path = os.path.join('data', 'dataset', 'train', '_annotations.coco.json')
        images_dir = os.path.join('data', 'dataset', 'train')
        output_dir = os.path.join('data', 'visualizations')
        main(coco_json_path, images_dir, output_dir)
    elif MODE == 'rectified':
        coco_json_path = os.path.join('rectification', 'output', 'dataset', 'train', '_annotations.coco.json')
        images_dir = os.path.join('rectification', 'output', 'dataset', 'train')
        output_dir = os.path.join('rectification', 'output', 'visualizations')
        main(coco_json_path, images_dir, output_dir)
    elif MODE == 'reprojection':
        coco_json_path = os.path.join('reprojection', 'output', 'reprojected_annotations.json')
        rectified_json_path = os.path.join('rectification', 'output', 'dataset', 'train', '_annotations.coco.json')
        images_dir = os.path.join('rectification', 'output', 'dataset', 'train')
        output_dir = os.path.join('reprojection', 'output', 'visualizations')
        main(coco_json_path, images_dir, output_dir, rectified_json_path)

    
