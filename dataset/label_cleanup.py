import os
from ultralytics import YOLO
import torch
from tqdm.notebook import tqdm

# --- CONFIGURATION ---
MODEL_PATH = '/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/weights/best.pt'
DATASET_ROOT = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b'
CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.50  # Overlap required to consider a label "already existing"
TROPHOZOITE_CLASS_ID = 0  # Assuming 0 based on previous data.yaml

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    """Converts normalized xywh to absolute xyxy pixel coordinates."""
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two absolute xyxy boxes.
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

# Load Model
model = YOLO(MODEL_PATH)

# Process Train and Val
splits = ['train', 'val']
total_added_labels = 0

print(f"Starting Auto-Labeling...")
print(f"Criteria: Class 'Trophozoite' AND Confidence > {CONF_THRESHOLD:.0%}")

for split in splits:
    img_dir = os.path.join(DATASET_ROOT, 'images', split)
    lbl_dir = os.path.join(DATASET_ROOT, 'labels', split)
    
    if not os.path.exists(img_dir):
        print(f"Skipping {split} (folder not found)")
        continue
        
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(image_files, desc=f"Processing {split}"):
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # 1. Run Inference
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        img_h, img_w = results.orig_shape
        
        # 2. Load Existing Ground Truth Labels
        existing_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        cls, x, y, w, h = parts
                        # Store as [class_id, absolute_x1, absolute_y1, absolute_x2, absolute_y2]
                        abs_box = xywhn_to_xyxy(x, y, w, h, img_w, img_h)
                        existing_boxes.append({'cls': int(cls), 'box': abs_box})
        
        # 3. Filter and Compare Predictions
        new_labels = []
        
        # Iterate through model detections
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            xywhn = box.xywhn[0].tolist() # Normalized x, y, w, h
            
            # CRITERIA 1: Must be Trophozoite (ID 0) and Conf > Threshold
            if cls_id == TROPHOZOITE_CLASS_ID and conf > CONF_THRESHOLD:
                
                # Convert pred to absolute xyxy for IoU check
                pred_abs_box = xywhn_to_xyxy(xywhn[0], xywhn[1], xywhn[2], xywhn[3], img_w, img_h)
                
                # CRITERIA 2: Must NOT overlap significantly with existing label
                is_duplicate = False
                for gt in existing_boxes:
                    # Only check against same class (though checking against all is safer to avoid overlap weirdness)
                    # Let's check against ALL existing boxes to be safe (don't label a WBC as a Trophozoite if it's already there)
                    iou = calculate_iou(pred_abs_box, gt['box'])
                    if iou > IOU_THRESHOLD:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # It's a new, high-conf Trophozoite! Add it.
                    # Format: class x_center y_center width height
                    new_label_line = f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                    new_labels.append(new_label_line)
                    # Add to existing_boxes temporarily so we don't add the SAME new box twice if model predicts duplicates
                    existing_boxes.append({'cls': cls_id, 'box': pred_abs_box})

        # 4. Append to File
        if new_labels:
            with open(lbl_path, 'a') as f:
                # Ensure we start on a new line if file doesn't end with one
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    with open(lbl_path, 'r') as f_read:
                        if not f_read.read().endswith('\n'):
                            f.write('\n')
                f.writelines(new_labels)
            total_added_labels += len(new_labels)

print(f"\nDone! Added {total_added_labels} new Trophozoite labels to the dataset.")