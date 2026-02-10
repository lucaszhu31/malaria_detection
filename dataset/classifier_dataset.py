
import os
import shutil
import yaml
import cv2
from ultralytics import YOLO

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.6  # Minimum IoU to consider a prediction as matching a label
THRESHOLD_STEP = 0.05  # Step to adjust confidence threshold
TARGET_PARASITE_RATIO_MIN = 0.48  # 48%
TARGET_PARASITE_RATIO_MAX = 0.52  # 52%
CROPS_PER_ITERATION = 100  # Check ratio after every 100 crops

MODEL_PATH = '/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/weights/best.pt'
DATA_YAML_PATH = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/data.yaml'
GDRIVE_INPUT_DIR = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/images/'
GDRIVE_LABELS_DIR = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/labels/'
LOCAL_INPUT_DIR = '/tmp/input_images'
CROPS_BASE_DIR = './crops_for_mobilenet'

# --- Helper Functions ---
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in xyxy format"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def load_labels_from_file(label_path, img_width, img_height):
    """Load labels from YOLO format .txt file and convert to xyxy pixel coordinates"""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                labels.append((cls_id, [x_min, y_min, x_max, y_max]))
    
    return labels

def check_match_with_original_labels(pred_box, original_labels, trophozoite_id):
    """Check if predicted box matches any original label with IoU threshold"""
    for label_cls_id, label_box in original_labels:
        # Only compare with trophozoite labels
        if label_cls_id == trophozoite_id:
            iou = calculate_iou(pred_box, label_box)
            if iou >= IOU_THRESHOLD:
                return True
    return False

def count_crops_in_folders(base_dir):
    """Count total files in parasites and artifacts_bg folders"""
    parasites_count = 0
    artifacts_count = 0
    
    for split in ['train', 'val']:
        parasites_dir = os.path.join(base_dir, split, 'parasites')
        artifacts_dir = os.path.join(base_dir, split, 'artifacts_bg')
        
        if os.path.exists(parasites_dir):
            parasites_count += len([f for f in os.listdir(parasites_dir) if os.path.isfile(os.path.join(parasites_dir, f))])
        if os.path.exists(artifacts_dir):
            artifacts_count += len([f for f in os.listdir(artifacts_dir) if os.path.isfile(os.path.join(artifacts_dir, f))])
    
    return parasites_count, artifacts_count

# Create folder structure
def create_folder_structure(base_dir):
    os.makedirs(os.path.join(base_dir, 'train', 'parasites'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'artifacts_bg'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'parasites'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'artifacts_bg'), exist_ok=True)

# --- 0. Copy images from Google Drive to local storage ---
print(f"Copying images from Google Drive to local storage...")
if os.path.exists(LOCAL_INPUT_DIR):
    shutil.rmtree(LOCAL_INPUT_DIR)
shutil.copytree(GDRIVE_INPUT_DIR, LOCAL_INPUT_DIR)
print(f"Copy complete!")

INPUT_DIR = LOCAL_INPUT_DIR

# --- 1. Identify Target Class ID ---
print(f"Loading class names from {DATA_YAML_PATH}...")
with open(DATA_YAML_PATH, 'r') as f:
    data_config = yaml.safe_load(f)

class_names = data_config.get('names')
trophozoite_id = None

# Handle both list and dict formats for 'names'
if isinstance(class_names, list):
    if 'Trophozoite' in class_names:
        trophozoite_id = class_names.index('Trophozoite')
elif isinstance(class_names, dict):
    for k, v in class_names.items():
        if v == 'Trophozoite':
            trophozoite_id = k
            break

if trophozoite_id is None:
    print(f"WARNING: 'Trophozoite' not found in {class_names}. Processing will check class names dynamically.")
else:
    print(f"Target Class found: 'Trophozoite' (ID: {trophozoite_id})")

# --- 2. Initialize YOLO Model ---
print("Initializing YOLO model...")
detection_model = YOLO(MODEL_PATH)

# --- 3. Process Images and Save Crops with Dynamic Threshold Tuning ---
current_threshold = CONFIDENCE_THRESHOLD
found_optimal_threshold = False

while not found_optimal_threshold:
    print(f"\n{'='*60}")
    print(f"Testing with Confidence Threshold: {current_threshold:.2f}")
    print(f"{'='*60}")
    
    # Reset/create folder structure
    if os.path.exists(CROPS_BASE_DIR):
        shutil.rmtree(CROPS_BASE_DIR)
    create_folder_structure(CROPS_BASE_DIR)
    
    count = 0

    # Process both train and val splits
    for split in ['train', 'val']:
        split_dir = os.path.join(INPUT_DIR, split)
        labels_dir = os.path.join(GDRIVE_LABELS_DIR, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found, skipping...")
            continue
        
        # Only get images, avoiding directories
        image_files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in image_files:
            # Stop after 100 crops
            if count >= CROPS_PER_ITERATION:
                break
            
            image_path = os.path.join(split_dir, filename)
            
            # Load image for cropping (OpenCV loads as BGR)
            original_img = cv2.imread(image_path)
            if original_img is None:
                print(f"Could not load image: {filename}")
                continue
            
            img_h, img_w = original_img.shape[:2]
            
            # Load original labels for this image
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            original_labels = load_labels_from_file(label_path, img_w, img_h)

            # Run YOLO prediction with current threshold
            results = detection_model(image_path, conf=current_threshold, verbose=False)
            
            # Filter and Save Crops
            for i, result in enumerate(results):
                for detection in result.boxes:
                    cls_id = int(detection.cls[0])
                    cls_name = result.names[cls_id]
                    
                    bbox = detection.xyxy[0]
                    pred_box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    
                    # Check if prediction matches any original label
                    matches_label = check_match_with_original_labels(pred_box, original_labels, trophozoite_id)
                    
                    # Determine category based on match with original labels
                    if matches_label:
                        category = 'parasites'
                    else:
                        category = 'artifacts_bg'
                    
                    # Ensure coordinates are valid integers within image bounds
                    x1 = max(0, int(bbox[0]))
                    y1 = max(0, int(bbox[1]))
                    x2 = min(img_w, int(bbox[2]))
                    y2 = min(img_h, int(bbox[3]))
                
                    # Only save if the crop has valid dimensions
                    if x2 > x1 and y2 > y1:
                        crop = original_img[y1:y2, x1:x2]
                        crop_filename = f"{os.path.splitext(filename)[0]}_crop_{i}.jpg"
                        crop_path = os.path.join(CROPS_BASE_DIR, split, category, crop_filename)
                        cv2.imwrite(crop_path, crop)
                        count += 1
                        
                        # Check after 100 crops
                        if count >= CROPS_PER_ITERATION:
                            break
        
        if count >= CROPS_PER_ITERATION:
            break
    
    # Analyze the ratio
    parasites_count, artifacts_count = count_crops_in_folders(CROPS_BASE_DIR)
    total_crops = parasites_count + artifacts_count
    
    if total_crops > 0:
        parasite_ratio = parasites_count / total_crops
        print(f"\nFinal counts at Confidence {current_threshold:.2f}:")
        print(f"  Parasites: {parasites_count} ({parasite_ratio*100:.1f}%)")
        print(f"  Artifacts: {artifacts_count} ({(1-parasite_ratio)*100:.1f}%)")
        print(f"  Total: {total_crops} crops")
        
        # Check if ratio is within target range
        if TARGET_PARASITE_RATIO_MIN <= parasite_ratio <= TARGET_PARASITE_RATIO_MAX:
            print(f"\n✓ OPTIMAL THRESHOLD FOUND: {current_threshold:.2f}")
            print(f"  Parasite ratio {parasite_ratio*100:.1f}% is within target range ({TARGET_PARASITE_RATIO_MIN*100:.0f}%-{TARGET_PARASITE_RATIO_MAX*100:.0f}%)")
            
            # Save to Google Drive
            print(f"\nSaving dataset to Google Drive...")
            gdrive_output_dir = '/content/drive/MyDrive/science_fair_2026/mbv3_classifier_dataset/t1'
            os.makedirs(gdrive_output_dir, exist_ok=True)
            
            # Copy the entire crops directory to Google Drive
            gdrive_crops_dir = os.path.join(gdrive_output_dir, 'crops_for_mobilenet')
            if os.path.exists(gdrive_crops_dir):
                shutil.rmtree(gdrive_crops_dir)
            shutil.copytree(CROPS_BASE_DIR, gdrive_crops_dir)
            print(f"✓ Dataset saved to {gdrive_crops_dir}")
            
            found_optimal_threshold = True
        else:
            # Adjust threshold
            if parasite_ratio > TARGET_PARASITE_RATIO_MAX:
                # Too many parasites, increase confidence threshold to reduce detections
                print(f"  → Too many parasites, increasing threshold...")
                current_threshold += THRESHOLD_STEP
            else:
                # Too few parasites, decrease confidence threshold to increase detections
                print(f"  → Too few parasites, decreasing threshold...")
                current_threshold -= THRESHOLD_STEP
            
            # Clamp threshold between reasonable values
            current_threshold = max(0.05, min(0.95, current_threshold))
    else:
        print(f"No crops generated. Decreasing threshold...")
        current_threshold -= THRESHOLD_STEP
        current_threshold = max(0.05, current_threshold)

print(f"\n{'='*60}")
print(f"Processing complete with Confidence Threshold: {current_threshold:.2f}")
print(f"Final dataset saved to {CROPS_BASE_DIR}")
print(f"{'='*60}")