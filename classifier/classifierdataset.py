
import os
import shutil
import yaml
import cv2
from ultralytics import YOLO
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5  # Minimum IoU to consider a prediction as matching a label

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

def check_match_with_label_class(pred_box, original_labels, label_class_id):
    """Check if predicted box matches any original label for a given class ID"""
    if label_class_id is None:
        return False
    for label_cls_id, label_box in original_labels:
        if label_cls_id == label_class_id:
            iou = calculate_iou(pred_box, label_box)
            if iou >= IOU_THRESHOLD:
                return True
    return False

# Create folder structure
def create_folder_structure(base_dir):
    os.makedirs(os.path.join(base_dir, 'train', 'parasites'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'artifacts_bg'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'parasites'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'artifacts_bg'), exist_ok=True)

# --- 0. Copy images from Google Drive to local storage ---
if not os.path.exists(LOCAL_INPUT_DIR):
    print(f"Copying images from Google Drive to local storage...")
    shutil.copytree(GDRIVE_INPUT_DIR, LOCAL_INPUT_DIR)
    print(f"Copy complete!")
else:
    print(f"Using existing local copy at {LOCAL_INPUT_DIR}")

INPUT_DIR = LOCAL_INPUT_DIR

# --- 1. Identify Target Class ID ---
print(f"Loading class names from {DATA_YAML_PATH}...")
with open(DATA_YAML_PATH, 'r') as f:
    data_config = yaml.safe_load(f)

class_names = data_config.get('names')
trophozoite_id = None
wbc_id = None

# Handle both list and dict formats for 'names'
if isinstance(class_names, list):
    if 'Trophozoite' in class_names:
        trophozoite_id = class_names.index('Trophozoite')
    if 'WBC' in class_names:
        wbc_id = class_names.index('WBC')
elif isinstance(class_names, dict):
    for k, v in class_names.items():
        if v == 'Trophozoite':
            trophozoite_id = k
            break
    for k, v in class_names.items():
        if v == 'WBC':
            wbc_id = k
            break

if trophozoite_id is None:
    print(f"WARNING: 'Trophozoite' not found in {class_names}. Processing will check class names dynamically.")
else:
    print(f"Target Class found: 'Trophozoite' (ID: {trophozoite_id})")

if wbc_id is None:
    print(f"WARNING: 'WBC' not found in {class_names}. WBC matches will not be ignored.")
else:
    print(f"Target Class found: 'WBC' (ID: {wbc_id})")

# --- 2. Initialize YOLO Model ---
print("Initializing YOLO model...")
detection_model = YOLO(MODEL_PATH)

# --- 3. Process Images and Save All Crops ---
print(f"\n{'='*60}")
print(f"Processing all images with Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
print(f"{'='*60}")

# Reset/create folder structure
if os.path.exists(CROPS_BASE_DIR):
    shutil.rmtree(CROPS_BASE_DIR)
create_folder_structure(CROPS_BASE_DIR)

total_saved = 0

# Process train and val splits
for split in ['train', 'val']:
    split_dir = os.path.join(INPUT_DIR, split)
    labels_dir = os.path.join(GDRIVE_LABELS_DIR, split)

    if not os.path.exists(split_dir):
        print(f"Warning: {split_dir} not found, skipping...")
        continue

    image_files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {split} folder")

    image_iter = tqdm(image_files, desc=f"{split} images") if tqdm else image_files

    for idx, filename in enumerate(image_iter, start=1):
        if not tqdm:
            print(f"{split}: {idx}/{len(image_files)} images")
        image_path = os.path.join(split_dir, filename)

        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Could not load image: {filename}")
            continue

        img_h, img_w = original_img.shape[:2]

        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        original_labels = load_labels_from_file(label_path, img_w, img_h)

        results = detection_model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for i, result in enumerate(results):
            for detection_idx, detection in enumerate(result.boxes):
                bbox = detection.xyxy[0]
                pred_box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

                matches_trophozoite = check_match_with_original_labels(pred_box, original_labels, trophozoite_id)
                matches_wbc = check_match_with_label_class(pred_box, original_labels, wbc_id)

                if matches_trophozoite:
                    category = 'parasites'
                elif matches_wbc:
                    continue
                else:
                    category = 'artifacts_bg'

                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(img_w, int(bbox[2]))
                y2 = min(img_h, int(bbox[3]))

                if x2 > x1 and y2 > y1:
                    crop = original_img[y1:y2, x1:x2]
                    crop_filename = f"{os.path.splitext(filename)[0]}_crop_{i}_{detection_idx}.jpg"
                    crop_path = os.path.join(CROPS_BASE_DIR, split, category, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    total_saved += 1
                    if tqdm:
                        try:
                            image_iter.set_postfix(saved=total_saved)
                        except Exception:
                            pass

print(f"\n{'='*60}")
print(f"Done. Saved {total_saved} crops to {CROPS_BASE_DIR}")
print(f"{'='*60}")

# --- 4. Download crops as a ZIP (Colab only) ---
try:
    from google.colab import files
    zip_name = 'crops_for_mobilenet.zip'
    if os.path.exists(zip_name):
        os.remove(zip_name)
    shutil.make_archive('crops_for_mobilenet', 'zip', CROPS_BASE_DIR)
    files.download(zip_name)
except Exception as e:
    print(f"Download skipped (not in Colab or failed): {e}")