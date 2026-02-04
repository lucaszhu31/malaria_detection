#KAGGLE ANNOTATION CONVERSION TO YOLOV8 FORMAT
import os
import pandas as pd
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Directory paths
image_dir = "/content/drive/My Drive/kaggle_datasets/datasets/rajsahu2004/lacuna-malaria-detection-dataset/versions/1/images"
train_csv = "/content/drive/My Drive/kaggle_datasets/datasets/rajsahu2004/lacuna-malaria-detection-dataset/versions/1/Train.csv"
test_csv = "/content/drive/My Drive/kaggle_datasets/datasets/rajsahu2004/lacuna-malaria-detection-dataset/versions/1/Test.csv"

# Output base directory - NEW YOLOV8 FOLDER
base_dir = "/content/drive/My Drive/kaggle_datasets/yolov8_transformed"
output_train_images = os.path.join(base_dir, "images/train")
output_val_images = os.path.join(base_dir, "images/val")
output_train_labels = os.path.join(base_dir, "labels/train")
output_val_labels = os.path.join(base_dir, "labels/val")

# Create output directories
print("Creating YOLOv8 directory structure...")
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_train_labels, exist_ok=True)
os.makedirs(output_val_labels, exist_ok=True)

# Class mapping
class_mapping = {
    'Trophozoite': 0,
    'WBC': 1
}

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """Convert bounding box from (xmin, ymin, xmax, ymax) to YOLO format"""
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height

def process_image(img_id, img_annotations, output_image_dir, output_label_dir):
    """Process a single image and create YOLO annotation"""
    # Construct image filename (check if .jpg already in img_id)
    if img_id.endswith('.jpg'):
        img_filename = img_id
    else:
        img_filename = f"{img_id}.jpg"
    img_path = os.path.join(image_dir, img_filename)
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return False
    
    # Copy image to output directory
    output_img_path = os.path.join(output_image_dir, img_filename)
    try:
        shutil.copy2(img_path, output_img_path)
    except Exception as e:
        print(f"Error copying {img_filename}: {e}")
        return False
    
    # Get image dimensions
    try:
        with Image.open(img_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error reading image {img_filename}: {e}")
        return False
    
    # Create label file (use base name without extension)
    label_filename = f"{os.path.splitext(img_filename)[0]}.txt"
    label_path = os.path.join(output_label_dir, label_filename)
    
    # Write annotations
    with open(label_path, 'w') as f:
        for _, row in img_annotations.iterrows():
            class_name = row['class']
            
            # Skip NEG samples (they get empty txt files)
            if class_name == 'NEG':
                continue
            
            # Get class ID
            if class_name not in class_mapping:
                print(f"Warning: Unknown class '{class_name}' in image {img_id}")
                continue
            
            class_id = class_mapping[class_name]
            
            # Get bounding box coordinates
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            # Skip invalid bounding boxes (like 0,0,0,0)
            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                continue
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True

def create_data_yaml():
    """Create data.yaml file for YOLOv8 training"""
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {base_dir}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: Trophozoite
  1: WBC

# Number of classes
nc: 2
"""
    
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at: {yaml_path}")
    return yaml_path

def create_readme():
    """Create README with dataset information"""
    readme_content = """# Malaria Detection YOLOv8 Dataset

## Dataset Structure
```
yolov8_transformed/
├── images/
│   ├── train/       # Training images (80%)
│   └── val/         # Validation images (20%)
├── labels/
│   ├── train/       # Training annotations
│   └── val/         # Validation annotations
├── data.yaml        # YOLOv8 configuration file
└── README.md        # This file
```

## Classes
- 0: Trophozoite (malaria parasite)
- 1: WBC (White Blood Cell)

## Annotation Format
YOLOv8 format: `class_id x_center y_center width height`
All values are normalized (0-1)

## Usage
Train YOLOv8 model:
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```
"""
    
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README.md at: {readme_path}")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("CSV to YOLOv8 Dataset Converter")
    print("="*60)
    print(f"\nSource image directory: {image_dir}")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV (to exclude): {test_csv}")
    print(f"\nOutput directory: {base_dir}")
    print("="*60)
    
    # Read CSVs
    print("\nReading CSV files...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"Train.csv shape: {train_df.shape}")
    print(f"Test.csv shape: {test_df.shape}")
    
    # Get test images to exclude
    test_images = set(test_df['Image_ID'].unique())
    print(f"\nTest set images to exclude: {len(test_images)}")
    
    # Get all train images and filter out test images
    all_train_images = train_df['Image_ID'].unique()
    available_images = [img for img in all_train_images if img not in test_images]
    
    print(f"Total images in Train.csv: {len(all_train_images)}")
    print(f"Available after excluding test: {len(available_images)}")
    
    if len(available_images) == 0:
        print("\nERROR: No images available after filtering! Check your CSV files.")
        exit(1)
    
    # Split 80/20 into train/val
    print(f"\nSplitting dataset (80% train / 20% val)...")
    train_images, val_images = train_test_split(
        available_images, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"  ✓ Train set: {len(train_images)} images ({len(train_images)/len(available_images)*100:.1f}%)")
    print(f"  ✓ Val set: {len(val_images)} images ({len(val_images)/len(available_images)*100:.1f}%)")
    
    # Process train set
    print("\n" + "="*60)
    print("Processing TRAIN set...")
    print("="*60)
    train_success = 0
    train_failed = 0
    for img_id in tqdm(train_images, desc="Converting train"):
        img_annotations = train_df[train_df['Image_ID'] == img_id]
        if process_image(img_id, img_annotations, output_train_images, output_train_labels):
            train_success += 1
        else:
            train_failed += 1
    
    # Process validation set
    print("\n" + "="*60)
    print("Processing VALIDATION set...")
    print("="*60)
    val_success = 0
    val_failed = 0
    for img_id in tqdm(val_images, desc="Converting val"):
        img_annotations = train_df[train_df['Image_ID'] == img_id]
        if process_image(img_id, img_annotations, output_val_images, output_val_labels):
            val_success += 1
        else:
            val_failed += 1
    
    # Create data.yaml and README
    print("\n" + "="*60)
    print("Creating configuration files...")
    print("="*60)
    yaml_path = create_data_yaml()
    create_readme()
    
    # Final summary
    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nDataset Statistics:")
    print(f"  Train: {train_success}/{len(train_images)} images processed ({train_failed} failed)")
    print(f"  Val:   {val_success}/{len(val_images)} images processed ({val_failed} failed)")
    print(f"\nOutput Structure:")
    print(f"   {base_dir}")
    print(f"     ├── images/")
    print(f"     │   ├── train/ ({train_success} images)")
    print(f"     │   └── val/ ({val_success} images)")
    print(f"     ├── labels/")
    print(f"     │   ├── train/ ({train_success} annotations)")
    print(f"     │   └── val/ ({val_success} annotations)")
    print(f"     ├── data.yaml")
    print(f"     └── README.md")
    print("\n" + "="*60)
    print(f"Ready for YOLOv8 training!")
    print(f"Use: model.train(data='{yaml_path}')")
    print("="*60)