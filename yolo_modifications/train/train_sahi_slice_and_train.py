"""
SAHI Slicing + YOLO11m Training - All in One Script

1. Slices original dataset from /tmp/malaria_data/yolov8_transformed/ using SAHI
2. Creates sliced dataset in /tmp/sliced_dataset/
3. Trains YOLO11m on sliced dataset (no augmentation)

Run this entire script in one Colab cell
"""

import os
import sys
from pathlib import Path
import yaml
import cv2
import shutil
from tqdm import tqdm

# Install SAHI if needed
os.system('pip install sahi -q')

from sahi.slicing import slice_coco
from ultralytics import YOLO


def slice_dataset(input_dir, output_dir, slice_size=1024, overlap_ratio=0.2):
    """Slice images and convert labels to YOLO format"""
    
    print(f"\n{'='*70}")
    print("SAHI SLICING: Converting yolov8_transformed to patches")
    print(f"{'='*70}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    stride = int(slice_size * (1 - overlap_ratio))
    
    # Process train and val splits
    for split in ['train', 'val']:
        img_dir = input_path / 'images' / split
        label_dir = input_path / 'labels' / split
        
        if not img_dir.exists():
            print(f"Skipping {split} (not found)")
            continue
        
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        print(f"\nProcessing {split}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"Slicing {split}"):
            # Read image
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Read labels
            label_path = label_dir / f"{img_path.stem}.txt"
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels = f.readlines()
            
            # Slice image and labels
            patch_id = 0
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y_end = min(y + slice_size, h)
                    x_end = min(x + slice_size, w)
                    
                    # Extract patch
                    patch = img[y:y_end, x:x_end]
                    
                    # Pad if needed
                    if patch.shape[0] < slice_size or patch.shape[1] < slice_size:
                        patch = cv2.copyMakeBorder(
                            patch, 0, slice_size - patch.shape[0],
                            0, slice_size - patch.shape[1],
                            cv2.BORDER_CONSTANT, value=[0, 0, 0]
                        )
                    
                    # Save patch
                    patch_name = f"{img_path.stem}_patch{patch_id}"
                    patch_path = output_path / 'images' / split / f"{patch_name}.jpg"
                    cv2.imwrite(str(patch_path), patch)
                    
                    # Convert and save labels for this patch
                    patch_labels = []
                    for label in labels:
                        parts = label.strip().split()
                        class_id = parts[0]
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Convert from normalized to pixel coords
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        # Check if box overlaps with patch
                        if x1 < x_end and x2 > x and y1 < y_end and y2 > y:
                            # Clip to patch
                            px1 = max(0, x1 - x)
                            py1 = max(0, y1 - y)
                            px2 = min(slice_size, x2 - x)
                            py2 = min(slice_size, y2 - y)
                            
                            # Convert back to YOLO format
                            patch_cx = (px1 + px2) / (2 * slice_size)
                            patch_cy = (py1 + py2) / (2 * slice_size)
                            patch_w = (px2 - px1) / slice_size
                            patch_h = (py2 - py1) / slice_size
                            
                            patch_labels.append(f"{class_id} {patch_cx} {patch_cy} {patch_w} {patch_h}\n")
                    
                    # Save patch labels (even if empty - YOLO expects all files)
                    label_path_out = output_path / 'labels' / split / f"{patch_name}.txt"
                    with open(label_path_out, 'w') as f:
                        f.writelines(patch_labels)
                    
                    patch_id += 1
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': {0: 'Parasite', 1: 'WBC'}
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\nSlicing complete! Dataset saved to {output_path}")


def train_on_sliced(data_yaml, epochs=50, imgsz=1024):
    """Train YOLO11m on sliced dataset (no augmentation)"""
    
    print(f"\n{'='*70}")
    print("TRAINING: YOLO11m on SAHI-Sliced Dataset (No Augmentation)")
    print(f"{'='*70}\n")
    
    model = YOLO('yolo11m.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
        device=0,
        project='/content/drive/MyDrive/kaggle_results',
        name='yolo11m_baseline_sahi',
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        verbose=True,
        plots=True,
        save=True,
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print("Results saved to Google Drive:")
    print("  /content/drive/MyDrive/kaggle_results/yolo11m_baseline_sahi/")


if __name__ == "__main__":
    # Configuration
    ORIGINAL_DATASET = '/tmp/malaria_data/yolov8_transformed'
    SLICED_DATASET = '/tmp/sliced_dataset'
    DATA_YAML = '/tmp/sliced_dataset/data.yaml'
    
    print("STEP 1: Checking original dataset...")
    if not Path(ORIGINAL_DATASET).exists():
        print(f"ERROR: Dataset not found at {ORIGINAL_DATASET}")
        sys.exit(1)
    
    print(f"OK: Found {ORIGINAL_DATASET}")
    
    # Step 1: Slice dataset
    print("\nSTEP 2: Starting SAHI slicing...")
    slice_dataset(
        input_dir=ORIGINAL_DATASET,
        output_dir=SLICED_DATASET,
        slice_size=1024,
        overlap_ratio=0.2,
    )
    
    # Step 2: Train
    print("\nSTEP 3: Starting training...")
    train_on_sliced(
        data_yaml=DATA_YAML,
        epochs=50,
        imgsz=1024,
    )
    
    print("\n" + "="*70)
    print("ALL STEPS COMPLETE")
    print("="*70)
