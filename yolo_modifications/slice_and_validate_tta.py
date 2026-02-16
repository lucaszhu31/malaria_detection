"""
Slice yolov8_transformed Dataset + Run Validation with TTA

1. Slices original dataset to local SSD
2. Validates trained model with TTA enabled
3. Reports metrics
"""

import os
import sys
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def slice_dataset(input_dir, output_dir, slice_size=1024, overlap_ratio=0.2):
    """Slice images and convert labels to YOLO format"""
    
    print(f"\n{'='*70}")
    print("SLICING DATASET FOR VALIDATION")
    print(f"{'='*70}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Slice:  {slice_size}x{slice_size}, overlap={overlap_ratio}")
    print(f"{'='*70}\n")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    stride = int(slice_size * (1 - overlap_ratio))
    
    # Count total images for progress tracking
    total_images = 0
    for split in ['train', 'val']:
        img_dir = input_path / 'images' / split
        if img_dir.exists():
            total_images += len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    
    print(f"Total images to slice: {total_images}")
    print(f"Estimated patches per image: ~{((1024 // stride) ** 2)}")
    print(f"Total estimated patches: ~{total_images * ((1024 // stride) ** 2)}\n")
    
    # Process train and val splits
    processed = 0
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
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
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
                        if len(parts) < 5:
                            continue
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
                    
                    # Save patch labels (even if empty)
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
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\n{'='*70}")
    print(f"Slicing complete! Dataset saved to {output_path}")
    print(f"data.yaml created at {yaml_path}")
    print(f"{'='*70}\n")
    
    return str(yaml_path)


def validate_with_tta(model_path, data_yaml, imgsz=1024):
    """Run validation with TTA enabled"""
    
    print(f"\n{'='*70}")
    print("RUNNING VALIDATION WITH TTA")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Data:  {data_yaml}")
    print(f"Size:  {imgsz}x{imgsz}")
    print(f"TTA:   Enabled (horizontal flip, vertical flip, scale variations)")
    print(f"{'='*70}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Validation with TTA
    print("Running validation with TTA...")
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=16,
        augment=True,  # Enable TTA
        verbose=True,
        project='/content/drive/MyDrive/kaggle_results/yolo11m_final',
        name='validation_tta',
        save=True,
    )
    
    # Extract metrics
    metrics = results.results_dict
    
    # Print results
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS (WITH TTA)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Value':<15}")
    print(f"{'-'*70}")
    
    metrics_to_display = [
        ('mAP@0.5', 'metrics/mAP50(B)'),
        ('mAP@0.5:0.95', 'metrics/mAP50-95(B)'),
        ('Precision', 'metrics/precision(B)'),
        ('Recall', 'metrics/recall(B)'),
    ]
    
    for name, key in metrics_to_display:
        value = metrics.get(key, 0)
        print(f"{name:<25} {value:<15.4f}")
    
    # Calculate F1 score
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{'F1 Score':<25} {f1:<15.4f}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    
    # Input: Original yolov8_transformed dataset
    ORIGINAL_DATASET = r'/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b'
    
    # Output: Sliced dataset on local SSD (fast storage)
    SLICED_DATASET = r'/tmp/sliced_dataset'
    
    # Model: Path to your trained model
    MODEL_PATH = r'/content/drive/MyDrive/kaggle_results/yolo11m_baseline_sahi/yolo11m_baseline_sahi/weights/best.pt'
    
    # Slicing parameters (match your training)
    SLICE_SIZE = 1024
    OVERLAP_RATIO = 0.2
    
    # =======================================================
    
    print("\n" + "="*70)
    print("SLICE + VALIDATE WITH TTA")
    print("="*70)
    
    # Step 1: Check inputs
    if not Path(ORIGINAL_DATASET).exists():
        print(f"ERROR: Original dataset not found: {ORIGINAL_DATASET}")
        sys.exit(1)
    
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Step 2: Slice dataset
    print("\nSTEP 1: Slicing dataset...")
    data_yaml = slice_dataset(
        input_dir=ORIGINAL_DATASET,
        output_dir=SLICED_DATASET,
        slice_size=SLICE_SIZE,
        overlap_ratio=OVERLAP_RATIO,
    )
    
    # Step 3: Run validation with TTA
    print("\nSTEP 2: Running validation with TTA...")
    results = validate_with_tta(
        model_path=MODEL_PATH,
        data_yaml=data_yaml,
        imgsz=SLICE_SIZE,
    )
    
    print("\n" + "="*70)
    print("ALL STEPS COMPLETE")
    print("="*70)
    print(f"Sliced dataset saved to: {SLICED_DATASET}")
    print(f"Validation results saved to: /content/drive/MyDrive/kaggle_results/yolo11m_final/validation_tta/")
    print(f"Metrics displayed above")
    print("="*70 + "\n")
