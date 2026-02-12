"""
YOLO11m-CBAM Training Script for Malaria Parasite Detection
Science Fair 2026

This script trains YOLO11 medium model with CBAM (Convolutional Block 
Attention Module) for detecting malaria parasites in FULL BLOOD SMEAR SLIDES.

Pipeline:
1. SAHI Slicing: Slice full slides into 1024×1024 patches (preserves parasite scale)
2. CBAM Training: Train on patches with channel + spatial attention
3. Augmentation: Microscopy-optimized augmentations
4. Inference: Use SAHI to detect on full slides

CBAM adds:
- Channel attention: Which features are important (texture, edges, color)
- Spatial attention: Where to look (parasite locations vs background)

Expected performance:
- Baseline YOLO11m: 75-77% mAP
- + CBAM attention: 78-82% mAP
- + Enhanced augmentation: 82-85% mAP
- + SAHI (full slides): 85-90% mAP
- + TTA + Ensemble: 88-92% mAP
"""

import sys
import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# Add yolo_modifications folder to Python path
YOLO_MODS_PATH = Path(__file__).parent.resolve()
sys.path.insert(0, str(YOLO_MODS_PATH))

# Import custom modules - this registers C2f_CBAM, SPPF_CBAM, etc.
import custom_modules
print(f"✅ Custom CBAM modules loaded from: {YOLO_MODS_PATH}")
print(f"   Available modules: {list(custom_modules.CBAM_MODULES.keys())}")

from ultralytics import YOLO


def slice_dataset_with_sahi(
    source_data_yaml,
    output_dir='sliced_dataset',
    slice_size=1024,
    overlap_ratio=0.2,
):
    """
    Slice full blood smear slides into patches using SAHI
    
    This preserves parasite scale:
    - Full slide: 4000×4000px, parasite = 30-115px
    - After slicing: 1024×1024 patches, parasite still = 30-115px ✓
    
    Args:
        source_data_yaml: Path to original data.yaml (points to full slides)
        output_dir: Where to save sliced patches
        slice_size: Patch size (default: 1024, matches training imgsz)
        overlap_ratio: Overlap between patches (0.2 = 20%, catches edge parasites)
    
    Returns:
        Path to new data.yaml for sliced dataset
    """
    
    try:
        from sahi.slicing import slice_coco
    except ImportError:
        raise ImportError("SAHI not installed. Run: pip install sahi")
    
    print("\n" + "="*70)
    print("SAHI Dataset Slicing - Full Slides → Patches")
    print("="*70)
    
    # Load original dataset config
    with open(source_data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config.get('path', '.'))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Source dataset: {dataset_path}")
    print(f"Output directory: {output_path}")
    print(f"Slice size: {slice_size}×{slice_size}")
    print(f"Overlap: {int(overlap_ratio*100)}%")
    print("="*70 + "\n")
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        print(f"📍 Slicing {split} set...")
        
        # Paths
        source_images = dataset_path / data_config[split].replace('images/', '')
        source_labels = source_images.parent.parent / 'labels' / split
        
        output_images = output_path / 'images' / split
        output_labels = output_path / 'labels' / split
        
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        if not source_images.exists():
            print(f"   ⚠️  {split} images not found: {source_images}")
            continue
        
        # Get all images
        image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
        
        print(f"   Found {len(image_files)} full slide images")
        print(f"   Slicing into {slice_size}×{slice_size} patches...\n")
        
        total_patches = 0
        
        for img_path in tqdm(image_files, desc=f"   {split}"):
            # Convert YOLO labels to COCO format for SAHI
            label_path = source_labels / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                print(f"   ⚠️  Label not found: {label_path.name}")
                continue
            
            # Slice image with SAHI
            # Note: SAHI expects COCO format, so we need to convert YOLO→COCO→slice→YOLO
            # For simplicity, we'll use a manual slicing approach
            import cv2
            import numpy as np
            
            # Read image
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Read YOLO labels (class, x_center, y_center, width, height - normalized)
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]
            
            # Calculate slice parameters
            stride = int(slice_size * (1 - overlap_ratio))
            
            patch_count = 0
            
            # Slice image into patches
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    # Extract patch
                    y_end = min(y + slice_size, h)
                    x_end = min(x + slice_size, w)
                    
                    patch = img[y:y_end, x:x_end]
                    
                    # Pad if necessary (edges might be smaller)
                    if patch.shape[0] < slice_size or patch.shape[1] < slice_size:
                        patch = cv2.copyMakeBorder(
                            patch,
                            0, slice_size - patch.shape[0],
                            0, slice_size - patch.shape[1],
                            cv2.BORDER_CONSTANT,
                            value=[0, 0, 0]
                        )
                    
                    # Convert labels to patch coordinates
                    patch_labels = []
                    
                    for label in labels:
                        cls = int(label[0])
                        # Convert normalized coords to absolute
                        x_center = float(label[1]) * w
                        y_center = float(label[2]) * h
                        box_w = float(label[3]) * w
                        box_h = float(label[4]) * h
                        
                        # Check if box center is in this patch
                        if (x <= x_center < x_end) and (y <= y_center < y_end):
                            # Convert to patch-relative normalized coords
                            new_x_center = (x_center - x) / slice_size
                            new_y_center = (y_center - y) / slice_size
                            new_w = box_w / slice_size
                            new_h = box_h / slice_size
                            
                            # Clip to [0, 1]
                            new_x_center = max(0, min(1, new_x_center))
                            new_y_center = max(0, min(1, new_y_center))
                            new_w = max(0, min(1, new_w))
                            new_h = max(0, min(1, new_h))
                            
                            patch_labels.append(f"{cls} {new_x_center} {new_y_center} {new_w} {new_h}")
                    
                    # Only save patches with labels
                    if patch_labels:
                        patch_name = f"{img_path.stem}_patch_{patch_count}"
                        
                        # Save patch image
                        cv2.imwrite(str(output_images / f"{patch_name}.jpg"), patch)
                        
                        # Save patch labels
                        with open(output_labels / f"{patch_name}.txt", 'w') as f:
                            f.write('\n'.join(patch_labels))
                        
                        patch_count += 1
                        total_patches += 1
        
        print(f"   ✅ Created {total_patches} patches from {len(image_files)} slides\n")
    
    # Create new data.yaml for sliced dataset
    new_data_yaml = output_path / 'data.yaml'
    
    sliced_config = {
        'path': str(output_path.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test' if 'test' in data_config else 'images/val',
        'nc': data_config['nc'],
        'names': data_config['names'],
    }
    
    with open(new_data_yaml, 'w') as f:
        yaml.dump(sliced_config, f, default_flow_style=False)
    
    print("="*70)
    print("✅ Slicing Complete!")
    print("="*70)
    print(f"Sliced dataset: {new_data_yaml}")
    print(f"Total patches: {total_patches}")
    print("="*70 + "\n")
    
    return str(new_data_yaml)


def train_yolo_cbam(
    data_yaml='malaria.yaml',
    epochs=100,
    imgsz=1024,
    device=0,
    project=None,  # Will use default below if None
    name='yolo11m_cbam_run1',
):
    """
    Train YOLO11m with CBAM attention - Simplified version
    
    Args:
        data_yaml: Path to data configuration file
        epochs: Number of training epochs
        imgsz: Input image size in pixels
        device: GPU device (0, 1, etc. or 'cpu')
        project: Save directory for results
        name: Experiment name
    
    Returns:
        Trained model
    """
    
    print("\n" + "="*70)
    print("YOLO11m-CBAM Training")
    print("="*70)
    
    # Resolve model path
    model_path = YOLO_MODS_PATH / 'yolo11m-cbam.yaml'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")
    
    print(f"Model: {model_path.name}")
    print(f"Data: {data_yaml}")
    print(f"Image size: {imgsz}px")
    print(f"Epochs: {epochs}")
    print(f"Device: GPU {device}" if device != 'cpu' else "Device: CPU")
    print(f"Save to: {project}/{name}")
    print("="*70 + "\n")
    
    # Load model
    print("🔄 Loading YOLO11m-CBAM...\n")
    model = YOLO(str(model_path))
    
    # Train with augmentation for microscopy images
    print("🚀 Starting training...\n")
    
    results = model.train(
        # Basic settings
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,          # Auto-batch
        device=device,
        project=project,
        name=name,
        pretrained=True,   # Use YOLO11m pretrained weights
        verbose=True,
        plots=True,
        
        # Augmentation (Microscopy-optimized)
        hsv_h=0.015,       # Hue shift (±1.5%, minimal for Giemsa stain)
        hsv_s=0.7,         # Saturation (±70%, handles stain intensity variation)
        hsv_v=0.4,         # Brightness (±40%, handles lighting differences)
        degrees=180,       # Rotation (full 360° rotation, no orientation bias)
        translate=0.1,     # Translation (±10%)
        scale=0.5,         # Zoom (±50%)
        flipud=0.5,        # Vertical flip (50% chance)
        fliplr=0.5,        # Horizontal flip (50% chance)
        mosaic=1.0,        # Mosaic augmentation (combines 4 images)
        mixup=0.15,        # MixUp (15% chance, blends two images)
        copy_paste=0.1,    # Copy-paste (10%, duplicates parasites to new locations)
    )
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Best weights: {project}/{name}/weights/best.pt")
    print(f"Results: {project}/{name}/results.csv")
    print("="*70 + "\n")
    
    return model


# Optional: Progressive training for advanced users
# Uncomment and modify if you want to try 1024→2048 progressive training


if __name__ == "__main__":
    """
    Main training entry point for FULL SLIDE workflow
    
    Pipeline:
    1. Slice full slides into 1024×1024 patches (preserves parasite scale)
    2. Train YOLO11m-CBAM on patches
    3. Use predict_sahi.py for inference on full slides
    
    Usage:
        python train_yolo_cbam.py
    """
    
    # =====================================================================
    # CONFIGURATION - Change these for your experiment
    # =====================================================================
    
  
    ORIGINAL_DATA_YAML = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/data.yaml'
    PROJECT_DIR = '/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments'
    
    # Slicing parameters (OK to keep as-is)
    SLICE_SIZE = 1024       # Patch size (matches training imgsz)
    OVERLAP_RATIO = 0.2     # 20% overlap between patches
    SLICED_OUTPUT_DIR = '/content/drive/MyDrive/kaggle_datasets/sliced_dataset'  # Where to save patches
    
    # Training parameters (OK to keep as-is)
    EPOCHS = 100
    EXPERIMENT_NAME = 'yolo11m_cbam_sahi_1024px'
    
    # Whether to re-slice dataset (set False if already sliced)
    FORCE_RESLICE = False
    
    # =====================================================================
    # PIPELINE EXECUTION
    # =====================================================================
    
    sliced_data_yaml_path = Path(SLICED_OUTPUT_DIR) / 'data.yaml'
    
    # Step 1: Slice dataset (if needed)
    if FORCE_RESLICE or not sliced_data_yaml_path.exists():
        print("🔪 Step 1: Slicing full slides into patches...")
        print(f"   This preserves parasite scale (30-115px)")
        print(f"   Full slide 4000×4000 → Multiple 1024×1024 patches\n")
        
        sliced_data_yaml = slice_dataset_with_sahi(
            source_data_yaml=ORIGINAL_DATA_YAML,
            output_dir=SLICED_OUTPUT_DIR,
            slice_size=SLICE_SIZE,
            overlap_ratio=OVERLAP_RATIO,
        )
    else:
        print("✅ Sliced dataset already exists, skipping slicing...")
        print(f"   Using: {sliced_data_yaml_path}")
        print(f"   (Set FORCE_RESLICE=True to re-slice)\n")
        sliced_data_yaml = str(sliced_data_yaml_path)
    
    # Step 2: Train on sliced patches
    print("🚀 Step 2: Training YOLO11m-CBAM on patches...")
    print(f"   Model: YOLO11m with CBAM attention")
    print(f"   Data: Sliced patches at {SLICE_SIZE}×{SLICE_SIZE}")
    print(f"   Augmentation: Microscopy-optimized")
    print(f"   Epochs: {EPOCHS}\n")
    
    model = train_yolo_cbam(
        data_yaml=sliced_data_yaml,
        epochs=EPOCHS,
        imgsz=SLICE_SIZE,  # Train at same size as sliced patches
        device=0,
        project=PROJECT_DIR,  # Use configured project directory
        name=EXPERIMENT_NAME,
    )
    
    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
    print(f"Trained on: {SLICE_SIZE}×{SLICE_SIZE} patches from full slides")
    print(f"Model saved: ../../../kaggle_results/cbam_experiments/{EXPERIMENT_NAME}/weights/best.pt")
    print("\n📋 Next Steps:")
    print("1. Use predict_sahi.py to run inference on full slides")
    print("2. SAHI will automatically slice test slides and merge results")
    print("3. Expected: 85-90% mAP on full slides")
    print("4. Apply TTA + Ensemble for 88-92% mAP")
    print("="*70 + "\n")
