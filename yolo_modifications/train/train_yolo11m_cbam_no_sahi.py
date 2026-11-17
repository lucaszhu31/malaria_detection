"""
YOLO11m-CBAM Training WITHOUT SAHI Slicing
Direct training on full blood smear slides (no patching)

This is a diagnostic script to test if:
1. SAHI slicing is causing the cutoff issue
2. CBAM works better on full slides vs patches
3. Direct training has better label preservation

Uses original full-resolution images directly from yolov8_transformed dataset
"""

import sys
import os
from pathlib import Path
import yaml

# Add yolo_modifications folder to Python path
YOLO_MODS_PATH = Path(__file__).parent.parent.resolve()  # yolo_modifications/
sys.path.insert(0, str(YOLO_MODS_PATH))

# Import custom modules
import custom_modules
print(f"Custom CBAM modules loaded from: {YOLO_MODS_PATH}")

from ultralytics import YOLO
from ultralytics.nn import tasks

# Register CBAM module in YOLO's parser for post-backbone attention
setattr(tasks, 'CBAM', custom_modules.CBAM if hasattr(custom_modules, 'CBAM') else None)
print(f"Registered CBAM module in ultralytics.nn.tasks")


def train_yolo11m_cbam_no_sahi(
    data_yaml='/tmp/malaria_data/yolov8_transformed/data.yaml',
    epochs=40,
    imgsz=1280,  # Full slides resized to 1280×1280 (preserves parasite detail)
    device=0,
    project=None,
    name='yolo11m_cbam_no_sahi_run1',
):
    """
    Train YOLO11m-CBAM on FULL blood smear slides WITHOUT SAHI slicing
    
    Args:
        data_yaml: Path to original data.yaml (full slides, not sliced)
        epochs: Number of training epochs
        imgsz: Input image size for training
        device: GPU device (0, 1, etc. or 'cpu')
        project: Save directory for results
        name: Experiment name
    """
    
    # Default project directory if not provided
    if project is None:
        project = '/content/drive/MyDrive/kaggle_results/cbam_experiments'
    
    print("\n" + "="*70)
    print("YOLO11m-CBAM Training (NO SAHI - Full Slides Direct)")
    print("="*70)
    print(f"Data: {data_yaml}")
    print(f"Model: YOLO11m with post-backbone CBAM")
    print(f"Image size: {imgsz}×{imgsz} (full slides resized, not sliced)")
    print(f"Epochs: {epochs}")
    print(f"Device: GPU {device}" if device != 'cpu' else "Device: CPU")
    print(f"Save to: {project}/{name}")
    print("="*70 + "\n")
    
    # Load model with post-backbone CBAM attention
    print("Loading YOLO11m-CBAM...\n")
    model = YOLO(str(Path(__file__).parent.resolve() / 'yolo11m-cbam.yaml'))
    
    # Train with augmentation for microscopy images
    print("Starting training on FULL SLIDES (no slicing)...\n")
    
    results = model.train(
        # Basic settings
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,          # Auto-batch
        device=device,
        project=project,
        name=name,
        pretrained=False,  # CBAM architecture requires training from scratch
        verbose=True,
        plots=True,
        
        # Optimizer (Force SGD for consistency)
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation - Microscopy optimized
        hsv_h=0.015,       # HSV-Hue augmentation
        hsv_s=0.7,         # HSV-Saturation
        hsv_v=0.4,         # HSV-Value (brightness)
        degrees=180,       # Rotation (parasites can appear at any angle)
        translate=0.1,
        scale=0.5,
        flipud=0.5,        # Flip upside down (parasites symmetric)
        fliplr=0.5,        # Flip left-right
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        
        # Save and logging
        save=True,
        save_period=5,
        patience=15,
        close_mosaic=10,
    )
    
    return model


if __name__ == "__main__":
    """
    Main entry point for training YOLO11m-CBAM on FULL SLIDES
    
    NOTE: This trains on full-resolution blood smear slides WITHOUT SAHI slicing
    
    Usage:
        python train_yolo11m_cbam_no_sahi.py
    """
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    
    # Path to ORIGINAL data.yaml (full slides, not sliced)
    DATA_YAML = '/tmp/malaria_data/yolov8_transformed/data.yaml'
    PROJECT_DIR = '/content/drive/MyDrive/kaggle_results/cbam_experiments'
    
    # Training parameters
    IMGSZ = 1024         # Full slide images resized to 1024×1024 (preserves parasite detail)
    EPOCHS = 40
    EXPERIMENT_NAME = 'yolo11m_cbam_no_sahi_diagnostic'
    
    # =====================================================================
    # TRAINING EXECUTION
    # =====================================================================
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: YOLO11m-CBAM WITHOUT SAHI SLICING")
    print("="*70)
    print("\nThis training will help diagnose:")
    print("1. Does SAHI slicing cause label cutoff issues?")
    print("2. Does CBAM work better on full slides or patches?")
    print("3. Is label preservation better without slicing?")
    print("\nResults will be compared to SAHI+CBAM version")
    print("="*70 + "\n")
    
    model = train_yolo11m_cbam_no_sahi(
        data_yaml=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=0,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved: {PROJECT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"\nResults to compare:")
    print(f"  - yolo11m_cbam_sahi_1024px (with SAHI slicing)")
    print(f"  - yolo11m_cbam_no_sahi_diagnostic (WITHOUT SAHI, full slides)")
    print("\nCompare mAP, precision, recall to see which approach is better")
    print("="*70 + "\n")
