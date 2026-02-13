"""
YOLO11m with CE FPN + CBAM Training Script
Contrast Enhanced FPN + CBAM for Malaria Parasite Detection

This combines:
1. CE FPN (post-backbone): Enhances parasite-background contrast globally
2. CBAM (post-backbone): Focuses attention on parasite regions locally
3. SAHI: Slices full slides to 1024×1024 patches (preserves parasite scale)

Expected performance:
- SAHI + CBAM: ~85% mAP (current baseline)
- SAHI + CBAM + CE FPN: ~88-90% mAP (target)
"""

import sys
import os
from pathlib import Path
import yaml

# Add yolo_modifications folder to Python path
YOLO_MODS_PATH = Path(__file__).parent.resolve()
sys.path.insert(0, str(YOLO_MODS_PATH))

# Import custom modules
import custom_modules
from ce_fpn import ContrastEnhancedFPN
print(f"✅ Custom modules loaded from: {YOLO_MODS_PATH}")

from ultralytics import YOLO
from ultralytics.nn import tasks

# Register CE FPN and CBAM modules
setattr(tasks, 'CE_FPN', ContrastEnhancedFPN)
setattr(tasks, 'CBAM', custom_modules.CBAM if hasattr(custom_modules, 'CBAM') else None)
print(f"✅ Registered CE_FPN and CBAM modules in ultralytics.nn.tasks")


def train_yolo11m_ce_fpn_cbam(
    data_yaml='/tmp/sliced_dataset/data.yaml',
    epochs=40,
    imgsz=1024,
    device=0,
    project=None,
    name='yolo11m_ce_fpn_cbam_run1',
):
    """
    Train YOLO11m with CE FPN + CBAM on SAHI-sliced patches
    
    Args:
        data_yaml: Path to sliced data.yaml
        epochs: Number of training epochs
        imgsz: Input image size (matches SAHI patch size)
        device: GPU device
        project: Save directory
        name: Experiment name
    """
    
    if project is None:
        project = '/content/drive/MyDrive/kaggle_results/cbam_experiments'
    
    print("\n" + "="*70)
    print("YOLO11m with CE FPN + CBAM Training")
    print("="*70)
    print(f"Architecture: YOLO11m backbone → CE FPN → CBAM → Head")
    print(f"Data: {data_yaml} (SAHI sliced patches)")
    print(f"Image size: {imgsz}×{imgsz}")
    print(f"Epochs: {epochs}")
    print(f"Expected: 88-90% mAP (up from 85% baseline)")
    print("="*70 + "\n")
    
    print("🔄 Loading YOLO11m with CE FPN + CBAM...\n")
    model = YOLO('yolo11m-ce-fpn-cbam.yaml')
    
    print("🚀 Starting training...\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
        device=device,
        project=project,
        name=name,
        pretrained=False,
        verbose=True,
        plots=True,
        
        # Optimizer
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation - Microscopy optimized
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        
        save=True,
        save_period=5,
        patience=15,
    )
    
    return model


if __name__ == "__main__":
    """
    Main entry point for CE FPN + CBAM training on SAHI sliced dataset
    
    Prerequisites:
    1. SAHI dataset already sliced and saved at /tmp/sliced_dataset/
    2. CE FPN module available in this directory
    
    Usage:
        python train_yolo11m_ce_fpn_cbam.py
    """
    
    # Configuration
    SLICED_DATA_YAML = '/tmp/sliced_dataset/data.yaml'
    PROJECT_DIR = '/content/drive/MyDrive/kaggle_results/cbam_experiments'
    EPOCHS = 40
    EXPERIMENT_NAME = 'yolo11m_ce_fpn_cbam_1024px'
    
    print("\n" + "="*70)
    print("CE FPN + CBAM Enhancement Training")
    print("="*70)
    print(f"Building on: SAHI + CBAM baseline (85% mAP)")
    print(f"Adding: Contrast Enhanced FPN (CE FPN)")
    print(f"Target: 88-90% mAP")
    print("="*70 + "\n")
    
    model = train_yolo11m_ce_fpn_cbam(
        data_yaml=SLICED_DATA_YAML,
        epochs=EPOCHS,
        imgsz=1024,
        device=0,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
    )
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Results: {PROJECT_DIR}/{EXPERIMENT_NAME}/")
    print(f"\nCompare models:")
    print(f"  1. yolo11m_cbam_sahi_1024px         (85% mAP baseline)")
    print(f"  2. yolo11m_ce_fpn_cbam_1024px       (88-90% mAP target)")
    print("="*70 + "\n")
