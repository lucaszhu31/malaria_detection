"""
YOLO11m with FPN (Contrast Enhanced Feature Pyramid Network) Training Script
Science Fair 2026

This script trains YOLO11 medium model with CE_FPN for parasite contrast enhancement.

Architecture:
- Base: YOLO11m backbone (Conv → C3k2 blocks → SPPF → C2PSA)
- CE_FPN: Integrated in backbone for contrast enhancement (512 channels)
- Design: FPN enhances parasite-background contrast before detection head
- Benefit: Better visibility of parasites in difficult staining conditions

Expected performance:
- Baseline YOLO11m: 75-77% mAP
- + CE_FPN: 77-82% mAP (enhanced contrast)
- + SAHI (full slides): 85-90% mAP
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

# Import custom modules
import custom_modules
print(f"✅ Custom FPN modules loaded from: {YOLO_MODS_PATH}")

from ultralytics import YOLO
from ultralytics.nn import tasks
from ce_fpn import ContrastEnhancedFPN

# Register CE_FPN module in YOLO's parser
setattr(tasks, 'CE_FPN', ContrastEnhancedFPN)
print(f"✅ Registered CE_FPN module in ultralytics.nn.tasks")


def train_yolo11m_fpn(
    data_yaml,
    epochs=100,
    imgsz=1024,
    device=0,
    project=None,
    name='yolo11m_fpn_run1',
):
    """
    Train YOLO11m with CE_FPN contrast enhancement
    
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
    print("YOLO11m-FPN (Contrast Enhanced) Training")
    print("="*70)
    
    # Resolve model path
    model_path = YOLO_MODS_PATH / 'yolo11m-fpn-integrated.yaml'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")
    
    print(f"Model: {model_path.name}")
    print(f"Data: {data_yaml}")
    print(f"Image size: {imgsz}px")
    print(f"Epochs: {epochs}")
    print(f"Device: GPU {device}" if device != 'cpu' else "Device: CPU")
    print(f"Save to: {project}/{name}")
    print("="*70 + "\n")
    
    # Load model with CE_FPN integrated in backbone
    print("🔄 Loading YOLO11m with CE_FPN...\n")
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
        pretrained=False,  # FPN requires training from scratch
        verbose=True,
        plots=True,
        
        # Optimizer
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation (Microscopy-optimized)
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
    )
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Best weights: {project}/{name}/weights/best.pt")
    print(f"Results: {project}/{name}/results.csv")
    print("="*70 + "\n")
    
    return model


if __name__ == "__main__":
    """
    Main training entry point for FPN variant
    """
    
    # Set paths
    DATA_YAML = r'g:\My Drive\science_fair_2026\dataset\data.yaml'
    RESULTS_DIR = r'g:\My Drive\science_fair_2026\training_results'
    
    # Verify data.yaml exists
    if not Path(DATA_YAML).exists():
        print(f"❌ ERROR: data.yaml not found at {DATA_YAML}")
        print("Please ensure dataset is prepared first.")
        sys.exit(1)
    
    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Train model
    model = train_yolo11m_fpn(
        data_yaml=DATA_YAML,
        epochs=100,
        imgsz=1024,
        device=0,
        project=RESULTS_DIR,
        name='yolo11m_fpn_run1',
    )
    
    print("\n🎉 FPN training complete! Results saved.")
