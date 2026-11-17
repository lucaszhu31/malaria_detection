"""
YOLO11m Training on SAHI-Created Dataset (Baseline - No CBAM)

This script trains vanilla YOLO11m on pre-sliced SAHI dataset.
No custom modules, no CBAM, just baseline YOLO.

Purpose: Establish baseline performance without attention mechanisms
"""

import sys
from pathlib import Path

# Add yolo_modifications folder to Python path
YOLO_MODS_PATH = Path(__file__).parent.parent.resolve()  # yolo_modifications/
sys.path.insert(0, str(YOLO_MODS_PATH))

from ultralytics import YOLO


def train_baseline(data_yaml, epochs=50, imgsz=1024, device=0, project=None, name='yolo11m_baseline'):
    """Train vanilla YOLO11m on SAHI dataset"""
    
    print("\n" + "="*70)
    print("YOLO11m Baseline Training (No CBAM)")
    print("="*70)
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}x{imgsz}")
    print(f"Optimizer: SGD")
    print("="*70 + "\n")
    
    # Load vanilla YOLO11m
    model = YOLO('yolo11m.pt')
    
    # Train with data augmentation
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
        device=device,
        project=project,
        name=name,
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Data Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
        plots=True,
        save=True,
    )
    
    return model


if __name__ == "__main__":
    # Configuration
    DATA_YAML = r'/tmp/sliced_dataset/data.yaml'
    RESULTS_DIR = r'/content/drive/MyDrive/kaggle_results/yolo11m_baseline_sahi'
    
    if not Path(DATA_YAML).exists():
        print(f"ERROR: Dataset not found: {DATA_YAML}")
        print("Expected: SAHI-sliced dataset with data.yaml")
        sys.exit(1)
    
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("BASELINE TRAINING: Vanilla YOLO11m on SAHI Dataset")
    print("="*70)
    
    model = train_baseline(
        data_yaml=DATA_YAML,
        epochs=50,
        imgsz=1024,
        device=0,
        project=RESULTS_DIR,
        name='yolo11m_baseline_sahi',
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to Google Drive:")
    print(f"  {RESULTS_DIR}/")
    print("="*70)
