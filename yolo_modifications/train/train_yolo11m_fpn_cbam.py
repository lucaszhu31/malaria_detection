"""
YOLO11m with CBAM + SAHI Integration
Science Fair 2026 - Training Pipeline

Single script that:
1. Trains YOLO11m with CBAM (Channel + Spatial Attention)
2. Uses SAHI for full blood smear slide predictions
3. Handles data slicing and inference on patches
"""

import sys
import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# Add yolo_modifications folder to Python path
YOLO_MODS_PATH = Path(__file__).parent.parent.resolve()  # yolo_modifications/
sys.path.insert(0, str(YOLO_MODS_PATH))

# Import YOLO
from ultralytics import YOLO
from ultralytics.nn import tasks

# Import CBAM
import custom_modules
CBAM = custom_modules.CBAM if hasattr(custom_modules, 'CBAM') else None
setattr(tasks, 'CBAM', CBAM)
print(f"Registered CBAM module")


def train_model(data_yaml, epochs=100, imgsz=1024, device=0, project=None, name='malaria_detection'):
    """Train with CBAM"""
    
    print("\n" + "="*70)
    print("YOLO11m Training with CBAM")
    print("="*70)
    
    model_path = Path(__file__).parent.resolve() / 'yolo11m-cbam.yaml'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Model: CBAM (Channel + Spatial Attention)")
    print(f"Config: {model_path.name}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}, Image Size: {imgsz}px")
    print("="*70 + "\n")
    
    # Load and train
    model = YOLO(str(model_path))
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
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Augmentation for microscopy
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
    
    return model


def slice_and_predict_sahi(model_path, slide_folder, output_folder, slice_size=1024, overlap=0.2):
    """Use SAHI for full-slide predictions on patches"""
    
    try:
        from sahi.slicing import slice_image
        from sahi.prediction import PredictionResult
    except ImportError:
        print("⚠️  SAHI not installed. Skipping full-slide inference.")
        print("    Run: pip install sahi")
        return
    
    print("\n" + "="*70)
    print("Full-Slide Prediction with SAHI")
    print("="*70)
    
    slide_folder = Path(slide_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model = YOLO(str(model_path))
    
    for slide_path in slide_folder.glob('*.jpg'):
        print(f"\n📄 Processing {slide_path.name}...")
        img = cv2.imread(str(slide_path))
        h, w = img.shape[:2]
        
        # Slice into patches
        stride = int(slice_size * (1 - overlap))
        predictions = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + slice_size, h)
                x_end = min(x + slice_size, w)
                patch = img[y:y_end, x:x_end]
                
                # Pad if needed
                if patch.shape[0] < slice_size or patch.shape[1] < slice_size:
                    patch = cv2.copyMakeBorder(
                        patch, 0, slice_size - patch.shape[0], 
                        0, slice_size - patch.shape[1],
                        cv2.BORDER_CONSTANT, value=[0,0,0]
                    )
                
                # Predict on patch
                results = model.predict(patch, verbose=False)
                
                # Adjust coordinates back to full slide
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            box.data[:, :4] += torch.tensor([x, y, x, y])
                            predictions.append(box)
        
        # Save predictions
        output_file = output_folder / f"{slide_path.stem}_predictions.txt"
        with open(output_file, 'w') as f:
            for pred in predictions:
                f.write(f"{pred.cls} {pred.conf} {pred.xywh[0]} {pred.xywh[1]} {pred.xywh[2]} {pred.xywh[3]}\n")
        
        print(f"   ✅ Saved {len(predictions)} predictions to {output_file}")


if __name__ == "__main__":
    # Configuration - Points to ALREADY-SLICED dataset on SSD
    DATA_YAML = r'/tmp/malaria_data/yolov8_transformed/data.yaml'  # Use pre-sliced patches from SSD
    # Save to Google Drive (Colab)
    RESULTS_DIR = r'/content/drive/MyDrive/kaggle_results/cbam_fpn_experiments'
    
    # If using different sliced dataset location, modify DATA_YAML above
    # Expected structure:
    #   data.yaml
    #   images/train/*.jpg
    #   images/val/*.jpg
    #   labels/train/*.txt
    #   labels/val/*.txt
    
    if not Path(DATA_YAML).exists():
        print(f"❌ Sliced dataset not found: {DATA_YAML}")
        print("\n📌 This script expects a PRE-SLICED dataset from SAHI")
        print("   Expected location: /tmp/malaria_data/yolov8_transformed/data.yaml")
        print("   (Or modify DATA_YAML path in this script)")
        sys.exit(1)
    
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Train on SAHI-sliced dataset
    print("\n" + "="*70)
    print("WORKFLOW: SAHI Slices -> Train CBAM -> Full-Slide Inference")
    print("="*70)
    print("\nSTEP 1: TRAINING MODEL on pre-sliced SAHI patches")
    print("="*70)
    
    # Step 1: Train on SAHI-sliced dataset
    print("\n" + "="*70)
    print("WORKFLOW: SAHI Slices -> Train CBAM -> Full-Slide Inference")
    print("="*70)
    print("\nSTEP 1: TRAINING MODEL on pre-sliced SAHI patches")
    print("="*70)
    
    model = train_model(
        data_yaml=DATA_YAML,  # Use sliced patches
        epochs=50,
        imgsz=1024,
        device=0,
        project=RESULTS_DIR,
        name='malaria_detector',
    )
    
    best_weights = Path(RESULTS_DIR) / 'malaria_detector' / 'weights' / 'best.pt'
    
    # Step 2: Full-slide inference with SAHI
    print("\n" + "="*70)
    print("STEP 2: FULL-SLIDE INFERENCE WITH SAHI")
    print("="*70)
    print("Using trained model to predict on FULL (un-sliced) test slides...")
    
    slide_folder = r'/content/test_slides'  # Colab test slides location
    if Path(slide_folder).exists():
        slice_and_predict_sahi(
            str(best_weights),
            slide_folder,
            Path(RESULTS_DIR) / 'malaria_detector' / 'sahi_predictions',
        )
    else:
        print(f"Test slides folder not found: {slide_folder}")
        print("Skipping full-slide inference (optional)")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Training results: {RESULTS_DIR}/malaria_detector/")
    print(f"Best weights: {best_weights}")
    print(f"Predictions: {RESULTS_DIR}/malaria_detector/sahi_predictions/")
    print("="*70)
