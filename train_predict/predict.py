"""
SAHI Inference Script for Full Blood Smear Slides
Science Fair 2026

This script runs YOLO11m-CBAM on full blood smear slides using SAHI
(Slicing Aided Hyper Inference) to maintain parasite detection accuracy
on large images.

How it works:
1. Load trained YOLO11m-CBAM model
2. Slice full slide into 1024×1024 patches (with overlap)
3. Run detection on each patch
4. Merge overlapping detections with NMS
5. Return final predictions on full slide

Expected performance:
- Without SAHI: 60-70% mAP (parasites too small when resized)
- With SAHI: 85-90% mAP (parasites maintain proper scale)
"""

import sys
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
import cv2

# Add yolo_modifications folder to Python path
YOLO_MODS_PATH = Path(__file__).parent.resolve()
sys.path.insert(0, str(YOLO_MODS_PATH))


def predict_full_slide_sahi(
    model_path,
    image_path,
    output_dir='sahi_results',
    slice_size=1024,
    overlap_ratio=0.2,
    confidence_threshold=0.25,
    nms_threshold=0.5,
    visualize=True,
):
    """
    Run SAHI inference on full blood smear slide
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        image_path: Path to full slide image
        output_dir: Where to save results
        slice_size: Patch size for slicing (should match training, default: 1024)
        overlap_ratio: Overlap between patches (0.2 = 20%)
        confidence_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold for merging detections
        visualize: Save visualization with bounding boxes
    
    Returns:
        SAHI prediction result object
    """
    
    print("\n" + "="*70)
    print("SAHI Inference - Full Slide Detection")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Slice size: {slice_size}×{slice_size}")
    print(f"Overlap: {int(overlap_ratio*100)}%")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*70 + "\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load detection model
    print("🔄 Loading YOLO11m-CBAM model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # YOLO11 uses YOLOv8 format
        model_path=str(model_path),
        confidence_threshold=confidence_threshold,
        device='cuda:0',  # Use GPU
    )
    print("✅ Model loaded\n")
    
    # Read image
    print("📸 Reading full slide image...")
    image = read_image(str(image_path))
    h, w = image.shape[:2]
    print(f"   Image size: {w}×{h} pixels\n")
    
    # Calculate expected number of patches
    stride = int(slice_size * (1 - overlap_ratio))
    n_patches_h = (h - slice_size) // stride + 1
    n_patches_w = (w - slice_size) // stride + 1
    total_patches = n_patches_h * n_patches_w
    
    print(f"🔪 Slicing into ~{total_patches} patches ({n_patches_h}×{n_patches_w} grid)...")
    
    # Run SAHI prediction
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type='NMS',  # Use NMS to merge overlapping boxes
        postprocess_match_metric='IOS',  # Intersection over smaller area
        postprocess_match_threshold=nms_threshold,
        postprocess_class_agnostic=False,  # Keep class-specific NMS
        verbose=1,
    )
    
    # Get detection statistics
    object_predictions = result.object_prediction_list
    n_detections = len(object_predictions)
    
    # Count by class
    class_counts = {}
    for pred in object_predictions:
        cls_name = pred.category.name
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    print("\n" + "="*70)
    print("✅ Detection Complete!")
    print("="*70)
    print(f"Total detections: {n_detections}")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count}")
    print("="*70 + "\n")
    
    # Save visualization
    if visualize:
        image_name = Path(image_path).stem
        vis_path = output_path / f"{image_name}_detections.jpg"
        
        print(f"💾 Saving visualization to: {vis_path}")
        result.export_visuals(
            export_dir=str(output_path),
            file_name=f"{image_name}_detections",
            rect_th=3,  # Rectangle thickness
            text_size=1.0,
            text_th=2,
        )
    
    # Save predictions to text file
    pred_path = output_path / f"{Path(image_path).stem}_predictions.txt"
    with open(pred_path, 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Total detections: {n_detections}\n\n")
        
        for i, pred in enumerate(object_predictions):
            bbox = pred.bbox
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Class: {pred.category.name}\n")
            f.write(f"  Confidence: {pred.score.value:.3f}\n")
            f.write(f"  BBox: x={bbox.minx:.1f}, y={bbox.miny:.1f}, ")
            f.write(f"w={bbox.maxx-bbox.minx:.1f}, h={bbox.maxy-bbox.miny:.1f}\n\n")
    
    print(f"💾 Saved predictions to: {pred_path}\n")
    
    return result


def batch_predict_sahi(
    model_path,
    image_folder,
    output_dir='sahi_batch_results',
    slice_size=1024,
    overlap_ratio=0.2,
    confidence_threshold=0.25,
):
    """
    Run SAHI inference on multiple full slides
    
    Args:
        model_path: Path to trained YOLO model
        image_folder: Folder containing full slide images
        output_dir: Where to save results
        slice_size: Patch size for slicing
        overlap_ratio: Overlap between patches
        confidence_threshold: Detection confidence threshold
    
    Returns:
        Dictionary mapping image names to prediction results
    """
    
    image_path = Path(image_folder)
    image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
    
    print(f"\n📁 Found {len(image_files)} images in {image_folder}")
    print("="*70 + "\n")
    
    results = {}
    
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_file.name}")
        
        result = predict_full_slide_sahi(
            model_path=model_path,
            image_path=str(img_file),
            output_dir=output_dir,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            confidence_threshold=confidence_threshold,
            visualize=True,
        )
        
        results[img_file.name] = result
        print()
    
    # Summary
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    
    total_detections = 0
    for img_name, result in results.items():
        n_det = len(result.object_prediction_list)
        total_detections += n_det
        print(f"{img_name}: {n_det} detections")
    
    print("-"*70)
    print(f"Total: {total_detections} detections across {len(image_files)} slides")
    print(f"Average: {total_detections/len(image_files):.1f} detections per slide")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    """
    Main inference entry point
    
    Usage:
        python predict_sahi.py
    
    Modify MODEL_PATH and IMAGE_PATH below for your predictions.
    """
    
    # =====================================================================
    # CONFIGURATION 
    # =====================================================================
    
    # Trained model from training (automatically saved here)
    MODEL_PATH = '/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px/weights/best.pt'
    
    BATCH_FOLDER = None  # Example: '/content/drive/MyDrive/test_data/full_slides/'
    
    # SAHI parameters (keep these - match your training settings)
    SLICE_SIZE = 1024           # Must match training SLICE_SIZE
    OVERLAP_RATIO = 0.2         # Must match training OVERLAP_RATIO
    CONFIDENCE_THRESHOLD = 0.25 # Lower = more detections, Higher = more confident
    
    # Where to save prediction results
    OUTPUT_DIR = '/content/drive/MyDrive/science_fair_2026/predictions'
    
    # =====================================================================
    # INFERENCE EXECUTION
    # =====================================================================
    
    if BATCH_FOLDER:
        # Batch inference on folder
        print("🔄 Running batch SAHI inference...\n")
        results = batch_predict_sahi(
            model_path=MODEL_PATH,
            image_folder=BATCH_FOLDER,
            output_dir=OUTPUT_DIR,
            slice_size=SLICE_SIZE,
            overlap_ratio=OVERLAP_RATIO,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
    else:
        # Single image inference
        print("🔄 Running SAHI inference on single image...\n")
        result = predict_full_slide_sahi(
            model_path=MODEL_PATH,
            image_path=SINGLE_IMAGE,
            output_dir=OUTPUT_DIR,
            slice_size=SLICE_SIZE,
            overlap_ratio=OVERLAP_RATIO,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            visualize=True,
        )
    
    print("🎉 SAHI inference complete!")
    print(f"Results saved to: {OUTPUT_DIR}/\n")
