import json
import cv2
from ultralytics import YOLO  # or whatever framework you're using

# Load your trained model
model = YOLO('path/to/your/model.pt')

# Load existing annotations (YOLO format example)
def load_annotations(annotation_path):
    """Load existing annotations from txt file"""
    annotations = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                annotations.append(line.strip())
    except FileNotFoundError:
        pass
    return annotations

def save_annotations(annotation_path, annotations):
    """Save annotations back to file"""
    with open(annotation_path, 'w') as f:
        for ann in annotations:
            f.write(ann + '\n')

def bbox_iou(box1, box2):
    """Calculate IoU between two boxes [x_center, y_center, width, height]"""
    # Convert to corners
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def add_high_confidence_predictions(
    image_path, 
    annotation_path, 
    model, 
    confidence_threshold=0.55,
    iou_threshold=0.3,  # If predicted box overlaps existing label, skip it
    class_id=0  # parasite class
):
    """
    Run inference and add high-confidence predictions as new labels
    if they don't overlap with existing annotations
    """
    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Run inference
    results = model(image_path, conf=confidence_threshold, verbose=False)[0]
    
    # Load existing annotations
    existing_annotations = load_annotations(annotation_path)
    existing_boxes = []
    for ann in existing_annotations:
        parts = ann.split()
        if len(parts) >= 5:
            cls = int(parts[0])
            box = [float(x) for x in parts[1:5]]
            existing_boxes.append((cls, box))
    
    # Track new annotations to add
    new_annotations = []
    
    # Process predictions
    for box in results.boxes:
        pred_class = int(box.cls[0])
        pred_conf = float(box.conf[0])
        
        # Only add parasites above threshold
        if pred_class == class_id and pred_conf >= confidence_threshold:
            # Convert to YOLO format (normalized)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            pred_box = [x_center, y_center, width, height]
            
            # Check if this prediction overlaps with existing annotation
            overlaps_existing = False
            for existing_cls, existing_box in existing_boxes:
                if existing_cls == class_id:  # Same class
                    iou = bbox_iou(pred_box, existing_box)
                    if iou > iou_threshold:
                        overlaps_existing = True
                        break
            
            # Only add if it doesn't overlap existing labels
            if not overlaps_existing:
                new_ann = f"{pred_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                new_annotations.append(new_ann)
                print(f"  Added prediction: class={pred_class}, conf={pred_conf:.2f}")
    
    # Save updated annotations
    if new_annotations:
        all_annotations = existing_annotations + new_annotations
        save_annotations(annotation_path, all_annotations)
        print(f"Added {len(new_annotations)} new labels to {annotation_path}")
        return len(new_annotations)
    
    return 0

# Process entire dataset
def process_dataset(images_dir, labels_dir, model, confidence_threshold=0.55):
    """Process all images in dataset"""
    import os
    from pathlib import Path
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(exist_ok=True)
    
    total_added = 0
    
    for img_path in images_dir.glob('*.jpg'):  # Adjust extension as needed
        print(f"\nProcessing {img_path.name}...")
        annotation_path = labels_dir / (img_path.stem + '.txt')
        
        added = add_high_confidence_predictions(
            str(img_path),
            str(annotation_path),
            model,
            confidence_threshold=confidence_threshold
        )
        total_added += added
    
    print(f"\n=== SUMMARY ===")
    print(f"Total new labels added: {total_added}")

# Usage
model = YOLO('path/to/your/best.pt')
process_dataset(
    images_dir='path/to/images',
    labels_dir='path/to/labels',
    model=model,
    confidence_threshold=0.55
)