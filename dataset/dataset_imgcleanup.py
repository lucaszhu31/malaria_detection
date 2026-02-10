import os
import cv2
import shutil
from tqdm.notebook import tqdm

def create_filtered_dataset(source_root, dest_root, exclude_w, exclude_h):
    """
    Copies a YOLO dataset to a new location, excluding images of specific dimensions.
    Resumes if files already exist in destination.
    """
    
    # Define splits to process
    splits = ['train', 'val']
    
    # Ensure destination root exists
    if os.path.exists(dest_root):
        print(f"Destination folder exists: {dest_root} (Resuming...)")
    else:
        os.makedirs(dest_root)
    
    # Copy data.yaml if it exists to make it a valid dataset
    yaml_src = os.path.join(source_root, 'data.yaml')
    if os.path.exists(yaml_src):
        shutil.copy2(yaml_src, os.path.join(dest_root, 'data.yaml'))
        print(f"Copied data.yaml to {dest_root}")

    total_kept = 0
    total_skipped = 0
    total_already_processed = 0

    for split in splits:
        print(f"\nProcessing split: {split}")
        
        # Define paths
        src_img_dir = os.path.join(source_root, 'images', split)
        src_lbl_dir = os.path.join(source_root, 'labels', split)
        
        dest_img_dir = os.path.join(dest_root, 'images', split)
        dest_lbl_dir = os.path.join(dest_root, 'labels', split)
        
        # Create destination directories
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        if not os.path.exists(src_img_dir):
            print(f"Source directory not found: {src_img_dir}, skipping.")
            continue

        # Get list of images
        image_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        for img_file in tqdm(image_files, desc=f"Filtering {split}"):
            src_img_path = os.path.join(src_img_dir, img_file)
            dest_img_path = os.path.join(dest_img_dir, img_file)
            
            # --- OPTIMIZATION: Check if already processed ---
            if os.path.exists(dest_img_path):
                total_already_processed += 1
                total_kept += 1
                continue
            
            # Check dimensions
            try:
                img = cv2.imread(src_img_path)
                if img is None:
                    print(f"Warning: Could not read {img_file}")
                    continue
                    
                h, w, _ = img.shape
                
                # Check exclusion criteria
                if w == exclude_w and h == exclude_h:
                    total_skipped += 1
                    continue # SKIP this file
                
                # If we get here, keep the file
                total_kept += 1
                
                # Copy Image
                shutil.copy2(src_img_path, dest_img_path)
                
                # Copy Label (if it exists)
                # Label file usually has same name but .txt extension
                lbl_file = os.path.splitext(img_file)[0] + '.txt'
                src_lbl_path = os.path.join(src_lbl_dir, lbl_file)
                
                if os.path.exists(src_lbl_path):
                    shutil.copy2(src_lbl_path, os.path.join(dest_lbl_dir, lbl_file))
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

    print(f"\n--- Summary ---")
    print(f"Total images already in new folder: {total_already_processed}")
    print(f"Total new images copied: {total_kept - total_already_processed}")
    print(f"Total images kept (Total): {total_kept}")
    print(f"Total images skipped (Found {exclude_w}x{exclude_h}): {total_skipped}")
    print(f"New dataset at: {dest_root}")

# --- Configuration ---
SOURCE_DATASET_ROOT = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed'
DEST_DATASET_ROOT = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b'

EXCLUDE_WIDTH = 4000
EXCLUDE_HEIGHT = 3000

# Run the filtering
create_filtered_dataset(SOURCE_DATASET_ROOT, DEST_DATASET_ROOT, EXCLUDE_WIDTH, EXCLUDE_HEIGHT)