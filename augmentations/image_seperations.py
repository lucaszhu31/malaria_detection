import os
import cv2

def count_images_by_dimensions(folder_path, target_width, target_height):
    """
    Counts images in a folder that match specific width and height.

    Args:
        folder_path (str): The path to the folder containing images.
        target_width (int): The target width in pixels.
        target_height (int): The target height in pixels.

    Returns:
        int: The count of images matching the dimensions.
    """
    count = 0
    # List of common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png')

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return 0

    print(f"Scanning folder: {folder_path}")
    print(f"Looking for images with dimensions: {target_width}x{target_height} pixels")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    # img.shape returns (height, width, channels)
                    height, width, _ = img.shape
                    if width == target_width and height == target_height:
                        count += 1
            except Exception as e:
                print(f"Could not read image {filename}: {e}")
    return count

# --- Configuration ---

images_folder = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed/images/' 

target_w = 4000
target_h = 3000

# Call the function and print the result
matching_images_count = count_images_by_dimensions(images_folder, target_w, target_h)
print(f"\nTotal images with {target_w}x{target_h} pixels: {matching_images_count}")
