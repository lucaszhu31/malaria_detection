import cv2
import os
from ultralytics import solutions

images_dir = "/content/drive/MyDrive/kaggle_datasets/test_imgs/sample2"
base_output = "/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/heatmap"
os.makedirs(base_output, exist_ok=True) 

heatmap = solutions.Heatmap(
    model="/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/weights/best.pt",
    colormap=cv2.COLORMAP_JET
)

for filename in os.listdir(images_dir):
    if filename.lower().endswith((".jpg", ".png")): 
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        
        # Generate the heatmap
        heatmap_img = heatmap.generate_heatmap(img)

        # Fix: Create a new variable for the file path so you don't ruin base_output
        save_file_path = os.path.join(base_output, f"heatmap_{filename}")
        cv2.imwrite(save_file_path, heatmap_img)
        print(f"Saved: {save_file_path}")