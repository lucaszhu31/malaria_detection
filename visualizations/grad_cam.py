import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- CONFIGURATION ---
MODEL_PATH = "/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/weights/best.pt"
INPUT_DIR = "/content/drive/MyDrive/kaggle_datasets/test_imgs/sample2"
OUTPUT_DIR = "/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/gradcam_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Model & Setup
model = YOLO(MODEL_PATH)
# Target the last layer of the backbone (most efficient for semantic features)
target_layers = [model.model.model[-2]]
cam = GradCAM(model=model.model, target_layers=target_layers)

print(f"Analyzing images in: {INPUT_DIR}")

# 2. Process Folder
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_DIR, filename)
        
        # Standard Prediction to get Confidence Score (Context)
        results = model.predict(img_path, conf=0.25, save=False)[0]
        
        # Load for Grad-CAM
        img = cv2.imread(img_path)
        img_float = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) / 255
        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        
        # Generate Heatmap
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        output_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # --- ADDING CONTEXT (Inside the Image) ---
        # Burning the filename and confidence score onto the top-left corner
        conf_score = results.boxes.conf[0].item() if len(results.boxes) > 0 else 0.0
        context_text = f"File: {filename} | Conf: {conf_score:.2%}"
        
        cv2.putText(output_img, context_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save Result
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{filename}")
        cv2.imwrite(save_path, output_img)
        print(f"Processed {filename} - Confidence: {conf_score:.2%}")

print(f"Done! Check your folder: {OUTPUT_DIR}")