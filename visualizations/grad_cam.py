import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. THE ADAPTER (Fixes the 'tuple' and 'backward' errors) ---
class YOLOv11TargetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # YOLOv11 returns a tuple; we take the first element (predictions)
        # and sum them to allow Grad-CAM to calculate gradients
        return self.model(x)[0].sum(dim=1)

# --- CONFIGURATION ---
MODEL_PATH = "/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/weights/best.pt"
INPUT_DIR = "/content/drive/MyDrive/kaggle_datasets/test_imgs/sample2"
OUTPUT_DIR = "/content/drive/MyDrive/science_fair_2026/kaggle_results/r_kaggle11/malaria_results_highres/gradcam_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model & Setup
# .model access the raw PyTorch network; .eval() is necessary for Grad-CAM
base_model = YOLO(MODEL_PATH).model.to(device).eval()
wrapped_model = YOLOv11TargetWrapper(base_model)

# Target the last layer of the backbone
target_layers = [base_model.model[-2]]
cam = GradCAM(model=wrapped_model, target_layers=target_layers)

print(f"Analyzing images in: {INPUT_DIR} on {device}")

# 2. Process Folder
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_DIR, filename)
        
        # Standard Prediction to get Confidence Score
        # We use the original YOLO wrapper here for convenience
        temp_model = YOLO(MODEL_PATH)
        results = temp_model.predict(img_path, conf=0.25, save=False, verbose=False)[0]
        
        # Load for Grad-CAM
        img = cv2.imread(img_path)
        
        # --- FIX 1: Resize to multiple of 32 (Standardizes Tensor sizes) ---
        img_resized = cv2.resize(img, (1024, 1024))
        img_float = np.float32(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)) / 255
        
        # --- FIX 2: Move Tensor to GPU (Fixes device mismatch error) ---
        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        
        # Generate Heatmap
        # eigen_smooth=True makes the parasite hotspots look more distinct
        with torch.enable_grad():
          grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)[0, :]
          
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        output_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # --- ADDING CONTEXT (Inside the Image) ---
        conf_score = results.boxes.conf[0].item() if len(results.boxes) > 0 else 0.0
        context_text = f"File: {filename} | Conf: {conf_score:.2%}"
        
        cv2.putText(output_img, context_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save Result
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{filename}")
        cv2.imwrite(save_path, output_img)
        print(f"Processed {filename} - Confidence: {conf_score:.2%}")

print(f"Done! Check your folder: {OUTPUT_DIR}")