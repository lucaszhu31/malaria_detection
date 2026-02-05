import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

image_dir = '/content/malaria_dataset_new/train/images'
microscope_files = [f for f in os.listdir(image_dir) if f.startswith('0')]
smartphone_files = [f for f in os.listdir(image_dir) if not f.startswith('0')]

def apply_augmentations_m(img_m):
    # 1. RANDOM BLUR (Must be odd)
    blur_size = random.choice([i for i in range(11, 23) if i % 2 != 0])
    img_m = cv2.GaussianBlur(img_m, (blur_size, blur_size), 0)

    # 2. GAUSSIAN NOISE
    row, col, ch = img_m.shape
    mean = 0
    sigma = random.uniform(3, 9)
    gauss = np.random.normal(mean, sigma, (row, col, ch))

    # Add the noise to the image and ensure pixels stay in 0-255 range
    img_m = np.clip(img_m.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    # Convert to HSV for Color and Brightness control
    hsv = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 2. RANDOM BRIGHTNESS -> v channel
    brightness_factor = random.uniform(0.8, 1.1)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    # 3. ADDING THE "TAN" TINT -> h channel
    hue_shift = random.uniform(-10, 10)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

    # 4. SATURATION BOOST
    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.4, 0.8)

    # Convert back to BGR for OpenCV saving
    img_m = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img_m


def apply_augmentations_s(img_s):

    blur_size = random.choice([i for i in range(1, 5) if i % 2 != 0])
    img_s = cv2.GaussianBlur(img_s, (blur_size, blur_size), 0)

    row, col, ch = img_s.shape
    mean = 0
    sigma = random.uniform(2, 5)
    gauss = np.random.normal(mean, sigma, (row, col, ch))

    img_s = np.clip(img_s.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue_shift = random.uniform(-10, 10)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.5, 1.0)

    img_s = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img_s

# 3. Apply heavy augmentation to ALL microscope images
print(f"Modifying {len(microscope_files)} microscope images...")
for filename in microscope_files:
    file_path = os.path.join(image_dir, filename)
    img = cv2.imread(file_path)
    if img is not None:
        # Use your finalized 'm' function here
        augmented = apply_augmentations_m(img)
        cv2.imwrite(file_path, augmented) # Overwrites the original file

# 4. Apply light augmentation to ALL smartphone images
print(f"Modifying {len(smartphone_files)} smartphone images...")
for filename in smartphone_files:
    file_path = os.path.join(image_dir, filename)
    img = cv2.imread(file_path)
    if img is not None:
        # Use your finalized 's' function here
        augmented = apply_augmentations_s(img)
        cv2.imwrite(file_path, augmented)

