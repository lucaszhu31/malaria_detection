# Paths Configuration Guide

## Directory Structure
```
science_fair_2026/
├── code/
│   ├── train_predict/
│   │   ├── train_yolo_cbam.py  ← Run from anywhere
│   │   └── predict.py           ← Run from anywhere
│   └── yolo_modifications/
│       ├── custom_modules.py    ← Imported by train_yolo_cbam.py
│       ├── se_block_cbam.py     ← Used by custom_modules.py
│       └── yolo11m-se.yaml      ← YOLO architecture config
└── PATHS_CONFIGURATION.md       ← This file
```

## Script Path Resolution

### train_yolo_cbam.py
**Auto-resolves paths**:
- `YOLO_MODS_PATH`: Automatically finds `code/yolo_modifications/` from script location
- `model_path`: Looks for `yolo11m-se.yaml` in `YOLO_MODS_PATH`

**Colab paths (hardcoded - must match your Colab setup)**:
```python
ORIGINAL_DATA_YAML = '/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/data.yaml'
PROJECT_DIR = '/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments'
SLICED_OUTPUT_DIR = '/content/drive/MyDrive/kaggle_datasets/sliced_dataset'
```

**Check these exist on Colab before running**:
- `/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/data.yaml`
- `/content/drive/MyDrive/kaggle_datasets/` (has write permission)

---

### predict.py
**Colab paths (hardcoded - must update for your setup)**:
```python
MODEL_PATH = '/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px/weights/best.pt'
SINGLE_IMAGE = '/content/drive/MyDrive/test_data/slide_001.jpg'  # TODO: Update!
BATCH_FOLDER = None  # Or set to folder path
OUTPUT_DIR = '/content/drive/MyDrive/science_fair_2026/predictions'
```

**Update SINGLE_IMAGE to your test slide path**:
- Example: `/content/drive/MyDrive/test_data/my_slide.jpg`
- Or use BATCH_FOLDER to process entire folder

---

## How Imports Work

### train_yolo_cbam.py imports custom_modules.py
```
train_yolo_cbam.py location: code/train_predict/train_yolo_cbam.py
                              ↓
Path(__file__).parent.parent = code/
                              ↓
/ 'yolo_modifications' = code/yolo_modifications/
                              ↓
imports custom_modules.py
```

**Works from any location**:
```powershell
# From root
python code/train_predict/train_yolo_cbam.py

# From code directory
cd code
python train_predict/train_yolo_cbam.py

# From train_predict directory
cd code/train_predict
python train_yolo_cbam.py
```

---

## File Dependencies

### train_yolo_cbam.py requires:
1. `code/yolo_modifications/custom_modules.py` (auto-found)
2. `code/yolo_modifications/yolo11m-se.yaml` (auto-found)
3. `/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/data.yaml` (Colab)

### predict.py requires:
1. Trained model at `MODEL_PATH` (created after training)
2. Test image at `SINGLE_IMAGE` or folder at `BATCH_FOLDER` (Colab)

---

## Running on Colab

### Step 1: Mount drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone repo
```python
!git clone https://github.com/YOUR_USERNAME/science_fair_2026.git
%cd science_fair_2026
```

### Step 3: Run training
```python
!python code/train_predict/train_yolo_cbam.py
```

Script will:
1. Auto-find custom_modules.py
2. Load yolo11m-se.yaml
3. Slice data from ORIGINAL_DATA_YAML
4. Train and save to PROJECT_DIR

### Step 4: Run inference
```python
!python code/train_predict/predict.py
```

Loads trained model and runs predictions.

---

## Checklist Before Running

### On Colab:
- [ ] Verify `/content/drive/MyDrive/kaggle_datasets/yolov8_transformed_b/data.yaml` exists
- [ ] Verify `/content/drive/MyDrive/kaggle_datasets/` has write permission
- [ ] Update `SINGLE_IMAGE` in predict.py to your test slide path
- [ ] Ensure A100 GPU selected (80GB VRAM)

### Local (before pushing to Colab):
- [ ] All Python files in correct directories
- [ ] No broken relative paths
- [ ] `.gitignore` excludes large files

---

## Path Summary Table

| File | Location | Auto-resolved? | Notes |
|------|----------|---|---|
| custom_modules.py | code/yolo_modifications/ | Yes | Found from YOLO_MODS_PATH |
| se_block_cbam.py | code/yolo_modifications/ | Yes | Imported by custom_modules.py |
| yolo11m-se.yaml | code/yolo_modifications/ | Yes | Loaded by YOLO |
| data.yaml | /content/drive/.../yolov8_transformed_b/ | No | Must exist on Colab |
| best.pt | /content/drive/.../cbam_experiments/.../ | No | Created after training |
| test slide | /content/drive/.../test_data/ | No | Must provide path |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'custom_modules'"
**Fix**: Ensure custom_modules.py is in `code/yolo_modifications/`

### "FileNotFoundError: yolo11m-se.yaml"
**Fix**: Check file exists at `code/yolo_modifications/yolo11m-se.yaml`

### "FileNotFoundError: data.yaml"
**Fix**: Update `ORIGINAL_DATA_YAML` in train_yolo_cbam.py to your Colab path

### "FileNotFoundError: best.pt"
**Fix**: Run training first, or update `MODEL_PATH` in predict.py

### "FileNotFoundError: test slide"
**Fix**: Update `SINGLE_IMAGE` in predict.py to your test slide path on Colab
