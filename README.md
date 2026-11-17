# Malaria Parasite Detection

Detects malaria parasites (trophozoites) and WBCs in full blood-smear slides.

**The final method is YOLO11m + CBAM attention, trained and run on SAHI slices.**
The repo also keeps scripts from experiments I ran along the way. They are kept for
reference and comparison, but the final results do not use them. This file shows which is which.

---

## Main pipeline (YOLO11m + SAHI + CBAM)

```
Kaggle Lacuna dataset
   │  augmentations/kaggle_annotation_conversion.py   (CSV boxes → YOLO format, train/val split)
   ▼
yolov8_transformed/  (full slides)
   │  yolo_modifications/train/train_yolo_cbam.py
   │    1. SAHI-style slicing → 1024×1024 patches (overlap 0.4)
   │    2. Train YOLO11m-CBAM on patches (yolo11m-cbam.yaml, 40 epochs, imgsz 1024)
   ▼
cbam_experiments/yolo11m_cbam_sahi_1024px/weights/best.pt
   │  yolo_modifications/predict/predict.py
   │    SAHI sliced inference on full slides → merged detections
   ▼
Predictions on full slides
```

| File | Role |
|---|---|
| `augmentations/kaggle_annotation_conversion.py` | Data prep: converts the Kaggle annotations to YOLO format |
| `yolo_modifications/train/train_yolo_cbam.py` | **Main training script** (slicing + CBAM training) |
| `yolo_modifications/train/yolo11m-cbam.yaml` | **Model architecture**: standard YOLO11m with one `CBAM` layer after the backbone (after C2PSA, before the neck) |
| `yolo_modifications/cbam.py` | CBAM implementation (channel + spatial attention) used by the yaml |
| `yolo_modifications/custom_modules.py` | Imports `CBAM` so the training/predict scripts can register it with Ultralytics |
| `yolo_modifications/predict/predict.py` | **Main inference script** (SAHI sliced prediction on full slides) |

Only the `CBAM` class does any work. The other modules in `custom_modules.py`
(`C2f_CBAM`, `Conv_CBAM`, `Bottleneck_CBAM`, `SPPF_CBAM`, `C2f_CBAM_Deep`) and
`SEBlock` / `SEBottleneck` (`squeeze_excitation.py`, `se_bottleneck.py`) are earlier attention-placement
ideas. The final yaml does not use them.

---

## Experiments and ablations (not the final method)

### Detector variants (compared against the main pipeline)

| File | What it tested | Difference from main |
|---|---|---|
| `train/train_yolo11m_sahi_only.py` | Baseline: vanilla YOLO11m on SAHI patches | **No CBAM** (ablation) |
| `train/train_sahi_slice_and_train.py` | Same baseline, slices data itself, no augmentation → `yolo11m_baseline_sahi` | **No CBAM**, no augmentation |
| `train/train_yolo11m_cbam_no_sahi.py` | CBAM trained on full slides resized to 1280 | **No SAHI** (ablation) |
| `train/train_yolo11m_fpn_cbam.py` | Earlier all-in-one CBAM + SAHI script, left over from the CE-FPN attempt | Superseded by `train_yolo_cbam.py` |
| `predict/slice_and_validate_tta.py` | Validation with test-time augmentation | Points at the **baseline** model; `augment=True` is commented out |
| `register_modules.py` | Old helper for registering CBAM | Unused; the train/predict scripts register CBAM themselves |

### Two-stage classifier (YOLO crops → classifier)

All of these use the older `r_kaggle11/malaria_results_highres` model, not the CBAM model.

| File | Purpose |
|---|---|
| `classifier/classifierdataset.py` | Crops YOLO detections into parasite / artifact folders |
| `classifier/confthres_optimizer.py` | Same, but tunes the confidence threshold to get a ~50/50 class balance |
| `classifier/mbnv3.py` | MobileNetV3 parasite-vs-artifact classifier |
| `classifier/customcnn.py` | Small custom CNN for 30–115 px crops |

### Dataset cleanup experiments

| File | Purpose |
|---|---|
| `dataset/dataset_imgcleanup.py` | Builds `yolov8_transformed_b` with the 4000×3000 images removed |
| `dataset/image_seperations.py` | Counts images by resolution |
| `dataset/label_cleanup.py` | Pseudo-labels: adds high-confidence trophozoite predictions to `yolov8_transformed_b` |
| `dataset/label_cleanup_test.py` | Draft of the above (placeholder model path) |
| `augmentations/lacuna_old_augmentation.py` | Old hand-made microscope/smartphone augmentation |

The main pipeline trains on `yolov8_transformed`, **not** `yolov8_transformed_b`.

### Visualization / analysis

| File | Purpose |
|---|---|
| `visualizations/grad_cam.py` | Grad-CAM heatmaps (currently points at the baseline SAHI model) |
| `visualizations/heatmap.py` | Ultralytics detection heatmap (points at `r_kaggle11`) |

---

## Known inconsistencies in the main pipeline

- **Slice overlap:** training uses `OVERLAP_RATIO = 0.4`, but `predict.py` uses `0.2`, even though the comment there says it must match training.
- **Weights path:** training saves to `/content/drive/MyDrive/kaggle_results/cbam_experiments/...`, but `predict.py` loads from `/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments/...`.
- `yolo11m-cbam.yaml` has the comment `nc: 2 # (parasite, artifact)`, but the dataset classes are Trophozoite / WBC. Ultralytics takes `nc` from `data.yaml`, so this only affects the comment.
- `PATHS_CONFIGURATION.md` describes an older layout (`code/train_predict/`, `yolo11m-se.yaml`) and is out of date.
