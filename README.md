# Malaria Parasite Detection

Detects malaria parasites (trophozoites) and WBCs in full blood-smear slides.

**The final method is YOLO11m + CBAM attention, trained and run on SAHI slices.**
Everything the final method uses is in `main_pipeline/`. Everything else is in
`experiments/`, grouped by the files that work together.

```
malaria_detection/
├── main_pipeline/            the final method, folders in run order
│   ├── 1_data_prep/
│   ├── 2_train/
│   ├── 3_predict/
│   └── model/                model files shared by training, prediction and side B
└── experiments/              not used by the final method
    ├── A_baseline_no_cbam/   YOLO11m without CBAM, plus its validation and Grad-CAM
    ├── B_cbam_ablations/     CBAM variants that reuse main_pipeline/model
    ├── C_r_kaggle11_model/   everything built on the older r_kaggle11 model
    └── D_unused/             dead code, kept for reference
```

All scripts run on Google Colab. Their data and weights paths point at Google
Drive (`/content/drive/MyDrive/...`), not at folders in this repo.

---

## Main pipeline

```
Kaggle Lacuna CSVs + images
  │
  │ 1  main_pipeline/1_data_prep/kaggle_annotation_conversion.py
  │      CSV boxes -> YOLO labels (0 = Trophozoite, 1 = WBC), train/val split
  ▼
kaggle_datasets/yolov8_transformed/                  full slides
  │
  │ 2  main_pipeline/2_train/train_yolo_cbam.py
  │      a. slices slides into 1024x1024 patches (overlap 0.4) -> /tmp/sliced_dataset
  │      b. trains YOLO11m-CBAM on the patches (40 epochs, imgsz 1024)
  │           model/yolo11m-cbam.yaml      architecture: YOLO11m + 1 CBAM layer after the backbone
  │             └ needs CBAM registered:   model/custom_modules.py -> model/cbam.py
  ▼
kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px/weights/best.pt
  │
  │ 3  main_pipeline/3_predict/predict.py
  │      imports model/custom_modules.py -> model/cbam.py to load best.pt,
  │      then runs SAHI sliced prediction on full slides
  ▼
Merged detections on full slides
```

| File | Role |
|---|---|
| `1_data_prep/kaggle_annotation_conversion.py` | Converts the Kaggle annotations to YOLO format and writes `data.yaml` |
| `2_train/train_yolo_cbam.py` | **Main training script**: slicing + CBAM training |
| `3_predict/predict.py` | **Main inference script**: SAHI sliced prediction on full slides |
| `model/yolo11m-cbam.yaml` | Model architecture: standard YOLO11m with one `CBAM` layer after C2PSA, before the neck |
| `model/cbam.py` | The `CBAM` layer: channel + spatial attention |
| `model/custom_modules.py` | Imports `CBAM` so scripts can register it with Ultralytics |
| `model/migrate_checkpoints.py` | One-off: rewrites `best.pt` files saved before the split so they reference `cbam.CBAM` (old name: `se_block_cbam.CBAM`) |

Only the `CBAM` class does any work in the model. The other wrappers in
`custom_modules.py` (`C2f_CBAM`, `Conv_CBAM`, `Bottleneck_CBAM`, `SPPF_CBAM`,
`C2f_CBAM_Deep`) are earlier attention-placement ideas that the yaml does not use.

---

## Experiments (not the final method)

Each group is linked by the data folder or model weights its files share.

### A. Baseline YOLO11m, no CBAM: `experiments/A_baseline_no_cbam/`
**Linked by:** the sliced dataset `/tmp/sliced_dataset` and the baseline weights `kaggle_results/yolo11m_baseline_sahi/.../best.pt`.

```
yolov8_transformed (full slides)
  │ train_sahi_slice_and_train.py     slices -> /tmp/sliced_dataset (also trains, no augmentation)
  ▼
/tmp/sliced_dataset
  │ train_yolo11m_sahi_only.py         vanilla YOLO11m, no CBAM (the ablation for main step 2)
  ▼
yolo11m_baseline_sahi/.../best.pt
  ├─> slice_and_validate_tta.py        validation on sliced yolov8_transformed_b (augment=True is commented out)
  └─> grad_cam.py                      Grad-CAM heatmaps on sample images
```

### B. CBAM ablations: `experiments/B_cbam_ablations/`
**Linked by:** both reuse the main model files in `main_pipeline/model/` (yaml, `custom_modules.py`, `cbam.py`).

| File | What it tested |
|---|---|
| `train_yolo11m_cbam_no_sahi.py` | Same CBAM model trained on full slides resized to 1280, **no SAHI slicing** |
| `train_yolo11m_fpn_cbam.py` | Earlier all-in-one CBAM + SAHI script from the CE-FPN attempt; replaced by `2_train/train_yolo_cbam.py` |

### C. Older `r_kaggle11` model: `experiments/C_r_kaggle11_model/`
**Linked by:** the older weights `r_kaggle11/malaria_results_highres/.../best.pt`, which were trained outside this repo, and the filtered dataset `yolov8_transformed_b`.

```
dataset_cleanup/image_seperations.py     counts images per resolution -> picks 4000x3000 to drop
dataset_cleanup/dataset_imgcleanup.py    yolov8_transformed -> yolov8_transformed_b (4000x3000 removed)
                                            │
r_kaggle11 best.pt ─────────────────────────┤
  ├─> dataset_cleanup/label_cleanup.py      adds high-confidence trophozoite pseudo-labels to yolov8_transformed_b
  ├─> two_stage_classifier/classifierdataset.py   ┐ crop detections into
  ├─> two_stage_classifier/confthres_optimizer.py ┘ crops_for_mobilenet/{parasites, artifacts_bg}
  │       (also tunes the confidence threshold toward a 50/50 balance)
  │                                    │
  │                                    ├─> two_stage_classifier/mbnv3.py      MobileNetV3 parasite vs artifact
  │                                    └─> two_stage_classifier/customcnn.py  small custom CNN for 30-115 px crops
  └─> heatmap.py                            Ultralytics detection heatmap
```

The main pipeline trains on `yolov8_transformed`, **not** `yolov8_transformed_b`.

### D. Unused: `experiments/D_unused/`
**Linked by:** nothing. None of these feed any other script.

| File | Why it's unused |
|---|---|
| `squeeze_excitation.py` | `SEBlock` (channel-only attention); never placed in the model |
| `se_bottleneck.py` | ResNet-style bottleneck built on `SEBlock` (imports `squeeze_excitation.py`) |
| `register_modules.py` | Old CBAM registration helper; each script now registers CBAM itself |
| `label_cleanup_test.py` | Draft of `label_cleanup.py` with placeholder model paths |
| `lacuna_old_augmentation.py` | Old hand-made microscope/smartphone augmentation |

---

## Folder change (before -> after)

```
BEFORE                                              AFTER
augmentations/kaggle_annotation_conversion.py   ->  main_pipeline/1_data_prep/
yolo_modifications/train/train_yolo_cbam.py     ->  main_pipeline/2_train/
yolo_modifications/predict/predict.py           ->  main_pipeline/3_predict/
yolo_modifications/train/yolo11m-cbam.yaml      ->  main_pipeline/model/
yolo_modifications/cbam.py                      ->  main_pipeline/model/
yolo_modifications/custom_modules.py            ->  main_pipeline/model/
yolo_modifications/migrate_checkpoints.py       ->  main_pipeline/model/
yolo_modifications/train/train_sahi_slice_and_train.py  ->  experiments/A_baseline_no_cbam/
yolo_modifications/train/train_yolo11m_sahi_only.py     ->  experiments/A_baseline_no_cbam/
yolo_modifications/predict/slice_and_validate_tta.py    ->  experiments/A_baseline_no_cbam/
visualizations/grad_cam.py                      ->  experiments/A_baseline_no_cbam/
yolo_modifications/train/train_yolo11m_cbam_no_sahi.py  ->  experiments/B_cbam_ablations/
yolo_modifications/train/train_yolo11m_fpn_cbam.py      ->  experiments/B_cbam_ablations/
dataset/{image_seperations,dataset_imgcleanup,label_cleanup}.py  ->  experiments/C_r_kaggle11_model/dataset_cleanup/
classifier/*.py                                 ->  experiments/C_r_kaggle11_model/two_stage_classifier/
visualizations/heatmap.py                       ->  experiments/C_r_kaggle11_model/
yolo_modifications/{squeeze_excitation,se_bottleneck,register_modules}.py  ->  experiments/D_unused/
dataset/label_cleanup_test.py                   ->  experiments/D_unused/
augmentations/lacuna_old_augmentation.py        ->  experiments/D_unused/
PATHS_CONFIGURATION.md                          ->  removed (described an older layout; this README replaces it)
```

**Why it's laid out this way**
- **The top level splits main from not-main**, so you can tell what the final method uses without opening a file.
- **Numbered folders** (`1_data_prep`, `2_train`, `3_predict`) give the main pipeline's run order.
- **`main_pipeline/model/` is shared.** Training, prediction and the side B ablations all load the same model files from this one place.
- **Experiments are grouped by what links them, not by file type.** For example, `grad_cam.py` sits with the baseline it visualizes, and `heatmap.py` sits with the other `r_kaggle11` scripts.

---

## Known inconsistencies in the main pipeline

- **Slice overlap:** training uses `OVERLAP_RATIO = 0.4`, but `3_predict/predict.py` uses `0.2`, even though the comment there says it must match training.
- **Weights path:** training saves to `/content/drive/MyDrive/kaggle_results/cbam_experiments/...`, but `predict.py` loads from `/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments/...`.
- `model/yolo11m-cbam.yaml` has the comment `nc: 2 # (parasite, artifact)`, but the dataset classes are Trophozoite / WBC. Ultralytics takes `nc` from `data.yaml`, so this only affects the comment.
