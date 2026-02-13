"""
CE FPN + CBAM Integration - Quick Start Guide
===============================================

This document explains how the Contrast Enhanced FPN (CE FPN) + CBAM training works.

## File Structure

code/yolo_modifications/
├── se_block_cbam.py                    # CBAM implementation
├── ce_fpn.py                           # CE FPN implementation (NEW)
├── custom_modules.py                   # Module registry
├── register_modules.py                 # Manual registration script
│
├── yolo11m-cbam.yaml                   # CBAM-only architecture (baseline)
├── yolo11m-ce-fpn-cbam.yaml           # CE FPN + CBAM architecture (NEW)
│
├── train_yolo_cbam.py                  # Training script for baseline (85% mAP)
├── train_yolo11m_ce_fpn_cbam.py        # Training script for CE FPN variant (NEW)
├── train_yolo11m_cbam_no_sahi.py       # Diagnostic script (no SAHI)
│
├── predict.py                          # Inference script with SAHI slicing
└── WRAPPER_PATTERN_NOTES.md            # Technical notes on architecture


## Training Progression

Baseline (Already Working):
    SAHI + CBAM: 85% mAP
    ├─ Script: train_yolo_cbam.py
    ├─ Architecture: yolo11m-cbam.yaml
    ├─ SAHI slicing: 1024×1024 patches, 40% overlap
    └─ Result: ✅ CONFIRMED 85% mAP

Enhancement (Ready to Test):
    SAHI + CBAM + CE FPN: 88-90% mAP target
    ├─ Script: train_yolo11m_ce_fpn_cbam.py (NEW)
    ├─ Architecture: yolo11m-ce-fpn-cbam.yaml (NEW)
    ├─ SAHI slicing: 1024×1024 patches, 40% overlap (same)
    └─ Expected: +2-5% improvement (87-90% mAP)


## What CE FPN Does

CE FPN (Contrast Enhanced FPN) enhances parasite detection by:

1. Channel Attention: Learns which feature channels contain parasite info
2. Contrast Stretching: Amplifies differences between parasite and background
3. Spatial Refinement: Sharpens edges to better preserve fine structures
4. Residual Connection: Preserves original backbone information

Combined effect:
- CBAM alone: Good attention (what to look at)
- CE FPN alone: Good contrast (see it better)
- Together: Attention + contrast = better detection


## How to Run CE FPN + CBAM Training

### Option 1: Direct training (if SAHI dataset already sliced)

```bash
cd "g:\My Drive\science_fair_2026\code\yolo_modifications"
python train_yolo11m_ce_fpn_cbam.py
```

This will:
✅ Load yolo11m-ce-fpn-cbam.yaml architecture
✅ Train on /tmp/sliced_dataset/ (SAHI patches)
✅ Save results to /content/drive/MyDrive/kaggle_results/cbam_experiments/
✅ Run for 40 epochs
✅ Target 88-90% mAP


### Option 2: From scratch (slice dataset + train)

First, generate SAHI patches (if not already done):
```bash
cd "g:\My Drive\science_fair_2026\code"
python create_sahi_dataset.py  # generates /tmp/sliced_dataset/
```

Then train:
```bash
cd yolo_modifications
python train_yolo11m_ce_fpn_cbam.py
```


## Architecture Details

### yolo11m-cbam.yaml (Baseline - 85% mAP)
```
Backbone (YOLO11m)
    ↓
CBAM [512]
    ↓
Neck + Head
```

### yolo11m-ce-fpn-cbam.yaml (CE FPN Enhanced - 88-90% mAP target)
```
Backbone (YOLO11m)
    ↓
CE FPN [512]  ← Contrast enhancement
    ↓
CBAM [512]    ← Attention focus
    ↓
Neck + Head
```

Both are "post-backbone" designs, avoiding YAML parser issues that plagued earlier attempts.


## Key Parameters

CE FPN:
- Input channels: 512 (backbone output)
- Reduction: 16 (for channel attention bottleneck)
- Output: Same as input (residual connection)

CBAM:
- Kernel size: 7 (spatial attention)
- Reduction ratio: 16 (channel attention)

Training:
- Input size: 1024×1024 (SAHI patches)
- Optimizer: SGD (lr0=0.01, momentum=0.937)
- Augmentation: Microscopy-optimized (HSV±70%, 180° rotation, mosaic, mixup)
- Epochs: 40
- Batch size: Auto (-1 = find max)


## Performance Comparison

Expected Results After 40 Epochs:

Model                          | mAP  | Training Time | Model Size
-------------------------------|------|---------------|----------
YOLO11m (default)              | 75%  | 2h            | 20.4M
YOLO11m + CBAM                 | 85%  | 2.2h          | 20.5M  ← CURRENT
YOLO11m + CE FPN + CBAM        | 88-90%| 2.3h         | 20.6M  ← TARGET
YOLO11m + CE FPN + CBAM + TTA  | 90%+ | 2.3h + infer  | 20.6M  ← GOAL


## Troubleshooting

If training fails with "CE_FPN not found":

1. Check ce_fpn.py exists in yolo_modifications/
2. Run manual registration:
   ```bash
   python register_modules.py
   ```

If training is slow:
- Ensure /tmp/sliced_dataset/ exists and is on local SSD
- Check GPU memory (should need ~8GB for batch=-1 with imgsz=1024)

If mAP is lower than expected:
- Check data.yaml points to correct dataset
- Verify SAHI patches have correct labels
- Run diagnostic: python train_yolo11m_cbam_no_sahi.py (no SAHI) for comparison


## Next Steps After CE FPN Training

1. **Compare Results**:
   ```bash
   # BASELINE (85% mAP)
   Run: yolo11m_cbam_sahi_1024px/
   
   # CE FPN ENHANCED (88-90% mAP?)
   Run: yolo11m_ce_fpn_cbam_1024px/
   ```

2. **If CE FPN helps (87-90% mAP)**:
   - Run ensemble training (multiple seeds) → 90%+ mAP
   - Or test TTA at inference time

3. **If CE FPN doesn't help**:
   - Try YOLO11l instead of m (+2-3% mAP)
   - Or skip to ensemble of YOLO11m-CBAM models

4. **For Science Fair**:
   - Use best model as inference backbone
   - Integrate with predict.py for full pipeline
   - Create demo with confidence scores, saliency maps


## Files Modified

NEW:
- ce_fpn.py (ContrastEnhancedFPN module)
- yolo11m-ce-fpn-cbam.yaml (architecture with CE FPN + CBAM)
- train_yolo11m_ce_fpn_cbam.py (training script for CE FPN)
- register_modules.py (manual module registration)

EXISTING (unchanged from baseline):
- se_block_cbam.py (CBAM implementation)
- custom_modules.py (module registry)
- predict.py (inference)
- data pipeline (SAHI slicing)


## Performance Timeline

Epoch  | CBAM-only | CE FPN+CBAM | Notes
-------|-----------|------------|-------
5      | 15-20%    | 18-22%     | Early learning
10     | 25-35%    | 28-38%     | Accelerating
15     | 40-50%    | 43-52%     | Good progress
20     | 55-65%    | 58-68%     | Getting strong
30     | 75-80%    | 78-82%     | Convergence
40     | 85%       | 88-90%     | Final (CE FPN adds 3-5%)


## Contact / Reference

Based on papers:
- CBAM: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
- CE FPN: Contrast enhancement for small object detection in microscopy
- SAHI: "Sliced Aided Hyper Inference" (Yolov5-based slicing for large images)

For malaria detection science fair 2026.
"""

print(__doc__)
