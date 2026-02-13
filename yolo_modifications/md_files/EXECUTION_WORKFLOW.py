#!/usr/bin/env python3
"""
EXECUTION WORKFLOW: CE FPN + CBAM Training Pipeline
====================================================

Complete end-to-end workflow for training YOLO11m with Contrast Enhanced FPN + CBAM
on SAHI-sliced malaria parasite detection dataset.

This document outlines the exact steps to reproduce the CE FPN enhancement training.
"""

WORKFLOW = """
═══════════════════════════════════════════════════════════════════════════════
                  EXECUTION WORKFLOW: CE FPN + CBAM TRAINING
═══════════════════════════════════════════════════════════════════════════════

## PHASE 0: PREPARATION (Already Complete)

✅ Dataset ready at: /tmp/malaria_data/yolov8_transformed/
✅ CBAM baseline trained: 85% mAP confirmed
✅ CE FPN module created: ce_fpn.py
✅ CE FPN architecture created: yolo11m-ce-fpn-cbam.yaml
✅ Training script created: train_yolo11m_ce_fpn_cbam.py
✅ Comparison tools created: compare_training_results.py

═══════════════════════════════════════════════════════════════════════════════

## PHASE 1: GENERATE SAHI PATCHES (if not already done)

Location: g:\My Drive\science_fair_2026\code\

Task 1.1: Verify dataset exists
   → Check: ls /tmp/malaria_data/yolov8_transformed/
   → Need: images/train/, images/val/, labels/train/, labels/val/

Task 1.2: Run SAHI slicing (if patches don't exist)
   → Check: ls /tmp/sliced_dataset/data.yaml
   → If missing, run: python create_sahi_dataset.py
   → Wait: ~30-60 minutes (4000×4000 slides → 1024×1024 patches @ 40% overlap)
   → Output: /tmp/sliced_dataset/ with ~3000-4000 total patches

═══════════════════════════════════════════════════════════════════════════════

## PHASE 2: TRAIN CE FPN + CBAM BASELINE

Location: g:\My Drive\science_fair_2026\code\yolo_modifications\

Task 2.1: Launch training
   Command: python train_yolo11m_ce_fpn_cbam.py
   
   What it does:
   ✓ Loads yolo11m-ce-fpn-cbam.yaml architecture
   ✓ Registers CE_FPN and CBAM modules with YOLO parser
   ✓ Trains on /tmp/sliced_dataset/ (SAHI patches)
   ✓ Saves to /content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_ce_fpn_cbam_1024px/
   ✓ Runs for 40 epochs with SGD optimizer
   ✓ Outputs: weights/best.pt, results.csv, training curves

Task 2.2: Monitor training
   Open results file (after ~30 min):
   → /content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_ce_fpn_cbam_1024px/results.csv
   → Watch column: metrics/mAP50(B)
   
   Expected progression:
   ┌─────────┬──────────────┬──────────────┐
   │ Epoch   │ CBAM-only    │ CE FPN+CBAM  │
   ├─────────┼──────────────┼──────────────┤
   │ 5       │ 15-20%       │ 18-22%       │ ← CE FPN helps slightly
   │ 10      │ 25-35%       │ 28-38%       │ ← Clear advantage
   │ 20      │ 55-65%       │ 58-68%       │ ← +3-5% difference
   │ 30      │ 75-80%       │ 78-82%       │ ← Maintaining lead
   │ 40      │ 85%          │ 88-90%       │ ← TARGET ACHIEVED
   └─────────┴──────────────┴──────────────┘

Task 2.3: Full training duration
   Expected: 45-60 minutes (on A100 GPU)
   Status indicators:
   ✓ Epoch 1: Training starting (GPU usage ~20GB)
   ✓ Epoch 10: Should see 25-38% mAP
   ✓ Epoch 20: Should see 55-68% mAP
   ✓ Epoch 30: Should see 75-82% mAP
   ✓ Epoch 40: Done! Final mAP recorded

═══════════════════════════════════════════════════════════════════════════════

## PHASE 3: ANALYZE RESULTS

Location: g:\My Drive\science_fair_2026\code\yolo_modifications\

Task 3.1: Generate comparison plot
   Command: python compare_training_results.py
   
   Output: /tmp/ce_fpn_vs_baseline_comparison.png
   Shows:
   ✓ mAP curves (CBAM-only vs CE FPN+CBAM)
   ✓ Training loss comparison
   ✓ Convergence speed (epochs to 80% mAP)
   ✓ mAP at key milestones (10, 20, 30, 40 epochs)

Task 3.2: Manual comparison
   Files to review:
   
   BASELINE (85% mAP):
   ✓ /content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px/
   ✓ results.csv: Check last row for final metrics
   ✓ runs/detect/predict_img/ : Predictions on test images
   
   CE FPN ENHANCED (target 88-90% mAP):
   ✓ /content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_ce_fpn_cbam_1024px/
   ✓ results.csv: Check final mAP50 value
   ✓ weights/best.pt: Best checkpoint (use for inference)

Task 3.3: Evaluation metrics to check
   From results.csv:
   ✓ metrics/mAP50(B): Confidence of correct detections
   ✓ metrics/mAP50-95(B): mAP across IOU thresholds
   ✓ val/box_loss: Bounding box accuracy
   ✓ val/cls_loss: Classification accuracy
   
   Success criteria:
   ✓ CE FPN should reach 87-90% mAP (3-5% improvement)
   ✓ Training loss should be smooth (no spikes)
   ✓ Final metrics > baseline metrics

═══════════════════════════════════════════════════════════════════════════════

## PHASE 4: NEXT STEPS (After CE FPN)

Based on CE FPN results:

SCENARIO A: CE FPN SUCCESSFUL (88-90% mAP achieved) ✅
   Next step: Test ensemble or TTA to reach 90%+ goal
   Actions:
   1. Run ensemble training:
      python train_yolo11m_ce_fpn_cbam.py (repeat 2-3 times with different seeds)
   2. Average predictions from multiple models → 90-92% mAP
   
   Or test TTA (test-time augmentation):
   1. Add TTA to predict.py
   2. Test on validation set with augmentation
   3. Should add 1-2% to mAP

SCENARIO B: CE FPN MINIMAL IMPROVEMENT (85-87% mAP) ⚠️
   Alternative approaches:
   1. Try YOLO11l (large) instead of m
   2. Extend training to 60-80 epochs
   3. Use ensemble of CBAM models (no CE FPN)
   4. Apply augmentation boost (increase mixup, copy_paste)

═══════════════════════════════════════════════════════════════════════════════

## DIAGNOSTIC SCRIPTS (Optional)

For troubleshooting or deeper analysis:

Script 1: No-SAHI baseline (test full-slide approach)
   Command: python train_yolo11m_cbam_no_sahi.py
   Purpose: Compare SAHI (85%) vs full-slide resizing
   Time: 45-60 minutes
   Use case: Verify SAHI is better than naive approach

Script 2: Module registration (if import errors)
   Command: python register_modules.py
   Purpose: Manually register CE_FPN and CBAM with YOLO
   Use case: Fix "CE_FPN not found" errors

═══════════════════════════════════════════════════════════════════════════════

## FILE LOCATIONS & STRUCTURE

Source Code:
  g:\My Drive\science_fair_2026\code\yolo_modifications\
  ├── train_yolo11m_ce_fpn_cbam.py    ← Main training script
  ├── ce_fpn.py                        ← CE FPN module (NEW)
  ├── yolo11m-ce-fpn-cbam.yaml        ← Architecture (NEW)
  ├── register_modules.py              ← Module setup
  ├── compare_training_results.py     ← Analysis tool
  └── [others: CBAM files, diagnostic scripts]

Data:
  /tmp/malaria_data/yolov8_transformed/  ← Original dataset
  /tmp/sliced_dataset/                   ← SAHI patches (generated)
  /content/drive/MyDrive/kaggle_results/cbam_experiments/
    ├── yolo11m_cbam_sahi_1024px/       ← Baseline (85% mAP)
    └── yolo11m_ce_fpn_cbam_1024px/     ← CE FPN (target 88-90%)

═══════════════════════════════════════════════════════════════════════════════

## TROUBLESHOOTING

Problem: "ModuleNotFoundError: No module named 'ce_fpn'"
Solution:
  1. Check file exists: ls g:\My Drive\science_fair_2026\code\yolo_modifications\ce_fpn.py
  2. Add to path: sys.path.insert(0, r'g:\My Drive\science_fair_2026\code\yolo_modifications')
  3. Or run: python register_modules.py

Problem: "YAML parser error: CE_FPN not recognized"
Solution:
  1. Ensure register_modules.py runs first
  2. Check yolo11m-ce-fpn-cbam.yaml has correct syntax
  3. Verify ce_fpn.py defines ContrastEnhancedFPN class

Problem: GPU out of memory
Solution:
  1. Reduce batch size: Change batch=-1 to batch=16 or batch=8
  2. Or use smaller model: yolo11s instead of m
  3. Check: nvidia-smi (should have <100MB free at peak)

Problem: Training too slow
Solution:
  1. Verify /tmp/sliced_dataset/ is on local SSD
  2. Check: ls -lh /tmp/sliced_dataset/ (should be <50GB)
  3. Ensure no other jobs using GPU: nvidia-smi
  4. If Google Drive: Use local /tmp copy (already done)

═══════════════════════════════════════════════════════════════════════════════

## QUICK REFERENCE: COMMAND EXECUTION ORDER

For a fresh start from scratch:

```bash
# 1. Generate SAHI patches (if needed)
cd g:\My Drive\science_fair_2026\code
python create_sahi_dataset.py
# Wait 30-60 minutes...

# 2. Train CE FPN + CBAM
cd yolo_modifications
python train_yolo11m_ce_fpn_cbam.py
# Wait 45-60 minutes...

# 3. Analyze results
python compare_training_results.py
# View: /tmp/ce_fpn_vs_baseline_comparison.png

# 4. Review metrics
cd /content/drive/MyDrive/kaggle_results/cbam_experiments/
ls yolo11m_ce_fpn_cbam_1024px/results.csv
```

Total time: ~2-3 hours (mostly training)

═══════════════════════════════════════════════════════════════════════════════

## SUCCESS CRITERIA

✅ CE FPN training successfully reaches:
   - Epoch 40: mAP ≥ 87% (improvement from 85%)
   - Ideally: mAP ≥ 88-90% (full target)

✅ Files generated:
   - weights/best.pt (best checkpoint)
   - results.csv (metrics log)
   - training curves (PNG plots)

✅ Comparison shows:
   - CE FPN + CBAM > CBAM alone
   - Clear mAP improvement visible in curves
   - Consistent advantage across epochs

═══════════════════════════════════════════════════════════════════════════════

## FINAL DELIVERABLES FOR SCIENCE FAIR

After successful CE FPN training:

1. **Best Model**: weights/best.pt from yolo11m_ce_fpn_cbam_1024px
2. **Performance**: 88-90% mAP on parasite detection
3. **Visualization**: Training curves + comparison plot
4. **Inference Pipeline**: predict.py with SAHI slicing
5. **Documentation**: This workflow + methodology

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(WORKFLOW)
    
    # Save to file
    with open('EXECUTION_WORKFLOW.txt', 'w') as f:
        f.write(WORKFLOW)
    
    print("\n✅ Workflow saved to: EXECUTION_WORKFLOW.txt")
