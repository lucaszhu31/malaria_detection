# Automated Malaria Detection with a SAHI-Enhanced YOLO-CBAM Model

Lucas Zhu, Science Fair 2026

## Motivation

Malaria is still one of the world's largest health burdens. In 2024 there were about
282 million cases, and roughly 95% of them were in resource-constrained countries.
Diagnosis usually means a trained microscopist examining blood smears by hand. That
expertise is scarce exactly where malaria is most common, so many patients never get
a proper diagnosis.

The project grew out of my work with Global Brigades, a non-profit that supports
sustainable healthcare in underserved regions. The goal is a lightweight, low-cost,
automated detector that can find malaria parasites in blood smear images in places
without specialist staff or expensive hardware.

## Report summary

### Approach
The detector is **YOLO11**, a single-stage object detection network chosen for its
balance of speed and accuracy, extended with two additions aimed at tiny objects:

- **Convolutional Block Attention Module (CBAM).** Placed after the backbone's C2PSA
  block. C2PSA captures spatial context but cannot weight feature channels. CBAM adds
  that in two steps:
  - **Channel attention:** boosts the feature maps that carry parasite-like textures.
  - **Spatial attention:** highlights where in the image those features appear.
- **Slicing Aided Hyper Inference (SAHI).** Splits each high-resolution smear into
  overlapping patches, with 20% overlap. Parasites then fill a larger share of each
  input, so less detail is lost when YOLO downsamples.

### Data
- **Source:** the public **Lacuna Malaria Dataset** (Makerere AI Lab): about 3,000
  thick blood smear images from patients in Uganda and Ghana.
- **How the images were taken:** with a smartphone held to the microscope, then
  annotated by medical experts for parasites (trophozoites) and white blood cells.
- **Why it suits this project:** the variation in staining and image noise makes it a
  realistic training set.
- **Preparation:** the images were sliced with SAHI, then augmented with rotations,
  flips and HSV color shifts to mimic staining differences and reduce overfitting.

### Results
Trained for 50 epochs on free Google Colab GPUs:

| Metric | SAHI + YOLO11 + CBAM |
|---|---|
| mAP | 91.6% |
| Precision | 88.1% |
| Recall | 85.0% |
| F1-score | 86.5% |
| Inference time | 72.5 ms per image |

- **Training curves:** box and classification losses fell quickly, and precision and
  recall levelled off around epoch 35.
- **Effect of SAHI:** compared with a baseline YOLO11, adding SAHI raised mAP and
  precision by about 5%. Keeping pixel density high matters for micro-scale targets.
- **Effect of CBAM:** adding CBAM improved every metric by about 2% more. This
  suggests the attention module helps separate parasites from background noise and
  staining artifacts.

### Limitations
- **Main source of error:** staining artifacts and image noise still cause false
  positives and false negatives.
- **No species distinction:** the model detects parasites but cannot tell *Plasmodium*
  species apart.

### Conclusion
A SAHI-enhanced YOLO11-CBAM model reaches high-precision malaria parasite detection
while staying fast and cheap to run. This points toward affordable field devices that
could make reliable malaria screening available where specialist diagnosis is not.

### Key references
- Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing aided hyper inference and fine-tuning for small object detection. *IEEE ICIP*, 966–970.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
- Jocher, G., Qiu, J., & Wang, A. (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
- Lacuna Malaria Detection Dataset. https://www.kaggle.com/datasets/rajsahu2004/lacuna-malariadetection-dataset
- World Health Organization. *World Malaria Report*.
