"""
Step 4: Grad-CAM explanations for the YOLO11m-CBAM model on full slides

Shows where the model finds evidence for its trophozoite detections:
1. Runs SAHI prediction (same function and settings as 3_predict/predict.py)
2. Cuts the slide into the same 1024x1024 tiles SAHI uses
3. Runs Grad-CAM on each tile for the chosen target layers
4. Stitches the tile heatmaps into one full-slide heatmap
5. Saves the heatmap with the detection boxes on top, plus a zoomed crop grid

Usage:
    python main_pipeline/4_explain/grad_cam.py
Edit the CONFIGURATION block at the bottom first.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add main_pipeline/model (CBAM + custom modules) to Python path
YOLO_MODS_PATH = (Path(__file__).parent.parent / 'model').resolve()  # main_pipeline/model/
sys.path.insert(0, str(YOLO_MODS_PATH))

# Register CBAM modules so the trained checkpoint can be loaded
import custom_modules
from ultralytics import YOLO
from ultralytics.nn import tasks

setattr(tasks, 'CBAM', custom_modules.CBAM)
for module_name, module_class in custom_modules.CBAM_MODULES.items():
    setattr(tasks, module_name, module_class)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sahi.slicing import get_slice_bboxes

# Reuse the SAHI prediction from step 3 (folder name "3_predict" can't be imported by name)
_PREDICT_PATH = Path(__file__).parent.parent / '3_predict' / 'predict.py'
_spec = importlib.util.spec_from_file_location('predict', _PREDICT_PATH)
predict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict)


# Layers that feed Detect in yolo11m-cbam.yaml (layers 22-23 are dead, see README)
DEFAULT_TARGET_LAYERS = {
    'cbam': 11,  # CBAM output: what the attention layer emphasizes
    'p3': 15,    # stride-8 input to Detect: finest map, best for small parasites
}


class YOLOClassScores(nn.Module):
    """Wraps the YOLO model so it returns only per-anchor class scores (B, nc, N)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        preds = out[0] if isinstance(out, (list, tuple)) else out  # (B, 4 + nc, N)
        return preds[:, 4:]  # drop box coordinates, keep class scores


class DetectedClassTarget:
    """
    Grad-CAM target: sum of one class's scores over anchors above the
    confidence threshold (falls back to the single best anchor if none).
    Records the tile's best score so empty tiles can be faded out.
    """

    def __init__(self, class_id, conf):
        self.class_id = class_id
        self.conf = conf
        self.max_score = 0.0

    def __call__(self, model_output):  # model_output: (nc, N) for one image
        scores = model_output[self.class_id]
        self.max_score = float(scores.max().detach())
        detected = scores[scores > self.conf]
        return detected.sum() if detected.numel() > 0 else scores.max()


def layers_feeding_detect(model):
    """Indices of layers whose output actually reaches the Detect head."""
    layers = model.model
    reachable, stack = set(), [len(layers) - 1]
    while stack:
        i = stack.pop()
        if i in reachable:
            continue
        reachable.add(i)
        sources = layers[i].f if isinstance(layers[i].f, list) else [layers[i].f]
        for f in sources:
            src = i + f if f < 0 else f  # -1 means "previous layer"
            if 0 <= src < len(layers):
                stack.append(src)
    return reachable


def _pad_to_multiple(tile, multiple=32, value=114):
    """Pad a tile (H, W, 3) on the bottom/right so H and W are multiples of 32."""
    h, w = tile.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        tile = cv2.copyMakeBorder(tile, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(value,) * 3)
    return tile


def gradcam_full_slide(model, image_rgb, layer_index, class_id=0, conf=0.25,
                       slice_size=1024, overlap_ratio=0.2, device='cpu'):
    """
    Grad-CAM over a full slide, tile by tile, using the same tiles as SAHI.

    Args:
        model: raw YOLO nn.Module (YOLO(...).model), in eval mode
        image_rgb: full slide, uint8 RGB (H, W, 3)
        layer_index: index into model.model of the target layer
        class_id: class to explain (0 = Trophozoite)
        conf: anchors above this score form the Grad-CAM target

    Returns:
        heatmap: float32 (H, W) in [0, 1]
    """
    if layer_index not in layers_feeding_detect(model):
        raise ValueError(
            f"Layer {layer_index} ({type(model.model[layer_index]).__name__}) does not feed the "
            f"Detect head, so it gets no gradient. Pick one of: {sorted(layers_feeding_detect(model))}"
        )

    h, w = image_rgb.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    wrapped = YOLOClassScores(model)
    cam = GradCAM(model=wrapped, target_layers=[model.model[layer_index]])

    tiles = get_slice_bboxes(
        image_height=h, image_width=w,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
    )
    for x0, y0, x1, y1 in tiles:
        tile = image_rgb[y0:y1, x0:x1]
        th, tw = tile.shape[:2]
        tile_float = np.float32(_pad_to_multiple(tile)) / 255.0

        input_tensor = torch.from_numpy(tile_float).permute(2, 0, 1).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        target = DetectedClassTarget(class_id, conf)
        with torch.enable_grad():
            tile_cam = cam(input_tensor=input_tensor, targets=[target])[0]

        # Grad-CAM rescales every tile to 0-1; weight by the tile's best score
        # so tiles without parasites stay dark
        tile_cam = tile_cam[:th, :tw] * target.max_score
        region = heatmap[y0:y1, x0:x1]
        np.maximum(region, tile_cam, out=region)

    return np.clip(heatmap, 0.0, 1.0)


def _draw_boxes(image_bgr, predictions, class_id):
    for pred in predictions:
        b = pred.bbox
        color = (0, 0, 255) if pred.category.id == class_id else (255, 255, 0)
        p1, p2 = (int(b.minx), int(b.miny)), (int(b.maxx), int(b.maxy))
        cv2.rectangle(image_bgr, p1, p2, color, 2)
        cv2.putText(image_bgr, f"{pred.category.name} {pred.score.value:.2f}",
                    (p1[0], max(p1[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_bgr


def _crop_grid(image_rgb, overlay_rgb, predictions, class_id, top_k=16, panel=160, cols=4):
    """Grid of the top-k detections: [image crop | Grad-CAM crop] per detection."""
    preds = sorted((p for p in predictions if p.category.id == class_id),
                   key=lambda p: p.score.value, reverse=True)[:top_k]
    if not preds:
        return None

    h, w = image_rgb.shape[:2]
    cells = []
    for pred in preds:
        b = pred.bbox
        cx, cy = (b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2
        half = max(b.maxx - b.minx, b.maxy - b.miny, 64)  # box plus context
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        pair = [cv2.resize(src[y0:y1, x0:x1], (panel, panel)) for src in (image_rgb, overlay_rgb)]
        cell = np.hstack(pair)
        cv2.putText(cell, f"{pred.score.value:.2f}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cells.append(cell)

    blank = np.zeros_like(cells[0])
    while len(cells) % cols:
        cells.append(blank)
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def explain_slide(model_path, image_path, output_dir, target_layers=None, class_id=0,
                  slice_size=1024, overlap_ratio=0.2, confidence_threshold=0.25,
                  device=None, top_k=16):
    """
    Run SAHI detection + Grad-CAM on one full slide and save the results.

    Saves, per target layer:
        {stem}_gradcam_{layer}.jpg        full-slide heatmap with detection boxes
        {stem}_gradcam_{layer}_crops.jpg  top-k detections, image crop | heatmap crop
    """
    target_layers = target_layers or DEFAULT_TARGET_LAYERS
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # 1. Detections, with exactly the settings step 3 uses
    result = predict.predict_full_slide_sahi(
        model_path=model_path,
        image_path=image_path,
        output_dir=str(output_dir),
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        confidence_threshold=confidence_threshold,
        visualize=False,
        device=device,
    )
    predictions = result.object_prediction_list

    # 2. Raw PyTorch model for Grad-CAM
    model = YOLO(str(model_path)).model.to(device).eval()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    saved = []
    for name, layer_index in target_layers.items():
        print(f"Grad-CAM on layer {layer_index} ({name})...")
        heatmap = gradcam_full_slide(
            model, image_rgb, layer_index, class_id=class_id, conf=confidence_threshold,
            slice_size=slice_size, overlap_ratio=overlap_ratio, device=device,
        )
        overlay_rgb = show_cam_on_image(np.float32(image_rgb) / 255.0, heatmap, use_rgb=True)

        overlay_path = output_dir / f"{stem}_gradcam_{name}.jpg"
        boxed = _draw_boxes(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), predictions, class_id)
        cv2.imwrite(str(overlay_path), boxed)
        saved.append(overlay_path)

        grid = _crop_grid(image_rgb, overlay_rgb, predictions, class_id, top_k=top_k)
        if grid is not None:
            crops_path = output_dir / f"{stem}_gradcam_{name}_crops.jpg"
            cv2.imwrite(str(crops_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            saved.append(crops_path)

    for p in saved:
        print(f"Saved: {p}")
    return saved


if __name__ == "__main__":
    # =====================================================================
    # CONFIGURATION (keep in sync with 3_predict/predict.py)
    # =====================================================================

    MODEL_PATH = '/content/drive/MyDrive/science_fair_2026/kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px/weights/best.pt'

    # Single image or batch folder (choose one, set other to None)
    SINGLE_IMAGE = '/content/drive/MyDrive/test_data/slide_001.jpg'  # TODO: Update to your test slide
    BATCH_FOLDER = None  # Example: '/content/drive/MyDrive/test_data/full_slides/'

    SLICE_SIZE = 1024
    OVERLAP_RATIO = 0.2
    CONFIDENCE_THRESHOLD = 0.25

    TARGET_CLASS = 0  # 0 = Trophozoite, 1 = WBC
    TARGET_LAYERS = DEFAULT_TARGET_LAYERS

    OUTPUT_DIR = '/content/drive/MyDrive/science_fair_2026/predictions/gradcam'

    # =====================================================================

    if BATCH_FOLDER:
        folder = Path(BATCH_FOLDER)
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
    else:
        images = [Path(SINGLE_IMAGE)]

    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img.name}")
        explain_slide(
            model_path=MODEL_PATH,
            image_path=img,
            output_dir=OUTPUT_DIR,
            target_layers=TARGET_LAYERS,
            class_id=TARGET_CLASS,
            slice_size=SLICE_SIZE,
            overlap_ratio=OVERLAP_RATIO,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

    print(f"\nGrad-CAM complete! Results saved to: {OUTPUT_DIR}/")
