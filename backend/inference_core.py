import os
import json
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

import gc
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.densenet import densenet121

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from huggingface_hub import hf_hub_download

# NEW: segmentation model
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor


# --------------------------------------------------------------------
# Paths, device & HF config
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../glaucoma_app
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = "SushantGiri/glaucoma-ensemble-weights"

# ✅ DenseNet121 + Disc/Cup SegFormer
HF_MODEL_FILES: Dict[str, str] = {
    "densenet121": "densenet121_best.pth",
    "disc_cup_segformer": "disc_cup_segformer_best.pth",
}

# Base pretrained SegFormer ID (for architecture + config)
# We load config/architecture from here, then load your .pth weights.
SEGFORMER_BASE_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def apply_clahe_rgb(img_np: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def validate_fundus_input(pil_img: Image.Image) -> Tuple[bool, str]:
    """
    Lightweight guardrail to reject obvious non-fundus inputs.
    Returns: (ok, reason)
    """
    img = np.array(pil_img.convert("RGB"))

    h, w = img.shape[:2]
    if h < 256 or w < 256:
        return False, "Invalid image: too small. Please upload a retinal fundus photograph."

    img_small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 60, 120)
    edge_density = float(np.mean(edges > 0))

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7,
    )
    thr = cv2.medianBlur(thr, 3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)

    small_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 12 <= area <= 900:
            small_components += 1

    center = gray[156:356, 156:356]
    border = np.concatenate(
        [gray[:60, :].ravel(), gray[-60:, :].ravel(), gray[:, :60].ravel(), gray[:, -60:].ravel()]
    )
    center_mean = float(np.mean(center))
    border_mean = float(np.mean(border))
    center_border_diff = center_mean - border_mean

    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    if mean_intensity > 205 and std_intensity < 40:
        return False, (
            "Invalid image: looks like a document/screenshot (very bright, low contrast). "
            "Please upload a retinal fundus photograph."
        )

    if edge_density > 0.14:
        return False, (
            "Invalid image: looks like text/screenshot (too many sharp edges). "
            "Please upload a retinal fundus photograph."
        )

    if small_components > 420:
        return False, (
            "Invalid image: looks like a screenshot/document (detected text-like patterns). "
            "Please upload a retinal fundus photograph."
        )

    if center_border_diff < 6:
        return False, (
            "Invalid image: does not resemble a retinal fundus photograph (missing fundus field-of-view cue). "
            "Please upload a macula-centered fundus image."
        )

    return True, "ok"


def load_config():
    class_mapping_path = os.path.join(MODELS_DIR, "class_mapping.json")
    prep_path = os.path.join(MODELS_DIR, "preprocessing.json")

    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)
    else:
        class_mapping = {"0": "Normal", "1": "Glaucoma"}

    if os.path.exists(prep_path):
        with open(prep_path, "r") as f:
            prep_cfg = json.load(f)
    else:
        prep_cfg = {}

    return class_mapping, prep_cfg


@lru_cache(maxsize=8)
def _download_weight_path(model_key: str) -> str:
    filename = HF_MODEL_FILES[model_key]

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN not set. Please add HF_TOKEN to your Streamlit secrets or env.")

    cache_dir = os.path.expanduser("~/.cache/huggingface")

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=hf_token,
        cache_dir=cache_dir,
    )


@lru_cache(maxsize=8)
def load_state_from_hf(model_key: str) -> Dict[str, torch.Tensor]:
    weight_path = _download_weight_path(model_key)
    state_dict = torch.load(weight_path, map_location="cpu")

    # handle common checkpoint formats
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict


# --------------------------------------------------------------------
# DenseNet121-only predictor
# --------------------------------------------------------------------
class DenseNet121Predictor:
    def __init__(self):
        self.class_mapping, self.prep_cfg = load_config()

        den = densenet121(weights=None)
        den.classifier = nn.Linear(den.classifier.in_features, 2)
        den.load_state_dict(load_state_from_hf("densenet121"))

        self.model = den.to(DEVICE)
        self.model.eval()

        self.target_layer = self.model.features[-1]
        self.input_size = int(self.prep_cfg.get("input_size", 512)) or 512

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def _preprocess(self, pil_img: Image.Image, input_size: int) -> torch.Tensor:
        img = pil_img.convert("RGB")
        img_np = np.array(img)
        img_np = apply_clahe_rgb(img_np)
        img_np = cv2.resize(img_np, (input_size, input_size))
        img_np = img_np.astype(np.float32) / 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        tensor = transform(img_np)
        return tensor.unsqueeze(0)

    def _preprocess_for_cam(self, pil_img: Image.Image, side: int) -> torch.Tensor:
        img = pil_img.convert("RGB")
        img_np = np.array(img)
        img_np = apply_clahe_rgb(img_np)
        img_np = cv2.resize(img_np, (side, side))
        img_np = img_np.astype(np.float32) / 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        tensor = transform(img_np)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, Any]:
        ok, reason = validate_fundus_input(pil_img)
        if not ok:
            raise ValueError(reason)

        x = self._preprocess(pil_img, self.input_size).to(DEVICE)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))

        result = {
            "model": "densenet121",
            "probs": probs.tolist(),
            "pred_class_index": pred_idx,
            "pred_label": self.class_mapping[str(pred_idx)],
        }

        del x, logits
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return result

    def gradcam(self, pil_img: Image.Image, output_dir: str, target_class_index: int) -> str:
        """
        Compute Grad-CAM quickly at cam_side, but SAVE overlay at ORIGINAL resolution (W,H).
        """
        ok, reason = validate_fundus_input(pil_img)
        if not ok:
            raise ValueError(reason)

        os.makedirs(output_dir, exist_ok=True)
        cam_side = 256  # fast CAM computation

        base_rgb = np.array(pil_img.convert("RGB"))
        base_rgb = apply_clahe_rgb(base_rgb)
        H, W = base_rgb.shape[:2]
        base_float = base_rgb.astype(np.float32) / 255.0

        input_tensor = self._preprocess_for_cam(pil_img, side=cam_side).to(DEVICE)

        cam = None
        try:
            cam = GradCAM(model=self.model, target_layers=[self.target_layer])
            targets = [ClassifierOutputTarget(int(target_class_index))]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (cam_side, cam_side)
            cam_up = cv2.resize(grayscale_cam, (W, H), interpolation=cv2.INTER_LINEAR)
            cam_img = show_cam_on_image(base_float, cam_up, use_rgb=True)

            save_path = os.path.join(output_dir, f"densenet121_gradcam_{os.getpid()}.png")
            Image.fromarray(cam_img).save(save_path)
            return save_path

        finally:
            try:
                if cam is not None and hasattr(cam, "activations_and_grads"):
                    cam.activations_and_grads.release()
            except Exception:
                pass

            del cam, input_tensor
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


# --------------------------------------------------------------------
# Optic disc/cup metrics + overlay helpers
# --------------------------------------------------------------------
def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    return (labels == i)


def compute_optic_disc_metrics(disc_mask: np.ndarray, cup_mask: np.ndarray) -> Dict[str, float]:
    disc_area = float(np.sum(disc_mask))
    cup_area = float(np.sum(cup_mask))
    rim_area = float(max(disc_area - cup_area, 0.0))

    cd_area_ratio = float(cup_area / disc_area) if disc_area > 0 else 0.0

    def _vertical_diameter(m: np.ndarray) -> float:
        ys = np.where(m)[0]
        if len(ys) == 0:
            return 0.0
        return float(ys.max() - ys.min() + 1)

    disc_v = _vertical_diameter(disc_mask)
    cup_v = _vertical_diameter(cup_mask)
    v_cd = float(cup_v / disc_v) if disc_v > 0 else 0.0

    return {
        "disc_area_px2": disc_area,
        "cup_area_px2": cup_area,
        "rim_area_px2": rim_area,
        "cd_area_ratio": cd_area_ratio,
        "vertical_cd_ratio": v_cd,
    }


def render_disc_cup_overlay(
    pil_img: Image.Image,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    alpha: float = 0.23,
) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    H, W = img.shape[:2]

    overlay = img.copy()

    disc_fill = np.zeros((H, W, 3), dtype=np.uint8)
    cup_fill = np.zeros((H, W, 3), dtype=np.uint8)
    disc_fill[:, :, 1] = 255  # green
    cup_fill[:, :, 0] = 255   # red

    overlay = np.where(disc_mask[..., None], (overlay * (1 - alpha) + disc_fill * alpha).astype(np.uint8), overlay)
    overlay = np.where(cup_mask[..., None], (overlay * (1 - alpha) + cup_fill * alpha).astype(np.uint8), overlay)

    disc_u8 = (disc_mask.astype(np.uint8) * 255)
    cup_u8 = (cup_mask.astype(np.uint8) * 255)
    disc_cnts, _ = cv2.findContours(disc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_cnts, _ = cv2.findContours(cup_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, disc_cnts, -1, (0, 255, 255), 2)
    cv2.drawContours(overlay, cup_cnts, -1, (255, 0, 255), 2)

    return Image.fromarray(overlay)


# --------------------------------------------------------------------
# NEW: Disc + Cup Segmentation (SegFormer)
# Output classes expected: 0=background, 1=disc, 2=cup
# --------------------------------------------------------------------
class DiscCupSegmenter:
    """
    SegFormer disc/cup segmentation.
    Uses the official HF image processor (correct resize/normalize).
    id2label mapping is read from config.
    """

    def __init__(self):
        # Load architecture + config
        self.model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_BASE_ID)

        # Load your uploaded weights
        sd = load_state_from_hf("disc_cup_segformer")
        _missing, _unexpected = self.model.load_state_dict(sd, strict=False)

        self.model = self.model.to(DEVICE)
        self.model.eval()

        # ✅ Correct preprocessing for this SegFormer checkpoint
        self.processor = AutoImageProcessor.from_pretrained(SEGFORMER_BASE_ID)

        # ----- label indices (from config) -----
        id2label = getattr(self.model.config, "id2label", None) or {}
        id2label_l = {int(k): str(v).lower() for k, v in id2label.items()} if isinstance(id2label, dict) else {}

        def _find_idx(keywords, default_idx):
            for idx, name in id2label_l.items():
                if any(k in name for k in keywords):
                    return int(idx)
            return int(default_idx)

        self.disc_idx = _find_idx(["disc", "optic disc", "optic_disc", "od"], default_idx=1)
        self.cup_idx  = _find_idx(["cup", "optic cup", "optic_cup", "oc"], default_idx=2)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def predict_masks_and_metrics(self, pil_img: Image.Image) -> Dict[str, Any]:
        ok, reason = validate_fundus_input(pil_img)
        if not ok:
            raise ValueError(reason)

        # ✅ Use official processor (handles resize + normalize)
        inputs = self.processor(images=pil_img.convert("RGB"), return_tensors="pt")
        x = inputs["pixel_values"].to(DEVICE)  # 1,3,h,w

        # Original size
        W, H = pil_img.size

        out = self.model(pixel_values=x)
        logits = out.logits  # 1,C,h,w

        # Upsample to original resolution
        logits_up = torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

        probs = torch.softmax(logits_up, dim=1)[0]  # C,H,W

        disc_prob = probs[self.disc_idx].cpu().numpy()
        cup_prob  = probs[self.cup_idx].cpu().numpy()

        # Disc: confident region
        disc = disc_prob > 0.5

        # Cup: lower threshold (VERY IMPORTANT)
        cup = cup_prob > 0.15

        # 🔹 REMOVE TINY SPECKLES (ADD THIS)
        cup = cv2.morphologyEx(
            cup.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        ).astype(bool)

        # Cleanup
        disc = _largest_component(disc)
        cup  = _largest_component(cup)

        # Ensure cup is inside disc
        if np.any(disc):
            cup = np.logical_and(cup, disc)



        # Keep largest component for disc; cup can be small—still keep largest (safe)
        disc = _largest_component(disc)
        cup  = _largest_component(cup)

        # Enforce cup inside disc only if disc exists
        if np.sum(disc) > 0:
            cup = np.logical_and(cup, disc)

        metrics = compute_optic_disc_metrics(disc, cup)
        overlay_pil = render_disc_cup_overlay(pil_img, disc, cup, alpha=0.23)

        # cleanup
        del x, inputs, out, logits, logits_up
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "disc_mask": disc,
            "cup_mask": cup,
            "metrics": metrics,
            "overlay_pil": overlay_pil,
            "debug": {
                "disc_idx": int(self.disc_idx),
                "cup_idx": int(self.cup_idx),
                "id2label": getattr(self.model.config, "id2label", None),
            }
        }
