# backend/optic_disc.py
import os
import gc
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import SegformerConfig, SegformerForSemanticSegmentation


HF_REPO_ID = "SushantGiri/glaucoma-ensemble-weights"
HF_SEG_FILE = "disc_cup_segformer_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- label mapping used by your debug screenshot ----
ID2LABEL = {0: "Background", 1: "Optic disc", 2: "Optic cup"}
LABEL2ID = {"Background": 0, "Optic disc": 1, "Optic cup": 2}
DISC_IDX = 1
CUP_IDX = 2


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only largest connected component in a boolean mask."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return labels == best


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a boolean mask."""
    m = mask.astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, flood_inv)
    return filled > 0


def _bbox_height(mask: np.ndarray) -> int:
    ys = np.where(mask)[0]
    if ys.size == 0:
        return 0
    return int(ys.max() - ys.min() + 1)


def _overlay_contours_fill(
    rgb: np.ndarray,
    disc: np.ndarray,
    cup: np.ndarray,
) -> np.ndarray:
    """
    Return overlay image with:
      - disc filled (green-ish)
      - cup filled (magenta-ish)
      - disc contour (cyan)
      - cup contour (magenta)
    """
    out = rgb.copy()

    # Fill colors (BGR for cv2)
    disc_fill = np.zeros_like(out)
    cup_fill = np.zeros_like(out)
    disc_fill[:] = (60, 200, 120)    # green-ish
    cup_fill[:] = (210, 80, 220)     # magenta-ish

    alpha_disc = 0.25
    alpha_cup = 0.28

    out = np.where(disc[..., None], (out * (1 - alpha_disc) + disc_fill * alpha_disc).astype(np.uint8), out)
    out = np.where(cup[..., None], (out * (1 - alpha_cup) + cup_fill * alpha_cup).astype(np.uint8), out)

    # Contours
    disc_u8 = (disc.astype(np.uint8) * 255)
    cup_u8 = (cup.astype(np.uint8) * 255)

    contours_d, _ = cv2.findContours(disc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_c, _ = cv2.findContours(cup_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(out, contours_d, -1, (80, 230, 230), 3)  # cyan-ish
    cv2.drawContours(out, contours_c, -1, (240, 120, 240), 3) # magenta-ish

    return out


def _overlay_heatmap(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """
    heat: float32 [0..1] same H,W. Apply a colormap on top of rgb.
    """
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    alpha = 0.35
    out = (rgb * (1 - alpha) + cmap * alpha).astype(np.uint8)
    return out


@lru_cache(maxsize=2)
def _download_seg_weight() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (needed to download segmentation weights).")
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_SEG_FILE,
        token=token,
        cache_dir=cache_dir,
    )


class OpticDiscSegmenter:
    def __init__(self, image_size: int = 512):
        self.image_size = int(image_size)

        SEGFORMER_BACKBONE = "nvidia/segformer-b1-finetuned-ade-512-512"  # backbone OK

        cfg = SegformerConfig.from_pretrained(SEGFORMER_BACKBONE)

        # ✅ match YOUR training/export settings
        cfg.num_labels = 3
        cfg.id2label = ID2LABEL
        cfg.label2id = LABEL2ID

        # ✅ THIS is the missing piece (your checkpoint uses 768-d decoder head)
        cfg.decoder_hidden_size = 768

        self.model = SegformerForSemanticSegmentation(cfg)

        wpath = _download_seg_weight()
        state = torch.load(wpath, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # Now it should load cleanly
        self.model.load_state_dict(state, strict=True)

        self.model.to(DEVICE)
        self.model.eval()

    """
    SegFormer segmentation model:
      class 0: background
      class 1: disc
      class 2: cup
    """

    def __init__(self, image_size: int = 512):
        self.image_size = int(image_size)

        # IMPORTANT:
        # We build a SegFormer config with 3 labels.
        # Your .pth must match the same architecture you trained/exported.
        SEGFORMER_BACKBONE = "nvidia/segformer-b1-finetuned-ade-512-512"  # ✅ B1 matches your checkpoint

        cfg = SegformerConfig.from_pretrained(SEGFORMER_BACKBONE)
        cfg.num_labels = 3
        cfg.id2label = ID2LABEL
        cfg.label2id = LABEL2ID

        self.model = SegformerForSemanticSegmentation(cfg)

        wpath = _download_seg_weight()
        state = torch.load(wpath, map_location="cpu")

        # If checkpoint is saved as {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Remove potential "module." prefix
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        self.model.load_state_dict(state, strict=False)

        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def segment(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Returns:
          - disc_mask, cup_mask (bool H,W) in ORIGINAL image size
          - overlay_contour_fill (PIL)
          - overlay_heatmap (PIL)
          - metrics dict
          - debug dict
        """
        rgb0 = np.array(pil_img.convert("RGB"))
        H0, W0 = rgb0.shape[:2]

        # Resize to model size
        rgb = cv2.resize(rgb0, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # 1,3,H,W

        out = self.model(pixel_values=x)
        logits = out.logits  # 1,C,H,W

        probs = torch.softmax(logits, dim=1)[0]  # C,H,W
        pred = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)  # H,W

        disc = (pred == DISC_IDX)
        cup = (pred == CUP_IDX)

        # Postprocess: keep largest components, fill holes
        disc = _largest_component(disc)
        disc = _fill_holes(disc)

        cup = _largest_component(cup)
        # remove tiny speckles (your line)
        cup = cv2.morphologyEx(cup.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        cup = _fill_holes(cup)

        # Ensure cup is within disc
        cup = np.logical_and(cup, disc)

        # Heatmap = confidence of disc/cup vs background (max prob of disc/cup)
        prob_disc = probs[DISC_IDX].detach().cpu().numpy()
        prob_cup = probs[CUP_IDX].detach().cpu().numpy()
        heat = np.maximum(prob_disc, prob_cup).astype(np.float32)  # 0..1

        # Overlays in model-res space
        overlay1 = _overlay_contours_fill(rgb, disc, cup)
        overlay2 = _overlay_heatmap(rgb, heat)

        # Resize masks + overlays back to original size
        disc0 = cv2.resize(disc.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        cup0 = cv2.resize(cup.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

        overlay1_0 = cv2.resize(overlay1, (W0, H0), interpolation=cv2.INTER_LINEAR)
        overlay2_0 = cv2.resize(overlay2, (W0, H0), interpolation=cv2.INTER_LINEAR)

        disc_area = int(np.sum(disc0))
        cup_area = int(np.sum(cup0))
        rim_area = int(max(disc_area - cup_area, 0))

        disc_h = _bbox_height(disc0)
        cup_h = _bbox_height(cup0)
        v_cd = float(cup_h / disc_h) if disc_h > 0 else 0.0
        cd_area = float(cup_area / disc_area) if disc_area > 0 else 0.0

        metrics = {
            "disc_area_px2": disc_area,
            "cup_area_px2": cup_area,
            "rim_area_px2": rim_area,
            "cd_area_ratio": cd_area,
            "vertical_cd_ratio": v_cd,

            # Fundus-only limitation (OCT/3D needed)
            "cup_volume_mm3": None,
            "min_cup_depth_mm": None,
            "max_cup_depth_mm": None,
            "ddls": None,
        }

        debug = {
            "disc_idx": DISC_IDX,
            "cup_idx": CUP_IDX,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
        }

        # cleanup
        del x, out, logits, probs
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "disc_mask": disc0,
            "cup_mask": cup0,
            "overlay_contours_fill": Image.fromarray(overlay1_0),
            "overlay_heatmap": Image.fromarray(overlay2_0),
            "metrics": metrics,
            "debug": debug,
        }
