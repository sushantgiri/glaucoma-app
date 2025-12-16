import os
import json
from typing import Dict, Any
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


# --------------------------------------------------------------------
# Paths, device & HF config
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../glaucoma_app
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = "SushantGiri/glaucoma-ensemble-weights"

# ✅ DenseNet121 only
HF_MODEL_FILES: Dict[str, str] = {
    "densenet121": "densenet121_best.pth",
}


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


@lru_cache(maxsize=4)
def _download_weight_path(model_key: str) -> str:
    """
    Download once per process and reuse path.
    """
    filename = HF_MODEL_FILES[model_key]

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN not set. Please add HF_TOKEN to your Streamlit secrets or env.")

    # Stable HF cache (faster after first run)
    cache_dir = os.path.expanduser("~/.cache/huggingface")

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=hf_token,
        cache_dir=cache_dir,
    )


@lru_cache(maxsize=4)
def load_state_from_hf(model_key: str) -> Dict[str, torch.Tensor]:
    """
    Load state_dict from cached downloaded file.
    """
    weight_path = _download_weight_path(model_key)
    state_dict = torch.load(weight_path, map_location="cpu")

    # Strip 'module.' if trained with DataParallel
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

        # Small speed hint (especially on GPU)
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

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
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

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        tensor = transform(img_np)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, Any]:
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
        Faster Grad-CAM:
        - smaller cam_side (256)
        - uses predicted class passed from Streamlit (no extra forward)
        """
        os.makedirs(output_dir, exist_ok=True)

        cam_side = 256  # 🔥 faster than 384

        img_rgb = np.array(pil_img.convert("RGB"))
        img_rgb = apply_clahe_rgb(img_rgb)
        img_rgb = cv2.resize(img_rgb, (cam_side, cam_side))
        img_rgb = img_rgb.astype(np.float32) / 255.0

        input_tensor = self._preprocess_for_cam(pil_img, side=cam_side).to(DEVICE)

        cam = None
        try:
            cam = GradCAM(model=self.model, target_layers=[self.target_layer])
            targets = [ClassifierOutputTarget(int(target_class_index))]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            cam_img = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
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
