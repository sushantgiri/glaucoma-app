import os
import json
from typing import Dict, Any, List

import gc
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.densenet import densenet121
from torchvision.models.efficientnet import efficientnet_b0
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from huggingface_hub import hf_hub_download


# --------------------------------------------------------------------
# Paths, device & HF config
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../glaucoma_app
MODELS_DIR = os.path.join(BASE_DIR, "models")          # still used for JSON configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = "SushantGiri/glaucoma-ensemble-weights"

HF_MODEL_FILES: Dict[str, str] = {
    "resnet50": "resnet50_best.pth",
    "efficientnet_b0": "efficientnet_b0_best.pth",
    "densenet121": "densenet121_best.pth",
    "vit_base_patch16_224": "vit_base_patch16_224_best.pth",
    "swin_base_patch4_window7_224": "swin_base_patch4_window7_224_best.pth",
}


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def apply_clahe_rgb(img_np: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB space.
    img_np: H x W x 3, RGB uint8 or float in [0,255].
    Returns RGB uint8.
    """
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def load_config():
    """
    Load class mapping & preprocessing config if present.
    Fallback: binary classification (Normal / Glaucoma).
    These files are tiny, so we still keep them in models/.
    """
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


def load_state_from_hf(model_key: str) -> Dict[str, torch.Tensor]:
    """
    Download weights for `model_key` from Hugging Face Hub and return state_dict.
    Cached locally so subsequent loads are fast.
    Requires HF_TOKEN to be set in environment (Streamlit/Spaces secret).
    """
    filename = HF_MODEL_FILES[model_key]

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError(
            "HF_TOKEN not set. Please add HF_TOKEN to your Streamlit/Spaces secrets."
        )

    weight_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=hf_token,  # 🔑 use your secret here
        cache_dir=os.path.join(BASE_DIR, "hf_models_cache"),
    )

    state_dict = torch.load(weight_path, map_location="cpu")

    # Strip 'module.' if trained with DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict



# --------------------------------------------------------------------
# Simple container for model info (Python 3.9 friendly)
# --------------------------------------------------------------------
class ModelInfo:
    def __init__(self, name: str, model: nn.Module, input_size: int, target_layer: nn.Module = None):
        self.name = name
        self.model = model
        self.input_size = input_size
        self.target_layer = target_layer  # None for models we don't Grad-CAM


# --------------------------------------------------------------------
# Main ensemble class
# --------------------------------------------------------------------
class GlaucomaEnsemble:
    def __init__(self):
        self.class_mapping, self.prep_cfg = load_config()
        self.models: List[ModelInfo] = []
        self._load_cnn_models()
        self._load_transformer_models()

        for mi in self.models:
            mi.model.to(DEVICE)
            mi.model.eval()

    # ------------------------ model loading ------------------------ #
    def _load_cnn_models(self):
        # -------- ResNet50 --------
        res = resnet50(weights=None)
        res.fc = nn.Linear(res.fc.in_features, 2)
        res.load_state_dict(load_state_from_hf("resnet50"))

        self.models.append(
            ModelInfo(
                name="resnet50",
                model=res,
                input_size=512,      # match your training size
                target_layer=res.layer4[-1],
            )
        )

        # -------- EfficientNet-B0 --------
        eff = efficientnet_b0(weights=None)
        eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, 2)
        eff.load_state_dict(load_state_from_hf("efficientnet_b0"))

        self.models.append(
            ModelInfo(
                name="efficientnet_b0",
                model=eff,
                input_size=512,
                target_layer=eff.features[-1],
            )
        )

        # -------- DenseNet121 --------
        den = densenet121(weights=None)
        den.classifier = nn.Linear(den.classifier.in_features, 2)
        den.load_state_dict(load_state_from_hf("densenet121"))

        self.models.append(
            ModelInfo(
                name="densenet121",
                model=den,
                input_size=512,
                target_layer=den.features[-1],
            )
        )

    def _load_transformer_models(self):
        # -------- ViT-Base --------
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
        vit.load_state_dict(load_state_from_hf("vit_base_patch16_224"))

        self.models.append(
            ModelInfo(
                name="vit_base_patch16_224",
                model=vit,
                input_size=224,
                target_layer=None,   # no Grad-CAM / attention for now
            )
        )

        # -------- Swin-Base --------
        swin = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
        swin.load_state_dict(load_state_from_hf("swin_base_patch4_window7_224"))

        self.models.append(
            ModelInfo(
                name="swin_base_patch4_window7_224",
                model=swin,
                input_size=224,
                target_layer=None,
            )
        )

    # ------------------------ preprocessing ------------------------ #
    def _preprocess(self, pil_img: Image.Image, input_size: int) -> torch.Tensor:
        """
        Match your validation transforms:
        - Convert to RGB
        - CLAHE
        - Resize to input_size
        - Normalize (ImageNet mean/std by default)
        """
        img = pil_img.convert("RGB")
        img_np = np.array(img)           # uint8, 0-255
        img_np = apply_clahe_rgb(img_np) # CLAHE
        img_np = cv2.resize(img_np, (input_size, input_size))
        img_np = img_np.astype(np.float32) / 255.0   # scale to [0,1]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        tensor = transform(img_np)  # (C, H, W)
        return tensor.unsqueeze(0)  # (1, C, H, W)

    def _preprocess_for_cam(self, pil_img: Image.Image, side: int = 384) -> torch.Tensor:
        """
        Lighter preprocessing just for Grad-CAM visualization.
        Uses a smaller resolution (default 384) to save memory.
        """
        img = pil_img.convert("RGB")
        img_np = np.array(img)
        img_np = apply_clahe_rgb(img_np)
        img_np = cv2.resize(img_np, (side, side))
        img_np = img_np.astype(np.float32) / 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        tensor = transform(img_np)
        return tensor.unsqueeze(0)

    # ------------------------ prediction / ensemble ------------------------ #
    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Run all 5 models on the image and return:
        - ensemble prediction
        - per-model predictions
        Structure matches your previous version so Streamlit code
        does not need to change.
        """
        per_model: Dict[str, Any] = {}
        ensemble_probs: List[np.ndarray] = []

        for mi in self.models:
            x = self._preprocess(pil_img, mi.input_size).to(DEVICE)
            logits = mi.model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            ensemble_probs.append(prob)

            pred_idx = int(np.argmax(prob))
            per_model[mi.name] = {
                "probs": prob.tolist(),
                "pred_class_index": pred_idx,
                "pred_label": self.class_mapping[str(pred_idx)],
            }

            del x, logits
            gc.collect()

        ensemble_probs = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
        ensemble_idx = int(np.argmax(ensemble_probs))

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "ensemble": {
                "probs": ensemble_probs.tolist(),
                "pred_class_index": ensemble_idx,
                "pred_label": self.class_mapping[str(ensemble_idx)],
            },
            "per_model": per_model,
        }

    # ------------------------ Grad-CAM for CNNs ------------------------ #
    def gradcam_for_cnn(self, pil_img: Image.Image, output_dir: str) -> Dict[str, str]:
        """
        Generate Grad-CAM overlays for a *subset* of CNN models.
        To keep memory under control on Streamlit Cloud, we only
        compute Grad-CAM for ResNet50 by default. All 5 models are
        still used for prediction.
        """
        os.makedirs(output_dir, exist_ok=True)

        cam_side = 384

        img_rgb = np.array(pil_img.convert("RGB"))
        img_rgb = apply_clahe_rgb(img_rgb)
        img_rgb = cv2.resize(img_rgb, (cam_side, cam_side))
        img_rgb = img_rgb.astype(np.float32) / 255.0  # [0,1]

        saved_paths: Dict[str, str] = {}

        cam_model_names = {"resnet50"}  # add "efficientnet_b0" here if you want two heatmaps

        for mi in self.models:
            if mi.name not in cam_model_names:
                continue

            input_tensor = self._preprocess_for_cam(pil_img, side=cam_side).to(DEVICE)

            cam = None
            try:
                cam = GradCAM(
                    model=mi.model,
                    target_layers=[mi.target_layer],
                )

                targets = [ClassifierOutputTarget(1)]  # class 1 = glaucoma
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

                cam_img = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

                filename = f"{mi.name}_gradcam.png"
                save_path = os.path.join(output_dir, filename)
                Image.fromarray(cam_img).save(save_path)
                saved_paths[mi.name] = save_path

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

        return saved_paths
