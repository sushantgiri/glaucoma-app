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


# --------------------------------------------------------------------
# Paths & device
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../glaucoma_app
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """
    class_mapping_path = os.path.join(MODELS_DIR, "class_mapping.json")
    prep_path = os.path.join(MODELS_DIR, "preprocessing.json")

    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)
    else:
        # Default labels if json not provided
        class_mapping = {"0": "Normal", "1": "Glaucoma"}

    if os.path.exists(prep_path):
        with open(prep_path, "r") as f:
            prep_cfg = json.load(f)
    else:
        prep_cfg = {}

    return class_mapping, prep_cfg


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
        res_path = os.path.join(MODELS_DIR, "resnet50_best.pth")
        if not os.path.exists(res_path):
            raise FileNotFoundError("Missing resnet50_best.pth in models/")

        res = resnet50(weights=None)
        res.fc = nn.Linear(res.fc.in_features, 2)
        res.load_state_dict(torch.load(res_path, map_location=DEVICE))

        self.models.append(
            ModelInfo(
                name="resnet50",
                model=res,
                input_size=512,      # match your training size
                target_layer=res.layer4[-1],
            )
        )

        # -------- EfficientNet-B0 --------
        eff_path = os.path.join(MODELS_DIR, "efficientnet_b0_best.pth")
        if not os.path.exists(eff_path):
            raise FileNotFoundError("Missing efficientnet_b0_best.pth in models/")

        eff = efficientnet_b0(weights=None)
        eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, 2)
        eff.load_state_dict(torch.load(eff_path, map_location=DEVICE))

        self.models.append(
            ModelInfo(
                name="efficientnet_b0",
                model=eff,
                input_size=512,
                target_layer=eff.features[-1],
            )
        )

        # -------- DenseNet121 --------
        den_path = os.path.join(MODELS_DIR, "densenet121_best.pth")
        if not os.path.exists(den_path):
            raise FileNotFoundError("Missing densenet121_best.pth in models/")

        den = densenet121(weights=None)
        den.classifier = nn.Linear(den.classifier.in_features, 2)
        den.load_state_dict(torch.load(den_path, map_location=DEVICE))

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
        vit_path = os.path.join(MODELS_DIR, "vit_base_patch16_224_best.pth")
        if not os.path.exists(vit_path):
            raise FileNotFoundError("Missing vit_base_patch16_224_best.pth in models/")

        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
        vit.load_state_dict(torch.load(vit_path, map_location=DEVICE))

        self.models.append(
            ModelInfo(
                name="vit_base_patch16_224",
                model=vit,
                input_size=224,
                target_layer=None,   # no Grad-CAM / attention here for now
            )
        )

        # -------- Swin-Base --------
        swin_path = os.path.join(MODELS_DIR, "swin_base_patch4_window7_224_best.pth")
        if not os.path.exists(swin_path):
            raise FileNotFoundError("Missing swin_base_patch4_window7_224_best.pth in models/")

        swin = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
        swin.load_state_dict(torch.load(swin_path, map_location=DEVICE))

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

        # You can customise these from preprocessing.json if needed
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

    # ------------------------ prediction / ensemble ------------------------ #
    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Run all 5 models on the image and return:
        - ensemble prediction
        - per-model predictions
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

            # free tensor
            del x, logits
            gc.collect()

        ensemble_probs = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
        ensemble_idx = int(np.argmax(ensemble_probs))

        # light cleanup
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
        Generate Grad-CAM overlays for CNN models (ResNet, EfficientNet, DenseNet).
        Returns a dict: {model_name: saved_image_path}

        Uses GradCAM as a context manager so hooks are removed after each run.
        This avoids memory leaks when running multiple assessments.
        """
        os.makedirs(output_dir, exist_ok=True)

        img_rgb = np.array(pil_img.convert("RGB"))
        img_rgb = apply_clahe_rgb(img_rgb)
        img_rgb = img_rgb.astype(np.float32) / 255.0  # [0,1]

        saved_paths: Dict[str, str] = {}

        for mi in self.models:
            if mi.target_layer is None:
                # skip ViT & Swin here
                continue

            input_tensor = self._preprocess(pil_img, mi.input_size).to(DEVICE)

            # NOTE: your installed pytorch-grad-cam does NOT accept use_cuda,
            # so we only pass model and target_layers.
            with GradCAM(
                model=mi.model,
                target_layers=[mi.target_layer],
            ) as cam:
                # here we target class 1 ("Glaucoma"); you can change to predicted class
                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            img_resized = cv2.resize(img_rgb, (mi.input_size, mi.input_size))
            cam_img = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

            filename = f"{mi.name}_gradcam.png"
            save_path = os.path.join(output_dir, filename)
            Image.fromarray(cam_img).save(save_path)
            saved_paths[mi.name] = save_path

            # free tensors & CAM
            del input_tensor, grayscale_cam, cam_img
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        return saved_paths
