"""
CNN Model Predictor
Loads the fine-tuned EfficientNet-B0 and runs inference on plant leaf images.
Falls back to demo mode if the model checkpoint is not found.
"""

import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ──────────────────────────────────────────────────────────────────────────
# Preprocessing constants  (must match train.py)
# ──────────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Fallback class list (authoritative order is always loaded from the checkpoint)
CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _parse_class(raw: str) -> Tuple[str, str]:
    """'Tomato___Early_blight' → ('Tomato', 'Early blight')"""
    clean = raw.replace("(including_sour)", "").replace(",_bell", " bell")
    parts = clean.split("___")
    plant   = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return plant, disease


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(num_classes: int) -> torch.nn.Module:
    import torch.nn as nn
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────
class CNNPredictor:
    def __init__(self, model_path: str):
        self.device      = _get_device()
        self.is_loaded   = False
        self.model       = None
        self.class_names: List[str] = list(CLASS_NAMES)
        self._load(model_path)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def _load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found at '{model_path}'. Running in DEMO mode.")
            return
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                if "classes" in checkpoint:
                    self.class_names = checkpoint["classes"]
                val_acc = checkpoint.get("val_acc", None)
            else:
                state_dict = checkpoint
                val_acc = None

            num_classes = len(self.class_names)
            self.model  = _build_model(num_classes).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True

            acc_str = f"  val_acc={val_acc:.2%}" if val_acc is not None else ""
            print(f"[OK] CNN loaded from '{model_path}' on {self.device}{acc_str}")
        except Exception as exc:
            print(f"[ERR] CNN load failed: {exc}")

    # ------------------------------------------------------------------
    def predict(self, image: Image.Image, topk: int = 5) -> Dict:
        if not self.is_loaded or self.model is None:
            return self._demo_result()

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

        top_probs, top_idxs  = probs.topk(topk)
        top_probs_list: List[float] = top_probs.cpu().tolist()
        top_idxs_list:  List[int]   = top_idxs.cpu().tolist()
        all_probs:      List[float] = probs.cpu().tolist()

        top1_class       = self.class_names[top_idxs_list[0]]
        plant, disease   = _parse_class(top1_class)

        return {
            "top1_class":      top1_class,
            "top1_plant":      plant,
            "top1_disease":    disease,
            "top1_confidence": top_probs_list[0],
            "top_k": [
                {
                    "class":      self.class_names[i],
                    "plant":      _parse_class(self.class_names[i])[0],
                    "disease":    _parse_class(self.class_names[i])[1],
                    "confidence": p,
                }
                for i, p in zip(top_idxs_list, top_probs_list)
            ],
            "all_probs":  all_probs,
            "model":      "MobileNetV3-Small (fine-tuned)",
            "demo_mode":  False,
        }

    def _demo_result(self) -> Dict:
        placeholder_class = self.class_names[0]
        plant, disease    = _parse_class(placeholder_class)
        return {
            "top1_class":      placeholder_class,
            "top1_plant":      plant,
            "top1_disease":    disease,
            "top1_confidence": 0.0,
            "top_k": [
                {"class": placeholder_class, "plant": plant, "disease": disease, "confidence": 0.0}
            ],
            "all_probs":  [0.0] * len(self.class_names),
            "model":      "MobileNetV3-Small — DEMO MODE (model .pth not found)",
            "demo_mode":  True,
        }
