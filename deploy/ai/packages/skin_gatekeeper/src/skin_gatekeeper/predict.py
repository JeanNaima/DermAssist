import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


_MODEL = None
_IDX_TO_CLASS = None
_SKIN_INDEX = None


def _default_weights_path() -> str:
    return os.path.join(
        "artifacts", "private", "skin_gatekeeper", "best_resnet18_skin_filter.pt"
    )


def _default_class_map_path() -> str:
    return os.path.join("artifacts", "private", "skin_gatekeeper", "class_to_idx.json")


def _infer_skin_index(idx_to_class: dict[int, str]) -> Optional[int]:
    # Try to find a class name that looks like "skin" but not "not_skin"
    candidates = [(i, name.lower()) for i, name in idx_to_class.items()]
    for i, n in candidates:
        if "skin" in n and "not" not in n and "no" not in n:
            return i

    # If binary and one class contains "not", pick the other
    if len(candidates) == 2:
        i0, n0 = candidates[0]
        i1, n1 = candidates[1]
        if "not" in n0 or "no" in n0:
            return i1
        if "not" in n1 or "no" in n1:
            return i0

    return None


def load(
    weights_path: Optional[str] = None,
    class_map_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, dict[int, str], Optional[int]]:
    global _MODEL, _IDX_TO_CLASS, _SKIN_INDEX

    if _MODEL is not None and _IDX_TO_CLASS is not None:
        return _MODEL, _IDX_TO_CLASS, _SKIN_INDEX

    if weights_path is None:
        weights_path = os.getenv("SKIN_GATEKEEPER_WEIGHTS") or _default_weights_path()

    if class_map_path is None:
        class_map_path = (
            os.getenv("SKIN_GATEKEEPER_CLASS_MAP") or _default_class_map_path()
        )

    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"class_to_idx.json not found at: {class_map_path}")

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # class_to_idx is usually {"class_name": idx}
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(weights_path, map_location="cpu")

    # Extract actual weights from common checkpoint formats
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("weights")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    # Strip common prefixes if present
    def _strip_prefix(sd: dict, prefix: str) -> dict:
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected state_dict type: {type(state_dict)}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = _strip_prefix(state_dict, "module.")
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = _strip_prefix(state_dict, "model.")

    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _MODEL = model
    _IDX_TO_CLASS = idx_to_class
    _SKIN_INDEX = _infer_skin_index(idx_to_class)

    return _MODEL, _IDX_TO_CLASS, _SKIN_INDEX


_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_image(image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    model, idx_to_class, skin_idx = load()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path).convert("RGB")
    x = _PREPROCESS(img).unsqueeze(0)

    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()

    predicted_index = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = idx_to_class.get(predicted_index, str(predicted_index))

    # Determine "skin score"
    if skin_idx is not None and 0 <= skin_idx < len(probs):
        skin_score = float(probs[skin_idx])
        is_skin = skin_score >= threshold
    else:
        # fallback: treat predicted prob as score
        skin_score = float(probs[predicted_index])
        is_skin = (
            ("skin" in label.lower())
            and ("not" not in label.lower())
            and (skin_score >= threshold)
        )

    return {
        "is_skin": bool(is_skin),
        "score": float(skin_score),
        "threshold": float(threshold),
        "label": label,
        "predicted_index": predicted_index,
        "probs": {idx_to_class[i]: float(probs[i]) for i in range(len(probs))},
    }
