import io
from pathlib import Path

import requests
import torch
from torch import nn
from torchvision import transforms, models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# --------- paths ----------
ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = ROOT / "runs" / "skin_filter_cnn" / "best_resnet18_skin_filter.pt"

# --------- app ----------
app = FastAPI(title="Skin Gatekeeper API", version="1.0")

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tfm = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
    ])

    return model, tfm, class_to_idx, idx_to_class

# Load once at startup
try:
    model, tfm, class_to_idx, idx_to_class = load_model(MODEL_PATH)
except Exception as e:
    # If the model path is wrong, you'll see it immediately
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

class PredictRequest(BaseModel):
    image_url: str
    threshold: float = 0.8

@app.get("/")
def root():
    return {"status": "ok", "device": device}


@app.post("/predict")
def predict(req: PredictRequest):
    # Download image with proper headers
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }

        # Add timeout and stream=False for better handling
        r = requests.get(
            req.image_url,
            timeout=15,
            headers=headers,
            stream=False
        )
        r.raise_for_status()

        # Verify it's actually an image
        content_type = r.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            print(f"Warning: Content-Type is {content_type}, but attempting to process anyway")

        # Try to open the image
        try:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as img_error:
            print(f"PIL failed to open image: {img_error}")
            # Try alternative approach - maybe the URL returns HTML?
            raise HTTPException(status_code=400, detail=f"URL content is not a valid image format")

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=400, detail="Image download timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unexpected error: {str(e)}")

    # Rest of your prediction code remains the same...
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    if "skin" not in class_to_idx or "non_skin" not in class_to_idx:
        raise HTTPException(status_code=500, detail=f"Model class_to_idx missing expected keys: {class_to_idx}")

    p_skin = float(probs[class_to_idx["skin"]].item())
    p_non_skin = float(probs[class_to_idx["non_skin"]].item())

    label = "skin" if p_skin >= req.threshold else "non_skin"

    return {
        "label": label,
        "threshold": req.threshold,
        "p_skin": p_skin,
        "p_non_skin": p_non_skin,
    }