# DermAssistAIModules

- **skin_gatekeeper**: binary classifier that checks whether an image contains skin.
- **dermassist**: skin lesion classifier (8-class checkpoint).
- **inference_api**: a single FastAPI service that runs **gatekeeper → dermassist** in the same process.

## Repo layout

- `packages/dermassist/` — DermAssist model package
- `packages/skin_gatekeeper/` — Gatekeeper model package
- `services/inference_api/app.py` — production inference API (single entrypoint)
- `scripts/` — local CLI helpers (smoke tests / batch inference)
- `artifacts/private/` — local model weights (not meant to be committed)


## Prerequisites

- Python **3.11** recommended
- `pip`
- `curl` for testing the API


## 1) Create a virtual environment

From repo root:

```bash
python -m venv .venv
```

### Activate it

**Git Bash / MSYS**

```bash
source .venv/Scripts/activate
```

**PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

Install the local packages in editable mode:

```bash
pip install -e packages/dermassist -e packages/skin_gatekeeper
```

## 3) Model weights

This repo expects model weights under:

- `artifacts/private/skin_gatekeeper/`
  - `best_resnet18_skin_filter.pt`
  - `class_to_idx.json`

- `artifacts/private/dermassist/saved_models/`
  - `best_model4.5.pth` (or whichever you prefer)

## 4) Run the inference API (single-process)

This service accepts an uploaded image and returns:

1. gatekeeper result
2. dermassist result (only if gatekeeper says skin)

Start the server:

```bash
uvicorn services.inference_api.app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/healthz
```
### For Deployment Run this
```bash
uvicorn services.inference_api.app:app --host 0.0.0.0 --port 8000
```

### Test inference (multipart upload)

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -F "image=@data/samples/test.jpg"
```

Expected response shape:

```json
{
  "status": "success",
  "latency_ms": 123,
  "gatekeeper": { "...": "..." },
  "dermassist": { "...": "..." }
}
```

If the image is not skin:

```json
{
  "status": "rejected",
  "reason": "NOT_SKIN",
  "gatekeeper": { "...": "..." }
}
```

## 5) Run local scripts (CLI smoke tests)

### Gatekeeper on a folder

```bash
python scripts/predict_folder_gatekeeper.py data/samples --out gate.jsonl
```

### Gatekeeper + DermAssist on one image

```bash
python scripts/predict_image_dermassist.py data/samples/test.jpg --json
```
