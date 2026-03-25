import os
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from skin_gatekeeper.predict import predict_image as gate_predict
from dermassist.predict import predict_image as derm_predict

try:
    from skin_gatekeeper.predict import load as gate_load  # type: ignore
except Exception:
    gate_load = None

try:
    from dermassist.predict import load as derm_load  # type: ignore
except Exception:
    derm_load = None

SKIN_CLASSIFICATION_THRESHOLD = 0.50
SKIN_CONFIDENCE_THRESHOLD = 0.80

app = FastAPI(title="DermAssistAIModules Inference API", version="1")


@app.on_event("startup")
def startup():
    if gate_load:
        gate_load()
    if derm_load:
        derm_load()
    print("Inference API ready. PORT=", os.getenv("PORT", "8000"))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def low_confidence_response(gate, latency_ms: int) -> JSONResponse:
    return JSONResponse(
        {
            "status": "needs_review",
            "reason": "LOW_SKIN_CONFIDENCE",
            "message": "Skin confidence is below 80%. Please retake the photo or continue at your own risk.",
            "latency_ms": latency_ms,
            "gatekeeper": gate,
            "threshold": SKIN_CONFIDENCE_THRESHOLD,
            "confidence": float(gate.get("score", 0.0)),
            "next_step": {
                "retake_recommended": True,
                "allow_continue_anyway": True,
                "bypass_param": "bypassSkinCheck=true",
            },
        }
    )


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def extract_is_skin(gate) -> bool:
    if isinstance(gate, dict):
        return bool(gate.get("is_skin", gate.get("skin", False)))
    if isinstance(gate, (tuple, list)) and len(gate) >= 1:
        return bool(gate[0])
    if isinstance(gate, bool):
        return gate
    return False


async def save_upload_to_temp(image: UploadFile) -> str:
    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(await image.read())
    return tmp_path


@app.post("/skin-check")
async def skin_check(image: UploadFile = File(...)) -> JSONResponse:
    start = time.perf_counter()
    tmp_path: Optional[str] = None

    try:
        tmp_path = await save_upload_to_temp(image)

        gate = gate_predict(tmp_path, threshold=SKIN_CLASSIFICATION_THRESHOLD)
        is_skin = extract_is_skin(gate)
        score = float(gate.get("score", 0.0))

        latency_ms = int((time.perf_counter() - start) * 1000)

        if not is_skin:
            return JSONResponse(
                {
                    "status": "rejected",
                    "reason": "NOT_SKIN",
                    "latency_ms": latency_ms,
                    "gatekeeper": gate,
                }
            )

        if score < SKIN_CONFIDENCE_THRESHOLD:
            return low_confidence_response(gate, latency_ms)

        return JSONResponse(
            {
                "status": "success",
                "latency_ms": latency_ms,
                "gatekeeper": gate,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{type(e).__name__}: {e}"},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/lesion-detection")
async def lesion_detection(
    image: UploadFile = File(...),
    bypassSkinCheck: str = Form("false"),
    forceLesionOnly: str = Form("false"),
) -> JSONResponse:
    start = time.perf_counter()
    tmp_path: Optional[str] = None

    try:
        tmp_path = await save_upload_to_temp(image)

        bypass_skin_check = to_bool(bypassSkinCheck)
        force_lesion_only = to_bool(forceLesionOnly)

        gate = None

        if not bypass_skin_check and not force_lesion_only:
            gate = gate_predict(tmp_path, threshold=SKIN_CLASSIFICATION_THRESHOLD)
            is_skin = extract_is_skin(gate)
            score = float(gate.get("score", 0.0))

            if not is_skin:
                latency_ms = int((time.perf_counter() - start) * 1000)
                return JSONResponse(
                    {
                        "status": "rejected",
                        "reason": "NOT_SKIN",
                        "latency_ms": latency_ms,
                        "gatekeeper": gate,
                    }
                )

            if score < SKIN_CONFIDENCE_THRESHOLD:
                latency_ms = int((time.perf_counter() - start) * 1000)
                return low_confidence_response(gate, latency_ms)
        derm = derm_predict(tmp_path)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return JSONResponse(
            {
                "status": "success",
                "latency_ms": latency_ms,
                "gatekeeper": gate or {},
                "dermassist": derm,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{type(e).__name__}: {e}"},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# keeping this for anything still using old calls
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    start = time.perf_counter()
    tmp_path: Optional[str] = None

    try:
        tmp_path = await save_upload_to_temp(image)

        gate = gate_predict(tmp_path, threshold=SKIN_CLASSIFICATION_THRESHOLD)
        is_skin = extract_is_skin(gate)
        score = float(gate.get("score", 0.0))

        if not is_skin:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                {
                    "status": "rejected",
                    "reason": "NOT_SKIN",
                    "latency_ms": latency_ms,
                    "gatekeeper": gate,
                }
            )

        if score < SKIN_CONFIDENCE_THRESHOLD:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return low_confidence_response(gate, latency_ms)

        derm = derm_predict(tmp_path)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return JSONResponse(
            {
                "status": "success",
                "latency_ms": latency_ms,
                "gatekeeper": gate,
                "dermassist": derm,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{type(e).__name__}: {e}"},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
