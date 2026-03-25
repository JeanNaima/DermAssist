import os
import tempfile
import time
from typing import Any, Literal, Optional, Tuple
from uuid import UUID
import json

import httpx
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

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


class HelloWord(BaseModel):
    status: str = "hello world"


class ImageRef(BaseModel):
    type: Literal["url"]
    url: HttpUrl


class AnalyzeRequest(BaseModel):
    request_id: UUID
    user_id: str = Field(min_length=1)
    image: ImageRef


class ErrorObj(BaseModel):
    code: str
    message: str
    retriable: bool


class ErrorResponse(BaseModel):
    request_id: Optional[UUID] = None
    status: Literal["error"] = "error"
    error: ErrorObj


class AnalyzeMeta(BaseModel):
    model: str
    latency_ms: int
    warnings: list[str] = []


class AnalyzeSuccessResponse(BaseModel):
    request_id: UUID
    status: Literal["success"] = "success"
    result: dict[str, Any]
    meta: AnalyzeMeta


app = FastAPI(title="DermAssistAIModules API", version="1")


def error_json(
    *,
    request_id: Optional[UUID],
    http_status: int,
    code: str,
    message: str,
    retriable: bool,
) -> JSONResponse:
    payload = ErrorResponse(
        request_id=request_id,
        error=ErrorObj(code=code, message=message, retriable=retriable),
    ).model_dump(mode="json")
    return JSONResponse(status_code=http_status, content=payload)


def _normalize_gatekeeper_output(out: Any) -> Tuple[bool, float, dict[str, Any]]:
    """
    Accept a few possible return shapes from skin_gatekeeper.predict.predict_image:
      - dict: {"is_skin": bool, "score": float, ...}
      - tuple/list: (is_skin, score) or (is_skin, score, extra_dict)
      - bool: is_skin
    Returns: (is_skin, score, extra)
    """
    if isinstance(out, dict):
        is_skin = bool(out.get("is_skin", out.get("skin", out.get("allowed", False))))
        score = float(out.get("score", out.get("confidence", out.get("prob", 0.0))))
        extra = dict(out)
        return is_skin, score, extra

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            return False, 0.0, {}
        is_skin = bool(out[0])
        score = float(out[1]) if len(out) >= 2 else 0.0
        extra = dict(out[2]) if len(out) >= 3 and isinstance(out[2], dict) else {}
        return is_skin, score, extra

    if isinstance(out, bool):
        return out, 1.0 if out else 0.0, {}

    # Unknown shape
    return False, 0.0, {"raw": str(out)}


def _normalize_dermassist_output(out: Any) -> dict[str, Any]:
    """
    Accept a few possible return shapes from dermassist.predict.predict_image:
      - dict: {"label": str, "confidence": float, "attributes"/"probs": {...}, ...}
      - tuple/list: (label, confidence, probs, predicted_index)
    Returns a dict compatible with your previous API output.
    """
    if isinstance(out, dict):
        label = out.get("label") or out.get("class_name") or out.get("class") or "unknown"
        confidence = float(out.get("confidence", out.get("score", 0.0)))
        attributes = out.get("attributes") or out.get("probs") or out.get("all_probs") or {}
        predicted_index = out.get("predicted_index", out.get("index", None))
        result = {
            "label": label,
            "confidence": confidence,
            "attributes": attributes,
            "predicted_index": predicted_index,
        }
        # Keep anything extra for debugging if present
        for k, v in out.items():
            if k not in result:
                result[k] = v
        return result

    if isinstance(out, (tuple, list)):
        label = out[0] if len(out) >= 1 else "unknown"
        confidence = float(out[1]) if len(out) >= 2 else 0.0
        attributes = out[2] if len(out) >= 3 else {}
        predicted_index = out[3] if len(out) >= 4 else None
        return {
            "label": label,
            "confidence": confidence,
            "attributes": attributes,
            "predicted_index": predicted_index,
        }

    return {"label": "unknown", "confidence": 0.0, "attributes": {}, "predicted_index": None}


@app.on_event("startup")
async def on_startup() -> None:
    print("Starting DermAssist API. PORT=", os.getenv("PORT", "8000"))

    # Optional: force model load at startup (recommended)
    if gate_load:
        try:
            gate_load()
            print("Gatekeeper model loaded.")
        except Exception as e:
            print("WARNING: gatekeeper load() failed:", repr(e))

    if derm_load:
        try:
            derm_load()
            print("DermAssist model loaded.")
        except Exception as e:
            print("WARNING: dermassist load() failed:", repr(e))

    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)
    )


@app.on_event("shutdown")
async def shutdown():
    await app.state.http.aclose()


@app.get("/helloword", response_model=HelloWord)
async def hellworld() -> HelloWord:
    return HelloWord()


@app.post("/analyzeimage", response_model=AnalyzeSuccessResponse)
async def analyze_image(
    req: AnalyzeRequest,
    _auth: Optional[str] = None,
    x_request_id: str = Header(default=""),
):
    start = time.perf_counter()
    client: httpx.AsyncClient = app.state.http

    url = str(req.image.url)

    # 1) Download image
    try:
        r = await client.get(url, timeout=15.0, follow_redirects=True)
    except httpx.TimeoutException:
        return error_json(
            request_id=req.request_id,
            http_status=504,
            code="MODEL_TIMEOUT",
            message="Timed out downloading image",
            retriable=True,
        )
    except httpx.HTTPError as e:
        return error_json(
            request_id=req.request_id,
            http_status=502,
            code="DEPENDENCY_FAILURE",
            message=f"Failed to download image: {type(e).__name__}",
            retriable=True,
        )

    print("Download URL:", url)
    print(
        "HTTP:",
        r.status_code,
        "ctype:",
        r.headers.get("content-type"),
        "bytes:",
        len(r.content) if r.content else 0,
    )

    if r.status_code >= 400:
        return error_json(
            request_id=req.request_id,
            http_status=400,
            code="INVALID_IMAGE_URL",
            message=f"Image URL returned HTTP {r.status_code}",
            retriable=r.status_code in (408, 429, 500, 502, 503, 504),
        )

    ctype = (r.headers.get("content-type") or "").lower()
    if not ctype.startswith("image/"):
        return error_json(
            request_id=req.request_id,
            http_status=400,
            code="NOT_AN_IMAGE",
            message=f"URL did not return an image (content-type={ctype})",
            retriable=False,
        )

    tmp_path = None

    # 2) Save to temp file (so both predictors can read it)
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)

        # 3) Gatekeeper first
        try:
            gate_out = gate_predict(tmp_path)
            is_skin, gate_score, gate_extra = _normalize_gatekeeper_output(gate_out)
        except Exception as e:
            print("GATEKEEPER_FAILED:", repr(e))
            return error_json(
                request_id=req.request_id,
                http_status=500,
                code="GATEKEEPER_FAILED",
                message=f"Gatekeeper failed: {type(e).__name__}: {e}",
                retriable=False,
            )

        if not is_skin:
            # "safe response": do NOT run derm model
            latency = int((time.perf_counter() - start) * 1000)
            result = {
                "label": "rejected_not_skin",
                "confidence": gate_score,
                "attributes": {"gatekeeper": gate_extra},
                "predicted_index": None,
                "rejected": True,
                "reason": "NOT_SKIN",
            }
            print("Gatekeeper rejection:", json.dumps(result, indent=2, default=str))
            return AnalyzeSuccessResponse(
                request_id=req.request_id,
                result=result,
                meta=AnalyzeMeta(model="Gatekeeper", latency_ms=latency, warnings=[]),
            )

        # 4) DermAssist inference (only if skin)
        try:
            derm_out = derm_predict(tmp_path)
            result = _normalize_dermassist_output(derm_out)
            # include gatekeeper details in response
            result["gatekeeper"] = {"is_skin": True, "score": gate_score, **gate_extra}
        except Exception as e:
            print("DERMASSIST_FAILED:", repr(e))
            return error_json(
                request_id=req.request_id,
                http_status=500,
                code="MODEL_INFERENCE_FAILED",
                message=f"Inference failed: {type(e).__name__}: {e}",
                retriable=False,
            )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    print("Inference result:", json.dumps(result, indent=2, default=str))
    latency = int((time.perf_counter() - start) * 1000)
    return AnalyzeSuccessResponse(
        request_id=req.request_id,
        result=result,
        meta=AnalyzeMeta(model="DermAssist", latency_ms=latency, warnings=[]),
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.dermassist_api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )