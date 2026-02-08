import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, Optional, Any
from uuid import UUID
import httpx

class HelloWord(BaseModel):
    status: str = "hello world"
class ImageRef(BaseModel):
    type: Literal["url"]
    url: HttpUrl
class AnalyzeRequest(BaseModel):
    request_id: UUID
    user_id: str = Field(min_length = 1)
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

app = FastAPI(title= "python api", version = "1")

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
    ).model_dump()
    return JSONResponse(status_code=http_status, content=payload)

@app.on_event("startup")
async def on_startup() -> None:
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    )
@app.on_event("shutdown")
async def on_shutdown() -> None:
    await app.state.http.aclose()

@app.get("/helloword", response_model=HelloWord)
async def hellworld() -> HelloWord:
    return HelloWord()

@app.post("/analyzeimage", response_model=AnalyzeSuccessResponse)
async def analyze_image(
    req: AnalyzeRequest,
    _auth: None,
    x_request_id: str = Header(default="")
):
    start = time.perf_counter()
    raw = await req.body()
    print(raw)
    client: httpx.AsyncClient = app.state.http
    try:
     r = await client.get(str(req.image.url))
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

    if r.status_code >= 400:
        return error_json(
            request_id=req.request_id,
            http_status=400,
            code="INVALID_IMAGE_URL",
            message=f"Image URL returned HTTP {r.status_code}",
            retriable=r.status_code in (408, 429, 500, 502, 503, 504),
        )
    # send image from request to ai model
    result = result = {
        "label": "cat",
        "confidence": 0.93,
        "attributes": {"bbox": [0.1, 0.2, 0.5, 0.6]},
    }
    latency = int((time.perf_counter() - start)*1000)
    return AnalyzeSuccessResponse(request_id = req.request_id, result = result, meta = AnalyzeMeta(model = "Classfication",latency_ms = latency, warnings = []),)

#curl.exe -i -X POST "http://127.0.0.1:8000/analyzeimage" `-H "Content-Type: application/json" `-H "X-Request-Id: 11111111-1111-1111-1111-111111111111" `-d '{"request_id": "11111111-1111-1111-1111-111111111111", "user_id": "local-test", "image": {"type": "url", "url": "https://loremipsum.imgix.net/gPyHKDGI0md4NkRDjs4k8/36be1e73008a0181c1980f727f29d002/avatar-placeholder-generator-500x500.jpg?w=1920&q=60&auto=format,compress"}, "analysis_type": "default", "options": {"return_debug": false}}'
