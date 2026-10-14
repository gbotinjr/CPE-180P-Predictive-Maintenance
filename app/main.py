oops_this_breaks_python
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from app.features import build_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LOG_FILE = Path(os.getenv("LOG_FILE", "logs/requests.jsonl"))
RATE_LIMIT, RATE_WINDOW, MAX_BODY = int(os.getenv("RATE_LIMIT", "300")), 60, 4096

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
req_log = logging.getLogger("requests")
req_log.setLevel(logging.INFO)
if not req_log.handlers:
    req_log.addHandler(RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5))
app_log = logging.getLogger("uvicorn.error")
state = {"session": None, "meta": {}}
hits = defaultdict(deque)


@asynccontextmanager
async def lifespan(_):
    try:
        state["session"] = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
        state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    except Exception as e:  # noqa: BLE001  (/healthz will report 503)
        app_log.error("model load failed: %s", e)
    yield


app = FastAPI(title="Predictive Maintenance API", lifespan=lifespan)


class PredictRequest(BaseModel):
    type: Literal["L", "M", "H"]
    air_temp_k: float
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    @model_validator(mode="after")
    def check_ranges(self):
        for name, (lo, hi) in state["meta"].get("ranges", {}).items():
            v = getattr(self, name)
            if not lo <= v <= hi:
                raise ValueError(f"{name}={v} outside accepted range [{lo:.1f}, {hi:.1f}]")
        return self


@app.middleware("http")
async def guards(request: Request, call_next):
    if int(request.headers.get("content-length", 0)) > MAX_BODY:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    ip, now = request.client.host if request.client else "?", time.time()
    q = hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
    q.append(now)
    return await call_next(request)


@app.get("/healthz")
def healthz():
    if state["session"] is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok"}


@app.get("/v1/model-info")
def model_info():
    if not state["meta"]:
        raise HTTPException(503, "model not loaded")
    return {k: state["meta"][k] for k in ("model_version", "training_date", "metrics", "sha256")}


@app.post("/v1/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    if state["session"] is None:
        raise HTTPException(503, "model not loaded")
    if req.process_temp_k < req.air_temp_k:
        app_log.warning("process temp below air temp: %s", req.model_dump())
    X = build_features(pd.DataFrame([req.model_dump()]))
    prob = float(state["session"].run(None, {"input": X})[1][0, 1])
    out = {"request_id": uuid.uuid4().hex, "prediction": int(prob >= 0.5), "failure_probability": round(prob, 4),
           "model_version": state["meta"]["model_version"], "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    req_log.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **out, "features": req.model_dump()}))
    return out
