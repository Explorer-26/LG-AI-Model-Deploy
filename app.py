"""
LGIDS V1 — FastAPI REST API
Deploy this on Render (or any cloud) to serve predictions via HTTP.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Ensure the package root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any
import uuid
import pandas as pd
import uvicorn

from lgids_v1.predict import V1Predictor
from lgids_v1.features import FEATURE_NAMES

# ---------------------------------------------------------------------------
app = FastAPI(
    title="LGIDS V1 — IoT Intrusion Detection",
    description=(
        "Behavior-routed XGBoost binary classifier for enterprise IoT network flows. "
        "Routes traffic through G1 (high-activity), G2 (low-activity), or GLOBAL specialist models."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
predictor = V1Predictor()

# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "LGIDS V1", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "models_loaded": list(predictor.models.keys())}


@app.get("/info", tags=["Info"])
def info():
    """Return model metadata, operating points and feature list."""
    return {
        "system": predictor.metadata.get("system"),
        "description": predictor.metadata.get("description"),
        "operating_points": predictor.operating_points,
        "models": {k: {
            "training_set": v.get("training_set"),
            "n_train": v.get("n_train"),
            "n_train_benign": v.get("n_train_benign"),
            "n_train_attack": v.get("n_train_attack"),
        } for k, v in predictor.metadata.get("models", {}).items()},
        "n_features": predictor.metadata.get("n_features", 70),
        "training_dataset": predictor.metadata.get("training_dataset"),
    }


@app.post("/predict/csv", tags=["Predict"])
async def predict_csv(
    file: UploadFile = File(...),
    operating_point: str = "T_house",
):
    """
    Upload a CSV file of network flows and get predictions back as JSON.

    Required columns:
    - flow_id (or Flow ID)
    - device (or device_name)
    - All 70 features from features_v1.json
    """
    if operating_point not in predictor.operating_points:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operating_point '{operating_point}'. Choose from: {list(predictor.operating_points.keys())}",
        )
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        result = predictor.predict(df, operating_point=operating_point)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return JSONResponse(content={
        "operating_point": operating_point,
        "threshold": float(predictor.operating_points[operating_point]),
        "n_flows": len(result),
        "n_attack": int((result["pred"] == "attack").sum()),
        "n_benign": int((result["pred"] == "benign").sum()),
        "predictions": result.to_dict(orient="records"),
    })


@app.post("/predict/json", tags=["Predict"])
async def predict_json(
    payload: Any = None,
    operating_point: str = "T_house",
    device: str | None = None,
):
    """
    POST flow records for inference. Supports:
    1. Single flow dict: {"device": "Netatmo Weather Station", "Protocol": 6, ...}
    2. List of flow dicts: [{"device": "..."}, ...]
    3. Batch object with 'flows': {"flows": [...]}

    Automatically populates missing flow_id with a UUID and fills missing 
    features with 0.0 (ThingsBoard-friendly).
    """
    if operating_point not in predictor.operating_points:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operating_point '{operating_point}'. Choose from: {list(predictor.operating_points.keys())}",
        )

    if payload is None:
        raise HTTPException(status_code=400, detail="Empty request body.")

    single_flow = False
    if isinstance(payload, list):
        flows = payload
    elif isinstance(payload, dict):
        if "flows" in payload and isinstance(payload["flows"], list):
            flows = payload["flows"]
        else:
            flows = [payload]
            single_flow = True
    else:
        raise HTTPException(status_code=400, detail="Payload must be a JSON object or list.")

    if not flows:
        raise HTTPException(status_code=400, detail="No flow records provided.")

    try:
        df = pd.DataFrame(flows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build DataFrame: {e}")

    # Auto-fill flow_id if missing
    if "flow_id" not in df.columns and "Flow ID" not in df.columns:
        df["flow_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Device column resolution
    if "device" not in df.columns and "device_name" not in df.columns:
        if "deviceName" in df.columns:
            df["device"] = df["deviceName"]
        elif device:
            df["device"] = device
        else:
            df["device"] = "Netatmo Weather Station"

    # Impute missing feature columns to 0.0
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0.0

    try:
        result = predictor.predict(df, operating_point=operating_point)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    predictions_list = result.to_dict(orient="records")

    response_data = {
        "operating_point": operating_point,
        "threshold": float(predictor.operating_points[operating_point]),
        "n_flows": len(result),
        "n_attack": int((result["pred"] == "attack").sum()),
        "n_benign": int((result["pred"] == "benign").sum()),
        "predictions": predictions_list,
    }

    # If single flow, flatten top-level keys for easy ThingsBoard telemetry ingestion
    if single_flow and len(predictions_list) == 1:
        first = predictions_list[0]
        response_data["p_attack"] = round(float(first["p_attack"]), 6)
        response_data["pred"] = str(first["pred"])
        response_data["is_attack"] = 1 if first["pred"] == "attack" else 0
        response_data["group"] = str(first["group"])
        response_data["model_used"] = str(first["model_used"])
        response_data["flow_id"] = str(first["flow_id"])
        response_data["device"] = str(first["device"])

    return response_data


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
