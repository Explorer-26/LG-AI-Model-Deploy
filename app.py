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
import pandas as pd
import uvicorn

from lgids_v1.predict import V1Predictor

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
async def predict_json(payload: dict, operating_point: str = "T_house"):
    """
    POST a JSON object with a 'flows' list of flow records (each record is a dict
    with all required columns). Returns predictions as JSON.

    Example payload:
    {
      "flows": [
        {"flow_id": "f1", "device": "Amazon Echo Show", "Protocol": 6, ...}
      ]
    }
    """
    if operating_point not in predictor.operating_points:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operating_point '{operating_point}'. Choose from: {list(predictor.operating_points.keys())}",
        )
    flows = payload.get("flows")
    if not flows or not isinstance(flows, list):
        raise HTTPException(status_code=400, detail="Payload must have a 'flows' list.")

    try:
        df = pd.DataFrame(flows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build DataFrame: {e}")

    try:
        result = predictor.predict(df, operating_point=operating_point)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "operating_point": operating_point,
        "threshold": float(predictor.operating_points[operating_point]),
        "n_flows": len(result),
        "n_attack": int((result["pred"] == "attack").sum()),
        "n_benign": int((result["pred"] == "benign").sum()),
        "predictions": result.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
