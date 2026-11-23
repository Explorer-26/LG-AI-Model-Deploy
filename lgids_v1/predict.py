"""lgids_v1.predict -- Core prediction pipeline for V1."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .features import extract_features, FEATURE_NAMES
from .routing import route_series

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PACKAGE_ROOT / "models"

class V1Predictor:
    def __init__(self, models_dir: Path | None = None):
        mdir = models_dir or MODELS_DIR
        with open(mdir / "MODELS.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.operating_points = self.metadata["operating_points"]
        
        self.models = {
            "G1": joblib.load(mdir / "G1.joblib"),
            "G2": joblib.load(mdir / "G2.joblib"),
            "GLOBAL": joblib.load(mdir / "GLOBAL.joblib")
        }

    def predict(self, df: pd.DataFrame, operating_point: str = "T_house") -> pd.DataFrame:
        if operating_point not in self.operating_points:
            valid = list(self.operating_points.keys())
            raise ValueError(f"Unknown operating point '{operating_point}'. Choose from: {valid}")
        
        threshold = float(self.operating_points[operating_point])
        
        # Check flow_id and device columns
        flow_id_col = None
        for candidate in ["flow_id", "Flow ID"]:
            if candidate in df.columns:
                flow_id_col = candidate
                break
        
        device_col = None
        for candidate in ["device", "device_name"]:
            if candidate in df.columns:
                device_col = candidate
                break
        
        missing_id = []
        if not flow_id_col:
            missing_id.append("flow_id (or 'Flow ID')")
        if not device_col:
            missing_id.append("device (or 'device_name')")
        
        if missing_id:
            raise ValueError(f"Input is missing required identification columns: {missing_id}")

        # Check features
        X = extract_features(df)
        
        # Routing
        groups, models_used = route_series(df[device_col])
        
        # Predict proba per model
        n_rows = len(df)
        p_attack = np.zeros(n_rows, dtype=np.float64)
        
        for m_name, model in self.models.items():
            mask = (models_used == m_name).to_numpy()
            if mask.any():
                sub_X = X[mask]
                proba = model.predict_proba(sub_X)[:, 1].astype(np.float64)
                p_attack[mask] = proba

        pred = np.where(p_attack >= threshold, "attack", "benign")
        
        return pd.DataFrame({
            "flow_id": df[flow_id_col].values,
            "device": df[device_col].values,
            "group": groups.values,
            "model_used": models_used.values,
            "p_attack": p_attack,
            "threshold": threshold,
            "pred": pred
        })
