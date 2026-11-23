"""lgids_v1.features -- Extract and format 70 features in features_v1 order."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "features_v1.json"

def load_feature_names() -> list[str]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["features"]

FEATURE_NAMES = load_feature_names()

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 70 features in features_v1 order as float32; fill NaN / Inf with 0.0."""
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns ({len(missing)} missing): {missing}")
    
    sub = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    # NaN and +-Inf -> 0.0 as in 08/09
    return np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
