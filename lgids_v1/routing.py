"""lgids_v1.routing -- Device to behavior group routing logic."""
import csv
import json
from pathlib import Path
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

def load_device_group_map() -> dict[str, str]:
    table_path = CONFIG_DIR / "device_table.csv"
    group_map = {}
    with open(table_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev_name = row.get("device_name", "").strip()
            group = row.get("behaviour_group", "").strip()
            if dev_name and group:
                group_map[dev_name] = group
    return group_map

DEVICE_GROUP_MAP = load_device_group_map()

def route_device(device_name: str) -> tuple[str, str]:
    """Route a device name to (group, model_used).
    G1 -> G1
    G2 -> G2
    Anything else -> (GEN, GLOBAL)
    """
    if not device_name or str(device_name).startswith("unregistered:"):
        return "GEN", "GLOBAL"
    group = DEVICE_GROUP_MAP.get(device_name, "GEN")
    if group == "G1":
        return "G1", "G1"
    elif group == "G2":
        return "G2", "G2"
    else:
        return "GEN", "GLOBAL"

def route_series(devices: pd.Series) -> tuple[pd.Series, pd.Series]:
    groups = []
    models = []
    for d in devices:
        g, m = route_device(d)
        groups.append(g)
        models.append(m)
    return pd.Series(groups, index=devices.index), pd.Series(models, index=devices.index)
