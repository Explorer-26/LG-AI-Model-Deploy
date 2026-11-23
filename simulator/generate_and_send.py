"""
LGIDS V1 — IoT Flow Simulator + ThingsBoard Telemetry Sender
============================================================
Generates realistic synthetic network flows for 5 IoT devices,
runs them through the LGIDS V1 inference API (or locally),
and pushes predictions as telemetry to ThingsBoard Cloud.

Usage:
    python simulator/generate_and_send.py \
        --api-url https://lgids-v1.onrender.com \
        --interval 10

Environment variables (or edit THINGSBOARD_DEVICES below):
    TB_HOST          ThingsBoard host (default: thingsboard.cloud)
    TB_TOKEN_<NAME>  Access token per device (e.g. TB_TOKEN_ECHO_SHOW)

Each device streams one prediction row per --interval seconds.
Benign traffic is generated most of the time; random attack bursts are injected
to demonstrate the IDS catching threats.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ---------------------------------------------------------------------------
# 5 Devices — All G2 (low-activity simple): sensors & bulbs
# Names match device_table.csv exactly so routing → G2 specialist model
# ---------------------------------------------------------------------------
DEVICES = [
    {"name": "Netatmo Weather Station", "group": "G2", "category": "Sensor"},
    {"name": "Lumiman Bulb",            "group": "G2", "category": "Bulb"},
    {"name": "LIFX Lightbulb",          "group": "G2", "category": "Bulb"},
    {"name": "Levoit Air Purifier",     "group": "G2", "category": "Sensor"},
    {"name": "Govee Smart Humidifier",  "group": "G2", "category": "Sensor"},
]

# ThingsBoard config — set TB_TOKEN_<NAME> env vars or hard-code below
TB_HOST = os.environ.get("TB_HOST", "thingsboard.cloud")
TB_DEVICE_TOKENS: dict[str, str] = {
    "Netatmo Weather Station": os.environ.get("TB_TOKEN_NETATMO",    "YOUR_TOKEN_1"),
    "Lumiman Bulb":            os.environ.get("TB_TOKEN_LUMIMAN",    "YOUR_TOKEN_2"),
    "LIFX Lightbulb":          os.environ.get("TB_TOKEN_LIFX",       "YOUR_TOKEN_3"),
    "Levoit Air Purifier":     os.environ.get("TB_TOKEN_LEVOIT",     "YOUR_TOKEN_4"),
    "Govee Smart Humidifier":  os.environ.get("TB_TOKEN_GOVEE",      "YOUR_TOKEN_5"),
}

# ---------------------------------------------------------------------------
# Feature generators — realistic IoT traffic profiles
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=None)   # non-deterministic each run

FEATURE_NAMES = [
    "Protocol","Flow Duration","Total Fwd Packet","Total Bwd packets",
    "Total Length of Fwd Packet","Total Length of Bwd Packet",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
    "Packet Length Min","Packet Length Max","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
    "CWR Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size",
    "Fwd Segment Size Avg","Bwd Segment Size Avg",
    "Bwd Bytes/Bulk Avg","Bwd Packet/Bulk Avg","Bwd Bulk Rate Avg",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "FWD Init Win Bytes","Bwd Init Win Bytes","Fwd Act Data Pkts","Fwd Seg Size Min",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
]

assert len(FEATURE_NAMES) == 70, f"Expected 70 features, got {len(FEATURE_NAMES)}"


def _benign_flow(group: str) -> dict[str, float]:
    """Generate a realistic benign IoT flow for G1 or G2 device."""
    r = RNG

    if group == "G1":
        # High-activity multi-peer: larger, more frequent flows
        proto = r.choice([6, 17], p=[0.6, 0.4])          # TCP or UDP
        duration = float(r.exponential(50_000_000))        # ~50 ms mean
        fwd_pkts = int(r.integers(5, 60))
        bwd_pkts = int(r.integers(5, 40))
        fwd_len_mean = float(r.uniform(200, 1400))
        bwd_len_mean = float(r.uniform(100, 800))
    else:
        # G2: low-activity simple; shorter, smaller flows
        proto = r.choice([6, 17], p=[0.4, 0.6])
        duration = float(r.exponential(10_000_000))        # ~10 ms mean
        fwd_pkts = int(r.integers(1, 10))
        bwd_pkts = int(r.integers(1, 8))
        fwd_len_mean = float(r.uniform(40, 300))
        bwd_len_mean = float(r.uniform(40, 200))

    fwd_len_std = float(r.uniform(0, fwd_len_mean * 0.3))
    bwd_len_std = float(r.uniform(0, bwd_len_mean * 0.3))
    total_fwd = fwd_pkts * fwd_len_mean
    total_bwd = bwd_pkts * bwd_len_mean
    total_bytes = total_fwd + total_bwd
    flow_bytes_s = total_bytes / max(duration / 1e6, 1e-6)
    total_pkts = fwd_pkts + bwd_pkts
    flow_pkts_s = total_pkts / max(duration / 1e6, 1e-6)
    iat_mean = duration / max(total_pkts - 1, 1)

    return {
        "Protocol": float(proto),
        "Flow Duration": duration,
        "Total Fwd Packet": float(fwd_pkts),
        "Total Bwd packets": float(bwd_pkts),
        "Total Length of Fwd Packet": total_fwd,
        "Total Length of Bwd Packet": total_bwd,
        "Fwd Packet Length Max": fwd_len_mean + fwd_len_std * 2,
        "Fwd Packet Length Min": max(0, fwd_len_mean - fwd_len_std * 2),
        "Fwd Packet Length Mean": fwd_len_mean,
        "Fwd Packet Length Std": fwd_len_std,
        "Bwd Packet Length Max": bwd_len_mean + bwd_len_std * 2,
        "Bwd Packet Length Min": max(0, bwd_len_mean - bwd_len_std * 2),
        "Bwd Packet Length Mean": bwd_len_mean,
        "Bwd Packet Length Std": bwd_len_std,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": iat_mean,
        "Flow IAT Std": iat_mean * 0.3,
        "Flow IAT Max": iat_mean * 3,
        "Flow IAT Min": iat_mean * 0.1,
        "Fwd IAT Total": duration * 0.6,
        "Fwd IAT Mean": iat_mean,
        "Fwd IAT Std": iat_mean * 0.2,
        "Fwd IAT Max": iat_mean * 2.5,
        "Fwd IAT Min": iat_mean * 0.1,
        "Bwd IAT Total": duration * 0.4,
        "Bwd IAT Mean": iat_mean * 0.8,
        "Bwd IAT Std": iat_mean * 0.2,
        "Bwd IAT Max": iat_mean * 2,
        "Bwd IAT Min": iat_mean * 0.1,
        "Fwd PSH Flags": float(r.integers(0, 2)),
        "Fwd Header Length": float(fwd_pkts * 20),
        "Bwd Header Length": float(bwd_pkts * 20),
        "Fwd Packets/s": fwd_pkts / max(duration / 1e6, 1e-6),
        "Bwd Packets/s": bwd_pkts / max(duration / 1e6, 1e-6),
        "Packet Length Min": max(0, min(fwd_len_mean, bwd_len_mean) - 20),
        "Packet Length Max": max(fwd_len_mean, bwd_len_mean) + 50,
        "Packet Length Mean": (fwd_len_mean + bwd_len_mean) / 2,
        "Packet Length Std": (fwd_len_std + bwd_len_std) / 2,
        "Packet Length Variance": ((fwd_len_std + bwd_len_std) / 2) ** 2,
        "FIN Flag Count": float(r.integers(0, 2)),
        "SYN Flag Count": float(r.integers(0, 2)),
        "RST Flag Count": 0.0,
        "PSH Flag Count": float(r.integers(0, 3)),
        "ACK Flag Count": float(r.integers(1, fwd_pkts + bwd_pkts)),
        "CWR Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": bwd_pkts / max(fwd_pkts, 1),
        "Average Packet Size": (fwd_len_mean + bwd_len_mean) / 2,
        "Fwd Segment Size Avg": fwd_len_mean,
        "Bwd Segment Size Avg": bwd_len_mean,
        "Bwd Bytes/Bulk Avg": total_bwd * 0.1,
        "Bwd Packet/Bulk Avg": float(bwd_pkts) * 0.1,
        "Bwd Bulk Rate Avg": flow_bytes_s * 0.1,
        "Subflow Fwd Packets": float(fwd_pkts),
        "Subflow Fwd Bytes": total_fwd,
        "Subflow Bwd Packets": float(bwd_pkts),
        "Subflow Bwd Bytes": total_bwd,
        "FWD Init Win Bytes": float(r.choice([512, 1024, 2048, 4096, 8192, 16384, 32768, 65535])),
        "Bwd Init Win Bytes": float(r.choice([512, 1024, 2048, 4096, 8192])),
        "Fwd Act Data Pkts": float(fwd_pkts - 1),
        "Fwd Seg Size Min": 20.0,
        "Active Mean": float(r.uniform(0, duration * 0.3)),
        "Active Std": 0.0,
        "Active Max": float(r.uniform(0, duration * 0.4)),
        "Active Min": 0.0,
        "Idle Mean": float(r.uniform(0, duration * 0.5)),
        "Idle Std": 0.0,
        "Idle Max": float(r.uniform(0, duration * 0.6)),
        "Idle Min": 0.0,
    }


def _syn_flood_flow() -> dict[str, float]:
    """Generate a SYN flood attack flow signature."""
    r = RNG
    duration = float(r.uniform(100, 5_000_000))   # very short or medium
    fwd_pkts = int(r.integers(1, 5))
    bwd_pkts = 0
    pkt_size = float(r.uniform(40, 80))            # small SYN packets

    return {
        "Protocol": 6.0,  # TCP
        "Flow Duration": duration,
        "Total Fwd Packet": float(fwd_pkts),
        "Total Bwd packets": float(bwd_pkts),
        "Total Length of Fwd Packet": fwd_pkts * pkt_size,
        "Total Length of Bwd Packet": 0.0,
        "Fwd Packet Length Max": pkt_size,
        "Fwd Packet Length Min": pkt_size,
        "Fwd Packet Length Mean": pkt_size,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 0.0,
        "Bwd Packet Length Min": 0.0,
        "Bwd Packet Length Mean": 0.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": (fwd_pkts * pkt_size) / max(duration / 1e6, 1e-6),
        "Flow Packets/s": fwd_pkts / max(duration / 1e6, 1e-6),
        "Flow IAT Mean": duration / max(fwd_pkts - 1, 1),
        "Flow IAT Std": 0.0,
        "Flow IAT Max": duration,
        "Flow IAT Min": 0.0,
        "Fwd IAT Total": duration,
        "Fwd IAT Mean": duration / max(fwd_pkts - 1, 1),
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": duration,
        "Fwd IAT Min": 0.0,
        "Bwd IAT Total": 0.0,
        "Bwd IAT Mean": 0.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 0.0,
        "Bwd IAT Min": 0.0,
        "Fwd PSH Flags": 0.0,
        "Fwd Header Length": float(fwd_pkts * 20),
        "Bwd Header Length": 0.0,
        "Fwd Packets/s": fwd_pkts / max(duration / 1e6, 1e-6),
        "Bwd Packets/s": 0.0,
        "Packet Length Min": pkt_size,
        "Packet Length Max": pkt_size,
        "Packet Length Mean": pkt_size,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "FIN Flag Count": 0.0,
        "SYN Flag Count": float(fwd_pkts),   # KEY: high SYN
        "RST Flag Count": 0.0,
        "PSH Flag Count": 0.0,
        "ACK Flag Count": 0.0,
        "CWR Flag Count": float(r.integers(0, 2)),
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 0.0,
        "Average Packet Size": pkt_size,
        "Fwd Segment Size Avg": pkt_size,
        "Bwd Segment Size Avg": 0.0,
        "Bwd Bytes/Bulk Avg": 0.0,
        "Bwd Packet/Bulk Avg": 0.0,
        "Bwd Bulk Rate Avg": 0.0,
        "Subflow Fwd Packets": float(fwd_pkts),
        "Subflow Fwd Bytes": fwd_pkts * pkt_size,
        "Subflow Bwd Packets": 0.0,
        "Subflow Bwd Bytes": 0.0,
        "FWD Init Win Bytes": 512.0,   # KEY: SYN flood fingerprint
        "Bwd Init Win Bytes": 0.0,
        "Fwd Act Data Pkts": 0.0,
        "Fwd Seg Size Min": 20.0,     # KEY: no TCP options
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0,
    }


def _port_scan_flow() -> dict[str, float]:
    """Generate a port scan attack flow signature."""
    r = RNG
    duration = float(r.uniform(100, 2_000_000))
    pkt_size = float(r.uniform(40, 60))

    return {
        "Protocol": 6.0,
        "Flow Duration": duration,
        "Total Fwd Packet": 1.0,
        "Total Bwd packets": 0.0,
        "Total Length of Fwd Packet": pkt_size,
        "Total Length of Bwd Packet": 0.0,
        "Fwd Packet Length Max": pkt_size,
        "Fwd Packet Length Min": pkt_size,
        "Fwd Packet Length Mean": pkt_size,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 0.0,
        "Bwd Packet Length Min": 0.0,
        "Bwd Packet Length Mean": 0.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": pkt_size / max(duration / 1e6, 1e-6),
        "Flow Packets/s": 1.0 / max(duration / 1e6, 1e-6),
        "Flow IAT Mean": 0.0,
        "Flow IAT Std": 0.0,
        "Flow IAT Max": 0.0,
        "Flow IAT Min": 0.0,
        "Fwd IAT Total": 0.0,
        "Fwd IAT Mean": 0.0,
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": 0.0,
        "Fwd IAT Min": 0.0,
        "Bwd IAT Total": 0.0,
        "Bwd IAT Mean": 0.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 0.0,
        "Bwd IAT Min": 0.0,
        "Fwd PSH Flags": 0.0,
        "Fwd Header Length": 20.0,
        "Bwd Header Length": 0.0,
        "Fwd Packets/s": 1.0 / max(duration / 1e6, 1e-6),
        "Bwd Packets/s": 0.0,
        "Packet Length Min": pkt_size,
        "Packet Length Max": pkt_size,
        "Packet Length Mean": pkt_size,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "FIN Flag Count": 0.0,
        "SYN Flag Count": 1.0,
        "RST Flag Count": float(r.integers(0, 2)),   # KEY RST for scan
        "PSH Flag Count": 0.0,
        "ACK Flag Count": 0.0,
        "CWR Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 0.0,
        "Average Packet Size": pkt_size,
        "Fwd Segment Size Avg": pkt_size,
        "Bwd Segment Size Avg": 0.0,
        "Bwd Bytes/Bulk Avg": 0.0,
        "Bwd Packet/Bulk Avg": 0.0,
        "Bwd Bulk Rate Avg": 0.0,
        "Subflow Fwd Packets": 1.0,
        "Subflow Fwd Bytes": pkt_size,
        "Subflow Bwd Packets": 0.0,
        "Subflow Bwd Bytes": 0.0,
        "FWD Init Win Bytes": 1024.0,   # KEY port scan fingerprint
        "Bwd Init Win Bytes": 0.0,
        "Fwd Act Data Pkts": 0.0,
        "Fwd Seg Size Min": 20.0,
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0,
    }


def generate_flow(device: dict, inject_attack: bool = False) -> dict[str, Any]:
    """Generate one flow row for the given device."""
    if inject_attack:
        attack_type = random.choice(["syn_flood", "port_scan"])
        if attack_type == "syn_flood":
            features = _syn_flood_flow()
        else:
            features = _port_scan_flow()
        true_label = attack_type
    else:
        features = _benign_flow(device["group"])
        true_label = "benign"

    flow_id = str(uuid.uuid4())
    row = {"flow_id": flow_id, "device": device["name"], **features}
    return row, true_label


# ---------------------------------------------------------------------------
# Inference — call the Render API or run locally
# ---------------------------------------------------------------------------

def infer_via_api(flows: list[dict], api_url: str, operating_point: str = "T_house") -> list[dict]:
    """Call the deployed Render API endpoint."""
    url = f"{api_url.rstrip('/')}/predict/json?operating_point={operating_point}"
    resp = requests.post(url, json={"flows": flows}, timeout=30)
    resp.raise_for_status()
    return resp.json()["predictions"]


def infer_locally(flows: list[dict], operating_point: str = "T_house") -> list[dict]:
    """Run inference in-process (no network call)."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lgids_v1.predict import V1Predictor
    predictor = V1Predictor()
    df = pd.DataFrame(flows)
    result = predictor.predict(df, operating_point=operating_point)
    return result.to_dict(orient="records")


# ---------------------------------------------------------------------------
# ThingsBoard telemetry sender
# ---------------------------------------------------------------------------

def send_to_thingsboard(device_name: str, telemetry: dict) -> bool:
    """POST telemetry to ThingsBoard HTTP API."""
    token = TB_DEVICE_TOKENS.get(device_name)
    if not token or token.startswith("YOUR_TOKEN"):
        return False   # skip if not configured

    url = f"https://{TB_HOST}/api/v1/{token}/telemetry"
    try:
        resp = requests.post(url, json=telemetry, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  ⚠ ThingsBoard send failed for {device_name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(api_url: str | None, interval: int, attack_rate: float, local: bool):
    print("=" * 60)
    print("LGIDS V1 — IoT Simulator")
    print(f"  Mode    : {'LOCAL' if local else f'API → {api_url}'}")
    print(f"  Devices : {len(DEVICES)}")
    print(f"  Interval: {interval}s | Attack rate: {attack_rate*100:.0f}%")
    print("=" * 60)

    tick = 0
    while True:
        tick += 1
        print(f"\n── Tick #{tick} ──────────────────────────────")

        flows = []
        true_labels = []
        for device in DEVICES:
            inject = (random.random() < attack_rate)
            flow, label = generate_flow(device, inject_attack=inject)
            flows.append(flow)
            true_labels.append(label)

        # Inference
        try:
            if local:
                predictions = infer_locally(flows)
            else:
                predictions = infer_via_api(flows, api_url)
        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(interval)
            continue

        # Print + send to ThingsBoard
        for pred, true_label, device in zip(predictions, true_labels, DEVICES):
            status_icon = "🔴 ATTACK" if pred["pred"] == "attack" else "🟢 benign"
            match_icon  = "✓" if pred["pred"] == true_label or true_label == "benign" else "✗"
            print(
                f"  {device['name']:<28} [{device['group']}]  "
                f"p={pred['p_attack']:.4f}  {status_icon}  {match_icon}"
            )

            telemetry = {
                "p_attack":        round(pred["p_attack"], 6),
                "prediction":      pred["pred"],
                "group":           pred["group"],
                "model_used":      pred["model_used"],
                "threshold":       pred["threshold"],
                "true_label":      true_label,
                "is_attack":       1 if pred["pred"] == "attack" else 0,
                "device_category": device["category"],
            }
            sent = send_to_thingsboard(device["name"], telemetry)
            if sent:
                print(f"    ↑ Sent to ThingsBoard ✓")

        time.sleep(interval)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LGIDS V1 IoT Simulator")
    parser.add_argument("--api-url", default=None,
                        help="Render API base URL (e.g. https://lgids-v1.onrender.com). "
                             "Omit to run inference locally.")
    parser.add_argument("--interval", type=int, default=10,
                        help="Seconds between simulation ticks (default: 10)")
    parser.add_argument("--attack-rate", type=float, default=0.2,
                        help="Fraction of flows that are injected attacks (default: 0.2)")
    parser.add_argument("--local", action="store_true",
                        help="Force local inference even if --api-url is set")
    args = parser.parse_args()

    local_mode = args.local or (args.api_url is None)
    run(api_url=args.api_url, interval=args.interval,
        attack_rate=args.attack_rate, local=local_mode)
