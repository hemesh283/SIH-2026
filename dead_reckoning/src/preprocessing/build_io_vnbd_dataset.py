"""Builds processed IO-VNBD datasets with native 10 Hz sampling and 20-sample windows.

Pipeline:
1. Load synchronized smartphone (S-*.csv) and vehicle ECU (V-*.csv) files.
2. Extract 6 features [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] with explicit physical mapping:
     acc = (ACCELEROMETER - GRAVITY) / 9.80665 (linear acceleration in g)
     gyro_x = GYROSCOPE Pitch (rad/s) (aligned with vehicle turning/yaw rate)
     gyro_y = GYROSCOPE Roll (rad/s) (aligned with vehicle roll axis)
     gyro_z = GYROSCOPE Yaw (rad/s) (aligned with vehicle pitch axis)
3. Generate 20-sample windows (2.0s duration at native 10 Hz) with 10-sample stride (1.0s).
4. Construct local displacement targets [dx, dy, dz] in meters from vehicle ECU reference GPS:
     R_start.T @ delta_ned
5. Partition by Driver/Sequence into strictly leakage-free splits:
     TRAIN: 53 sequences
     VAL:   10 sequences
     TEST:   9 sequences
6. Compute normalization statistics ONLY on the TRAIN split.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/dead_reckoning")
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "IO-VNBD"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 20  # 20 samples @ 10 Hz = 2.0 seconds
STRIDE = 10       # 10 samples @ 10 Hz = 1.0 second update
MEAN_EARTH_RADIUS = 6371000.0


def heading_to_dcm(heading_deg: float) -> np.ndarray:
    """Direction Cosine Matrix rotating body vector to NED: v_ned = R_bn @ v_b."""
    psi = np.deg2rad(heading_deg)
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def extract_features_and_targets(
    s_csv_path: Path,
    v_csv_path: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Process one synchronized sequence into windows X and local displacement targets y."""
    df_s = pd.read_csv(s_csv_path, encoding="latin-1")
    df_v = pd.read_csv(v_csv_path, encoding="latin-1")

    n = min(len(df_s), len(df_v))
    if n < window_size:
        return np.zeros((0, window_size, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), pd.DataFrame()

    # 1. Smartphone Features
    # Linear acceleration: (ACCEL - GRAVITY) / 9.80665 in g
    ax_col = [c for c in df_s.columns if "accel" in c.lower() and "x" in c.lower()][0]
    ay_col = [c for c in df_s.columns if "accel" in c.lower() and "y" in c.lower()][0]
    az_col = [c for c in df_s.columns if "accel" in c.lower() and "z" in c.lower()][0]

    gx_col = [c for c in df_s.columns if "gravity" in c.lower() and "x" in c.lower()][0]
    gy_col = [c for c in df_s.columns if "gravity" in c.lower() and "y" in c.lower()][0]
    gz_col = [c for c in df_s.columns if "gravity" in c.lower() and "z" in c.lower()][0]

    ax_raw = pd.to_numeric(df_s[ax_col], errors="coerce").fillna(0.0).values[:n]
    ay_raw = pd.to_numeric(df_s[ay_col], errors="coerce").fillna(0.0).values[:n]
    az_raw = pd.to_numeric(df_s[az_col], errors="coerce").fillna(0.0).values[:n]

    gx_raw = pd.to_numeric(df_s[gx_col], errors="coerce").fillna(0.0).values[:n]
    gy_raw = pd.to_numeric(df_s[gy_col], errors="coerce").fillna(0.0).values[:n]
    gz_raw = pd.to_numeric(df_s[gz_col], errors="coerce").fillna(9.81).values[:n]

    lin_ax = (ax_raw - gx_raw) / 9.80665
    lin_ay = (ay_raw - gy_raw) / 9.80665
    lin_az = (az_raw - gz_raw) / 9.80665

    # Gyroscope channels (with verified Pitch=turning yaw rate mapping)
    # Col 1: GYROSCOPE Pitch (rad/s) (vehicle turning yaw rate)
    # Col 2: GYROSCOPE Roll (rad/s)  (vehicle roll axis)
    # Col 3: GYROSCOPE Yaw (rad/s)   (vehicle transverse pitch axis)
    g_pitch_col = [c for c in df_s.columns if "gyro" in c.lower() and "pitch" in c.lower()][0]
    g_roll_col = [c for c in df_s.columns if "gyro" in c.lower() and "roll" in c.lower()][0]
    g_yaw_col = [c for c in df_s.columns if "gyro" in c.lower() and "yaw" in c.lower()][0]

    gyro_1 = pd.to_numeric(df_s[g_pitch_col], errors="coerce").fillna(0.0).values[:n]
    gyro_2 = pd.to_numeric(df_s[g_roll_col], errors="coerce").fillna(0.0).values[:n]
    gyro_3 = pd.to_numeric(df_s[g_yaw_col], errors="coerce").fillna(0.0).values[:n]

    features = np.stack([lin_ax, lin_ay, lin_az, gyro_1, gyro_2, gyro_3], axis=1).astype(np.float32)

    # 2. Vehicle Reference Ground Truth
    lat_col = [c for c in df_v.columns if "lat" in c.lower()][0]
    lon_col = [c for c in df_v.columns if "lon" in c.lower()][0]
    hdg_col = [c for c in df_v.columns if "head" in c.lower() or "bearing" in c.lower()][0]
    spd_col = [c for c in df_v.columns if "velocity" in c.lower() or "speed" in c.lower()][0]

    v_lat = pd.to_numeric(df_v[lat_col], errors="coerce").ffill().bfill().values[:n]
    v_lon = pd.to_numeric(df_v[lon_col], errors="coerce").ffill().bfill().values[:n]
    v_hdg = pd.to_numeric(df_v[hdg_col], errors="coerce").ffill().bfill().values[:n]
    v_spd_kmh = pd.to_numeric(df_v[spd_col], errors="coerce").ffill().bfill().values[:n]

    # 3. Create Windows and Local Targets
    starts = range(0, n - window_size + 1, stride)
    windows_X = []
    targets_y = []
    metadata = []

    for s_idx in starts:
        e_idx = s_idx + window_size - 1
        w_feat = features[s_idx : s_idx + window_size]

        # Start and end geodetic coordinates
        lat_start, lon_start = v_lat[s_idx], v_lon[s_idx]
        lat_end, lon_end = v_lat[e_idx], v_lon[e_idx]
        hdg_start = v_hdg[s_idx]

        # Metric displacement in NED frame
        d_lat_rad = np.deg2rad(lat_end - lat_start)
        d_lon_rad = np.deg2rad(lon_end - lon_start)
        lat_mean_rad = np.deg2rad(lat_start)

        dp_north = d_lat_rad * MEAN_EARTH_RADIUS
        dp_east = d_lon_rad * MEAN_EARTH_RADIUS * np.cos(lat_mean_rad)
        dp_down = 0.0
        dp_ned = np.array([dp_north, dp_east, dp_down], dtype=np.float64)

        # Rotate into local initial frame: dx = R_start.T @ dp_ned
        R_start = heading_to_dcm(hdg_start)
        dp_local = R_start.T @ dp_ned

        # Filter out extreme GPS jumps (> 150m in 2s = > 270 km/h)
        if np.linalg.norm(dp_local) > 150.0:
            continue

        windows_X.append(w_feat)
        targets_y.append(dp_local.astype(np.float32))

        metadata.append({
            "start_sample": s_idx,
            "end_sample": e_idx,
            "start_lat": lat_start,
            "start_lon": lon_start,
            "start_hdg": hdg_start,
            "dx_m": dp_local[0],
            "dy_m": dp_local[1],
            "dz_m": dp_local[2],
            "vehicle_speed_mps": v_spd_kmh[s_idx] / 3.6,
        })

    X_arr = np.array(windows_X, dtype=np.float32)
    y_arr = np.array(targets_y, dtype=np.float32)
    df_meta = pd.DataFrame(metadata)

    return X_arr, y_arr, df_meta


def partition_sequences() -> Dict[str, List[Tuple[Path, Path, str]]]:
    """Define deterministic, leakage-free sequence assignment."""
    sync_csv = RESULTS_DIR / "synchronization_audit.csv"
    if not sync_csv.is_file():
        raise FileNotFoundError(f"{sync_csv} not found. Run audit script first.")

    df_sync = pd.read_csv(sync_csv)
    print(f"Total synchronized pairs available: {len(df_sync)}")

    splits: Dict[str, List[Tuple[Path, Path, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    manifest_rows = []

    for _, row in df_sync.iterrows():
        key = row["sequence_key"].lower()
        s_file = row["smartphone_file"]
        v_file = row["vehicle_file"]
        cat = row["category"]

        s_path = list(DATA_DIR.rglob(s_file))[0]
        v_path = list(DATA_DIR.rglob(v_file))[0]

        # Explicit Driver Partition:
        # TEST: Driver B ('m'), Driver D ('y1'), Driver E last 7 runs ('vw14c', 'vw15', 'vw16a', 'vw16b', 'vw17', 'vfa01', 'vfa02')
        # VAL:  Driver A ('s1', 's2', 's3a', 's3b', 's3c', 's4'), Driver E 4 runs ('vw12', 'vw13', 'vw14a', 'vw14b')
        # TRAIN: Driver C ('vta1' to 'vta30', 'vtb1' to 'vtb12') + Driver E first 11 runs ('vw1' to 'vw11')
        if key in ["m", "y1", "vw14c", "vw15", "vw16a", "vw16b", "vw17", "vfa01", "vfa02"]:
            split_name = "test"
        elif key in ["s1", "s2", "s3a", "s3b", "s3c", "s4", "vw12", "vw13", "vw14a", "vw14b"]:
            split_name = "val"
        else:
            split_name = "train"

        splits[split_name].append((s_path, v_path, key))

        manifest_rows.append({
            "sequence_key": key,
            "smartphone_file": s_file,
            "vehicle_file": v_file,
            "assigned_split": split_name,
            "category": cat,
            "duration_s": row["smartphone_duration_s"],
        })

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = RESULTS_DIR / "split_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Saved {manifest_csv}")
    print("Sequence counts per split:")
    print(df_manifest["assigned_split"].value_counts())

    return splits


def build_datasets():
    print("=" * 75)
    print("BUILDING REAL IO-VNBD PROCESSED DATASETS (10 Hz, 20 SAMPLES)")
    print("=" * 75)

    splits = partition_sequences()

    processed_data = {}

    for split_name, seq_list in splits.items():
        print(f"\nProcessing {split_name.upper()} split ({len(seq_list)} sequences)...")
        all_X = []
        all_y = []

        for s_path, v_path, key in seq_list:
            X, y, _ = extract_features_and_targets(s_path, v_path)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if all_X:
            X_concat = np.concatenate(all_X, axis=0)
            y_concat = np.concatenate(all_y, axis=0)
        else:
            X_concat = np.zeros((0, WINDOW_SIZE, 6), dtype=np.float32)
            y_concat = np.zeros((0, 3), dtype=np.float32)

        out_path = PROCESSED_DIR / f"io_vnbd_{split_name}_local.npz"
        np.savez_compressed(out_path, X=X_concat, y=y_concat)
        print(f"  Saved {out_path}: X shape {X_concat.shape}, y shape {y_concat.shape}")
        processed_data[split_name] = (X_concat, y_concat)

    # Compute normalization parameters strictly from TRAIN split
    print("\nComputing Normalization Parameters strictly from TRAIN split...")
    X_train, y_train = processed_data["train"]

    # Flatten windows for feature statistics: (N * 20, 6)
    features_flat = X_train.reshape(-1, 6)
    X_mean = np.mean(features_flat, axis=0).astype(float)
    X_std = np.std(features_flat, axis=0).astype(float)

    # Protect against zero std
    X_std = np.where(X_std < 1e-4, 1.0, X_std)

    # Target statistics
    y_mean = np.mean(y_train, axis=0).astype(float)
    y_std = np.std(y_train, axis=0).astype(float)
    y_std = np.where(y_std < 1e-4, 1.0, y_std)

    norm_dict = {
        "dataset": "IO-VNBD Native 10Hz",
        "window_size": WINDOW_SIZE,
        "feature_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "feature_units": ["g", "g", "g", "rad/s", "rad/s", "rad/s"],
        "mean": X_mean.tolist(),
        "std": X_std.tolist(),
        "target_names": ["dx", "dy", "dz"],
        "target_units": ["m", "m", "m"],
        "target_mean": y_mean.tolist(),
        "target_std": y_std.tolist(),
        "training_windows_count": int(len(X_train)),
        "notes": "Computed strictly on train split without validation/test data leakage."
    }

    norm_json = MODELS_DIR / "io_vnbd_normalization.json"
    with open(norm_json, "w", encoding="utf-8") as f:
        json.dump(norm_dict, f, indent=2)
    print(f"Saved normalization statistics to: {norm_json}")

    print("\nTRAIN Target Displacements [dx, dy, dz]:")
    print(f"  Mean (m): {y_mean}")
    print(f"  Std  (m): {y_std}")
    print(f"  Mean 2s displacement magnitude: {np.mean(np.linalg.norm(y_train, axis=1)):.2f} meters")
    print(f"  Max  2s displacement magnitude: {np.max(np.linalg.norm(y_train, axis=1)):.2f} meters")


if __name__ == "__main__":
    build_datasets()
