"""Reproducible Real Vehicle-Domain GNSS Outage Benchmark for IO-VNBD.

Evaluates the complete 7-tier navigation ladder on real held-out IO-VNBD driving data:
  1. Pure INS (Kinematic double-integration)
  2. INS + EKF (6-state filter)
  3. ML Only (Native 10 Hz GRU displacement)
  4. ML + INS (Blended kinematics)
  5. ML + INS + EKF (Filter fusion)
  6. ML + INS + EKF + NHC (Non-holonomic velocity constraints)
  7. ML + INS + EKF + NHC + ZUPT (Zero-velocity updates)

Simulates deterministic GNSS blackouts: 10s, 30s, 60s, 120s.
Ground truth: Vehicle ECU reference trajectory (CAN bus / dual-antenna GPS).
All results explicitly labeled: REAL IO-VNBD.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/dead_reckoning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import NavigationMetrics, OutageEvaluationResult
from src.evaluation.outage_simulator import OutageSimulator, OutageInterval
from src.inference.gru_inference import DeadReckoningInference
from src.navigation.coordinate_frames import CoordinateTransformer, MEAN_EARTH_RADIUS
from src.navigation.ekf import NavigationEKF
from src.navigation.nhc import NhcConstraint
from src.navigation.zupt import ZuptDetector

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "IO-VNBD"
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"
MODELS_DIR = PROJECT_ROOT / "models"


def ned_to_geodetic_vec(pn: np.ndarray, pe: np.ndarray, origin_lat: float, origin_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert metric NED displacement arrays to WGS-84 latitude and longitude."""
    lat_rad = np.deg2rad(origin_lat)
    d_lat_deg = np.rad2deg(pn / MEAN_EARTH_RADIUS)
    d_lon_deg = np.rad2deg(pe / (MEAN_EARTH_RADIUS * np.cos(lat_rad)))
    return origin_lat + d_lat_deg, origin_lon + d_lon_deg


def geodetic_to_ned_vec(lat: np.ndarray, lon: np.ndarray, origin_lat: float, origin_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WGS-84 geodetic coordinate arrays to metric NED displacement."""
    lat_rad = np.deg2rad(origin_lat)
    pn = np.deg2rad(lat - origin_lat) * MEAN_EARTH_RADIUS
    pe = np.deg2rad(lon - origin_lon) * (MEAN_EARTH_RADIUS * np.cos(lat_rad))
    return pn, pe


class RealIOVNBDSequence:
    """Holds synchronized 10 Hz smartphone sensors and vehicle ECU ground truth."""

    def __init__(self, s_csv_path: Path, v_csv_path: Path):
        self.s_path = s_csv_path
        self.v_path = v_csv_path
        self._load()

    def _load(self):
        df_s = pd.read_csv(self.s_path, encoding="latin-1")
        df_v = pd.read_csv(self.v_path, encoding="latin-1")
        n = min(len(df_s), len(df_v))

        # Time
        self.n_samples = n
        self.dt = 0.1  # Native 10 Hz
        self.timestamps_s = np.arange(n) * self.dt

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

        # Gyroscope channels (Pitch=turning yaw rate mapping)
        g_pitch_col = [c for c in df_s.columns if "gyro" in c.lower() and "pitch" in c.lower()][0]
        g_roll_col = [c for c in df_s.columns if "gyro" in c.lower() and "roll" in c.lower()][0]
        g_yaw_col = [c for c in df_s.columns if "gyro" in c.lower() and "yaw" in c.lower()][0]

        gyro_pitch = pd.to_numeric(df_s[g_pitch_col], errors="coerce").fillna(0.0).values[:n]
        gyro_roll = pd.to_numeric(df_s[g_roll_col], errors="coerce").fillna(0.0).values[:n]
        gyro_yaw = pd.to_numeric(df_s[g_yaw_col], errors="coerce").fillna(0.0).values[:n]

        self.features = np.stack([lin_ax, lin_ay, lin_az, gyro_pitch, gyro_roll, gyro_yaw], axis=1).astype(np.float32)

        # Vehicle ECU Reference
        lat_col = [c for c in df_v.columns if "lat" in c.lower()][0]
        lon_col = [c for c in df_v.columns if "lon" in c.lower()][0]
        hdg_col = [c for c in df_v.columns if "head" in c.lower() or "bearing" in c.lower()][0]
        spd_col = [c for c in df_v.columns if "velocity" in c.lower() or "speed" in c.lower()][0]

        self.gt_lat = pd.to_numeric(df_v[lat_col], errors="coerce").ffill().bfill().values[:n]
        self.gt_lon = pd.to_numeric(df_v[lon_col], errors="coerce").ffill().bfill().values[:n]
        self.gt_hdg = pd.to_numeric(df_v[hdg_col], errors="coerce").ffill().bfill().values[:n]
        # Speed in m/s (from km/h)
        self.gt_spd = (pd.to_numeric(df_v[spd_col], errors="coerce").ffill().bfill().values[:n]) / 3.6


def precompute_ml_displacements(
    features: np.ndarray,
    infer_engine: DeadReckoningInference,
    window_size: int = 20,
    stride: int = 10,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Generate windows and run batch ML inference."""
    n = len(features)
    starts = list(range(0, n - window_size + 1, stride))
    ends = [s + window_size - 1 for s in starts]

    windows = np.zeros((len(starts), window_size, 6), dtype=np.float32)
    for i, s in enumerate(starts):
        windows[i] = features[s : s + window_size]

    batch_size = 256
    preds = []
    for i in range(0, len(windows), batch_size):
        preds.append(infer_engine.predict_batch(windows[i : i + batch_size]))

    if preds:
        displacements = np.concatenate(preds, axis=0)
    else:
        displacements = np.zeros((0, 3), dtype=np.float32)

    return displacements, starts, ends


def run_benchmark():
    parser = argparse.ArgumentParser(description="Real IO-VNBD GNSS Outage Benchmark")
    parser.add_argument("--sequence", type=str, default="vfa01", help="Test sequence key (vfa01, y1, m)")
    parser.add_argument("--outages", nargs="+", type=float, default=[10.0, 30.0, 60.0, 120.0], help="Outage durations (s)")
    args = parser.parse_args()

    print("=" * 75)
    print("REAL IO-VNBD VEHICLE DEAD RECKONING BENCHMARK (NATIVE 10 Hz)")
    print("=" * 75)

    # 1. Locate Sequence Files
    key = args.sequence.lower()
    s_files = list(RAW_DATA_DIR.rglob(f"S-*{key.upper()}*.csv")) or list(RAW_DATA_DIR.rglob(f"S-*{key}*.csv"))
    v_files = list(RAW_DATA_DIR.rglob(f"V-*{key.upper()}*.csv")) or list(RAW_DATA_DIR.rglob(f"V-*{key}*.csv"))

    # Precise match for vfa01
    if key == "vfa01":
        s_files = [p for p in RAW_DATA_DIR.rglob("S-Vfa01.csv")]
        v_files = [p for p in RAW_DATA_DIR.rglob("V-Vfa01.csv")]
    elif key == "y1":
        s_files = [p for p in RAW_DATA_DIR.rglob("S-Y1.csv")]
        v_files = [p for p in RAW_DATA_DIR.rglob("V-Y1.csv")]
    elif key == "m":
        s_files = [p for p in RAW_DATA_DIR.rglob("S-M.csv")]
        v_files = [p for p in RAW_DATA_DIR.rglob("V-M.csv")]

    if not s_files or not v_files:
        raise FileNotFoundError(f"Could not find matching real data files for sequence '{key}'")

    s_path, v_path = s_files[0], v_files[0]
    print(f"Loading REAL IO-VNBD Sequence: {s_path.name} & {v_path.name}")
    seq = RealIOVNBDSequence(s_path, v_path)
    print(f"Loaded {seq.n_samples:,} samples at native 10 Hz ({seq.n_samples * 0.1:.1f} seconds / {seq.n_samples * 0.1 / 60:.1f} min)")
    print(f"Ground-truth speed: mean = {seq.gt_spd.mean() * 3.6:.1f} km/h, max = {seq.gt_spd.max() * 3.6:.1f} km/h")

    # 2. Load Native 10 Hz Inference Engine
    onnx_path = MODELS_DIR / "gru_io_vnbd.onnx"
    norm_path = MODELS_DIR / "io_vnbd_normalization.json"
    infer = DeadReckoningInference(
        model_path=onnx_path,
        normalization_path=norm_path,
        backend="onnx",
        window_size=20,
    )
    print(f"Initialized inference engine: {infer}")

    # 3. Precompute ML Displacements (20 samples window, 10 samples stride = 1.0s)
    ml_displacements, w_starts, w_ends = precompute_ml_displacements(seq.features, infer, window_size=20, stride=10)
    print(f"Precomputed {len(ml_displacements):,} ML window displacements.")

    # 4. Map ML Window Displacements to 10 Hz Per-Sample Increments
    # In a 20-sample window (2.0s duration), the model predicts [dx, dy, dz] total displacement in meters.
    # Therefore, the increment per 10 Hz sample (0.1s) is dp = [dx, dy, dz] / 20.0
    step_dp_ned = np.zeros((seq.n_samples, 3), dtype=np.float64)
    for w_idx, s_idx in enumerate(w_starts):
        e_idx = w_ends[w_idx]
        stride_len = max(1, e_idx - s_idx + 1)
        dx, dy, dz = ml_displacements[w_idx]
        hdg = float(seq.gt_hdg[s_idx])
        R_bn = CoordinateTransformer.heading_to_dcm(hdg)
        dp_ned = R_bn @ np.array([dx, dy, dz], dtype=np.float64)
        # Per-sample increment
        step_dp_ned[s_idx : e_idx + 1] = dp_ned / stride_len

    # 5. Outage Simulation
    simulator = OutageSimulator(
        outage_durations_s=args.outages,
        warmup_duration_s=60.0,
        recovery_duration_s=30.0,
        random_seed=42,
    )

    zupt_det = ZuptDetector(
        acc_dev_thresh_g=0.10,
        acc_var_thresh_g2=0.02,
        gyro_mag_thresh_rads=0.08,
        gyro_var_thresh_rads2=0.01,
        gnss_speed_thresh_mps=0.3,
        window_size=10,
    )
    nhc_mod = NhcConstraint(noise_lateral_mps=0.20, noise_vertical_mps=0.20)

    origin_lat = float(seq.gt_lat[0])
    origin_lon = float(seq.gt_lon[0])
    dt = seq.dt
    n = seq.n_samples

    # Precompute ground truth in NED frame
    gt_pn, gt_pe = geodetic_to_ned_vec(seq.gt_lat, seq.gt_lon, origin_lat, origin_lon)
    init_pn, init_pe = float(gt_pn[0]), float(gt_pe[0])
    init_hdg = float(seq.gt_hdg[0])
    init_spd = float(seq.gt_spd[0])

    all_results: List[OutageEvaluationResult] = []
    solutions_120s: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    interval_120s: OutageInterval | None = None

    for dur in args.outages:
        intervals = simulator.plan_outages(seq.timestamps_s, dur)
        if not intervals:
            print(f"Warning: Sequence too short for {dur}s outage.")
            continue
        interval = intervals[0]
        if dur == 120.0:
            interval_120s = interval

        gnss_mask = simulator.create_gnss_mask(seq.timestamps_s, interval)
        print(f"\nEvaluating {dur}s GNSS Outage (Interval: {interval.start_time_s:.1f}s - {interval.end_time_s:.1f}s, indices {interval.start_idx}:{interval.end_idx})...")

        # ------------------------------------------------------------------
        # BASELINE 1: Pure INS (Kinematic Double Integration)
        # ------------------------------------------------------------------
        pn_1 = np.zeros(n)
        pe_1 = np.zeros(n)
        vn_1 = np.zeros(n)
        ve_1 = np.zeros(n)
        for i in range(n):
            hdg = float(seq.gt_hdg[i])
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            a_body = seq.features[i, 0:3] * 9.80665
            a_ned = R_bn @ a_body

            if gnss_mask[i]:
                pn_1[i] = gt_pn[i]
                pe_1[i] = gt_pe[i]
                spd = float(seq.gt_spd[i])
                vn_1[i] = spd * np.cos(np.deg2rad(hdg))
                ve_1[i] = spd * np.sin(np.deg2rad(hdg))
            else:
                vn_1[i] = vn_1[i - 1] + a_ned[0] * dt
                ve_1[i] = ve_1[i - 1] + a_ned[1] * dt
                pn_1[i] = pn_1[i - 1] + vn_1[i] * dt
                pe_1[i] = pe_1[i - 1] + ve_1[i] * dt

        lat_1, lon_1 = ned_to_geodetic_vec(pn_1, pe_1, origin_lat, origin_lon)
        spd_1 = np.sqrt(vn_1 ** 2 + ve_1 ** 2)
        res_1 = NavigationMetrics.evaluate_outage(
            lat_1, lon_1, seq.gt_lat, seq.gt_lon,
            interval.start_idx, interval.end_idx, dur,
            "BASELINE 1: Pure INS", seq.timestamps_s, spd_1, seq.gt_spd
        )
        all_results.append(res_1)

        # ------------------------------------------------------------------
        # BASELINE 2: INS + EKF (6-State Filter Kinematic Propagation)
        # ------------------------------------------------------------------
        ekf_2 = NavigationEKF()
        ekf_2.initialize(init_pn, init_pe, 0.0, init_spd * np.cos(np.deg2rad(init_hdg)), init_spd * np.sin(np.deg2rad(init_hdg)), 0.0)

        pn_2 = np.zeros(n)
        pe_2 = np.zeros(n)
        spd_2 = np.zeros(n)
        for i in range(n):
            hdg = float(seq.gt_hdg[i])
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            a_body = seq.features[i, 0:3] * 9.80665
            a_ned = R_bn @ a_body

            # Kinematic propagation
            v_curr = ekf_2.velocity_ned
            dp = v_curr * dt + 0.5 * a_ned * (dt ** 2)
            ekf_2.predict(dp, dt)

            if gnss_mask[i]:
                ekf_2.update_gnss_position(gt_pn[i], gt_pe[i], 0.0, noise_m=2.0)
                spd = float(seq.gt_spd[i])
                ekf_2.update_gnss_velocity(spd * np.cos(np.deg2rad(hdg)), spd * np.sin(np.deg2rad(hdg)), 0.0, noise_mps=0.2)

            pos = ekf_2.position_ned
            pn_2[i] = pos[0]
            pe_2[i] = pos[1]
            vel = ekf_2.velocity_ned
            spd_2[i] = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

        lat_2, lon_2 = ned_to_geodetic_vec(pn_2, pe_2, origin_lat, origin_lon)
        res_2 = NavigationMetrics.evaluate_outage(
            lat_2, lon_2, seq.gt_lat, seq.gt_lon,
            interval.start_idx, interval.end_idx, dur,
            "BASELINE 2: INS + EKF", seq.timestamps_s, spd_2, seq.gt_spd
        )
        all_results.append(res_2)

        # ------------------------------------------------------------------
        # BASELINE 3: ML Only (Native 10 Hz GRU Displacement)
        # ------------------------------------------------------------------
        pn_3 = np.zeros(n)
        pe_3 = np.zeros(n)
        spd_3 = np.zeros(n)
        for i in range(n):
            if gnss_mask[i]:
                pn_3[i] = gt_pn[i]
                pe_3[i] = gt_pe[i]
                spd_3[i] = seq.gt_spd[i]
            else:
                pn_3[i] = pn_3[i - 1] + step_dp_ned[i, 0]
                pe_3[i] = pe_3[i - 1] + step_dp_ned[i, 1]
                spd_3[i] = float(np.sqrt(step_dp_ned[i, 0] ** 2 + step_dp_ned[i, 1] ** 2) / dt)

        lat_3, lon_3 = ned_to_geodetic_vec(pn_3, pe_3, origin_lat, origin_lon)
        res_3 = NavigationMetrics.evaluate_outage(
            lat_3, lon_3, seq.gt_lat, seq.gt_lon,
            interval.start_idx, interval.end_idx, dur,
            "BASELINE 3: ML Only", seq.timestamps_s, spd_3, seq.gt_spd
        )
        all_results.append(res_3)

        # ------------------------------------------------------------------
        # BASELINE 4: ML + INS (Blended Kinematics)
        # ------------------------------------------------------------------
        alpha = 0.85
        lat_4 = alpha * lat_3 + (1.0 - alpha) * lat_1
        lon_4 = alpha * lon_3 + (1.0 - alpha) * lon_1
        spd_4 = alpha * spd_3 + (1.0 - alpha) * spd_1
        res_4 = NavigationMetrics.evaluate_outage(
            lat_4, lon_4, seq.gt_lat, seq.gt_lon,
            interval.start_idx, interval.end_idx, dur,
            "BASELINE 4: ML + INS", seq.timestamps_s, spd_4, seq.gt_spd
        )
        all_results.append(res_4)

        # ------------------------------------------------------------------
        # BASELINES 5, 6, 7: EKF Fusion (ML + INS + EKF +/- NHC +/- ZUPT)
        # ------------------------------------------------------------------
        for b_num, use_nhc, use_zupt, b_name in [
            (5, False, False, "BASELINE 5: ML + INS + EKF"),
            (6, True,  False, "BASELINE 6: ML + INS + EKF + NHC"),
            (7, True,  True,  "BASELINE 7: ML + INS + EKF + NHC + ZUPT"),
        ]:
            ekf = NavigationEKF()
            ekf.initialize(init_pn, init_pe, 0.0, init_spd * np.cos(np.deg2rad(init_hdg)), init_spd * np.sin(np.deg2rad(init_hdg)), 0.0)

            pn_k = np.zeros(n)
            pe_k = np.zeros(n)
            spd_k = np.zeros(n)

            for i in range(n):
                hdg = float(seq.gt_hdg[i])
                dp = step_dp_ned[i]

                # Prediction step with ML displacement increment
                ekf.predict(dp, dt)

                # ZUPT update
                if use_zupt and i >= 10:
                    recent_acc = seq.features[i - 10 : i, 0:3]
                    recent_gyro = seq.features[i - 10 : i, 3:6]
                    g_spd = float(seq.gt_spd[i]) if gnss_mask[i] else None
                    if zupt_det.is_stationary(recent_acc, recent_gyro, g_spd):
                        ekf.update_zupt(noise_mps=0.05)

                # NHC update
                if use_nhc:
                    ekf.update_nhc(hdg, noise_lat_mps=0.15, noise_vert_mps=0.15)

                # GNSS measurement outside outage
                if gnss_mask[i]:
                    ekf.update_gnss_position(gt_pn[i], gt_pe[i], 0.0, noise_m=2.0)
                    spd = float(seq.gt_spd[i])
                    ekf.update_gnss_velocity(spd * np.cos(np.deg2rad(hdg)), spd * np.sin(np.deg2rad(hdg)), 0.0, noise_mps=0.2)

                pos = ekf.position_ned
                pn_k[i] = pos[0]
                pe_k[i] = pos[1]
                vel = ekf.velocity_ned
                spd_k[i] = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

            lat_k, lon_k = ned_to_geodetic_vec(pn_k, pe_k, origin_lat, origin_lon)
            res_k = NavigationMetrics.evaluate_outage(
                lat_k, lon_k, seq.gt_lat, seq.gt_lon,
                interval.start_idx, interval.end_idx, dur,
                b_name, seq.timestamps_s, spd_k, seq.gt_spd
            )
            all_results.append(res_k)

            if dur == 120.0:
                solutions_120s[b_name] = (lat_k, lon_k, spd_k)

        if dur == 120.0:
            solutions_120s["BASELINE 1: Pure INS"] = (lat_1, lon_1, spd_1)
            solutions_120s["BASELINE 2: INS + EKF"] = (lat_2, lon_2, spd_2)
            solutions_120s["BASELINE 3: ML Only"] = (lat_3, lon_3, spd_3)
            solutions_120s["BASELINE 4: ML + INS"] = (lat_4, lon_4, spd_4)

        # Print summary for this duration
        dur_results = [r for r in all_results if r.outage_duration_s == dur]
        print(f"--- Results for {dur}s Outage ---")
        for r in dur_results:
            print(f"  {r.baseline_name:<38} | RMSE: {r.position_rmse_m:6.2f}m | End: {r.endpoint_error_m:6.2f}m | Drift: {r.drift_percent_endpoint:5.2f}%")

    # ------------------------------------------------------------------
    # 6. Save Machine-Readable CSVs
    # ------------------------------------------------------------------
    rows = [asdict(r) for r in all_results]
    df_results = pd.DataFrame(rows)
    df_results.insert(0, "dataset_type", "REAL IO-VNBD")
    df_results.insert(1, "sequence_key", key)

    out_csv = RESULTS_DIR / "real_benchmark_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved real benchmark results to {out_csv}")

    # Summary table
    summary_cols = [
        "dataset_type", "sequence_key", "baseline_name", "outage_duration_s",
        "distance_travelled_m", "position_rmse_m", "position_mae_m",
        "endpoint_error_m", "max_position_error_m", "drift_percent_endpoint", "drift_percent_rmse"
    ]
    df_summary = df_results[summary_cols]
    summary_csv = RESULTS_DIR / "real_benchmark_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved real benchmark summary to {summary_csv}")

    # ------------------------------------------------------------------
    # 7. Generate Professional Visualizations (REAL IO-VNBD)
    # ------------------------------------------------------------------
    print("\nGenerating Real IO-VNBD Benchmark Figures...")

    # Plot 1: 120s Trajectory Comparison
    if interval_120s and solutions_120s:
        fig, ax = plt.subplots(figsize=(10, 8))
        s_idx, e_idx = interval_120s.start_idx, interval_120s.end_idx
        # Context padding (+/- 10s)
        pad = 100
        p_s = max(0, s_idx - pad)
        p_e = min(n, e_idx + pad)

        # Ground Truth
        ax.plot(seq.gt_lon[p_s:p_e], seq.gt_lat[p_s:p_e], "k-", linewidth=2.5, label="Ground Truth (Vehicle ECU)")
        ax.plot(seq.gt_lon[s_idx:e_idx], seq.gt_lat[s_idx:e_idx], "k--", linewidth=3.0, label="120s Outage Segment (GNSS Lost)")

        colors = {
            "BASELINE 1: Pure INS": "#E74C3C",
            "BASELINE 2: INS + EKF": "#E67E22",
            "BASELINE 3: ML Only": "#9B59B6",
            "BASELINE 4: ML + INS": "#3498DB",
            "BASELINE 5: ML + INS + EKF": "#1ABC9C",
            "BASELINE 6: ML + INS + EKF + NHC": "#F1C40F",
            "BASELINE 7: ML + INS + EKF + NHC + ZUPT": "#2ECC71",
        }

        for b_name, (b_lat, b_lon, _) in solutions_120s.items():
            ax.plot(b_lon[s_idx:e_idx], b_lat[s_idx:e_idx], label=b_name, color=colors.get(b_name, "#333333"), linewidth=1.8)

        ax.set_title(f"REAL IO-VNBD: 120s GNSS Outage Trajectory Comparison\nSequence: {key.upper()} (Driver E, Native 10 Hz)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Longitude (degrees)", fontsize=11)
        ax.set_ylabel("Latitude (degrees)", fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        traj_path = RESULTS_DIR / "real_trajectory_comparison_120s.png"
        plt.savefig(traj_path, dpi=300)
        plt.close()
        print(f"Saved {traj_path.name}")

    # Plot 2: Position Error vs Time during 120s Outage
    if interval_120s and solutions_120s:
        fig, ax = plt.subplots(figsize=(10, 5))
        s_idx, e_idx = interval_120s.start_idx, interval_120s.end_idx
        t_out = seq.timestamps_s[s_idx:e_idx] - seq.timestamps_s[s_idx]

        for b_name, (b_lat, b_lon, _) in solutions_120s.items():
            err = np.zeros(len(t_out))
            for i_pt in range(len(t_out)):
                err[i_pt] = CoordinateTransformer.haversine_distance(
                    b_lat[s_idx + i_pt], b_lon[s_idx + i_pt],
                    seq.gt_lat[s_idx + i_pt], seq.gt_lon[s_idx + i_pt]
                )
            ax.plot(t_out, err, label=b_name, color=colors.get(b_name, "#333333"), linewidth=1.8)

        ax.set_title("REAL IO-VNBD: Position Error Growth Over 120s GNSS Outage", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time in Outage (seconds)", fontsize=11)
        ax.set_ylabel("Horizontal Position Error (meters)", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        err_time_path = RESULTS_DIR / "real_position_error_vs_time_120s.png"
        plt.savefig(err_time_path, dpi=300)
        plt.close()
        print(f"Saved {err_time_path.name}")

    # Plot 3: Drift Percentage vs Outage Duration
    fig, ax = plt.subplots(figsize=(10, 5))
    for b_name in sorted(list(set(df_results["baseline_name"]))):
        sub = df_results[df_results["baseline_name"] == b_name].sort_values("outage_duration_s")
        ax.plot(sub["outage_duration_s"], sub["drift_percent_endpoint"], marker="o", label=b_name, linewidth=2.0)

    ax.axhline(10.0, color="r", linestyle="--", alpha=0.7, label="10% Drift Threshold Target")
    ax.set_title("REAL IO-VNBD: Endpoint Drift Percentage vs Outage Duration", fontsize=13, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=11)
    ax.set_ylabel("Endpoint Drift (%)", fontsize=11)
    ax.set_xticks(args.outages)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    drift_path = RESULTS_DIR / "real_drift_percent_vs_duration.png"
    plt.savefig(drift_path, dpi=300)
    plt.close()
    print(f"Saved {drift_path.name}")

    # Plot 4: Baseline Ladder Bar Comparison across 120s
    fig, ax = plt.subplots(figsize=(11, 5))
    df_120 = df_results[df_results["outage_duration_s"] == 120.0]
    if not df_120.empty:
        short_names = [name.replace("BASELINE ", "B") for name in df_120["baseline_name"]]
        x = np.arange(len(short_names))
        w = 0.35
        rects1 = ax.bar(x - w/2, df_120["position_rmse_m"], w, label="Position RMSE (m)", color="#4A90E2")
        rects2 = ax.bar(x + w/2, df_120["endpoint_error_m"], w, label="Endpoint Error (m)", color="#E67E22")

        ax.set_title("REAL IO-VNBD: 120s GNSS Outage Error Across Navigation Ladder", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Error (meters)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")

        for rects in [rects1, rects2]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}m", xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        bar_path = RESULTS_DIR / "real_baseline_comparison_120s.png"
        plt.savefig(bar_path, dpi=300)
        plt.close()
        print(f"Saved {bar_path.name}")

    # Plot 5: Position RMSE (meters) vs Outage Duration
    fig, ax = plt.subplots(figsize=(10, 5))
    for b_name in sorted(list(set(df_results["baseline_name"]))):
        sub = df_results[df_results["baseline_name"] == b_name].sort_values("outage_duration_s")
        ax.plot(sub["outage_duration_s"], sub["position_rmse_m"], marker="s", label=b_name, linewidth=2.0)

    ax.set_title("REAL IO-VNBD: Position RMSE (meters) vs GNSS Outage Duration", fontsize=13, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=11)
    ax.set_ylabel("Position RMSE (meters)", fontsize=11)
    ax.set_xticks(args.outages)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    err_dur_path = RESULTS_DIR / "real_error_vs_outage_duration.png"
    plt.savefig(err_dur_path, dpi=300)
    plt.close()
    print(f"Saved {err_dur_path.name}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
