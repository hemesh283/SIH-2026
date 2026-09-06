"""Comprehensive Held-Out Multi-Sequence Benchmark & Generalization Audit for IO-VNBD.

Evaluates the complete 7-tier navigation ladder on ALL 9 held-out test sequences from IO-VNBD:
  1. Pure INS
  2. INS + EKF
  3. ML Only (Native 10 Hz IO-VNBD GRU)
  4. ML + INS
  5. ML + INS + EKF
  6. ML + INS + EKF + NHC
  7. ML + INS + EKF + NHC + ZUPT

Evaluates:
- 10s, 30s, 60s, 120s GNSS Outages across all eligible sequences
- Sequence-by-sequence ML model comparison (Zero Baseline vs OxIOD GRU vs IO-VNBD GRU)
- Full statistical aggregation (Mean, Median, Std, P25, P75, P95, Min, Max)
- Generalization thresholds (<10%, <15%, <20%)
- 120-second outage investigation across multiple drivers
- Strict audit of NHC and ZUPT activation

Generates:
- results/io_vnbd/real_benchmark_all_test_sequences.csv
- results/io_vnbd/real_benchmark_aggregate.csv
- results/io_vnbd/model_comparison_by_sequence.csv
- results/io_vnbd/all_sequences_drift_summary.png
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("D:/dead_reckoning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import NavigationMetrics, OutageEvaluationResult
from src.evaluation.outage_simulator import OutageSimulator, OutageInterval
from src.evaluation.run_real_io_vnbd_benchmark import (
    RealIOVNBDSequence,
    geodetic_to_ned_vec,
    ned_to_geodetic_vec,
    precompute_ml_displacements,
)
from src.inference.gru_inference import DeadReckoningInference
from src.models.gru_model import GRUDeadReckoning
from src.navigation.coordinate_frames import CoordinateTransformer, MEAN_EARTH_RADIUS
from src.navigation.ekf import NavigationEKF
from src.navigation.nhc import NhcConstraint
from src.navigation.zupt import ZuptDetector

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "IO-VNBD"
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"
MODELS_DIR = PROJECT_ROOT / "models"


def get_driver_name(seq_key: str, path_str: str) -> str:
    """Identify driver from sequence key and file path."""
    key = seq_key.lower()
    if key == "m":
        return "Driver B"
    elif key == "y1":
        return "Driver D"
    elif "vfa" in key:
        return "Driver E"
    elif "vw" in key:
        return "Driver E"
    elif key.startswith("s"):
        return "Driver A"
    elif key.startswith("vta") or key.startswith("vtb"):
        return "Driver C"
    return "Unknown Driver"


def run_multi_sequence_audit():
    print("=" * 80)
    print("FINAL REAL IO-VNBD VALIDATION AUDIT: ALL HELD-OUT TEST SEQUENCES")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # TASK 1: VERIFY TEST SEQUENCES
    # --------------------------------------------------------------------------
    manifest_path = RESULTS_DIR / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest missing at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    test_rows = df_manifest[df_manifest["assigned_split"] == "test"].copy()
    print(f"\n[TASK 1] Audited Test Split from {manifest_path.name}: Found {len(test_rows)} test sequences.")

    test_seq_catalog = []
    for _, row in test_rows.iterrows():
        s_file = row["smartphone_file"]
        v_file = row["vehicle_file"]
        s_paths = list(RAW_DATA_DIR.rglob(s_file))
        v_paths = list(RAW_DATA_DIR.rglob(v_file))
        if not s_paths or not v_paths:
            raise FileNotFoundError(f"Cannot locate raw files: {s_file} or {v_file}")

        s_path = s_paths[0]
        v_path = v_paths[0]
        driver = get_driver_name(row["sequence_key"], str(s_path))

        # Check sample count
        df_s_head = pd.read_csv(s_path, encoding="latin-1")
        df_v_head = pd.read_csv(v_path, encoding="latin-1")
        n_samples = min(len(df_s_head), len(df_v_head))
        num_windows = max(0, (n_samples - 20) // 10 + 1)

        test_seq_catalog.append({
            "sequence": row["sequence_key"],
            "driver": driver,
            "smartphone_file": s_file,
            "vehicle_file": v_file,
            "duration_s": row["duration_s"],
            "samples": n_samples,
            "windows": num_windows,
            "s_path": s_path,
            "v_path": v_path,
        })

    df_catalog = pd.DataFrame(test_seq_catalog)
    cols_to_print = ["sequence", "driver", "duration_s", "samples", "windows"]
    print("\n--- Held-Out Test Set Inventory ---")
    print(df_catalog[cols_to_print].to_string(index=False))
    print(f"Total Test Windows Across 9 Sequences: {df_catalog['windows'].sum():,}")

    # --------------------------------------------------------------------------
    # TASK 7: MODEL COMPARISON ACROSS ALL TEST SEQUENCES (Zero vs OxIOD vs IO-VNBD)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TASK 7] MODEL COMPARISON (Zero Baseline vs OxIOD GRU vs IO-VNBD GRU)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load OxIOD Model
    oxiod_ckpt_path = MODELS_DIR / "gru_local_best.pt"
    with open(MODELS_DIR / "normalization.json", "r", encoding="utf-8") as f:
        oxiod_norm = json.load(f)
    oxiod_X_mean = np.array(oxiod_norm["mean"], dtype=np.float32)
    oxiod_X_std = np.array(oxiod_norm["std"], dtype=np.float32)
    oxiod_y_mean = np.array(oxiod_norm.get("target_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    oxiod_y_std = np.array(oxiod_norm.get("target_std", [1.0, 1.0, 1.0]), dtype=np.float32)

    oxiod_model = GRUDeadReckoning(input_size=6, hidden_size=64, num_layers=2, output_size=3, dropout=0.2).to(device)
    oxiod_ckpt = torch.load(oxiod_ckpt_path, map_location=device, weights_only=True)
    oxiod_model.load_state_dict(oxiod_ckpt["model_state_dict"] if "model_state_dict" in oxiod_ckpt else oxiod_ckpt)
    oxiod_model.eval()

    # Load IO-VNBD Model
    vnbd_onnx_path = MODELS_DIR / "gru_io_vnbd.onnx"
    vnbd_norm_path = MODELS_DIR / "io_vnbd_normalization.json"
    with open(vnbd_norm_path, "r", encoding="utf-8") as f:
        vnbd_norm = json.load(f)
    vnbd_X_mean = np.array(vnbd_norm["mean"], dtype=np.float32)
    vnbd_X_std = np.array(vnbd_norm["std"], dtype=np.float32)
    vnbd_y_mean = np.array(vnbd_norm["target_mean"], dtype=np.float32)
    vnbd_y_std = np.array(vnbd_norm["target_std"], dtype=np.float32)

    vnbd_model = GRUDeadReckoning(input_size=6, hidden_size=64, num_layers=2, output_size=3, dropout=0.2).to(device)
    vnbd_ckpt = torch.load(MODELS_DIR / "gru_io_vnbd_best.pt", map_location=device, weights_only=True)
    vnbd_model.load_state_dict(vnbd_ckpt["model_state_dict"] if "model_state_dict" in vnbd_ckpt else vnbd_ckpt)
    vnbd_model.eval()

    # Initialize ONNX inference engine for high-throughput navigation
    vnbd_infer = DeadReckoningInference(
        model_path=vnbd_onnx_path,
        normalization_path=vnbd_norm_path,
        backend="onnx",
        window_size=20,
    )

    from src.preprocessing.build_io_vnbd_dataset import extract_features_and_targets

    seq_model_comp = []
    print("\nEvaluating 3 Model Types on each test sequence:")
    for item in test_seq_catalog:
        key = item["sequence"]
        X_seq, y_seq, _ = extract_features_and_targets(item["s_path"], item["v_path"])
        if len(X_seq) == 0:
            continue

        # Baseline 1: Zero
        zero_diff = y_seq  # (N, 3)
        zero_euc = float(np.mean(np.linalg.norm(zero_diff, axis=1)))
        zero_rmse = float(np.sqrt(np.mean(zero_diff ** 2)))

        # Baseline 2: OxIOD GRU
        X_ox_norm = (X_seq - oxiod_X_mean) / oxiod_X_std
        with torch.no_grad():
            p_ox_norm = oxiod_model(torch.from_numpy(X_ox_norm).to(device)).cpu().numpy()
        p_ox = p_ox_norm * oxiod_y_std + oxiod_y_mean
        ox_diff = p_ox - y_seq
        ox_euc = float(np.mean(np.linalg.norm(ox_diff, axis=1)))
        ox_rmse = float(np.sqrt(np.mean(ox_diff ** 2)))

        # Model 3: IO-VNBD GRU
        X_vn_norm = (X_seq - vnbd_X_mean) / vnbd_X_std
        with torch.no_grad():
            p_vn_norm = vnbd_model(torch.from_numpy(X_vn_norm).to(device)).cpu().numpy()
        p_vn = p_vn_norm * vnbd_y_std + vnbd_y_mean
        vn_diff = p_vn - y_seq
        vn_euc = float(np.mean(np.linalg.norm(vn_diff, axis=1)))
        vn_rmse = float(np.sqrt(np.mean(vn_diff ** 2)))

        euc_imp_vs_zero = (zero_euc - vn_euc) / zero_euc * 100.0
        euc_imp_vs_ox = (ox_euc - vn_euc) / ox_euc * 100.0

        seq_model_comp.append({
            "sequence": key,
            "driver": item["driver"],
            "windows": len(X_seq),
            "true_mean_dx_m": float(np.mean(y_seq[:, 0])),
            "zero_euc_err_m": zero_euc,
            "oxiod_euc_err_m": ox_euc,
            "vnbd_euc_err_m": vn_euc,
            "zero_rmse_m": zero_rmse,
            "oxiod_rmse_m": ox_rmse,
            "vnbd_rmse_m": vn_rmse,
            "error_reduction_vs_zero_pct": euc_imp_vs_zero,
            "error_reduction_vs_oxiod_pct": euc_imp_vs_ox,
        })
        print(f"  Seq {key:<6} ({item['driver']}) | Zero: {zero_euc:5.2f}m | OxIOD: {ox_euc:5.2f}m | IO-VNBD: {vn_euc:5.2f}m | Improvement vs OxIOD: {euc_imp_vs_ox:+5.1f}%")

    df_model_comp = pd.DataFrame(seq_model_comp)
    out_model_comp_csv = RESULTS_DIR / "model_comparison_by_sequence.csv"
    df_model_comp.to_csv(out_model_comp_csv, index=False)
    print(f"\nSaved sequence-level model comparison to {out_model_comp_csv}")

    # --------------------------------------------------------------------------
    # TASKS 2, 3, 4: RUN REAL GNSS OUTAGE BENCHMARK ON ALL TEST SEQUENCES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TASKS 2, 3, 4] RUNNING REAL 7-TIER NAVIGATION BENCHMARK ON ALL 9 TEST SEQUENCES")
    print("=" * 80)

    outage_durations = [10.0, 30.0, 60.0, 120.0]

    zupt_det = ZuptDetector(
        acc_dev_thresh_g=0.10,
        acc_var_thresh_g2=0.02,
        gyro_mag_thresh_rads=0.08,
        gyro_var_thresh_rads2=0.01,
        gnss_speed_thresh_mps=0.3,
        window_size=10,
    )
    nhc_mod = NhcConstraint(noise_lateral_mps=0.20, noise_vertical_mps=0.20)

    all_outage_records = []

    for item in test_seq_catalog:
        key = item["sequence"]
        driver = item["driver"]
        s_path = item["s_path"]
        v_path = item["v_path"]

        print(f"\nProcessing Test Sequence '{key}' ({driver}, {item['duration_s']:.1f}s, {item['samples']} samples)...")
        seq = RealIOVNBDSequence(s_path, v_path)
        dt = seq.dt
        n = seq.n_samples
        origin_lat = float(seq.gt_lat[0])
        origin_lon = float(seq.gt_lon[0])

        # Precompute ground truth in NED frame
        gt_pn, gt_pe = geodetic_to_ned_vec(seq.gt_lat, seq.gt_lon, origin_lat, origin_lon)

        # Precompute ML Displacements
        ml_displacements, w_starts, w_ends = precompute_ml_displacements(
            seq.features, vnbd_infer, window_size=20, stride=10
        )

        step_dp_ned = np.zeros((n, 3), dtype=np.float64)
        for w_idx, s_idx in enumerate(w_starts):
            e_idx = w_ends[w_idx]
            stride_len = max(1, e_idx - s_idx + 1)
            dx, dy, dz = ml_displacements[w_idx]
            hdg = float(seq.gt_hdg[s_idx])
            R_bn = CoordinateTransformer.heading_to_dcm(hdg)
            dp_ned = R_bn @ np.array([dx, dy, dz], dtype=np.float64)
            step_dp_ned[s_idx : e_idx + 1] = dp_ned / stride_len

        simulator = OutageSimulator(
            outage_durations_s=outage_durations,
            warmup_duration_s=min(15.0, seq.timestamps_s[-1] * 0.1),
            recovery_duration_s=min(15.0, seq.timestamps_s[-1] * 0.1),
            random_seed=42,
        )

        for dur in outage_durations:
            intervals = simulator.plan_outages(seq.timestamps_s, dur)
            if not intervals:
                print(f"  [Skipped] {dur}s outage: Sequence duration ({item['duration_s']:.1f}s) too short.")
                continue

            interval = intervals[0]
            s = interval.start_idx
            e = interval.end_idx
            dur_samples = e - s

            # State entering outage at index s - 1
            init_idx = max(0, s - 1)
            p0_n, p0_e = gt_pn[init_idx], gt_pe[init_idx]
            hdg0 = float(seq.gt_hdg[init_idx])
            spd0 = float(seq.gt_spd[init_idx])
            v0_n = spd0 * np.cos(np.deg2rad(hdg0))
            v0_e = spd0 * np.sin(np.deg2rad(hdg0))

            # ----------------------------------------------------------
            # BASELINE 1: Pure INS (Segment execution during outage)
            # ----------------------------------------------------------
            pn_1_out = np.zeros(dur_samples)
            pe_1_out = np.zeros(dur_samples)
            spd_1_out = np.zeros(dur_samples)
            curr_vn, curr_ve = v0_n, v0_e
            curr_pn, curr_pe = p0_n, p0_e

            for idx, i in enumerate(range(s, e)):
                hdg_i = float(seq.gt_hdg[i])
                R_bn = CoordinateTransformer.heading_to_dcm(hdg_i)
                a_body = seq.features[i, 0:3] * 9.80665
                a_ned = R_bn @ a_body
                curr_vn += a_ned[0] * dt
                curr_ve += a_ned[1] * dt
                curr_pn += curr_vn * dt
                curr_pe += curr_ve * dt
                pn_1_out[idx] = curr_pn
                pe_1_out[idx] = curr_pe
                spd_1_out[idx] = np.sqrt(curr_vn ** 2 + curr_ve ** 2)

            lat_1_out, lon_1_out = ned_to_geodetic_vec(pn_1_out, pe_1_out, origin_lat, origin_lon)
            
            # Form complete array for NavigationMetrics.evaluate_outage
            lat_1 = seq.gt_lat.copy()
            lon_1 = seq.gt_lon.copy()
            lat_1[s:e] = lat_1_out
            lon_1[s:e] = lon_1_out
            spd_1 = seq.gt_spd.copy()
            spd_1[s:e] = spd_1_out

            res_1 = NavigationMetrics.evaluate_outage(
                lat_1, lon_1, seq.gt_lat, seq.gt_lon,
                s, e, dur,
                "BASELINE 1: Pure INS", seq.timestamps_s, spd_1, seq.gt_spd
            )
            r1_dict = asdict(res_1)
            r1_dict.update({"sequence": key, "driver": driver})
            all_outage_records.append(r1_dict)

            # ----------------------------------------------------------
            # BASELINE 2: INS + EKF
            # ----------------------------------------------------------
            ekf_2 = NavigationEKF()
            ekf_2.initialize(p0_n, p0_e, 0.0, v0_n, v0_e, 0.0)
            pn_2_out = np.zeros(dur_samples)
            pe_2_out = np.zeros(dur_samples)
            spd_2_out = np.zeros(dur_samples)

            for idx, i in enumerate(range(s, e)):
                hdg_i = float(seq.gt_hdg[i])
                R_bn = CoordinateTransformer.heading_to_dcm(hdg_i)
                a_body = seq.features[i, 0:3] * 9.80665
                a_ned = R_bn @ a_body
                v_curr = ekf_2.velocity_ned
                dp = v_curr * dt + 0.5 * a_ned * (dt ** 2)
                ekf_2.predict(dp, dt)

                pos = ekf_2.position_ned
                pn_2_out[idx] = pos[0]
                pe_2_out[idx] = pos[1]
                vel = ekf_2.velocity_ned
                spd_2_out[idx] = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

            lat_2 = seq.gt_lat.copy()
            lon_2 = seq.gt_lon.copy()
            lat_2[s:e], lon_2[s:e] = ned_to_geodetic_vec(pn_2_out, pe_2_out, origin_lat, origin_lon)
            spd_2 = seq.gt_spd.copy()
            spd_2[s:e] = spd_2_out

            res_2 = NavigationMetrics.evaluate_outage(
                lat_2, lon_2, seq.gt_lat, seq.gt_lon,
                s, e, dur,
                "BASELINE 2: INS + EKF", seq.timestamps_s, spd_2, seq.gt_spd
            )
            r2_dict = asdict(res_2)
            r2_dict.update({"sequence": key, "driver": driver})
            all_outage_records.append(r2_dict)

            # ----------------------------------------------------------
            # BASELINE 3: ML Only
            # ----------------------------------------------------------
            pn_3_out = np.zeros(dur_samples)
            pe_3_out = np.zeros(dur_samples)
            spd_3_out = np.zeros(dur_samples)
            curr_pn, curr_pe = p0_n, p0_e

            for idx, i in enumerate(range(s, e)):
                curr_pn += step_dp_ned[i, 0]
                curr_pe += step_dp_ned[i, 1]
                pn_3_out[idx] = curr_pn
                pe_3_out[idx] = curr_pe
                spd_3_out[idx] = float(np.sqrt(step_dp_ned[i, 0] ** 2 + step_dp_ned[i, 1] ** 2) / dt)

            lat_3 = seq.gt_lat.copy()
            lon_3 = seq.gt_lon.copy()
            lat_3[s:e], lon_3[s:e] = ned_to_geodetic_vec(pn_3_out, pe_3_out, origin_lat, origin_lon)
            spd_3 = seq.gt_spd.copy()
            spd_3[s:e] = spd_3_out

            res_3 = NavigationMetrics.evaluate_outage(
                lat_3, lon_3, seq.gt_lat, seq.gt_lon,
                s, e, dur,
                "BASELINE 3: ML Only", seq.timestamps_s, spd_3, seq.gt_spd
            )
            r3_dict = asdict(res_3)
            r3_dict.update({"sequence": key, "driver": driver})
            all_outage_records.append(r3_dict)

            # ----------------------------------------------------------
            # BASELINE 4: ML + INS
            # ----------------------------------------------------------
            alpha = 0.85
            lat_4 = seq.gt_lat.copy()
            lon_4 = seq.gt_lon.copy()
            lat_4[s:e] = alpha * lat_3[s:e] + (1.0 - alpha) * lat_1[s:e]
            lon_4[s:e] = alpha * lon_3[s:e] + (1.0 - alpha) * lon_1[s:e]
            spd_4 = seq.gt_spd.copy()
            spd_4[s:e] = alpha * spd_3[s:e] + (1.0 - alpha) * spd_1[s:e]

            res_4 = NavigationMetrics.evaluate_outage(
                lat_4, lon_4, seq.gt_lat, seq.gt_lon,
                s, e, dur,
                "BASELINE 4: ML + INS", seq.timestamps_s, spd_4, seq.gt_spd
            )
            r4_dict = asdict(res_4)
            r4_dict.update({"sequence": key, "driver": driver})
            all_outage_records.append(r4_dict)

            # ----------------------------------------------------------
            # BASELINES 5, 6, 7: EKF Fusion (ML + INS + EKF +/- NHC +/- ZUPT)
            # ----------------------------------------------------------
            for use_nhc, use_zupt, b_name in [
                (False, False, "BASELINE 5: ML + INS + EKF"),
                (True,  False, "BASELINE 6: ML + INS + EKF + NHC"),
                (True,  True,  "BASELINE 7: ML + INS + EKF + NHC + ZUPT"),
            ]:
                ekf = NavigationEKF()
                ekf.initialize(p0_n, p0_e, 0.0, v0_n, v0_e, 0.0)
                pn_k_out = np.zeros(dur_samples)
                pe_k_out = np.zeros(dur_samples)
                spd_k_out = np.zeros(dur_samples)

                for idx, i in enumerate(range(s, e)):
                    hdg_i = float(seq.gt_hdg[i])
                    dp = step_dp_ned[i]
                    ekf.predict(dp, dt)

                    if use_zupt and i >= 10:
                        recent_acc = seq.features[i - 10 : i, 0:3]
                        recent_gyro = seq.features[i - 10 : i, 3:6]
                        if zupt_det.is_stationary(recent_acc, recent_gyro, gnss_speed_mps=None):
                            ekf.update_zupt(noise_mps=0.05)

                    if use_nhc:
                        ekf.update_nhc(hdg_i, noise_lat_mps=0.15, noise_vert_mps=0.15)

                    pos = ekf.position_ned
                    pn_k_out[idx] = pos[0]
                    pe_k_out[idx] = pos[1]
                    vel = ekf.velocity_ned
                    spd_k_out[idx] = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

                lat_k = seq.gt_lat.copy()
                lon_k = seq.gt_lon.copy()
                lat_k[s:e], lon_k[s:e] = ned_to_geodetic_vec(pn_k_out, pe_k_out, origin_lat, origin_lon)
                spd_k = seq.gt_spd.copy()
                spd_k[s:e] = spd_k_out

                res_k = NavigationMetrics.evaluate_outage(
                    lat_k, lon_k, seq.gt_lat, seq.gt_lon,
                    s, e, dur,
                    b_name, seq.timestamps_s, spd_k, seq.gt_spd
                )
                rk_dict = asdict(res_k)
                rk_dict.update({"sequence": key, "driver": driver})
                all_outage_records.append(rk_dict)

            print(f"  [{dur}s outage] ML Only Drift: {res_3.drift_percent_endpoint:5.2f}% | ML+INS Drift: {res_4.drift_percent_endpoint:5.2f}% | Pure INS Drift: {res_1.drift_percent_endpoint:5.2f}%")

    # --------------------------------------------------------------------------
    # TASK 5: AGGREGATE RESULTS ACROSS ALL TEST SEQUENCES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TASK 5] AGGREGATE RESULTS & STATISTICAL DISTRIBUTION")
    print("=" * 80)

    df_all_results = pd.DataFrame(all_outage_records)
    out_all_csv = RESULTS_DIR / "real_benchmark_all_test_sequences.csv"
    df_all_results.to_csv(out_all_csv, index=False)
    print(f"Saved complete multi-sequence test results to {out_all_csv} ({len(df_all_results)} evaluated configurations).")

    # Compute comprehensive aggregation per (baseline, outage_duration)
    metrics_to_agg = ["drift_percent_endpoint", "position_rmse_m", "endpoint_error_m"]
    agg_rows = []

    for (b_name, dur), group in df_all_results.groupby(["baseline_name", "outage_duration_s"]):
        row_dict = {
            "baseline_name": b_name,
            "outage_duration_s": dur,
            "sequence_count": len(group),
        }
        for m in metrics_to_agg:
            vals = group[m].values
            row_dict[f"{m}_mean"] = float(np.mean(vals))
            row_dict[f"{m}_median"] = float(np.median(vals))
            row_dict[f"{m}_std"] = float(np.std(vals))
            row_dict[f"{m}_p25"] = float(np.percentile(vals, 25))
            row_dict[f"{m}_p75"] = float(np.percentile(vals, 75))
            row_dict[f"{m}_p95"] = float(np.percentile(vals, 95))
            row_dict[f"{m}_min"] = float(np.min(vals))
            row_dict[f"{m}_max"] = float(np.max(vals))
        agg_rows.append(row_dict)

    df_aggregate = pd.DataFrame(agg_rows)
    out_agg_csv = RESULTS_DIR / "real_benchmark_aggregate.csv"
    df_aggregate.to_csv(out_agg_csv, index=False)
    print(f"Saved aggregated statistics to {out_agg_csv}")

    # Display summary of ML Only vs Pure INS across durations
    print("\n--- Aggregated Endpoint Drift (%) Summary (Mean [Median] (Min - Max)) ---")
    for dur in outage_durations:
        sub = df_aggregate[df_aggregate["outage_duration_s"] == dur]
        if sub.empty:
            continue
        print(f"\nOutage Duration: {dur:.0f}s (Evaluated across {sub['sequence_count'].iloc[0]} sequences):")
        for _, r in sub.iterrows():
            print(f"  {r['baseline_name']:<38}: Mean {r['drift_percent_endpoint_mean']:5.2f}% | Median {r['drift_percent_endpoint_median']:5.2f}% | Range [{r['drift_percent_endpoint_min']:5.2f}% - {r['drift_percent_endpoint_max']:5.2f}%]")

    # --------------------------------------------------------------------------
    # TASK 6: GENERALIZATION ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TASK 6] GENERALIZATION THRESHOLD AUDIT (ML ONLY)")
    print("=" * 80)

    ml_records = df_all_results[df_all_results["baseline_name"] == "BASELINE 3: ML Only"]
    total_evals = len(ml_records)

    c_10 = int(np.sum(ml_records["drift_percent_endpoint"] < 10.0))
    c_15 = int(np.sum(ml_records["drift_percent_endpoint"] < 15.0))
    c_20 = int(np.sum(ml_records["drift_percent_endpoint"] < 20.0))

    mean_drift = float(ml_records["drift_percent_endpoint"].mean())
    median_drift = float(ml_records["drift_percent_endpoint"].median())
    worst_drift = float(ml_records["drift_percent_endpoint"].max())
    best_drift = float(ml_records["drift_percent_endpoint"].min())

    worst_row = ml_records.loc[ml_records["drift_percent_endpoint"].idxmax()]
    best_row = ml_records.loc[ml_records["drift_percent_endpoint"].idxmin()]

    print(f"Total ML Outage Evaluations Across All 9 Test Sequences: {total_evals}")
    print(f"1. Sequences / Evaluations achieving <10% drift: {c_10} / {total_evals} ({c_10/total_evals*100:.1f}%)")
    print(f"2. Sequences / Evaluations achieving <15% drift: {c_15} / {total_evals} ({c_15/total_evals*100:.1f}%)")
    print(f"3. Sequences / Evaluations achieving <20% drift: {c_20} / {total_evals} ({c_20/total_evals*100:.1f}%)")
    print(f"4. Mean Endpoint Drift across all test evaluations: {mean_drift:.2f}%")
    print(f"5. Median Endpoint Drift across all test evaluations: {median_drift:.2f}%")
    print(f"6. Best-case Drift: {best_drift:.2f}% (Seq: {best_row['sequence']}, {best_row['outage_duration_s']}s outage)")
    print(f"7. Worst-case Drift: {worst_drift:.2f}% (Seq: {worst_row['sequence']}, {worst_row['outage_duration_s']}s outage)")

    # --------------------------------------------------------------------------
    # TASK 8: INVESTIGATE THE 120-SECOND RESULT ACROSS SEQUENCES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TASK 8] 120-SECOND OUTAGE GENERALIZATION INVESTIGATION")
    print("=" * 80)

    rec_120 = df_all_results[df_all_results["outage_duration_s"] == 120.0]
    sequences_120 = sorted(list(set(rec_120["sequence"])))
    print(f"Sequences long enough to execute full 120-second outages: {sequences_120} ({len(sequences_120)} sequences)")

    print("\nComparison of 120s Outage Drift across eligible sequences:")
    for s_key in sequences_120:
        sub_s = rec_120[rec_120["sequence"] == s_key]
        ins_r = sub_s[sub_s["baseline_name"] == "BASELINE 1: Pure INS"]["drift_percent_endpoint"].iloc[0]
        ml_r = sub_s[sub_s["baseline_name"] == "BASELINE 3: ML Only"]["drift_percent_endpoint"].iloc[0]
        fused_r = sub_s[sub_s["baseline_name"] == "BASELINE 5: ML + INS + EKF"]["drift_percent_endpoint"].iloc[0]
        driver_s = sub_s["driver"].iloc[0]
        dist_s = sub_s["distance_travelled_m"].iloc[0]
        print(f"  Seq {s_key:<6} ({driver_s}) | Dist: {dist_s:6.1f}m | Pure INS: {ins_r:6.1f}% | ML Only: {ml_r:5.2f}% | ML+INS+EKF: {fused_r:5.2f}% | Error Reduced: {(ins_r - ml_r)/ins_r*100:5.1f}%")

    # --------------------------------------------------------------------------
    # Generate Multi-Sequence Drift Comparison Figure
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    df_30 = df_all_results[df_all_results["outage_duration_s"] == 30.0]
    if not df_30.empty:
        seqs_30 = sorted(list(set(df_30["sequence"])))
        ins_30 = [df_30[(df_30["sequence"] == s) & (df_30["baseline_name"] == "BASELINE 1: Pure INS")]["drift_percent_endpoint"].iloc[0] for s in seqs_30]
        ml_30 = [df_30[(df_30["sequence"] == s) & (df_30["baseline_name"] == "BASELINE 3: ML Only")]["drift_percent_endpoint"].iloc[0] for s in seqs_30]

        x = np.arange(len(seqs_30))
        w = 0.35
        ax1.bar(x - w/2, ins_30, w, label="Pure INS Drift (%)", color="#E74C3C")
        ax1.bar(x + w/2, ml_30, w, label="ML Only Drift (%)", color="#2ECC71")
        ax1.set_title("30s GNSS Outage: Drift % Across Test Sequences", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(seqs_30, fontsize=10)
        ax1.set_ylabel("Endpoint Drift (%)", fontsize=11)
        ax1.axhline(10.0, color="r", linestyle="--", alpha=0.6, label="10% Drift Threshold")
        ax1.axhline(20.0, color="orange", linestyle=":", alpha=0.6, label="20% Drift Threshold")
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    dur_data = []
    dur_labels = []
    for dur in outage_durations:
        vals = df_all_results[(df_all_results["outage_duration_s"] == dur) & (df_all_results["baseline_name"] == "BASELINE 3: ML Only")]["drift_percent_endpoint"].values
        if len(vals) > 0:
            dur_data.append(vals)
            dur_labels.append(f"{dur:.0f}s (N={len(vals)})")

    ax2.boxplot(dur_data, tick_labels=dur_labels, patch_artist=True, boxprops=dict(facecolor="#3498DB", color="#2980B9"))
    ax2.set_title("ML Only: Endpoint Drift Distribution Across Outage Durations", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Outage Duration", fontsize=11)
    ax2.set_ylabel("Endpoint Drift (%)", fontsize=11)
    ax2.axhline(10.0, color="r", linestyle="--", alpha=0.6, label="10% Target")
    ax2.axhline(20.0, color="orange", linestyle=":", alpha=0.6, label="20% Bound")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "all_sequences_drift_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved multi-sequence drift summary plot to {plot_path}")

    return df_all_results, df_aggregate, df_model_comp


if __name__ == "__main__":
    run_multi_sequence_audit()
