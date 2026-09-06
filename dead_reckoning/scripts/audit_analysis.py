"""Comprehensive Statistical Analysis and Aggregation for IO-VNBD Audit.

Computes:
1. Full 7-baseline aggregate table (all 32 evaluations).
2. Moving-only aggregate table (distance_travelled_m > 100 m).
3. B3 (ML Only), B4 (ML+INS), B5 (ML+INS+EKF) breakdown (<10%, <15%, <20%, median, mean, P95).
4. Best baseline determination per outage duration.
5. Claims verification against raw CSV (8a through 8f).
6. 120s moving sequence metrics (best, worst, median, mean, P95).
7. ZUPT vs NHC difference analysis (identifying where B6 == B7 and where B6 != B7).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/dead_reckoning")
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"


def run_audit_analysis():
    csv_path = RESULTS_DIR / "real_benchmark_all_test_sequences.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing benchmark CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print("=" * 80)
    print(f"RAW BENCHMARK CSV AUDIT: {len(df)} total rows")
    print("=" * 80)

    # 1. Verification of 224 evaluations
    baselines = [
        "BASELINE 1: Pure INS",
        "BASELINE 2: INS + EKF",
        "BASELINE 3: ML Only",
        "BASELINE 4: ML + INS",
        "BASELINE 5: ML + INS + EKF",
        "BASELINE 6: ML + INS + EKF + NHC",
        "BASELINE 7: ML + INS + EKF + NHC + ZUPT",
    ]
    assert len(df) == 224, f"Expected 224 rows, got {len(df)}"
    for b in baselines:
        count = len(df[df["baseline_name"] == b])
        assert count == 32, f"Baseline {b} has {count} rows, expected 32"
    print("Task 1 & 2 Verified: Exactly 224 evaluations across all 7 baselines (32 evaluations each).\n")

    # 2. Comprehensive 7-Baseline Aggregate Table (All 32 evaluations)
    outage_durations = [10.0, 30.0, 60.0, 120.0]
    agg_all = []
    for dur in outage_durations:
        for b in baselines:
            sub = df[(df["outage_duration_s"] == dur) & (df["baseline_name"] == b)]
            if sub.empty:
                continue
            drifts = sub["drift_percent_endpoint"].values
            rmses = sub["position_rmse_m"].values
            errs = sub["endpoint_error_m"].values

            agg_all.append({
                "outage_duration_s": dur,
                "baseline_name": b,
                "eval_count": len(sub),
                "drift_mean": float(np.mean(drifts)),
                "drift_median": float(np.median(drifts)),
                "drift_std": float(np.std(drifts)),
                "drift_p25": float(np.percentile(drifts, 25)),
                "drift_p75": float(np.percentile(drifts, 75)),
                "drift_p95": float(np.percentile(drifts, 95)),
                "drift_min": float(np.min(drifts)),
                "drift_max": float(np.max(drifts)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_median": float(np.median(rmses)),
                "endpoint_err_mean": float(np.mean(errs)),
                "endpoint_err_median": float(np.median(errs)),
            })

    df_agg_all = pd.DataFrame(agg_all)
    df_agg_all.to_csv(RESULTS_DIR / "real_benchmark_aggregate.csv", index=False)
    print("Task 3 Completed: Saved updated real_benchmark_aggregate.csv with all 7 baselines.")

    # 3. Moving Sequences Aggregate Table (distance_travelled_m > 100 m)
    df_moving = df[df["distance_travelled_m"] > 100.0].copy()
    print(f"\nTask 4: Moving sequences criterion (distance > 100m): {len(df_moving)} / {len(df)} evaluations ({len(df_moving)/len(df)*100:.1f}%).")
    
    agg_mov = []
    for dur in outage_durations:
        for b in baselines:
            sub = df_moving[(df_moving["outage_duration_s"] == dur) & (df_moving["baseline_name"] == b)]
            if sub.empty:
                continue
            drifts = sub["drift_percent_endpoint"].values
            rmses = sub["position_rmse_m"].values
            errs = sub["endpoint_error_m"].values

            agg_mov.append({
                "outage_duration_s": dur,
                "baseline_name": b,
                "eval_count": len(sub),
                "drift_mean": float(np.mean(drifts)),
                "drift_median": float(np.median(drifts)),
                "drift_std": float(np.std(drifts)),
                "drift_p25": float(np.percentile(drifts, 25)),
                "drift_p75": float(np.percentile(drifts, 75)),
                "drift_p95": float(np.percentile(drifts, 95)),
                "drift_min": float(np.min(drifts)),
                "drift_max": float(np.max(drifts)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_median": float(np.median(rmses)),
                "endpoint_err_mean": float(np.mean(errs)),
                "endpoint_err_median": float(np.median(errs)),
            })

    df_agg_mov = pd.DataFrame(agg_mov)
    df_agg_mov.to_csv(RESULTS_DIR / "real_benchmark_aggregate_moving.csv", index=False)
    print("Saved real_benchmark_aggregate_moving.csv.")

    # 4. Task 5 & 6: ML Only (B3), ML+INS (B4), ML+INS+EKF (B5) Analysis
    print("\n" + "=" * 80)
    print("TASKS 5 & 6: B3 (ML Only), B4 (ML+INS), B5 (ML+INS+EKF) THRESHOLD BREAKDOWN")
    print("=" * 80)

    for b_code, b_name in [("B3", "BASELINE 3: ML Only"), ("B4", "BASELINE 4: ML + INS"), ("B5", "BASELINE 5: ML + INS + EKF")]:
        sub_all = df[df["baseline_name"] == b_name]
        sub_mov = df_moving[df_moving["baseline_name"] == b_name]

        for scope_name, sub in [("ALL EVALUATIONS", sub_all), ("MOVING ONLY (>100m)", sub_mov)]:
            drifts = sub["drift_percent_endpoint"].values
            n_eval = len(drifts)
            c10 = int(np.sum(drifts < 10.0))
            c15 = int(np.sum(drifts < 15.0))
            c20 = int(np.sum(drifts < 20.0))
            med = float(np.median(drifts))
            mean_v = float(np.mean(drifts))
            p95_v = float(np.percentile(drifts, 95))

            print(f"\n{b_code} ({b_name}) — {scope_name} (N = {n_eval}):")
            print(f"  <10% Drift: {c10} / {n_eval} ({c10/n_eval*100:5.1f}%)")
            print(f"  <15% Drift: {c15} / {n_eval} ({c15/n_eval*100:5.1f}%)")
            print(f"  <20% Drift: {c20} / {n_eval} ({c20/n_eval*100:5.1f}%)")
            print(f"  Median Drift: {med:6.2f}% | Mean Drift: {mean_v:6.2f}% | P95 Drift: {p95_v:6.2f}%")

    # 5. Task 7: Best Baseline per Outage Duration
    print("\n" + "=" * 80)
    print("TASK 7: BEST BASELINE PER OUTAGE DURATION")
    print("=" * 80)

    for dur in outage_durations:
        sub_dur_all = df_agg_all[df_agg_all["outage_duration_s"] == dur]
        sub_dur_mov = df_agg_mov[df_agg_mov["outage_duration_s"] == dur]

        best_med_drift_all = sub_dur_all.loc[sub_dur_all["drift_median"].idxmin()]
        best_mean_drift_all = sub_dur_all.loc[sub_dur_all["drift_mean"].idxmin()]
        best_med_rmse_all = sub_dur_all.loc[sub_dur_all["rmse_median"].idxmin()]

        best_med_drift_mov = sub_dur_mov.loc[sub_dur_mov["drift_median"].idxmin()]
        best_mean_drift_mov = sub_dur_mov.loc[sub_dur_mov["drift_mean"].idxmin()]
        best_med_rmse_mov = sub_dur_mov.loc[sub_dur_mov["rmse_median"].idxmin()]

        print(f"\n--- Outage Duration: {dur:.0f}s ---")
        print(f"  [ALL (N={best_med_drift_all['eval_count']})]")
        print(f"    Best by Median Drift: {best_med_drift_all['baseline_name']} ({best_med_drift_all['drift_median']:.2f}%)")
        print(f"    Best by Mean Drift  : {best_mean_drift_all['baseline_name']} ({best_mean_drift_all['drift_mean']:.2f}%)")
        print(f"    Best by Median RMSE : {best_med_rmse_all['baseline_name']} ({best_med_rmse_all['rmse_median']:.2f}m)")
        print(f"  [MOVING ONLY (N={best_med_drift_mov['eval_count']})]")
        print(f"    Best by Median Drift: {best_med_drift_mov['baseline_name']} ({best_med_drift_mov['drift_median']:.2f}%)")
        print(f"    Best by Mean Drift  : {best_mean_drift_mov['baseline_name']} ({best_mean_drift_mov['drift_mean']:.2f}%)")
        print(f"    Best by Median RMSE : {best_med_rmse_mov['baseline_name']} ({best_med_rmse_mov['rmse_median']:.2f}m)")

    # 6. Task 8: Verification of Specific Claims
    print("\n" + "=" * 80)
    print("TASK 8: VERIFICATION OF SPECIFIC CLAIMS")
    print("=" * 80)

    model_comp_path = RESULTS_DIR / "model_comparison_by_sequence.csv"
    df_mc = pd.read_csv(model_comp_path)

    # Claim a: 8/9 test sequences improved by IO-VNBD GRU vs Zero baseline
    imp_zero = df_mc[df_mc["error_reduction_vs_zero_pct"] > 0]
    claim_a = len(imp_zero) == 8
    print(f"Claim a: 8/9 sequences improved vs Zero baseline? {'TRUE' if claim_a else 'FALSE'} ({len(imp_zero)}/9 sequences: {list(imp_zero['sequence'])})")

    # Claim b: IO-VNBD GRU improves over OxIOD on 8/9 sequences
    imp_oxiod = df_mc[df_mc["error_reduction_vs_oxiod_pct"] > 0]
    claim_b = len(imp_oxiod) == 8
    print(f"Claim b: IO-VNBD improves over OxIOD on 8/9 sequences? {'TRUE' if claim_b else 'FALSE'} ({len(imp_oxiod)}/9 sequences: {list(imp_oxiod['sequence'])})")

    # Claim c: Only 4/32 ML-only outage evaluations achieve <10% drift
    b3_all = df[df["baseline_name"] == "BASELINE 3: ML Only"]
    c10_b3 = int(np.sum(b3_all["drift_percent_endpoint"] < 10.0))
    claim_c = c10_b3 == 4
    print(f"Claim c: Only 4/32 ML-only evaluations achieve <10% drift? {'TRUE' if claim_c else 'FALSE'} (Exactly {c10_b3}/32)")

    # Claim d: Worst stationary outlier is vw15
    worst_seq = b3_all.loc[b3_all["drift_percent_endpoint"].idxmax()]["sequence"]
    claim_d = worst_seq == "vw15"
    print(f"Claim d: Worst stationary outlier is vw15? {'TRUE' if claim_d else 'FALSE'} (Worst is '{worst_seq}' with drift {b3_all['drift_percent_endpoint'].max():.2f}%)")

    # Claim e: y1 is a case where ML does not outperform Pure INS at 120s
    y1_120_ins = df[(df["sequence"] == "y1") & (df["outage_duration_s"] == 120.0) & (df["baseline_name"] == "BASELINE 1: Pure INS")]["drift_percent_endpoint"].iloc[0]
    y1_120_ml = df[(df["sequence"] == "y1") & (df["outage_duration_s"] == 120.0) & (df["baseline_name"] == "BASELINE 3: ML Only")]["drift_percent_endpoint"].iloc[0]
    claim_e = y1_120_ml > y1_120_ins
    print(f"Claim e: y1 ML does not outperform Pure INS at 120s? {'TRUE' if claim_e else 'FALSE'} (Pure INS: {y1_120_ins:.2f}%, ML: {y1_120_ml:.2f}%)")

    # Claim f: vw16a has the best 120s moving-sequence ML result
    b3_120_mov = df_moving[(df_moving["outage_duration_s"] == 120.0) & (df_moving["baseline_name"] == "BASELINE 3: ML Only")]
    best_120_seq = b3_120_mov.loc[b3_120_mov["drift_percent_endpoint"].idxmin()]["sequence"]
    claim_f = best_120_seq == "vw16a"
    print(f"Claim f: vw16a has the best 120s moving ML result? {'TRUE' if claim_f else 'FALSE'} (Best is '{best_120_seq}' with drift {b3_120_mov['drift_percent_endpoint'].min():.2f}%)")

    # 7. Task 9: NHC and ZUPT Trajectory Verification
    print("\n" + "=" * 80)
    print("TASK 9: NHC AND ZUPT TRAJECTORY VERIFICATION (B6 vs B7)")
    print("=" * 80)

    b6_rows = df[df["baseline_name"] == "BASELINE 6: ML + INS + EKF + NHC"].sort_values(["sequence", "outage_duration_s"]).reset_index(drop=True)
    b7_rows = df[df["baseline_name"] == "BASELINE 7: ML + INS + EKF + NHC + ZUPT"].sort_values(["sequence", "outage_duration_s"]).reset_index(drop=True)

    diff_count = 0
    same_count = 0
    for i in range(len(b6_rows)):
        s_name = b6_rows.loc[i, "sequence"]
        dur = b6_rows.loc[i, "outage_duration_s"]
        drift6 = b6_rows.loc[i, "drift_percent_endpoint"]
        drift7 = b7_rows.loc[i, "drift_percent_endpoint"]
        rmse6 = b6_rows.loc[i, "position_rmse_m"]
        rmse7 = b7_rows.loc[i, "position_rmse_m"]
        diff = abs(drift6 - drift7)

        if diff > 1e-4:
            diff_count += 1
            print(f"  [ZUPT ACTIVE] Seq: {s_name:<6} | {dur:3.0f}s outage | B6 Drift: {drift6:8.2f}% -> B7 Drift: {drift7:8.2f}% | RMSE: {rmse6:6.2f}m -> {rmse7:6.2f}m")
        else:
            same_count += 1

    print(f"\nTotal Outages: {len(b6_rows)} | ZUPT Changed Trajectory: {diff_count} | Identical (No Stop Occurred): {same_count}")

    # 8. Task 13: 120s Moving Sequence Detailed Metrics
    print("\n" + "=" * 80)
    print("TASK 13: 120s OUTAGE DETAILED METRICS FOR MOVING SEQUENCES (>100m)")
    print("=" * 80)

    b3_120_m = df_moving[(df_moving["outage_duration_s"] == 120.0) & (df_moving["baseline_name"] == "BASELINE 3: ML Only")]
    drifts_120 = b3_120_m["drift_percent_endpoint"].values
    best_s = b3_120_m.loc[b3_120_m["drift_percent_endpoint"].idxmin()]
    worst_s = b3_120_m.loc[b3_120_m["drift_percent_endpoint"].idxmax()]

    print(f"Moving 120s sequences evaluated: {len(b3_120_m)} ({list(b3_120_m['sequence'])})")
    print(f"  Best Moving Sequence : {best_s['sequence']} ({best_s['driver']}) with Drift: {best_s['drift_percent_endpoint']:.2f}% (Dist: {best_s['distance_travelled_m']:.1f}m, RMSE: {best_s['position_rmse_m']:.2f}m)")
    print(f"  Worst Moving Sequence: {worst_s['sequence']} ({worst_s['driver']}) with Drift: {worst_s['drift_percent_endpoint']:.2f}% (Dist: {worst_s['distance_travelled_m']:.1f}m, RMSE: {worst_s['position_rmse_m']:.2f}m)")
    print(f"  Median Moving Drift  : {np.median(drifts_120):.2f}%")
    print(f"  Mean Moving Drift    : {np.mean(drifts_120):.2f}%")
    print(f"  P95 Moving Drift     : {np.percentile(drifts_120, 95):.2f}%")
    print(f"  Min Moving Drift     : {np.min(drifts_120):.2f}%")
    print(f"  Max Moving Drift     : {np.max(drifts_120):.2f}%")


if __name__ == "__main__":
    run_audit_analysis()
