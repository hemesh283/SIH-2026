"""Score baseline (and optionally baseline+residual-corrected) trajectories
against ground truth -- the run_eval.py evaluation/README.md has promised
since Phase 1 but never had code behind it (see
evaluation/AUDIT_Phase1-4_Report.md Section 4).

This scores whatever CSVs you hand it; it does not itself run the physics
baseline or apply a trained residual model to produce those CSVs -- see
physics-baseline/strapdown.py (baseline) and ml-residual/train.py (residual
model) for that. As of this writing there is no real baseline/corrected/
ground-truth CSV anywhere in this repo (see the audit above, Section 1) --
run this once a real recorded trace and its ground-truth GPS file exist.

Usage:
    python run_eval.py --truth ground_truth.csv --baseline baseline.csv \
        [--corrected corrected.csv] [--rte-window-s 60] [--out results.json]

ground_truth.csv / baseline.csv / corrected.csv use the same column
conventions as ml-residual/common.py's loaders (timestamp, pos_x, pos_y[,
pos_z, fix_available]) so a real trace's existing physics-baseline output
and ground-truth file can be pointed at directly, no reformatting needed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml-residual"))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402


def _score(label, estimate_df, truth_df, rte_window_s):
    truth_fix = truth_df[truth_df["fix_available"] == 1]
    result = metrics.evaluate_against_ground_truth(
        estimate_df["t"].to_numpy(), estimate_df["pos_x"].to_numpy(), estimate_df["pos_y"].to_numpy(),
        truth_fix["t"].to_numpy(), truth_fix["pos_x"].to_numpy(), truth_fix["pos_y"].to_numpy(),
        rte_window_s=rte_window_s,
    )
    result["label"] = label
    print(f"[{label}] ATE={result['ate_m']:.3f}m  RTE={result['rte_m']:.3f}m  "
          f"CEP50={result['cep50_m']:.3f}m  CEP90={result['cep90_m']:.3f}m  "
          f"drift={result['drift_rate_pct']:.2f}%  "
          f"(n={result['n_samples']}, distance={result['distance_m']:.1f}m)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", required=True, help="Ground-truth GPS CSV (timestamp, pos_x, pos_y[, fix_available])")
    parser.add_argument("--baseline", required=True, help="physics-baseline output CSV to score")
    parser.add_argument("--corrected", help="Optional baseline+residual-corrected trajectory CSV (same columns), scored the same way for comparison")
    parser.add_argument("--rte-window-s", type=float, default=metrics.DEFAULT_RTE_WINDOW_S)
    parser.add_argument("--time-unit", choices=["auto", "s", "ms", "ns"], default="auto")
    parser.add_argument("--out", help="Optional path to save results as JSON")
    args = parser.parse_args()

    baseline_raw = pd.read_csv(args.baseline)
    raw_ts = baseline_raw["timestamp"].to_numpy(dtype=np.float64)
    time_unit = args.time_unit if args.time_unit != "auto" else common._resolve_time_unit(raw_ts, "auto")
    origin_raw = float(raw_ts.min())

    baseline_df = common.load_baseline_csv(args.baseline, origin_raw=origin_raw, time_unit=time_unit)
    truth_df = common.load_ground_truth_csv(args.truth, origin_raw=origin_raw, time_unit=time_unit)

    if not (truth_df["fix_available"] == 1).any():
        raise SystemExit("ground truth has no rows with fix_available == 1 -- nothing to score against")

    results = {"baseline": _score("baseline", baseline_df, truth_df, args.rte_window_s)}

    if args.corrected:
        corrected_df = common.load_baseline_csv(args.corrected, origin_raw=origin_raw, time_unit=time_unit)
        results["corrected"] = _score("corrected", corrected_df, truth_df, args.rte_window_s)
        d_ate = results["baseline"]["ate_m"] - results["corrected"]["ate_m"]
        print(f"\nATE improvement (baseline - corrected): {d_ate:+.3f}m")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
