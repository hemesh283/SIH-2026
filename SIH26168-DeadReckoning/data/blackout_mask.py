#!/usr/bin/env python3
"""
blackout_mask.py — SIH26168 data/ pipeline, GNSS-blackout simulation
(dossier Section 5.1 / 5.3)

Takes a ground_truth.csv produced by convert_trace.py (continuous, good
GPS: timestamp, pos_x, pos_y, pos_z, fix_available==1 throughout) and
flips `fix_available` to 0 for one or more configurable windows, to
simulate a GNSS blackout for evaluation.

IMPORTANT — this does NOT null out or delete pos_x/pos_y/pos_z. Per
ml-residual/common.py's load_ground_truth_csv docstring: "Position is
required even where fix_available=0 -- that is the withheld true label
the residual model is trained and scored against." The physics baseline
(physics-baseline/strapdown.py) never reads this file at all — it dead-
reckons from the IMU file alone — so "masking" here only controls what
ml-residual/features.py treats as an available fix (for
time_since_last_gnss_fix and motion-mode/uncertainty calibration), not
what strapdown.py can see. The real positions stay in the file so
features.py can still score the baseline+residual estimate against them
during the blackout window.

No accelerometer/gyroscope/GPS value is ever fabricated or removed —
only which rows count as "available" changes, per the dossier's Section
5.3 Data Integrity Statement.

OUTPUT (for input `foo_ground_truth.csv` and `--out-prefix data/raw_traces/foo_blackout`)
-------------------------------------------------------------------------
<out_prefix>_ground_truth.csv
    Same file, with fix_available flipped to 0 inside each blackout
    window. Positions are untouched. Feed THIS into ml-residual/ instead
    of the un-masked ground_truth.csv when you want a blackout-evaluation
    run; keep the original around too, since it's needed to train on
    non-blackout data in the first place.

<out_prefix>_windows.json
    The blackout windows actually applied (start_s, end_s, duration_s,
    n_rows_flipped), for bookkeeping / reproducibility.

USAGE
-----
Single window (seconds elapsed since the trace's first sample):
    python blackout_mask.py data/raw_traces/walk01_ground_truth.csv \
        --out-prefix data/raw_traces/walk01_blackout \
        --window 60 120

Multiple windows (repeat --window):
    python blackout_mask.py data/raw_traces/walk01_ground_truth.csv \
        --out-prefix data/raw_traces/walk01_blackout \
        --window 60 120 --window 300 360

Random window of a given duration, from a segment that currently has
continuous fix_available==1:
    python blackout_mask.py data/raw_traces/walk01_ground_truth.csv \
        --out-prefix data/raw_traces/walk01_blackout \
        --random-duration 90 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = {"timestamp", "pos_x", "pos_y"}


def resolve_time_unit(raw_ts: np.ndarray) -> str:
    """Mirrors physics-baseline/strapdown.py's _normalize_time auto-detection."""
    magnitude = np.median(np.abs(raw_ts))
    if magnitude > 1e17:
        return "ns"
    if magnitude > 1e12:
        return "ms"
    return "s"


def elapsed_seconds(raw_ts: np.ndarray, time_unit: str) -> np.ndarray:
    scale = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}[time_unit]
    t = raw_ts * scale
    return t - t[0]


def validate_window(t_s: np.ndarray, fix_available: np.ndarray, start_s: float, end_s: float) -> None:
    if start_s >= end_s:
        raise ValueError(f"Window start ({start_s}) must be < end ({end_s}).")
    if start_s < t_s[0] or end_s > t_s[-1]:
        raise ValueError(f"Window [{start_s}, {end_s}] falls outside the trace's time range [{t_s[0]:.1f}, {t_s[-1]:.1f}]s.")
    mask = (t_s >= start_s) & (t_s <= end_s)
    if not fix_available[mask].any():
        raise ValueError(
            f"Window [{start_s}, {end_s}] has no rows with fix_available==1 to withhold in the "
            f"first place — pick a window from a segment with continuous GPS lock."
        )


def pick_random_window(t_s: np.ndarray, fix_available: np.ndarray, duration_s: float, rng: random.Random) -> tuple[float, float]:
    span = t_s[-1] - t_s[0]
    if duration_s >= span:
        raise ValueError(f"Requested duration {duration_s}s >= trace length {span:.1f}s.")
    candidates = []
    for i in range(len(t_s)):
        start = t_s[i]
        end = start + duration_s
        if end > t_s[-1]:
            break
        seg = (t_s >= start) & (t_s <= end)
        if fix_available[seg].all() and seg.sum() > 1:
            candidates.append(start)
    if not candidates:
        raise ValueError(f"No {duration_s}s segment with fully continuous fix_available==1 was found in this trace.")
    start = rng.choice(candidates)
    return float(start), float(start + duration_s)


def apply_blackout(df: pd.DataFrame, t_s: np.ndarray, windows: list[tuple[float, float]]) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    window_info = []
    for start_s, end_s in windows:
        mask = (t_s >= start_s) & (t_s <= end_s)
        n_flipped = int((mask & (out["fix_available"] == 1)).sum())
        out.loc[mask, "fix_available"] = 0
        window_info.append({
            "start_s": start_s, "end_s": end_s,
            "duration_s": round(end_s - start_s, 3),
            "n_rows_flipped": n_flipped,
        })
    return out, window_info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ground_truth_csv", type=Path, help="ground_truth.csv from convert_trace.py")
    ap.add_argument("--out-prefix", type=Path, required=True, help="Output path prefix, e.g. data/raw_traces/walk01_blackout")
    ap.add_argument("--window", nargs=2, type=float, action="append", metavar=("START_S", "END_S"),
                     help="Blackout window [start_s, end_s], elapsed seconds since the trace's first sample. Repeatable.")
    ap.add_argument("--random-duration", type=float, default=None,
                     help="Instead of --window, pick a random window of this many seconds from a continuously-available segment.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random-duration (default 0)")
    args = ap.parse_args()

    df = pd.read_csv(args.ground_truth_csv).sort_values("timestamp").reset_index(drop=True)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{args.ground_truth_csv} is missing columns {missing} — is this a convert_trace.py ground_truth CSV?")
    if "fix_available" not in df.columns:
        df["fix_available"] = 1

    raw_ts = df["timestamp"].to_numpy(dtype=np.float64)
    time_unit = resolve_time_unit(raw_ts)
    t_s = elapsed_seconds(raw_ts, time_unit)
    fix_available = df["fix_available"].to_numpy()

    if bool(args.window) == bool(args.random_duration):
        ap.error("Pass exactly one of --window (one or more) or --random-duration.")

    if args.random_duration is not None:
        rng = random.Random(args.seed)
        start_s, end_s = pick_random_window(t_s, fix_available, args.random_duration, rng)
        windows = [(start_s, end_s)]
        print(f"Randomly selected blackout window: [{start_s:.1f}, {end_s:.1f}]s (seed={args.seed})")
    else:
        windows = [(s, e) for s, e in args.window]
        for start_s, end_s in windows:
            validate_window(t_s, fix_available, start_s, end_s)

    masked, window_info = apply_blackout(df, t_s, windows)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    gt_path = Path(str(args.out_prefix) + "_ground_truth.csv")
    windows_path = Path(str(args.out_prefix) + "_windows.json")

    masked.to_csv(gt_path, index=False)
    windows_path.write_text(json.dumps(window_info, indent=2))

    print(f"Ground-truth with blackout applied (positions kept, fix_available flipped) -> {gt_path}")
    print(f"Window bookkeeping -> {windows_path}")
    for w in window_info:
        print(f"  blackout [{w['start_s']:.1f}, {w['end_s']:.1f}]s ({w['duration_s']:.1f}s, {w['n_rows_flipped']} rows flipped to fix_available=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
