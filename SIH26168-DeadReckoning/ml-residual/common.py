"""Shared IO helpers for the ml-residual pipeline.

physics-baseline/ and ml-residual/ are sibling directories with hyphens in
their names, so `import physics_baseline` isn't available as a normal
package import -- load strapdown.py by file path instead. This also keeps
both stages agreeing on the same column names, units, and the
timestamp-normalization rule, instead of a second, drifting copy of that
logic living here.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_BASELINE_PATH = Path(__file__).resolve().parent.parent / "physics-baseline" / "strapdown.py"


def _load_strapdown_module():
    spec = importlib.util.spec_from_file_location("strapdown", _BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strapdown = _load_strapdown_module()


def load_imu_csv(path, time_unit="auto", gyro_in_degrees=False):
    """Raw IMU trace -- delegates to physics-baseline's loader.

    Returns (df, has_mag, time_unit) where time_unit is whichever unit was
    used (resolved from "auto" if needed), so callers can normalize other
    files from the same recording session onto the same origin/scale.
    """
    df = pd.read_csv(path)
    raw_ts = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df.columns else None
    resolved_unit = _resolve_time_unit(raw_ts, time_unit) if raw_ts is not None else time_unit
    loaded, has_mag = strapdown.load_imu_csv(path, time_unit=time_unit, gyro_in_degrees=gyro_in_degrees)
    return loaded, has_mag, resolved_unit


def _resolve_time_unit(raw_ts, time_unit):
    if time_unit != "auto":
        return time_unit
    magnitude = np.median(np.abs(raw_ts))
    if magnitude > 1e17:
        return "ns"
    if magnitude > 1e12:
        return "ms"
    return "s"


def normalize_to_origin(raw_ts, origin_raw, time_unit):
    """Seconds elapsed since `origin_raw`, in `raw_ts`'s own raw units.

    Unlike strapdown._normalize_time (which anchors each array to its own
    first sample), this anchors every file to one shared origin -- the
    first IMU sample's raw timestamp -- so IMU, baseline, and ground-truth
    timestamps land on the same clock.
    """
    scale = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}[time_unit]
    return (raw_ts - origin_raw) * scale


def load_baseline_csv(path, origin_raw=None, time_unit="s"):
    """physics-baseline output: timestamp, pos_x/y/z, velocity (speed), orientation (deg).

    strapdown.run_pipeline() writes back the *original* raw timestamp
    column unchanged, so it needs the same origin/unit normalization as the
    IMU file it came from.
    """
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "pos_x", "pos_y", "pos_z", "velocity", "orientation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"baseline CSV missing required columns: {sorted(missing)}")
    if origin_raw is not None:
        df["t"] = normalize_to_origin(df["timestamp"].to_numpy(dtype=np.float64), origin_raw, time_unit)
    return df


def load_ground_truth_csv(path, origin_raw=None, time_unit="s"):
    """Ground-truth GPS, resolved to the same local ENU frame as the baseline.

    Columns: timestamp, pos_x, pos_y[, pos_z], and optionally
    `fix_available` (1 = a real GNSS fix was available at this timestamp,
    0 = this row falls inside a simulated blackout window).

    Position is required even where fix_available=0 -- that is the withheld
    true label the residual model is trained and scored against, per the
    dossier's Section 5.1 methodology: GPS labels are masked from the
    *baseline's* view, never deleted from the dataset. If `fix_available`
    is absent, every row is treated as available (no blackout to learn
    from -- fine for a smoke test, not for real training).
    """
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "pos_x", "pos_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground-truth CSV missing required columns: {sorted(missing)}")
    if "pos_z" not in df.columns:
        df["pos_z"] = 0.0
    if "fix_available" not in df.columns:
        df["fix_available"] = 1
    if origin_raw is not None:
        df["t"] = normalize_to_origin(df["timestamp"].to_numpy(dtype=np.float64), origin_raw, time_unit)
    return df
