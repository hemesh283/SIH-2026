"""Mock stand-in for the real pipeline (physics-baseline/ + ml-residual/).

This is the ONLY module that will need to change once the real strapdown
mechanization (`physics-baseline/strapdown.py`) and the trained LightGBM
residual model (`ml-residual/`) are ready: every router calls this module's
functions, never numpy directly, so swapping mock -> real is a drop-in
replacement behind the same function signatures.

Everything here is deterministic per trace_id (seeded RNG), so repeated GET
requests for the same trace return byte-identical trajectories -- important
for a stable API contract while the Android client is built against it.

Simulation model (all mock, not measured):
  1. A smooth "true" path: a pedestrian-speed random walk with slowly
     wandering heading and occasional stops (stands in for ground truth).
  2. The "baseline" (naive strapdown/ZUPT output): the true path plus a
     compounding heading-drift bias and sqrt(t)-scaled position noise --
     the qualitative failure mode strapdown.py's own docstring describes
     (yaw drift without a magnetometer; residual double-integration noise
     ZUPT doesn't fully cancel).
  3. The "corrected" path (baseline + mock ML residual): the true path
     plus a small residual error -- i.e. the mock correction cancels most,
     not all, of the baseline's drift, so the two endpoints are visibly
     different (dossier Section 7's "naive vs corrected path").
  4. Evaluation metrics (ATE/RTE/CEP/drift-rate) are computed from the
     corrected path against the true path, so the numbers stay internally
     consistent with what /corrected actually returns.
"""
import hashlib
from datetime import datetime, timezone

import numpy as np

from timeutil import to_raw

RTE_WINDOW_S = 60.0
UNCERTAINTY_COVERAGE = 0.9

# Mock calibration -- stand-in for ml-residual/uncertainty.py's
# cross-validated, held-out CEP-90-per-bucket radii (dossier Section 4.3).
UNCERTAINTY_BUCKETS = [
    {"label": "0-30s", "lo_s": 0.0, "hi_s": 30.0, "radius_m": 3.5},
    {"label": "30-90s", "lo_s": 30.0, "hi_s": 90.0, "radius_m": 9.0},
    {"label": "90s+", "lo_s": 90.0, "hi_s": None, "radius_m": 22.0},
]

MAX_TRAJECTORY_POINTS = 3000
OUTPUT_HZ = 5.0  # mock output cadence; the real baseline runs at IMU rate


def _seed(trace_id: str) -> int:
    return int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16)


def _rng(trace_id: str) -> np.random.Generator:
    return np.random.default_rng(_seed(trace_id))


def time_grid(duration_s: float) -> np.ndarray:
    """Shared t-axis for baseline/corrected/uncertainty, so all three
    responses for one trace line up point-for-point."""
    duration_s = max(duration_s, 0.0)
    n = int(min(MAX_TRAJECTORY_POINTS, max(2, round(duration_s * OUTPUT_HZ) + 1)))
    return np.linspace(0.0, duration_s, n)


def _smoothed_series(rng: np.random.Generator, n: int, step_std: float, window: int) -> np.ndarray:
    raw = rng.normal(0.0, step_std, size=n)
    if window > 1 and n > 1:
        w = min(window, n)
        kernel = np.ones(w) / w
        raw = np.convolve(raw, kernel, mode="same")
    return raw


def _true_path(rng: np.random.Generator, t: np.ndarray):
    """Pedestrian-speed smooth random walk: heading wanders slowly, speed
    dips toward zero at a few random 'stops'. Stands in for ground truth."""
    n = len(t)
    dt = np.diff(t, prepend=t[0])

    heading_rate = _smoothed_series(rng, n, step_std=0.12, window=25)  # rad/s wander
    heading = np.cumsum(heading_rate * dt)

    speed_wander = _smoothed_series(rng, n, step_std=0.06, window=25)
    base_speed = 1.3 + speed_wander

    n_stops = max(0, int(rng.integers(0, 3)))
    for _ in range(n_stops):
        center = rng.uniform(t[0], t[-1] + 1e-6)
        half_width = rng.uniform(1.0, 3.5)
        dip = np.exp(-0.5 * ((t - center) / max(half_width, 1e-6)) ** 2)
        base_speed = base_speed * (1.0 - 0.95 * dip)

    speed = np.clip(base_speed, 0.0, None)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    pos_x = np.concatenate(([0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * np.diff(t))))
    pos_y = np.concatenate(([0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * np.diff(t))))
    pos_z = _smoothed_series(rng, n, step_std=0.03, window=15).cumsum() * 0.02

    return pos_x, pos_y, pos_z, speed, heading


def _baseline_from_true(rng: np.random.Generator, t: np.ndarray, true_pos_x, true_pos_y, true_pos_z, true_speed, true_heading):
    """True path + compounding heading-drift bias + sqrt(t)-scaled noise."""
    n = len(t)
    heading_bias_rate = rng.uniform(-0.03, 0.03)  # rad/s constant yaw-drift bias
    speed_bias = rng.uniform(-0.04, 0.06)

    drift_heading = true_heading + heading_bias_rate * t
    drift_speed = np.clip(true_speed * (1.0 + speed_bias), 0.0, None)

    vx = drift_speed * np.cos(drift_heading)
    vy = drift_speed * np.sin(drift_heading)
    pos_x = np.concatenate(([0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * np.diff(t))))
    pos_y = np.concatenate(([0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * np.diff(t))))

    growth = np.sqrt(np.maximum(t, 0.0))
    pos_x = pos_x + _smoothed_series(rng, n, step_std=0.15, window=9) * growth
    pos_y = pos_y + _smoothed_series(rng, n, step_std=0.15, window=9) * growth
    pos_z = true_pos_z + _smoothed_series(rng, n, step_std=0.02, window=9) * growth

    return pos_x, pos_y, pos_z, drift_speed, np.degrees(drift_heading) % 360.0


def _corrected_from_baseline(rng: np.random.Generator, true_pos_x, true_pos_y, true_pos_z, true_speed, true_heading,
                              base_pos_x, base_pos_y, base_pos_z):
    """True path + a small residual (the mock ML correction cancels most,
    not all, of the baseline's drift)."""
    n = len(true_pos_x)
    residual_fraction = rng.uniform(0.12, 0.25)
    noise = _smoothed_series(rng, n, step_std=0.08, window=9)

    pos_x = true_pos_x + residual_fraction * (base_pos_x - true_pos_x) + noise
    pos_y = true_pos_y + residual_fraction * (base_pos_y - true_pos_y) + noise[::-1] * 0.5
    pos_z = true_pos_z + residual_fraction * (base_pos_z - true_pos_z) * 0.3

    speed = np.clip(true_speed + _smoothed_series(rng, n, step_std=0.03, window=9), 0.0, None)
    heading_deg = np.degrees(true_heading) % 360.0

    return pos_x, pos_y, pos_z, speed, heading_deg


def _to_points(t, timestamp_raw, pos_x, pos_y, pos_z, velocity, orientation) -> list[dict]:
    return [
        {
            "t": float(t[i]),
            "timestamp": float(timestamp_raw[i]),
            "pos_x": float(pos_x[i]),
            "pos_y": float(pos_y[i]),
            "pos_z": float(pos_z[i]),
            "velocity": float(velocity[i]),
            "orientation": float(orientation[i]),
        }
        for i in range(len(t))
    ]


def _simulate(trace_id: str, duration_s: float):
    """Core deterministic simulation, shared by baseline/corrected/evaluation
    so the three stay numerically consistent with each other."""
    rng = _rng(trace_id)
    t = time_grid(duration_s)

    true_pos_x, true_pos_y, true_pos_z, true_speed, true_heading = _true_path(rng, t)
    base_pos_x, base_pos_y, base_pos_z, base_speed, base_heading_deg = _baseline_from_true(
        rng, t, true_pos_x, true_pos_y, true_pos_z, true_speed, true_heading
    )
    corr_pos_x, corr_pos_y, corr_pos_z, corr_speed, corr_heading_deg = _corrected_from_baseline(
        rng, true_pos_x, true_pos_y, true_pos_z, true_speed, true_heading,
        base_pos_x, base_pos_y, base_pos_z,
    )

    return {
        "t": t,
        "true": (true_pos_x, true_pos_y, true_pos_z),
        "baseline": (base_pos_x, base_pos_y, base_pos_z, base_speed, base_heading_deg),
        "corrected": (corr_pos_x, corr_pos_y, corr_pos_z, corr_speed, corr_heading_deg),
    }


def generate_baseline(trace_id: str, duration_s: float, start_timestamp_raw: float, time_unit: str) -> list[dict]:
    sim = _simulate(trace_id, duration_s)
    t = sim["t"]
    timestamp_raw = to_raw(t, start_timestamp_raw, time_unit)
    pos_x, pos_y, pos_z, speed, heading_deg = sim["baseline"]
    return _to_points(t, timestamp_raw, pos_x, pos_y, pos_z, speed, heading_deg)


def generate_corrected(trace_id: str, duration_s: float, start_timestamp_raw: float, time_unit: str) -> list[dict]:
    sim = _simulate(trace_id, duration_s)
    t = sim["t"]
    timestamp_raw = to_raw(t, start_timestamp_raw, time_unit)
    pos_x, pos_y, pos_z, speed, heading_deg = sim["corrected"]
    return _to_points(t, timestamp_raw, pos_x, pos_y, pos_z, speed, heading_deg)


def _blackout_duration(t: np.ndarray, fix_times: list[float]) -> np.ndarray:
    """Seconds since the most recent GNSS fix at or before each t (or since
    trace start, if no fix has occurred yet) -- mirrors
    ml-residual/features.py's time_since_last_fix, but never NaN: with no
    fix at all, the whole trace is treated as one continuous blackout."""
    if not fix_times:
        return t.copy()
    fix_t = np.sort(np.asarray(fix_times, dtype=np.float64))
    idx = np.searchsorted(fix_t, t, side="right") - 1
    out = t.copy()
    has_fix = idx >= 0
    out[has_fix] = t[has_fix] - fix_t[idx[has_fix]]
    return out


def _bucket_for(duration_s: float) -> dict:
    for b in UNCERTAINTY_BUCKETS:
        if b["hi_s"] is None or duration_s < b["hi_s"]:
            if duration_s >= b["lo_s"]:
                return b
    return UNCERTAINTY_BUCKETS[-1]


def generate_uncertainty(trace_id: str, duration_s: float, start_timestamp_raw: float, time_unit: str,
                          fix_times: list[float]) -> dict:
    rng = _rng(trace_id)
    t = time_grid(duration_s)
    timestamp_raw = to_raw(t, start_timestamp_raw, time_unit)
    blackout = _blackout_duration(t, fix_times)

    jitter = rng.normal(1.0, 0.05, size=len(t))
    points = []
    for i in range(len(t)):
        bucket = _bucket_for(float(blackout[i]))
        radius = max(0.5, bucket["radius_m"] * float(jitter[i]))
        points.append({
            "t": float(t[i]),
            "timestamp": float(timestamp_raw[i]),
            "blackout_duration_s": float(blackout[i]),
            "bucket": bucket["label"],
            "radius_m": radius,
        })

    return {
        "coverage": UNCERTAINTY_COVERAGE,
        "buckets": UNCERTAINTY_BUCKETS,
        "points": points,
    }


def generate_evaluation_metrics(trace_id: str, duration_s: float) -> dict:
    sim = _simulate(trace_id, duration_s)
    t = sim["t"]
    true_x, true_y, _ = sim["true"]
    corr_x, corr_y, _, _, _ = sim["corrected"]

    err = np.sqrt((corr_x - true_x) ** 2 + (corr_y - true_y) ** 2)
    ate_m = float(np.sqrt(np.mean(err ** 2)))
    cep50_m = float(np.percentile(err, 50))
    cep90_m = float(np.percentile(err, 90))

    rte_vals = []
    if len(t) > 1 and t[-1] > 0:
        window_starts = np.arange(t[0], t[-1], RTE_WINDOW_S)
        for ws in window_starts:
            we = ws + RTE_WINDOW_S
            mask = (t >= ws) & (t <= we)
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue
            dx_true = true_x[idx[-1]] - true_x[idx[0]]
            dy_true = true_y[idx[-1]] - true_y[idx[0]]
            dx_corr = corr_x[idx[-1]] - corr_x[idx[0]]
            dy_corr = corr_y[idx[-1]] - corr_y[idx[0]]
            rte_vals.append(np.hypot(dx_corr - dx_true, dy_corr - dy_true))
    rte_m = float(np.mean(rte_vals)) if rte_vals else 0.0

    distance_m = float(np.sum(np.hypot(np.diff(true_x), np.diff(true_y))))
    final_err = float(err[-1]) if len(err) else 0.0
    drift_rate_pct = float(100.0 * final_err / distance_m) if distance_m > 1e-6 else 0.0

    return {
        "ate_m": ate_m,
        "rte_m": rte_m,
        "rte_window_s": RTE_WINDOW_S,
        "cep50_m": cep50_m,
        "cep90_m": cep90_m,
        "drift_rate_pct": drift_rate_pct,
        "distance_m": distance_m,
    }
