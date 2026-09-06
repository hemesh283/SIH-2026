"""Windowed IMU feature engineering for the residual model (dossier Section 4.2).

Each output row summarizes one tumbling window of raw IMU samples ending at
`t_end`, plus the residual target at that same instant:

    resid_pos_x  = true_pos_x(t_end)  - baseline_pos_x(t_end)
    resid_pos_y  = true_pos_y(t_end)  - baseline_pos_y(t_end)
    resid_speed  = true_speed(t_end)  - baseline_speed(t_end)

i.e. the baseline's own error against ground truth -- the thing the
LightGBM model is trained to predict and cancel out. `true_speed` is the
finite-difference speed of the ground-truth position track (GPS doesn't
usually ship a clean instantaneous speed, so we derive one the same way the
baseline's speed was derived: from position).

Feature groups (why each one is here):

- accel/gyro mean & std, per axis and magnitude: the direct MEMS-noise and
  motion-level statistics the dossier calls for. Mean captures sustained
  acceleration/rotation (cornering, braking); std captures how much the
  signal is bouncing around within the window.
- turn-rate signature: mean/max |yaw rate| plus net heading change over the
  window. Mean-vs-max distinguishes a sustained turn from a single sharp
  swerve; net heading change distinguishes an actual turn from steering
  oscillation that cancels out (both can have similar mean |yaw rate|).
- vibration/noise: std of the *detrended* accel/gyro magnitude -- i.e. the
  high-frequency component left after removing the window's own rolling
  mean. This isolates road/engine vibration from the bulk acceleration
  change already captured by acc_std, which would otherwise conflate "the
  car sped up" with "the road is bumpy".
- time_since_last_gnss_fix: elapsed time since the most recent available
  fix. This is the single feature that lets the model learn how baseline
  error grows with blackout duration, and it is also the axis Section 4.3's
  uncertainty buckets are defined over.
- motion_mode: a cheap rule-based classifier (stationary / pedestrian /
  vehicle), same spirit as strapdown.py's ZUPT variance gate -- not learned,
  and its thresholds need the same "tune against a real recorded stationary
  vs. moving segment" caveat before trusting it on real data.
- baseline_speed: the baseline's own current speed estimate. Included
  because drift compounds with distance travelled, not just elapsed time
  (see the drift-rate-per-distance metric in evaluation/), so the model
  needs to know how fast the baseline thinks it's going, not just for how
  long it's been guessing.

Deliberately NOT included: baseline heading. Heading is circular (wraps at
360deg), and folding it in cleanly (sin/cos features) is real but optional
work -- the four listed categories above are the ones the dossier commits
to, so heading is left as a documented future extension rather than
half-implemented here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "acc_mean_x", "acc_mean_y", "acc_mean_z",
    "acc_std_x", "acc_std_y", "acc_std_z",
    "acc_mag_mean", "acc_mag_std",
    "gyro_mean_x", "gyro_mean_y", "gyro_mean_z",
    "gyro_std_x", "gyro_std_y", "gyro_std_z",
    "gyro_mag_mean", "gyro_mag_std",
    "turn_rate_mean_abs", "turn_rate_max_abs", "heading_change_abs",
    "vibration_acc", "vibration_gyro",
    "time_since_last_gnss_fix", "motion_mode", "baseline_speed",
    "window_n_samples",
]

TARGET_COLUMNS = ["resid_pos_x", "resid_pos_y", "resid_speed"]

MOTION_STATIONARY = "stationary"
MOTION_PEDESTRIAN = "pedestrian"
MOTION_VEHICLE = "vehicle"


def _detrended_std(x, smooth_window):
    """Std of x after removing its own rolling mean -- the "noise left over
    once you subtract the slow trend" proxy for vibration."""
    if len(x) < 2:
        return 0.0
    trend = pd.Series(x).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    return float(np.std(x - trend))


def classify_motion_mode(acc_std_mag, gyro_std_mag, baseline_speed,
                          stationary_acc_std=0.03, stationary_gyro_std=0.02,
                          vehicle_speed_thresh=3.0):
    """Rule-based motion mode for one window.

    Thresholds are a starting point, not a calibrated result -- fit against
    real recorded stationary/walking/driving segments from your own data
    before trusting this on a real trace (same caveat strapdown.py's ZUPT
    gate carries).
    """
    if acc_std_mag < stationary_acc_std and gyro_std_mag < stationary_gyro_std:
        return MOTION_STATIONARY
    if baseline_speed >= vehicle_speed_thresh:
        return MOTION_VEHICLE
    return MOTION_PEDESTRIAN


def time_since_last_fix(sample_t, fix_t):
    """Seconds since the most recent fix in `fix_t` at or before each time in
    `sample_t`. NaN before the first fix (nothing to measure "since" yet)."""
    fix_t = np.sort(np.asarray(fix_t, dtype=np.float64))
    sample_t = np.asarray(sample_t, dtype=np.float64)
    idx = np.searchsorted(fix_t, sample_t, side="right") - 1
    out = np.full(len(sample_t), np.nan)
    valid = idx >= 0
    out[valid] = sample_t[valid] - fix_t[idx[valid]]
    return out


def _window_bounds(t, window_s, stride_s):
    t0, t1 = t[0], t[-1]
    starts = np.arange(t0, t1 - window_s + 1e-9, stride_s)
    if len(starts) == 0:
        starts = np.array([t0])
    return starts, starts + window_s


def build_features_and_targets(
    imu_df,
    baseline_df,
    gnss_df,
    window_s=2.0,
    stride_s=1.0,
    trace_id=None,
    stationary_acc_std=0.03,
    stationary_gyro_std=0.02,
    vehicle_speed_thresh=3.0,
):
    """Build one row of features + residual targets per tumbling window.

    imu_df, baseline_df, gnss_df must already share a common "t" column
    (seconds since the same origin) -- see common.py's loaders, which
    normalize all three onto the IMU file's start time.

    Windows ending before the first available GNSS fix are dropped: without
    a fix yet, "time since last fix" is undefined, and in real deployment
    the system always initializes from a trustworthy fix (dossier Q12)
    rather than running blind from t=0.
    """
    t = imu_df["t"].to_numpy()
    acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = imu_df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    fix_t = gnss_df.loc[gnss_df["fix_available"] == 1, "t"].to_numpy()
    if len(fix_t) == 0:
        raise ValueError("ground-truth data has no rows with fix_available == 1 -- "
                          "nothing to anchor time_since_last_gnss_fix or baseline "
                          "re-initialization to")

    bt = baseline_df["t"].to_numpy()
    gt = gnss_df["t"].to_numpy()

    starts, ends = _window_bounds(t, window_s, stride_s)
    rows = []

    for start, end in zip(starts, ends):
        if end < fix_t[0]:
            continue

        lo = np.searchsorted(t, start, side="left")
        hi = np.searchsorted(t, end, side="right")
        if hi - lo < 2:
            continue

        a = acc[lo:hi]
        g = gyro[lo:hi]
        a_mag = acc_mag[lo:hi]
        g_mag = gyro_mag[lo:hi]
        dt = np.diff(t[lo:hi])

        smooth_window = max(3, (hi - lo) // 4)
        yaw_rate = g[:, 2]
        heading_change_abs = float(np.abs(np.sum(yaw_rate[:-1] * dt))) if len(dt) else 0.0

        baseline_speed = float(np.interp(end, bt, baseline_df["velocity"].to_numpy()))
        acc_std_mag = float(np.std(a_mag))
        gyro_std_mag = float(np.std(g_mag))

        motion_mode = classify_motion_mode(
            acc_std_mag, gyro_std_mag, baseline_speed,
            stationary_acc_std, stationary_gyro_std, vehicle_speed_thresh,
        )

        feat = {
            "t_end": end,
            "acc_mean_x": float(np.mean(a[:, 0])), "acc_mean_y": float(np.mean(a[:, 1])), "acc_mean_z": float(np.mean(a[:, 2])),
            "acc_std_x": float(np.std(a[:, 0])), "acc_std_y": float(np.std(a[:, 1])), "acc_std_z": float(np.std(a[:, 2])),
            "acc_mag_mean": float(np.mean(a_mag)), "acc_mag_std": acc_std_mag,
            "gyro_mean_x": float(np.mean(g[:, 0])), "gyro_mean_y": float(np.mean(g[:, 1])), "gyro_mean_z": float(np.mean(g[:, 2])),
            "gyro_std_x": float(np.std(g[:, 0])), "gyro_std_y": float(np.std(g[:, 1])), "gyro_std_z": float(np.std(g[:, 2])),
            "gyro_mag_mean": float(np.mean(g_mag)), "gyro_mag_std": gyro_std_mag,
            "turn_rate_mean_abs": float(np.mean(np.abs(yaw_rate))),
            "turn_rate_max_abs": float(np.max(np.abs(yaw_rate))),
            "heading_change_abs": heading_change_abs,
            "vibration_acc": _detrended_std(a_mag, smooth_window),
            "vibration_gyro": _detrended_std(g_mag, smooth_window),
            "time_since_last_gnss_fix": float(time_since_last_fix(np.array([end]), fix_t)[0]),
            "motion_mode": motion_mode,
            "baseline_speed": baseline_speed,
            "window_n_samples": int(hi - lo),
        }

        true_pos_x = np.interp(end, gt, gnss_df["pos_x"].to_numpy())
        true_pos_y = np.interp(end, gt, gnss_df["pos_y"].to_numpy())
        base_pos_x = np.interp(end, bt, baseline_df["pos_x"].to_numpy())
        base_pos_y = np.interp(end, bt, baseline_df["pos_y"].to_numpy())

        eps = 1e-3
        true_speed = float(np.linalg.norm([
            np.interp(end + eps, gt, gnss_df["pos_x"].to_numpy()) - np.interp(end - eps, gt, gnss_df["pos_x"].to_numpy()),
            np.interp(end + eps, gt, gnss_df["pos_y"].to_numpy()) - np.interp(end - eps, gt, gnss_df["pos_y"].to_numpy()),
        ]) / (2 * eps))

        feat["resid_pos_x"] = float(true_pos_x - base_pos_x)
        feat["resid_pos_y"] = float(true_pos_y - base_pos_y)
        feat["resid_speed"] = float(true_speed - baseline_speed)
        feat["trace_id"] = trace_id

        rows.append(feat)

    if not rows:
        raise ValueError("no windows produced -- check that imu/baseline/gnss traces overlap "
                          "and that a GNSS fix occurs before the trace ends")

    return pd.DataFrame(rows)
