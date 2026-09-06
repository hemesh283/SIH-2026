#!/usr/bin/env python3
"""
convert_trace.py — SIH26168 data/ pipeline

Takes a raw export from the "Sensor Logger" app (Kelvin Choi / tszheichoi)
and produces the two files physics-baseline/ and ml-residual/ actually
consume (see ml-residual/common.py and physics-baseline/strapdown.py for
the authoritative loaders this mirrors):

  <out_prefix>_imu.csv
      timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z[, mag_x, mag_y, mag_z]
      Raw epoch timestamp (same units strapdown.py auto-detects: s/ms/ns by
      magnitude), calibrated accelerometer in m/s^2 (includes gravity),
      calibrated gyroscope in rad/s. This is exactly what
      `python physics-baseline/strapdown.py <this file> baseline.csv` and
      `common.load_imu_csv(...)` expect.

  <out_prefix>_ground_truth.csv
      timestamp, pos_x, pos_y, pos_z, fix_available
      GPS resolved to a local ENU (East-North-Up) frame, meters, with the
      origin at the trace's first fix -- NOT lat/lon degrees, because
      ml-residual/common.load_ground_truth_csv and features.py both expect
      metric positions directly comparable to the physics baseline's
      pos_x/pos_y. fix_available is 1 for every row here; blackout_mask.py
      is what flips windows of it to 0 for GNSS-blackout evaluation.

Also writes, for the verification discipline in dossier Section 5.1:
  <out_prefix>_sanity.png   axis-order / stationarity plot
  <out_prefix>_report.json  unit/axis/timestamp sanity-check results

INPUT
-----
A folder produced by Sensor Logger's CSV export (unzip the .zip it saves),
containing at least Accelerometer.csv, Gyroscope.csv, Location.csv, and
optionally Magnetometer.csv. Columns: time (ns Unix epoch), seconds_elapsed,
x, y, z (accel m/s^2, gyro rad/s, both calibrated); Location.csv has time,
seconds_elapsed, latitude, longitude, altitude, speed, bearing, and
accuracy columns.

USAGE
-----
    python convert_trace.py data/raw_traces/20260904_walk_hostel_pixel7 \
        --out-prefix data/raw_traces/20260904_walk_hostel_pixel7 \
        --stationary-start 2.0 --stationary-end 2.0

(--out-prefix defaults to <export_folder>/<export_folder's own name>, i.e.
the two output CSVs land inside the same raw_traces/ subfolder as the
export they came from, per data/raw_traces/README.md's naming convention.)

(--stationary-start/--stationary-end: seconds at the very start/end of the
recording the phone was held still -- required by your collection protocol
and used here only for the unit/axis sanity checks below.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665  # standard gravity, m/s^2
EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius, for the local ENU approximation


def load_sensor_csv(folder: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    path = folder / name
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Expected '{name}' in {folder} (Sensor Logger CSV export). "
                f"Found: {sorted(p.name for p in folder.glob('*.csv'))}"
            )
        return None
    df = pd.read_csv(path)
    missing = {"time", "seconds_elapsed"} - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing expected columns: {missing}")
    return df


def to_epoch_seconds(time_raw: pd.Series) -> pd.Series:
    """Sensor Logger 'time' is a Unix epoch timestamp in nanoseconds."""
    return time_raw.astype(np.float64) / 1e9


def sampling_rate_stats(t_s: np.ndarray) -> dict:
    dt = np.diff(t_s)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {"median_hz": float("nan"), "dt_std_ms": float("nan"), "n_gaps_gt_5x_median": 0}
    median_dt = float(np.median(dt))
    return {
        "median_hz": round(1.0 / median_dt, 2) if median_dt > 0 else float("nan"),
        "dt_std_ms": round(float(np.std(dt)) * 1000, 3),
        "max_gap_ms": round(float(np.max(dt)) * 1000, 3),
        "n_gaps_gt_5x_median": int(np.sum(dt > 5 * median_dt)) if median_dt > 0 else 0,
    }


def run_sanity_checks(accel: pd.DataFrame, gyro: pd.DataFrame, stat_start_s: float, stat_end_s: float) -> dict:
    report: dict = {"checks": []}

    def add(name: str, passed: bool, detail: str):
        report["checks"].append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    t0, t_last = accel["t_s"].iloc[0], accel["t_s"].iloc[-1]
    stat_mask = (accel["t_s"] <= t0 + stat_start_s) | (accel["t_s"] >= t_last - stat_end_s)

    if stat_mask.sum() < 5:
        add("accel_unit_check", False, "Not enough samples in the declared stationary window to check units.")
    else:
        mean_norm = float(np.sqrt(accel.loc[stat_mask, ["x", "y", "z"]].pow(2).sum(axis=1)).mean())
        if 8.5 <= mean_norm <= 11.0:
            add("accel_unit_check", True, f"Stationary |a| = {mean_norm:.2f} m/s^2, consistent with m/s^2 incl. gravity.")
        elif 0.85 <= mean_norm <= 1.15:
            add("accel_unit_check", False, f"Stationary |a| = {mean_norm:.3f} — looks like G's, not m/s^2. Re-export with calibrated accelerometer in m/s^2.")
        elif mean_norm < 0.5:
            add("accel_unit_check", False, f"Stationary |a| = {mean_norm:.3f} — looks like gravity-removed 'linear acceleration'. Export 'Accelerometer', not 'LinearAcceleration' (strapdown.py double-subtracts gravity otherwise).")
        else:
            add("accel_unit_check", False, f"Stationary |a| = {mean_norm:.2f} m/s^2 — outside the expected ~9.8 m/s^2 band; check sensor/units.")

    gt0, gt_last = gyro["t_s"].iloc[0], gyro["t_s"].iloc[-1]
    gstat_mask = (gyro["t_s"] <= gt0 + stat_start_s) | (gyro["t_s"] >= gt_last - stat_end_s)
    if gstat_mask.sum() < 5:
        add("gyro_unit_check", False, "Not enough samples in the declared stationary window to check units.")
    else:
        mean_abs = float(gyro.loc[gstat_mask, ["x", "y", "z"]].abs().to_numpy().mean())
        if mean_abs <= 0.3:
            add("gyro_unit_check", True, f"Stationary mean |gyro| = {mean_abs:.4f} rad/s, consistent with a still phone in rad/s.")
        else:
            add("gyro_unit_check", False, f"Stationary mean |gyro| = {mean_abs:.4f} — too large for a still phone in rad/s (looks like deg/s, or the phone moved). strapdown.py has --gyro-in-degrees if this is genuinely deg/s.")

    accel_rate = sampling_rate_stats(accel["t_s"].to_numpy())
    gyro_rate = sampling_rate_stats(gyro["t_s"].to_numpy())
    report["accel_sampling"] = accel_rate
    report["gyro_sampling"] = gyro_rate
    add("accel_sampling_regular", accel_rate["n_gaps_gt_5x_median"] == 0,
        f"{accel_rate['n_gaps_gt_5x_median']} gap(s) > 5x median dt (median {accel_rate['median_hz']} Hz).")
    add("gyro_sampling_regular", gyro_rate["n_gaps_gt_5x_median"] == 0,
        f"{gyro_rate['n_gaps_gt_5x_median']} gap(s) > 5x median dt (median {gyro_rate['median_hz']} Hz).")

    clock_skew = abs(accel["t_s"].iloc[0] - gyro["t_s"].iloc[0])
    add("accel_gyro_clock_alignment", clock_skew < 1.0,
        f"Accel/gyro start times differ by {clock_skew:.3f}s (both use the phone's epoch clock; large skew suggests a bad export).")

    report["all_pass"] = all(c["status"] == "PASS" for c in report["checks"])
    return report


def make_sanity_plot(accel: pd.DataFrame, gyro: pd.DataFrame, loc: pd.DataFrame | None, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3 if loc is not None else 2, 1, figsize=(11, 8))

    ax0 = axes[0]
    for col, label in zip(["x", "y", "z"], ["acc_x", "acc_y", "acc_z"]):
        ax0.plot(accel["t_s"] - accel["t_s"].iloc[0], accel[col], label=label, linewidth=0.7)
    ax0.axhline(G, color="gray", linestyle="--", linewidth=0.8, label="+g")
    ax0.axhline(-G, color="gray", linestyle="--", linewidth=0.8)
    ax0.set_title("Accelerometer (m/s^2) — axis-order / sign sanity")
    ax0.set_ylabel("m/s^2")
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    for col, label in zip(["x", "y", "z"], ["gyro_x", "gyro_y", "gyro_z"]):
        ax1.plot(gyro["t_s"] - gyro["t_s"].iloc[0], gyro[col], label=label, linewidth=0.7)
    ax1.set_title("Gyroscope (rad/s)")
    ax1.set_ylabel("rad/s")
    ax1.legend(loc="upper right", fontsize=8)

    if loc is not None:
        ax2 = axes[2]
        ax2.plot(loc["pos_x"], loc["pos_y"], marker=".", linewidth=0.6, markersize=2)
        ax2.set_title("GPS track, local ENU meters (east vs north) — check it looks like the real route")
        ax2.set_xlabel("east (m)")
        ax2.set_ylabel("north (m)")
        ax2.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def latlon_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local tangent-plane (East-North-Up) approximation, origin = first fix.

    Flat-earth approximation using meters-per-degree at the origin latitude.
    Accurate to well under a meter of distortion over trace lengths of a
    few km (walk/drive test scale); do not reuse this for anything spanning
    tens of kilometers or near the poles.
    """
    lat0, lon0, alt0 = lat[0], lon[0], (alt[0] if not np.isnan(alt[0]) else 0.0)
    lat0_rad = np.radians(lat0)
    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat0_rad)

    east = (lon - lon0) * m_per_deg_lon
    north = (lat - lat0) * m_per_deg_lat
    up = np.nan_to_num(alt - alt0, nan=0.0)
    return east, north, up


def build_imu_csv(accel_raw: pd.DataFrame, gyro_raw: pd.DataFrame, mag_raw: pd.DataFrame | None) -> pd.DataFrame:
    accel = pd.DataFrame({
        "t_s": to_epoch_seconds(accel_raw["time"]),
        "timestamp": accel_raw["time"],
        "acc_x": accel_raw["x"], "acc_y": accel_raw["y"], "acc_z": accel_raw["z"],
    }).sort_values("t_s").reset_index(drop=True)

    gyro = pd.DataFrame({
        "t_s": to_epoch_seconds(gyro_raw["time"]),
        "gyro_x": gyro_raw["x"], "gyro_y": gyro_raw["y"], "gyro_z": gyro_raw["z"],
    }).sort_values("t_s").reset_index(drop=True)

    merged = pd.merge_asof(accel, gyro, on="t_s", direction="nearest", tolerance=0.05)

    if mag_raw is not None:
        mag = pd.DataFrame({
            "t_s": to_epoch_seconds(mag_raw["time"]),
            "mag_x": mag_raw["x"], "mag_y": mag_raw["y"], "mag_z": mag_raw["z"],
        }).sort_values("t_s").reset_index(drop=True)
        merged = pd.merge_asof(merged, mag, on="t_s", direction="nearest", tolerance=0.05)

    required = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    merged = merged.dropna(subset=required).reset_index(drop=True)

    cols = ["timestamp"] + required + (["mag_x", "mag_y", "mag_z"] if mag_raw is not None else [])
    return merged[cols]


def build_ground_truth_csv(loc_raw: pd.DataFrame) -> pd.DataFrame:
    loc_raw = loc_raw.sort_values("time").reset_index(drop=True)
    lat = loc_raw["latitude"].to_numpy(dtype=np.float64)
    lon = loc_raw["longitude"].to_numpy(dtype=np.float64)
    alt = loc_raw["altitude"].to_numpy(dtype=np.float64) if "altitude" in loc_raw.columns else np.full(len(loc_raw), np.nan)

    east, north, up = latlon_to_enu(lat, lon, alt)

    out = pd.DataFrame({
        "timestamp": loc_raw["time"],
        "pos_x": east,
        "pos_y": north,
        "pos_z": up,
        "fix_available": 1,
    })
    # Informational extras -- harmless to downstream loaders (they only
    # require timestamp/pos_x/pos_y[/pos_z]/fix_available), useful for
    # sanity-checking the ENU conversion or debugging a trace later.
    if "speed" in loc_raw.columns:
        out["speed_mps_reported"] = loc_raw["speed"]
    if "bearing" in loc_raw.columns:
        out["bearing_deg_reported"] = loc_raw["bearing"]
    if "horizontalAccuracy" in loc_raw.columns:
        out["horizontal_accuracy_m"] = loc_raw["horizontalAccuracy"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_folder", type=Path, help="Unzipped Sensor Logger export folder")
    ap.add_argument("--out-prefix", type=Path, default=None,
                     help="Output path prefix. Default: <export_folder>/<export_folder name> "
                          "(so outputs land next to the raw export inside data/raw_traces/).")
    ap.add_argument("--stationary-start", type=float, default=2.0, help="Seconds at trace start the phone was held still (default 2.0)")
    ap.add_argument("--stationary-end", type=float, default=2.0, help="Seconds at trace end the phone was held still (default 2.0)")
    ap.add_argument("--allow-fail", action="store_true", help="Write outputs even if sanity checks FAIL (default: abort)")
    args = ap.parse_args()

    folder = args.export_folder
    out_prefix = args.out_prefix or (folder / folder.name)

    accel_raw = load_sensor_csv(folder, "Accelerometer.csv")
    gyro_raw = load_sensor_csv(folder, "Gyroscope.csv")
    mag_raw = load_sensor_csv(folder, "Magnetometer.csv", required=False)
    loc_raw = load_sensor_csv(folder, "Location.csv", required=False)
    if loc_raw is None:
        print("WARNING: no Location.csv found — no ground_truth file will be written. "
              "The physics baseline can still run on the IMU file alone, but nothing can be evaluated against it.", file=sys.stderr)

    accel_t = pd.DataFrame({"t_s": to_epoch_seconds(accel_raw["time"]), "x": accel_raw["x"], "y": accel_raw["y"], "z": accel_raw["z"]})
    gyro_t = pd.DataFrame({"t_s": to_epoch_seconds(gyro_raw["time"]), "x": gyro_raw["x"], "y": gyro_raw["y"], "z": gyro_raw["z"]})
    report = run_sanity_checks(accel_t, gyro_t, args.stationary_start, args.stationary_end)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    imu_df = build_imu_csv(accel_raw, gyro_raw, mag_raw)
    gt_df = build_ground_truth_csv(loc_raw) if loc_raw is not None else None

    plot_loc = gt_df[["pos_x", "pos_y"]] if gt_df is not None else None
    make_sanity_plot(accel_t, gyro_t, plot_loc, Path(str(out_prefix) + "_sanity.png"))

    report_path = Path(str(out_prefix) + "_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Sanity plot -> {out_prefix}_sanity.png")
    print(f"Sanity report -> {report_path}")
    for c in report["checks"]:
        print(f"  [{c['status']}] {c['check']}: {c['detail']}")

    if not report["all_pass"] and not args.allow_fail:
        print("\nOne or more sanity checks FAILED. Fix the issue and re-export, or re-run with --allow-fail "
              "to write outputs anyway (not recommended for training data — Section 5.1 verification discipline).",
              file=sys.stderr)
        return 1

    imu_path = Path(str(out_prefix) + "_imu.csv")
    imu_df.to_csv(imu_path, index=False)
    print(f"IMU file ({len(imu_df)} rows) -> {imu_path}")

    if gt_df is not None:
        gt_path = Path(str(out_prefix) + "_ground_truth.csv")
        gt_df.to_csv(gt_path, index=False)
        print(f"Ground-truth file ({len(gt_df)} rows, local ENU meters) -> {gt_path}")
        span_m = float(np.hypot(gt_df["pos_x"].iloc[-1], gt_df["pos_y"].iloc[-1]))
        print(f"  net displacement from start: {span_m:.1f} m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
