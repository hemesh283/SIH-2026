"""
Strapdown inertial mechanization -- physics baseline for SIH26168.

Pipeline (dossier Section 4.1):
  1. Orientation: gyro integration fused with accel (roll/pitch) and,
     if present, magnetometer (yaw) via a complementary filter.
  2. Rotate body-frame accelerometer readings into the nav frame and
     subtract gravity to recover true linear acceleration.
  3. Trapezoidal double integration -> velocity, position.
  4. ZUPT: variance-gated stationary detection resets velocity to zero.

Frame conventions:
  Body frame  -- phone axes, Android/Sensor-Logger convention (x-right,
                 y-up-the-screen, z-out-of-the-screen). Flat on a table,
                 screen up, a stationary accelerometer reads ~[0,0,+g].
  Nav frame   -- local ENU (East, North, Up). Gravity is [0, 0, -g].

Known simplifications (this is the "start simple" baseline, not the
Section-4.2 EKF upgrade):
  - Gyro rates are integrated directly as roll/pitch/yaw rates. This is
    the standard small-angle complementary-filter approximation used in
    most introductory implementations; it is not the exact Euler-rate
    transform and degrades near pitch = +/-90 deg. Fine for phones
    carried upright/in-pocket; not fine for a phone tumbling in free
    fall.
  - Without a magnetometer, yaw has no absolute reference and is pure
    gyro integration -- it *will* drift. Sensor Logger exports without
    mag_x/mag_y/mag_z columns still run, degraded, rather than failing.
  - acc_x/y/z must be the raw accelerometer (specific force, includes
    gravity, ~9.8 m/s^2 magnitude at rest) -- not a pre-fused "Linear
    Acceleration" virtual sensor, which has already removed gravity and
    would double-subtract it here.
"""
import argparse

import numpy as np
import pandas as pd

GRAVITY = 9.80665  # m/s^2, standard gravity magnitude


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_imu_csv(path, time_unit="auto", gyro_in_degrees=False):
    df = pd.read_csv(path)

    required = {"timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing required columns: {sorted(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["t"] = _normalize_time(df["timestamp"].to_numpy(dtype=np.float64), time_unit)

    if gyro_in_degrees:
        for c in ("gyro_x", "gyro_y", "gyro_z"):
            df[c] = np.radians(df[c])

    has_mag = {"mag_x", "mag_y", "mag_z"}.issubset(df.columns)
    return df, has_mag


def _normalize_time(raw, time_unit):
    """Convert timestamps to seconds elapsed since the first sample.

    Sensor Logger-style exports show up as unix seconds, milliseconds, or
    nanoseconds depending on app/version; auto-detect by magnitude.
    """
    if time_unit == "auto":
        magnitude = np.median(np.abs(raw))
        if magnitude > 1e17:
            time_unit = "ns"
        elif magnitude > 1e12:
            time_unit = "ms"
        else:
            time_unit = "s"
    scale = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}[time_unit]
    t = raw * scale
    return t - t[0]


# --------------------------------------------------------------------------
# Orientation -- complementary filter
# --------------------------------------------------------------------------

def _tilt_from_accel(a):
    """Roll/pitch implied by the gravity direction in a stationary accel reading."""
    ax, ay, az = a
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    return roll, pitch


def _yaw_from_mag(m, roll, pitch):
    """Tilt-compensated magnetic heading (uncorrected for magnetic declination)."""
    mx, my, mz = m
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    mx_h = mx * cp + mz * sp
    my_h = mx * sr * sp + my * cr - mz * sr * cp
    return np.arctan2(-my_h, mx_h)


def _blend_angle(alpha, angle_gyro, angle_ref):
    """Complementary blend of two angles, wraparound-aware near +/-pi."""
    diff = np.arctan2(np.sin(angle_ref - angle_gyro), np.cos(angle_ref - angle_gyro))
    return angle_gyro + (1.0 - alpha) * diff


def complementary_filter_orientation(df, has_mag, alpha=0.98):
    """Fuse gyro/accel(/mag) into roll, pitch, yaw (radians) at every sample."""
    n = len(df)
    t = df["t"].to_numpy()
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    mag = df[["mag_x", "mag_y", "mag_z"]].to_numpy() if has_mag else None

    roll = np.zeros(n)
    pitch = np.zeros(n)
    yaw = np.zeros(n)

    roll[0], pitch[0] = _tilt_from_accel(acc[0])
    yaw[0] = _yaw_from_mag(mag[0], roll[0], pitch[0]) if has_mag else 0.0

    for k in range(1, n):
        dt = t[k] - t[k - 1]

        roll_gyro = roll[k - 1] + gyro[k, 0] * dt
        pitch_gyro = pitch[k - 1] + gyro[k, 1] * dt
        yaw_gyro = yaw[k - 1] + gyro[k, 2] * dt

        roll_acc, pitch_acc = _tilt_from_accel(acc[k])
        roll[k] = _blend_angle(alpha, roll_gyro, roll_acc)
        pitch[k] = _blend_angle(alpha, pitch_gyro, pitch_acc)

        if has_mag:
            yaw_mag = _yaw_from_mag(mag[k], roll[k], pitch[k])
            yaw[k] = _blend_angle(alpha, yaw_gyro, yaw_mag)
        else:
            yaw[k] = yaw_gyro  # no absolute reference -- will drift

    return roll, pitch, yaw


# --------------------------------------------------------------------------
# Body -> nav rotation and gravity removal
# --------------------------------------------------------------------------

def rotation_matrix_body_to_nav(roll, pitch, yaw):
    """ZYX Euler rotation matrix (yaw * pitch * roll), body axes -> ENU nav axes."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def specific_force_to_nav_accel(acc_body, roll, pitch, yaw):
    """a_true_nav = R(roll,pitch,yaw) . f_body + g_nav, per sample."""
    n = len(acc_body)
    accel_nav = np.zeros((n, 3))
    g_nav = np.array([0.0, 0.0, -GRAVITY])
    for k in range(n):
        r = rotation_matrix_body_to_nav(roll[k], pitch[k], yaw[k])
        accel_nav[k] = r @ acc_body[k] + g_nav
    return accel_nav


# --------------------------------------------------------------------------
# ZUPT -- variance-gated stationary detection
# --------------------------------------------------------------------------

def detect_stationary(t, acc, gyro, window_s=0.25, acc_var_thresh=0.0005, gyro_var_thresh=0.0003):
    """Flag samples as stationary where accel/gyro magnitude variance is low
    over a trailing window (variance, not magnitude, so a constant sensor
    bias offset doesn't defeat detection).

    Caveat: ||acc|| is dominated by gravity, so this is fairly insensitive
    to smooth, low-magnitude, non-rotating accelerations (e.g. a gentle
    sideways push barely moves sqrt(9.8^2 + a^2) away from 9.8). In
    practice genuine motion -- footsteps, road vibration, steering --
    also perturbs the gyro, which is why both channels are ANDed together.
    The defaults below were fit against one synthetic noise profile only;
    plot acc/gyro variance for a real recorded stationary segment vs. a
    real moving segment from your own data and set the thresholds between
    them before trusting this on a real trace. Use calibrate_zupt.py in
    this directory to do that measurement -- it labels stationary/moving
    windows in a real trace, reports whether a separating threshold exists,
    and proposes one if so.
    """
    dt_med = np.median(np.diff(t)) if len(t) > 1 else 0.01
    window = max(3, int(round(window_s / dt_med)))

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    acc_var = pd.Series(acc_mag).rolling(window, center=True, min_periods=1).var().fillna(0).to_numpy()
    gyro_var = pd.Series(gyro_mag).rolling(window, center=True, min_periods=1).var().fillna(0).to_numpy()

    return (acc_var < acc_var_thresh) & (gyro_var < gyro_var_thresh)


# --------------------------------------------------------------------------
# Double integration with ZUPT
# --------------------------------------------------------------------------

def integrate_trajectory(t, accel_nav, stationary):
    """Trapezoidal double integration; velocity is reset to zero on ZUPT hits."""
    n = len(t)
    vel = np.zeros((n, 3))
    pos = np.zeros((n, 3))

    for k in range(1, n):
        dt = t[k] - t[k - 1]
        vel[k] = vel[k - 1] + 0.5 * (accel_nav[k] + accel_nav[k - 1]) * dt

        if stationary[k]:
            vel[k] = 0.0

        pos[k] = pos[k - 1] + 0.5 * (vel[k] + vel[k - 1]) * dt

    return vel, pos


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_pipeline(
    input_csv,
    output_csv,
    alpha=0.98,
    zupt_window_s=0.25,
    acc_var_thresh=0.0005,
    gyro_var_thresh=0.0003,
    time_unit="auto",
    gyro_in_degrees=False,
):
    df, has_mag = load_imu_csv(input_csv, time_unit=time_unit, gyro_in_degrees=gyro_in_degrees)

    roll, pitch, yaw = complementary_filter_orientation(df, has_mag, alpha=alpha)

    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    t = df["t"].to_numpy()

    accel_nav = specific_force_to_nav_accel(acc, roll, pitch, yaw)
    stationary = detect_stationary(t, acc, gyro, zupt_window_s, acc_var_thresh, gyro_var_thresh)
    vel, pos = integrate_trajectory(t, accel_nav, stationary)

    speed = np.linalg.norm(vel, axis=1)
    heading_deg = np.degrees(yaw) % 360.0

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "pos_x": pos[:, 0],
            "pos_y": pos[:, 1],
            "pos_z": pos[:, 2],
            "velocity": speed,
            "orientation": heading_deg,
        }
    )
    out.to_csv(output_csv, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Strapdown inertial dead-reckoning physics baseline")
    parser.add_argument("input_csv", help="Sensor Logger export: timestamp, acc_x/y/z, gyro_x/y/z[, mag_x/y/z]")
    parser.add_argument("output_csv", help="Output path: timestamp, pos_x/y/z, velocity, orientation")
    parser.add_argument("--alpha", type=float, default=0.98, help="complementary filter weight on gyro (default 0.98)")
    parser.add_argument("--zupt-window-s", type=float, default=0.25, help="ZUPT variance window, seconds")
    parser.add_argument("--acc-var-thresh", type=float, default=0.0005, help="ZUPT accel-magnitude variance threshold")
    parser.add_argument("--gyro-var-thresh", type=float, default=0.0003, help="ZUPT gyro-magnitude variance threshold")
    parser.add_argument("--time-unit", choices=["auto", "s", "ms", "ns"], default="auto")
    parser.add_argument("--gyro-in-degrees", action="store_true", help="set if gyro columns are deg/s, not rad/s")
    args = parser.parse_args()

    run_pipeline(
        args.input_csv,
        args.output_csv,
        alpha=args.alpha,
        zupt_window_s=args.zupt_window_s,
        acc_var_thresh=args.acc_var_thresh,
        gyro_var_thresh=args.gyro_var_thresh,
        time_unit=args.time_unit,
        gyro_in_degrees=args.gyro_in_degrees,
    )


if __name__ == "__main__":
    main()
