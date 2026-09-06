"""IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark Dataset) Loader.

Reference:
- Official repository: https://github.com/onyekpeu/IO-VNBD
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7907232/

Handles both smartphone recordings (S-*.csv via AndroSensor at ~10 Hz)
and vehicle CAN/ECU recordings (V-*.csv at ~10 Hz).
Extracts:
  - Smartphone IMU: linear acceleration (converted to g), gyroscope (rad/s)
  - Timestamps (converted to seconds elapsed)
  - Smartphone GNSS: latitude, longitude, speed (m/s), bearing (deg)
  - Vehicle Ground Truth: GPS latitude, longitude, velocity (m/s), heading (deg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

GRAVITY_MS2 = 9.80665


@dataclass
class VehicleSequenceData:
    """Loaded and aligned vehicle sequence ready for preprocessing/evaluation."""

    sequence_id: str
    timestamps_s: np.ndarray  # Shape: (N,), monotonically increasing seconds
    acc_g: np.ndarray         # Shape: (N, 3), linear acceleration in g: [ax, ay, az]
    gyro_rads: np.ndarray     # Shape: (N, 3), angular velocity in rad/s: [wx, wy, wz]
    gnss_lat: np.ndarray      # Shape: (N,), WGS-84 latitude in degrees
    gnss_lon: np.ndarray      # Shape: (N,), WGS-84 longitude in degrees
    gnss_speed_mps: np.ndarray  # Shape: (N,), speed in m/s
    gnss_heading_deg: np.ndarray # Shape: (N,), heading in degrees [0, 360)
    gt_lat: np.ndarray        # Shape: (N,), Ground-truth latitude
    gt_lon: np.ndarray        # Shape: (N,), Ground-truth longitude
    gt_speed_mps: np.ndarray  # Shape: (N,), Ground-truth velocity in m/s
    gt_heading_deg: np.ndarray # Shape: (N,), Ground-truth heading in degrees
    is_mock: bool = False     # Flag indicating whether sequence is synthetic fixture


class IOVNBDLoader:
    """Loader and indexer for the IO-VNBD dataset."""

    def __init__(self, raw_data_dir: str | Path | None = None) -> None:
        if raw_data_dir is None:
            self.data_dir = Path("D:/dead_reckoning/data/raw")
        else:
            self.data_dir = Path(raw_data_dir)

    def find_dataset_dir(self) -> Optional[Path]:
        """Locate the root of IO-VNBD dataset if present."""
        candidates = [
            self.data_dir / "IO-VNBD",
            self.data_dir / "io_vnbd",
            self.data_dir / "IOVNB",
            self.data_dir / "Synchronised V abd S datasets",
            self.data_dir / "Synchronised V and S datasets",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        # Search recursively one level down
        if self.data_dir.is_dir():
            for p in self.data_dir.glob("**/Synchronised*"):
                if p.is_dir():
                    return p
            for p in self.data_dir.glob("**/IO-VNBD*"):
                if p.is_dir():
                    return p
        return None

    def is_available(self) -> bool:
        """Check if any valid IO-VNBD data files exist."""
        root = self.find_dataset_dir()
        if root is None:
            return False
        csv_files = list(root.glob("**/*.csv"))
        return len(csv_files) > 0

    def list_sequences(self) -> List[Tuple[Path, Path]]:
        """Find paired smartphone (S-) and vehicle (V-) CSV files.
        
        Returns:
            List of (smartphone_path, vehicle_path) tuples.
        """
        root = self.find_dataset_dir()
        if root is None:
            return []

        s_files = sorted(list(root.glob("**/S-*.csv")) + list(root.glob("**/s-*.csv")))
        pairs = []
        for s_path in s_files:
            # Look for matching V- file in the same directory
            v_name = s_path.name.replace("S-", "V-").replace("s-", "v-")
            v_path = s_path.parent / v_name
            if v_path.is_file():
                pairs.append((s_path, v_path))
            else:
                # Also try looking in corresponding vehicle folder if separated
                v_candidates = list(s_path.parent.glob("V-*.csv"))
                if v_candidates:
                    pairs.append((s_path, v_candidates[0]))

        return pairs

    def load_sequence(self, s_csv_path: str | Path, v_csv_path: str | Path) -> VehicleSequenceData:
        """Load and harmonize a smartphone CSV and vehicle ECU CSV."""
        s_path = Path(s_csv_path)
        v_path = Path(v_csv_path)

        df_s = pd.read_csv(s_path)
        df_v = pd.read_csv(v_path)

        # Normalize column headers (strip spaces, lowercase)
        s_cols = {c: c.strip().lower() for c in df_s.columns}
        v_cols = {c: c.strip().lower() for c in df_v.columns}

        # Helper to find column by regex
        def find_col(df: pd.DataFrame, pattern: str) -> Optional[str]:
            for col in df.columns:
                if re.search(pattern, col, re.IGNORECASE):
                    return col
            return None

        # 1. Smartphone IMU features
        # Look for Linear Acceleration (gravity removed) or fallback to raw Accelerometer
        ax_col = find_col(df_s, r"linear.*accel.*x") or find_col(df_s, r"accel.*x")
        ay_col = find_col(df_s, r"linear.*accel.*y") or find_col(df_s, r"accel.*y")
        az_col = find_col(df_s, r"linear.*accel.*z") or find_col(df_s, r"accel.*z")

        gx_col = find_col(df_s, r"gyro.*x")
        gy_col = find_col(df_s, r"gyro.*y")
        gz_col = find_col(df_s, r"gyro.*z")

        # Location columns in smartphone
        s_lat_col = find_col(df_s, r"lat")
        s_lon_col = find_col(df_s, r"lon")
        s_spd_col = find_col(df_s, r"speed")
        s_brg_col = find_col(df_s, r"bearing")

        # Vehicle ECU ground truth columns
        v_lat_col = find_col(df_v, r"lat")
        v_lon_col = find_col(df_v, r"lon")
        v_spd_col = find_col(df_v, r"speed|vel")
        v_hdg_col = find_col(df_v, r"head|bearing")
        v_time_col = find_col(df_v, r"time")

        # Time column in smartphone
        s_time_col = find_col(df_s, r"time")

        n_rows = min(len(df_s), len(df_v))
        if n_rows < 10:
            raise ValueError(f"Sequence has too few samples ({n_rows})")

        # Extract timestamps
        if s_time_col and pd.to_numeric(df_s[s_time_col], errors="coerce").notna().all():
            raw_time = df_s[s_time_col].to_numpy()[:n_rows].astype(np.float64)
            # If in milliseconds, convert to seconds
            if np.median(np.diff(raw_time)) > 10.0:
                t_s = (raw_time - raw_time[0]) / 1000.0
            else:
                t_s = raw_time - raw_time[0]
        else:
            # Fallback nominal 10 Hz
            t_s = np.arange(n_rows, dtype=np.float64) * 0.1

        # Extract and convert acceleration m/s² -> g
        if ax_col and ay_col and az_col:
            ax = df_s[ax_col].to_numpy()[:n_rows].astype(np.float32) / GRAVITY_MS2
            ay = df_s[ay_col].to_numpy()[:n_rows].astype(np.float32) / GRAVITY_MS2
            az = df_s[az_col].to_numpy()[:n_rows].astype(np.float32) / GRAVITY_MS2
        else:
            ax = np.zeros(n_rows, dtype=np.float32)
            ay = np.zeros(n_rows, dtype=np.float32)
            az = np.zeros(n_rows, dtype=np.float32)
        acc_g = np.stack([ax, ay, az], axis=1)

        # Extract gyroscope (rad/s)
        if gx_col and gy_col and gz_col:
            gx = df_s[gx_col].to_numpy()[:n_rows].astype(np.float32)
            gy = df_s[gy_col].to_numpy()[:n_rows].astype(np.float32)
            gz = df_s[gz_col].to_numpy()[:n_rows].astype(np.float32)
        else:
            gx = np.zeros(n_rows, dtype=np.float32)
            gy = np.zeros(n_rows, dtype=np.float32)
            gz = np.zeros(n_rows, dtype=np.float32)
        gyro_rads = np.stack([gx, gy, gz], axis=1)

        # Smartphone GNSS
        gnss_lat = df_s[s_lat_col].to_numpy()[:n_rows].astype(np.float64) if s_lat_col else np.zeros(n_rows)
        gnss_lon = df_s[s_lon_col].to_numpy()[:n_rows].astype(np.float64) if s_lon_col else np.zeros(n_rows)
        
        if s_spd_col:
            spd_raw = df_s[s_spd_col].to_numpy()[:n_rows].astype(np.float32)
            # If speed is km/h, convert to m/s
            if np.nanmean(spd_raw) > 30.0:
                gnss_spd = spd_raw / 3.6
            else:
                gnss_spd = spd_raw
        else:
            gnss_spd = np.zeros(n_rows, dtype=np.float32)

        gnss_hdg = df_s[s_brg_col].to_numpy()[:n_rows].astype(np.float32) if s_brg_col else np.zeros(n_rows, dtype=np.float32)

        # Ground Truth from Vehicle ECU
        gt_lat = df_v[v_lat_col].to_numpy()[:n_rows].astype(np.float64) if v_lat_col else gnss_lat.copy()
        gt_lon = df_v[v_lon_col].to_numpy()[:n_rows].astype(np.float64) if v_lon_col else gnss_lon.copy()
        if v_spd_col:
            v_spd_raw = df_v[v_spd_col].to_numpy()[:n_rows].astype(np.float32)
            gt_spd = v_spd_raw / 3.6 if np.nanmean(v_spd_raw) > 30.0 else v_spd_raw
        else:
            gt_spd = gnss_spd.copy()

        gt_hdg = df_v[v_hdg_col].to_numpy()[:n_rows].astype(np.float32) if v_hdg_col else gnss_hdg.copy()

        # Handle NaNs via forward/backward fill
        def clean_series(arr: np.ndarray) -> np.ndarray:
            s = pd.Series(arr)
            return s.ffill().bfill().fillna(0.0).to_numpy()

        acc_g = np.stack([clean_series(acc_g[:, i]) for i in range(3)], axis=1).astype(np.float32)
        gyro_rads = np.stack([clean_series(gyro_rads[:, i]) for i in range(3)], axis=1).astype(np.float32)
        gnss_lat = clean_series(gnss_lat)
        gnss_lon = clean_series(gnss_lon)
        gnss_spd = clean_series(gnss_spd).astype(np.float32)
        gnss_hdg = clean_series(gnss_hdg).astype(np.float32)
        gt_lat = clean_series(gt_lat)
        gt_lon = clean_series(gt_lon)
        gt_spd = clean_series(gt_spd).astype(np.float32)
        gt_hdg = clean_series(gt_hdg).astype(np.float32)

        return VehicleSequenceData(
            sequence_id=s_path.stem,
            timestamps_s=t_s,
            acc_g=acc_g,
            gyro_rads=gyro_rads,
            gnss_lat=gnss_lat,
            gnss_lon=gnss_lon,
            gnss_speed_mps=gnss_spd,
            gnss_heading_deg=gnss_hdg,
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            gt_speed_mps=gt_spd,
            gt_heading_deg=gt_hdg,
            is_mock=False,
        )


def generate_mock_io_vnbd_sequence(
    duration_s: float = 180.0,
    sampling_rate_hz: float = 10.0,
    origin_lat: float = 52.4862,
    origin_lon: float = -1.8904,
    random_seed: int = 42,
) -> VehicleSequenceData:
    """Generate a high-fidelity synthetic vehicle sequence adhering to IO-VNBD schema.
    
    Includes realistic vehicle acceleration, deceleration, stationary stops (ZUPT intervals),
    coordinated turns, and matching WGS-84 trajectory.
    """
    rng = np.random.default_rng(random_seed)
    num_samples = int(duration_s * sampling_rate_hz)
    dt = 1.0 / sampling_rate_hz
    t = np.arange(num_samples) * dt

    # Motion profile:
    # 0 - 15s: stationary stop (ZUPT active)
    # 15 - 45s: accelerate and cruise forward at 12 m/s
    # 45 - 65s: 90-degree right turn
    # 65 - 100s: straight cruise at 15 m/s
    # 100 - 120s: 90-degree right turn
    # 120 - 140s: decelerate to 0
    # 140 - 180s: stationary stop (ZUPT active)
    speed = np.zeros(num_samples, dtype=np.float64)
    yaw_rate = np.zeros(num_samples, dtype=np.float64) # rad/s

    for i, ts in enumerate(t):
        if ts < 15.0:
            speed[i] = 0.0
            yaw_rate[i] = 0.0
        elif 15.0 <= ts < 25.0:
            # Accelerating from 0 to 12 m/s
            speed[i] = 1.2 * (ts - 15.0)
        elif 25.0 <= ts < 45.0:
            speed[i] = 12.0
        elif 45.0 <= ts < 65.0:
            speed[i] = 12.0
            yaw_rate[i] = np.deg2rad(90.0 / 20.0) # 4.5 deg/s
        elif 65.0 <= ts < 100.0:
            speed[i] = 15.0
        elif 100.0 <= ts < 120.0:
            speed[i] = 15.0
            yaw_rate[i] = np.deg2rad(90.0 / 20.0)
        elif 120.0 <= ts < 140.0:
            # Decelerating to 0
            speed[i] = max(0.0, 15.0 - 0.75 * (ts - 120.0))
        else:
            speed[i] = 0.0
            yaw_rate[i] = 0.0

    # Integrate heading and position
    heading_rad = np.cumsum(yaw_rate * dt)
    heading_deg = np.rad2deg(heading_rad) % 360.0

    # NED displacements
    v_n = speed * np.cos(heading_rad)
    v_e = speed * np.sin(heading_rad)
    p_n = np.cumsum(v_n * dt)
    p_e = np.cumsum(v_e * dt)

    # Convert to geodetic WGS-84
    earth_r = 6378137.0
    lat = origin_lat + np.rad2deg(p_n / earth_r)
    lon = origin_lon + np.rad2deg(p_e / (earth_r * np.cos(np.deg2rad(origin_lat))))

    # Simulate Accelerations (longitudinal + centripetal + vibration/noise)
    dv = np.gradient(speed, dt)
    # in vehicle frame: x=forward, y=lateral, z=vertical
    acc_x_ms2 = dv + rng.normal(0.0, 0.02, size=num_samples)
    acc_y_ms2 = speed * yaw_rate + rng.normal(0.0, 0.02, size=num_samples)
    acc_z_ms2 = rng.normal(0.0, 0.02, size=num_samples) # gravity removed in linear accel

    # Convert to g
    acc_g = np.stack([
        acc_x_ms2 / GRAVITY_MS2,
        acc_y_ms2 / GRAVITY_MS2,
        acc_z_ms2 / GRAVITY_MS2
    ], axis=1).astype(np.float32)

    # Simulate Gyro (rad/s)
    gyro_rads = np.stack([
        rng.normal(0.0, 0.005, size=num_samples),
        rng.normal(0.0, 0.005, size=num_samples),
        yaw_rate + rng.normal(0.0, 0.005, size=num_samples),
    ], axis=1).astype(np.float32)

    # Simulate GNSS fixes (with standard 2.5m noise when active)
    gnss_lat = lat + np.rad2deg(rng.normal(0.0, 1.5, size=num_samples) / earth_r)
    gnss_lon = lon + np.rad2deg(rng.normal(0.0, 1.5, size=num_samples) / (earth_r * np.cos(np.deg2rad(origin_lat))))
    gnss_spd = np.clip(speed + rng.normal(0.0, 0.15, size=num_samples), 0.0, None).astype(np.float32)
    gnss_hdg = heading_deg.astype(np.float32)

    return VehicleSequenceData(
        sequence_id="IO_VNBD_SYNTHETIC_MOCK_01",
        timestamps_s=t,
        acc_g=acc_g,
        gyro_rads=gyro_rads,
        gnss_lat=gnss_lat,
        gnss_lon=gnss_lon,
        gnss_speed_mps=gnss_spd,
        gnss_heading_deg=gnss_hdg,
        gt_lat=lat,
        gt_lon=lon,
        gt_speed_mps=speed.astype(np.float32),
        gt_heading_deg=heading_deg.astype(np.float32),
        is_mock=True,
    )
