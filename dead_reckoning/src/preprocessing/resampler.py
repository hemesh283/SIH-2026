"""Timestamp-aware 100 Hz resampling and window generation for vehicle sequences.

The GRU dead reckoning model strictly expects:
  - Uniform 100 Hz sampling rate (dt = 0.01 s)
  - Window size: 200 samples (2.0 seconds duration)
  - Stride: 100 samples (1.0 second invocation step)
  - Feature dimensions: 6 [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

TARGET_RATE_HZ = 100.0
TARGET_DT = 1.0 / TARGET_RATE_HZ  # 0.01 s
WINDOW_SIZE = 200
STRIDE = 100


@dataclass
class ResampledVehicleStream:
    """100 Hz uniformly resampled and synchronized vehicle sensor stream."""

    t_100hz: np.ndarray       # Shape: (M,), seconds at 100 Hz
    features_100hz: np.ndarray # Shape: (M, 6), [ax, ay, az, gx, gy, gz]
    gnss_lat_100hz: np.ndarray # Shape: (M,)
    gnss_lon_100hz: np.ndarray # Shape: (M,)
    gnss_spd_100hz: np.ndarray # Shape: (M,)
    gnss_hdg_100hz: np.ndarray # Shape: (M,)
    gt_lat_100hz: np.ndarray   # Shape: (M,)
    gt_lon_100hz: np.ndarray   # Shape: (M,)
    gt_spd_100hz: np.ndarray   # Shape: (M,)
    gt_hdg_100hz: np.ndarray   # Shape: (M,)
    windows: np.ndarray        # Shape: (num_windows, 200, 6)
    window_start_indices: np.ndarray # Shape: (num_windows,) index into 100 Hz stream
    window_end_indices: np.ndarray   # Shape: (num_windows,) index into 100 Hz stream


class StreamResampler:
    """Resamples variable or low-rate IMU and GNSS streams to uniform 100 Hz."""

    def __init__(self, target_rate_hz: float = TARGET_RATE_HZ) -> None:
        self.target_rate_hz = target_rate_hz
        self.target_dt = 1.0 / target_rate_hz

    def resample_and_window(
        self,
        t_raw: np.ndarray,
        acc_g: np.ndarray,
        gyro_rads: np.ndarray,
        gnss_lat: np.ndarray,
        gnss_lon: np.ndarray,
        gnss_spd: np.ndarray,
        gnss_hdg: np.ndarray,
        gt_lat: np.ndarray,
        gt_lon: np.ndarray,
        gt_spd: np.ndarray,
        gt_hdg: np.ndarray,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
    ) -> ResampledVehicleStream:
        """Interpolates input data to exact 100 Hz grid and slices 200-sample windows."""
        # 1. Validate timestamps
        t_raw = np.asarray(t_raw, dtype=np.float64)
        if len(t_raw) < 2:
            raise ValueError("Input stream must have at least 2 samples")
        
        diffs = np.diff(t_raw)
        if np.any(diffs <= 0):
            # Sort and remove duplicates if non-monotonic
            unique_indices = np.where(diffs > 0)[0]
            unique_indices = np.concatenate([[0], unique_indices + 1])
            t_raw = t_raw[unique_indices]
            acc_g = acc_g[unique_indices]
            gyro_rads = gyro_rads[unique_indices]
            gnss_lat = gnss_lat[unique_indices]
            gnss_lon = gnss_lon[unique_indices]
            gnss_spd = gnss_spd[unique_indices]
            gnss_hdg = gnss_hdg[unique_indices]
            gt_lat = gt_lat[unique_indices]
            gt_lon = gt_lon[unique_indices]
            gt_spd = gt_spd[unique_indices]
            gt_hdg = gt_hdg[unique_indices]

        t_start = t_raw[0]
        t_end = t_raw[-1]
        t_100hz = np.arange(t_start, t_end, self.target_dt)

        if len(t_100hz) < window_size:
            raise ValueError(f"Resampled stream duration too short for window_size={window_size}")

        # 2. Linear interpolation for continuous vectors
        acc_100hz = np.zeros((len(t_100hz), 3), dtype=np.float32)
        gyro_100hz = np.zeros((len(t_100hz), 3), dtype=np.float32)
        for i in range(3):
            acc_100hz[:, i] = np.interp(t_100hz, t_raw, acc_g[:, i]).astype(np.float32)
            gyro_100hz[:, i] = np.interp(t_100hz, t_raw, gyro_rads[:, i]).astype(np.float32)

        features_100hz = np.concatenate([acc_100hz, gyro_100hz], axis=1) # (M, 6)

        # 3. Position and velocity interpolation
        gnss_lat_100hz = np.interp(t_100hz, t_raw, gnss_lat)
        gnss_lon_100hz = np.interp(t_100hz, t_raw, gnss_lon)
        gnss_spd_100hz = np.interp(t_100hz, t_raw, gnss_spd).astype(np.float32)

        gt_lat_100hz = np.interp(t_100hz, t_raw, gt_lat)
        gt_lon_100hz = np.interp(t_100hz, t_raw, gt_lon)
        gt_spd_100hz = np.interp(t_100hz, t_raw, gt_spd).astype(np.float32)

        # 4. Heading interpolation (unwrapped to handle 0/360 wrap-around)
        gnss_hdg_unwrapped = np.unwrap(np.deg2rad(gnss_hdg))
        gnss_hdg_100hz = (np.rad2deg(np.interp(t_100hz, t_raw, gnss_hdg_unwrapped)) % 360.0).astype(np.float32)

        gt_hdg_unwrapped = np.unwrap(np.deg2rad(gt_hdg))
        gt_hdg_100hz = (np.rad2deg(np.interp(t_100hz, t_raw, gt_hdg_unwrapped)) % 360.0).astype(np.float32)

        # 5. Slicing windows of size 200 with specified stride
        starts = range(0, len(t_100hz) - window_size + 1, stride)
        windows = []
        start_indices = []
        end_indices = []
        for s in starts:
            e = s + window_size
            windows.append(features_100hz[s:e])
            start_indices.append(s)
            end_indices.append(e - 1)

        windows_arr = np.array(windows, dtype=np.float32)
        start_indices_arr = np.array(start_indices, dtype=np.int64)
        end_indices_arr = np.array(end_indices, dtype=np.int64)

        return ResampledVehicleStream(
            t_100hz=t_100hz,
            features_100hz=features_100hz,
            gnss_lat_100hz=gnss_lat_100hz,
            gnss_lon_100hz=gnss_lon_100hz,
            gnss_spd_100hz=gnss_spd_100hz,
            gnss_hdg_100hz=gnss_hdg_100hz,
            gt_lat_100hz=gt_lat_100hz,
            gt_lon_100hz=gt_lon_100hz,
            gt_spd_100hz=gt_spd_100hz,
            gt_hdg_100hz=gt_hdg_100hz,
            windows=windows_arr,
            window_start_indices=start_indices_arr,
            window_end_indices=end_indices_arr,
        )
