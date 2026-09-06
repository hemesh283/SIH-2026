"""Multi-Signal Zero Velocity Update (ZUPT) Detector.

Does NOT rely on acceleration variance alone.
Combines:
  1. Acceleration magnitude deviation from 1g
  2. Acceleration variance
  3. Gyroscope magnitude
  4. Gyroscope variance
  5. GNSS speed (when GNSS fix is available)
"""

from __future__ import annotations

from typing import Optional
import numpy as np


class ZuptDetector:
    """Configurable multi-signal stationary detector for land vehicles."""

    def __init__(
        self,
        acc_dev_thresh_g: float = 0.08,
        acc_var_thresh_g2: float = 0.015,
        gyro_mag_thresh_rads: float = 0.05,
        gyro_var_thresh_rads2: float = 0.005,
        gnss_speed_thresh_mps: float = 0.3,
        window_size: int = 15,
        is_enabled: bool = True,
    ) -> None:
        self.acc_dev_thresh_g = acc_dev_thresh_g
        self.acc_var_thresh_g2 = acc_var_thresh_g2
        self.gyro_mag_thresh_rads = gyro_mag_thresh_rads
        self.gyro_var_thresh_rads2 = gyro_var_thresh_rads2
        self.gnss_speed_thresh_mps = gnss_speed_thresh_mps
        self.window_size = window_size
        self.is_enabled = is_enabled

    def is_stationary(
        self,
        recent_acc_g: np.ndarray,
        recent_gyro_rads: np.ndarray,
        gnss_speed_mps: Optional[float] = None,
    ) -> bool:
        """Evaluate whether current motion is stationary based on multiple signals.
        
        Args:
            recent_acc_g: Array of shape (K, 3) representing recent acceleration samples in g.
            recent_gyro_rads: Array of shape (K, 3) representing recent gyroscope samples in rad/s.
            gnss_speed_mps: Optional GNSS speed in m/s (if GNSS fix is available).
            
        Returns:
            True if vehicle is stationary, False otherwise.
        """
        if not self.is_enabled:
            return False

        if len(recent_acc_g) < 2 or len(recent_gyro_rads) < 2:
            return False

        # 1. Acceleration magnitude and variance
        acc_norms = np.linalg.norm(recent_acc_g, axis=1) # (K,)
        mean_acc = float(np.mean(acc_norms))
        var_acc = float(np.var(acc_norms))

        # Note: In linear acceleration (gravity-removed), stationary mean is ~0.
        # If total acceleration with gravity is supplied, mean is ~1.0g.
        # We test deviation from 0.0 (for linear accel) or from 1.0 (for raw accel).
        is_linear = mean_acc < 0.5
        acc_dev = mean_acc if is_linear else abs(mean_acc - 1.0)

        cond_acc = (acc_dev < self.acc_dev_thresh_g) and (var_acc < self.acc_var_thresh_g2)

        # 2. Gyroscope magnitude and variance
        gyro_norms = np.linalg.norm(recent_gyro_rads, axis=1)
        mean_gyro = float(np.mean(gyro_norms))
        var_gyro = float(np.var(gyro_norms))

        cond_gyro = (mean_gyro < self.gyro_mag_thresh_rads) and (var_gyro < self.gyro_var_thresh_rads2)

        # 3. GNSS speed condition
        cond_gnss = True
        if gnss_speed_mps is not None and not np.isnan(gnss_speed_mps):
            cond_gnss = (gnss_speed_mps < self.gnss_speed_thresh_mps)

        return bool(cond_acc and cond_gyro and cond_gnss)
