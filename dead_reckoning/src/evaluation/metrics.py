"""Comprehensive evaluation metrics for vehicle dead reckoning and GNSS outage benchmarks.

Computes:
  1. Position RMSE (m)
  2. Position MAE (m)
  3. Endpoint Error (m)
  4. Maximum Position Error (m)
  5. Velocity RMSE (m/s)
  6. Heading Error (deg)
  7. Distance Travelled (m)
  8. Drift Percentages:
     - Endpoint Drift % = (Endpoint Error / Distance) * 100
     - RMSE Drift % = (Position RMSE / Distance) * 100
     - Maximum Drift % = (Max Error / Distance) * 100
  9. Recovery metrics:
     - Pre-recovery error (m)
     - Post-recovery error at +5s (m)
     - Time to stable recovery (s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.navigation.coordinate_frames import CoordinateTransformer


@dataclass
class OutageEvaluationResult:
    """Quantitative results for a single GNSS outage interval evaluation."""

    outage_duration_s: float
    baseline_name: str
    distance_travelled_m: float
    position_rmse_m: float
    position_mae_m: float
    endpoint_error_m: float
    max_position_error_m: float
    drift_percent_endpoint: float
    drift_percent_rmse: float
    drift_percent_max: float
    velocity_rmse_mps: Optional[float] = None
    heading_error_deg: Optional[float] = None
    pre_recovery_error_m: Optional[float] = None
    post_recovery_error_m: Optional[float] = None
    recovery_time_s: Optional[float] = None


class NavigationMetrics:
    """Calculates rigorous trajectory error metrics during GNSS outages."""

    @staticmethod
    def evaluate_outage(
        est_lat: np.ndarray,
        est_lon: np.ndarray,
        gt_lat: np.ndarray,
        gt_lon: np.ndarray,
        outage_start_idx: int,
        outage_end_idx: int,
        outage_duration_s: float,
        baseline_name: str,
        timestamps_s: np.ndarray,
        est_speed_mps: Optional[np.ndarray] = None,
        gt_speed_mps: Optional[np.ndarray] = None,
        est_heading_deg: Optional[np.ndarray] = None,
        gt_heading_deg: Optional[np.ndarray] = None,
        recovery_window_samples: int = 500, # 5s at 100 Hz
    ) -> OutageEvaluationResult:
        """Evaluate performance over the outage segment."""
        s = outage_start_idx
        e = outage_end_idx

        est_lat_out = est_lat[s:e]
        est_lon_out = est_lon[s:e]
        gt_lat_out = gt_lat[s:e]
        gt_lon_out = gt_lon[s:e]

        n_samples = len(est_lat_out)
        if n_samples < 2:
            raise ValueError(f"Outage segment has too few samples ({n_samples})")

        # 1. Point-by-point horizontal position errors
        pos_errors_m = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            pos_errors_m[i] = CoordinateTransformer.haversine_distance(
                est_lat_out[i], est_lon_out[i],
                gt_lat_out[i], gt_lon_out[i]
            )

        pos_rmse = float(np.sqrt(np.mean(pos_errors_m ** 2)))
        pos_mae = float(np.mean(pos_errors_m))
        endpoint_err = float(pos_errors_m[-1])
        max_err = float(np.max(pos_errors_m))

        # 2. Cumulative ground-truth distance travelled during outage
        dist_m = 0.0
        for i in range(1, n_samples):
            dist_m += CoordinateTransformer.haversine_distance(
                gt_lat_out[i - 1], gt_lon_out[i - 1],
                gt_lat_out[i], gt_lon_out[i]
            )
        dist_m = max(dist_m, 1.0) # Avoid div by zero if vehicle stopped

        # 3. Drift percentages
        drift_endpoint = (endpoint_err / dist_m) * 100.0
        drift_rmse = (pos_rmse / dist_m) * 100.0
        drift_max = (max_err / dist_m) * 100.0

        # 4. Velocity error
        vel_rmse = None
        if est_speed_mps is not None and gt_speed_mps is not None:
            vel_diff = est_speed_mps[s:e] - gt_speed_mps[s:e]
            vel_rmse = float(np.sqrt(np.mean(vel_diff ** 2)))

        # 5. Heading error
        hdg_err = None
        if est_heading_deg is not None and gt_heading_deg is not None:
            diff_deg = (est_heading_deg[s:e] - gt_heading_deg[s:e] + 180.0) % 360.0 - 180.0
            hdg_err = float(np.mean(np.abs(diff_deg)))

        # 6. Recovery metrics (behavior after outage ends)
        pre_rec_err = endpoint_err
        post_rec_err = None
        rec_time_s = None

        if len(est_lat) > e:
            rec_end = min(len(est_lat), e + recovery_window_samples)
            rec_errors = []
            for i in range(e, rec_end):
                rec_errors.append(CoordinateTransformer.haversine_distance(
                    est_lat[i], est_lon[i], gt_lat[i], gt_lon[i]
                ))
            rec_errors = np.array(rec_errors)
            if len(rec_errors) > 0:
                post_rec_err = float(rec_errors[-1])
                # Find time to reach within 2.5m (GNSS accuracy)
                stable_indices = np.where(rec_errors <= 3.5)[0]
                if len(stable_indices) > 0:
                    first_stable = stable_indices[0]
                    rec_time_s = float(timestamps_s[e + first_stable] - timestamps_s[e])
                else:
                    rec_time_s = float(timestamps_s[rec_end - 1] - timestamps_s[e])

        return OutageEvaluationResult(
            outage_duration_s=outage_duration_s,
            baseline_name=baseline_name,
            distance_travelled_m=dist_m,
            position_rmse_m=pos_rmse,
            position_mae_m=pos_mae,
            endpoint_error_m=endpoint_err,
            max_position_error_m=max_err,
            drift_percent_endpoint=drift_endpoint,
            drift_percent_rmse=drift_rmse,
            drift_percent_max=drift_max,
            velocity_rmse_mps=vel_rmse,
            heading_error_deg=hdg_err,
            pre_recovery_error_m=pre_rec_err,
            post_recovery_error_m=post_rec_err,
            recovery_time_s=rec_time_s,
        )
