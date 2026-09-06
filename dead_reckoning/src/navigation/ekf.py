"""Simplified 6-State MVP Navigation Filter (EKF).

================================================================================
IMPORTANT ARCHITECTURAL & THEORETICAL CLASSIFICATION:
================================================================================
This filter is explicitly a SIMPLIFIED 6-STATE MVP NAVIGATION FILTER,
and is NOT a full 15-state error-state inertial navigation EKF (ES-EKF).

State Vector:
  x = [p_North, p_East, p_Down, v_North, v_East, v_Down]^T

Explicit Limitations:
  - Does NOT explicitly estimate accelerometer biases (b_a).
  - Does NOT explicitly estimate gyroscope biases (b_g).
  - Does NOT explicitly estimate attitude/orientation errors (delta_theta).

Supported Fused Measurements:
  - ML displacement / motion estimate
  - GNSS position when available
  - GNSS velocity when available
  - Non-Holonomic Constraints (NHC) pseudo-measurements
  - Zero Velocity Updates (ZUPT) pseudo-measurements
================================================================================
"""

from __future__ import annotations

from typing import Optional
import numpy as np

STATE_DIM = 6


class NavigationEKF:
    """Simplified 6-state MVP navigation Extended Kalman Filter."""

    def __init__(
        self,
        process_noise_pos: float = 0.5,
        process_noise_vel: float = 0.2,
        gnss_pos_noise: float = 2.5,
        gnss_vel_noise: float = 0.2,
    ) -> None:
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.gnss_pos_noise = gnss_pos_noise
        self.gnss_vel_noise = gnss_vel_noise

        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.P = np.eye(STATE_DIM, dtype=np.float64) * 10.0
        self.is_initialized = False

    def initialize(
        self,
        p_north: float = 0.0,
        p_east: float = 0.0,
        p_down: float = 0.0,
        v_north: float = 0.0,
        v_east: float = 0.0,
        v_down: float = 0.0,
        init_pos_std: float = 3.0,
        init_vel_std: float = 0.5,
    ) -> None:
        """Initialize filter state and covariance."""
        self.state = np.array([p_north, p_east, p_down, v_north, v_east, v_down], dtype=np.float64)
        self.P = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        for i in range(3):
            self.P[i, i] = init_pos_std ** 2
            self.P[i + 3, i + 3] = init_vel_std ** 2
        self.is_initialized = True

    def predict(self, delta_p_ned: np.ndarray, dt: float) -> None:
        """Propagate state and covariance using motion increment delta_p_ned over dt."""
        if not self.is_initialized:
            return

        dt = max(dt, 0.001)
        dp = np.asarray(delta_p_ned, dtype=np.float64)

        # 1. State propagation
        self.state[0:3] += dp
        self.state[3:6] = dp / dt

        # 2. State transition matrix F
        # [ I_3   dt * I_3 ]
        # [ 0_3     I_3    ]
        F = np.eye(STATE_DIM, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # 3. Process noise Q
        Q = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        q_pos = (self.process_noise_pos ** 2) * dt
        q_vel = (self.process_noise_vel ** 2) * dt
        for i in range(3):
            Q[i, i] = q_pos
            Q[i + 3, i + 3] = q_vel

        # 4. Covariance propagation
        self.P = F @ self.P @ F.T + Q

    def update_measurement(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Generic Kalman measurement update with Joseph form covariance update."""
        if not self.is_initialized:
            return

        z = np.asarray(z, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        # Innovation: y = z - H @ x
        y = z - H @ self.state

        # Innovation covariance: S = H @ P @ H^T + R
        S = H @ self.P @ H.T + R

        # Kalman gain: K = P @ H^T @ S^-1
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        K = self.P @ H.T @ S_inv

        # State update: x = x + K @ y
        self.state += K @ y

        # Joseph form covariance update for numerical stability:
        # P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
        I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update_gnss_position(self, p_north: float, p_east: float, p_down: float, noise_m: float | None = None) -> None:
        """Update filter with GNSS position measurement."""
        z = np.array([p_north, p_east, p_down], dtype=np.float64)
        H = np.zeros((3, STATE_DIM), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        r_val = (noise_m or self.gnss_pos_noise) ** 2
        R = np.eye(3, dtype=np.float64) * r_val
        self.update_measurement(z, H, R)

    def update_gnss_velocity(self, v_north: float, v_east: float, v_down: float, noise_mps: float | None = None) -> None:
        """Update filter with GNSS velocity measurement."""
        z = np.array([v_north, v_east, v_down], dtype=np.float64)
        H = np.zeros((3, STATE_DIM), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        r_val = (noise_mps or self.gnss_vel_noise) ** 2
        R = np.eye(3, dtype=np.float64) * r_val
        self.update_measurement(z, H, R)

    def update_zupt(self, noise_mps: float = 0.05) -> None:
        """Apply Zero Velocity Update pseudo-measurement (velocity = 0)."""
        z = np.zeros(3, dtype=np.float64)
        H = np.zeros((3, STATE_DIM), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        R = np.eye(3, dtype=np.float64) * (noise_mps ** 2)
        self.update_measurement(z, H, R)

    def update_nhc(self, heading_deg: float, noise_lat_mps: float = 0.2, noise_vert_mps: float = 0.2) -> None:
        """Apply Non-Holonomic Constraint pseudo-measurements in vehicle frame.
        
        Assumes planar motion without sideslip:
        v_lateral = -sin(psi)*v_N + cos(psi)*v_E = 0
        v_vertical = v_D = 0
        """
        psi = np.deg2rad(heading_deg)
        c, s = np.cos(psi), np.sin(psi)

        z = np.zeros(2, dtype=np.float64)
        H = np.zeros((2, STATE_DIM), dtype=np.float64)
        # Row 0: lateral velocity = 0
        H[0, 3] = -s
        H[0, 4] = c
        # Row 1: vertical velocity = 0
        H[1, 5] = 1.0

        R = np.diag([noise_lat_mps ** 2, noise_vert_mps ** 2]).astype(np.float64)
        self.update_measurement(z, H, R)

    @property
    def position_ned(self) -> np.ndarray:
        return self.state[0:3].copy()

    @property
    def velocity_ned(self) -> np.ndarray:
        return self.state[3:6].copy()
