"""Non-Holonomic Constraints (NHC) for ground vehicle navigation.

Assumptions under which NHC is valid:
1. Land wheeled vehicle moving on conventional road surfaces.
2. Vehicle tires adhere to the ground without significant lateral side-slip (skidding/drifting).
3. The vehicle does not jump, roll over, or experience airborne vertical motion.
4. The velocity perpendicular to the vehicle forward axis (lateral and vertical) is approximately zero:
     v_lateral ≈ 0
     v_vertical ≈ 0
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


class NhcConstraint:
    """Non-Holonomic Constraints module for land vehicles."""

    def __init__(
        self,
        noise_lateral_mps: float = 0.2,
        noise_vertical_mps: float = 0.2,
        is_enabled: bool = True,
    ) -> None:
        self.noise_lateral_mps = noise_lateral_mps
        self.noise_vertical_mps = noise_vertical_mps
        self.is_enabled = is_enabled

    def get_measurement_matrix(self, heading_deg: float, state_dim: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute NHC pseudo-measurement vector z, matrix H, and covariance R.
        
        Args:
            heading_deg: Current vehicle heading in degrees clockwise from North.
            state_dim: Dimension of the state vector (default 6: [pN, pE, pD, vN, vE, vD]).
            
        Returns:
            (z, H, R) matrices for EKF measurement update.
        """
        psi = np.deg2rad(heading_deg)
        c, s = np.cos(psi), np.sin(psi)

        # z: expected velocity = [0, 0]
        z = np.zeros(2, dtype=np.float64)

        # H: transforms NED velocity [vN, vE, vD] to [v_lat, v_vert]
        # v_lat  = -sin(psi)*vN + cos(psi)*vE
        # v_vert = vD
        H = np.zeros((2, state_dim), dtype=np.float64)
        H[0, 3] = -s
        H[0, 4] = c
        H[1, 5] = 1.0

        R = np.diag([self.noise_lateral_mps ** 2, self.noise_vertical_mps ** 2]).astype(np.float64)
        return z, H, R
