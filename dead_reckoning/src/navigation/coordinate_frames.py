"""Coordinate frame definitions and transformations for vehicle navigation.

Coordinate Frames:
1. PHONE FRAME (p): Sensor axes as defined by Android hardware (x=right, y=up, z=out).
2. VEHICLE FRAME (v): Vehicle body axes (x=forward, y=right/lateral, z=down).
3. LOCAL FRAME (b): Initial orientation frame of the 200-sample window where ML predicts [dx, dy, dz].
4. NED FRAME (n): Local Cartesian navigation tangent frame (North, East, Down).
5. GEODETIC FRAME (g): WGS-84 Ellipsoid (Latitude, Longitude, Altitude).
"""

from __future__ import annotations

from typing import Tuple, Sequence
import numpy as np

WGS84_A = 6378137.0         # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563 # Flattening
MEAN_EARTH_RADIUS = 6371000.0


class CoordinateTransformer:
    """Rigorous coordinate frame transformations for inertial vehicle navigation."""

    def __init__(self, phone_mounting_matrix: np.ndarray | None = None) -> None:
        """Initialize with optional phone-to-vehicle mounting alignment matrix.
        
        Default assumes phone mounted horizontally facing vehicle forward:
        Vehicle Forward (X) = Phone Y (Up)
        Vehicle Right (Y)   = Phone X (Right)
        Vehicle Down (Z)    = -Phone Z (Back)
        """
        if phone_mounting_matrix is None:
            # Standard dashboard phone mount (landscape/portrait facing front)
            # Default identity if already aligned
            self.R_pv = np.eye(3, dtype=np.float64)
        else:
            self.R_pv = np.asarray(phone_mounting_matrix, dtype=np.float64)
            if self.R_pv.shape != (3, 3):
                raise ValueError(f"Mounting matrix must be (3, 3), got {self.R_pv.shape}")

    def transform_phone_to_vehicle(self, vec_phone: np.ndarray) -> np.ndarray:
        """Rotate vector from phone sensor frame to vehicle body frame."""
        v = np.asarray(vec_phone, dtype=np.float64)
        if v.ndim == 1:
            return self.R_pv @ v
        elif v.ndim == 2:
            return (self.R_pv @ v.T).T
        else:
            raise ValueError(f"Input must be 1D or 2D array, got {v.ndim}D")

    @staticmethod
    def heading_to_dcm(heading_deg: float) -> np.ndarray:
        """Create a 2D/3D Direction Cosine Matrix for pure yaw/heading rotation.
        
        Transforms from vehicle body frame to NED frame: v_ned = R_bn @ v_b
        where heading is clockwise from North in degrees.
        """
        psi = np.deg2rad(heading_deg)
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def euler_to_dcm(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """Create 3-2-1 Direction Cosine Matrix (Body -> Navigation NED)."""
        phi = np.deg2rad(roll_deg)
        theta = np.deg2rad(pitch_deg)
        psi = np.deg2rad(yaw_deg)

        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        R = np.zeros((3, 3), dtype=np.float64)
        R[0, 0] = c_th * c_psi
        R[0, 1] = s_phi * s_th * c_psi - c_phi * s_psi
        R[0, 2] = c_phi * s_th * c_psi + s_phi * s_psi

        R[1, 0] = c_th * s_psi
        R[1, 1] = s_phi * s_th * s_psi + c_phi * c_psi
        R[1, 2] = c_phi * s_th * s_psi - s_phi * c_psi

        R[2, 0] = -s_th
        R[2, 1] = s_phi * c_th
        R[2, 2] = c_phi * c_th

        return R

    @staticmethod
    def ned_to_geodetic(
        p_north_m: float,
        p_east_m: float,
        origin_lat_deg: float,
        origin_lon_deg: float,
    ) -> Tuple[float, float]:
        """Convert local metric NED displacement to WGS-84 latitude and longitude."""
        lat_rad = np.deg2rad(origin_lat_deg)
        d_lat_rad = p_north_m / MEAN_EARTH_RADIUS
        d_lon_rad = p_east_m / (MEAN_EARTH_RADIUS * np.cos(lat_rad))

        new_lat = origin_lat_deg + np.rad2deg(d_lat_rad)
        new_lon = origin_lon_deg + np.rad2deg(d_lon_rad)
        return float(new_lat), float(new_lon)

    @staticmethod
    def geodetic_to_ned(
        lat_deg: float,
        lon_deg: float,
        origin_lat_deg: float,
        origin_lon_deg: float,
    ) -> Tuple[float, float]:
        """Convert WGS-84 latitude and longitude to local metric NED displacement."""
        lat_rad = np.deg2rad(origin_lat_deg)
        d_lat_deg = lat_deg - origin_lat_deg
        d_lon_deg = lon_deg - origin_lon_deg

        p_north = np.deg2rad(d_lat_deg) * MEAN_EARTH_RADIUS
        p_east = np.deg2rad(d_lon_deg) * MEAN_EARTH_RADIUS * np.cos(lat_rad)
        return float(p_north), float(p_east)

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute great-circle distance between two geodetic coordinates in meters."""
        phi1 = np.deg2rad(lat1)
        phi2 = np.deg2rad(lat2)
        delta_phi = np.deg2rad(lat2 - lat1)
        delta_lambda = np.deg2rad(lon2 - lon1)

        a = (np.sin(delta_phi / 2.0) ** 2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(MEAN_EARTH_RADIUS * c)
