"""Local-frame temporal targets for synchronized OxIOD sequences.

This module preserves the global-target window implementation and derives a
parallel target: ``R_start.T @ (p_end - p_start)``. Quaternions use OxIOD's
``[x, y, z, w]`` order, and only the quaternion at the first sample of each
window is used. Components are named local X/Y/Z; axis semantics are not
otherwise assumed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.features.create_features import SequenceFeatures
from src.features.create_windows import create_temporal_windows


QUATERNION_COLUMNS = ["quat_x", "quat_y", "quat_z", "quat_w"]


def quaternion_to_rotation_matrix(quaternion: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the validated OxIOD rotation matrix for quaternion ``[x, y, z, w]``."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values in [x, y, z, w] order")
    x, y, z, w = q
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(rotation).all():
        raise ValueError("quaternion produced a non-finite rotation matrix")
    return rotation


def create_temporal_windows_local(
    sequence: SequenceFeatures,
    quaternions: pd.DataFrame | np.ndarray,
    window_size: int = 200,
    stride: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Create unchanged IMU windows with initial-orientation local displacement targets.

    The global window creator supplies raw ``X`` and enforces all its input
    checks. For each window ``[start, end]``, this function uses only
    ``quaternions[start]`` and computes ``R_start.T @ (p_end - p_start)``.
    """
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    if quaternion_array.ndim != 2 or quaternion_array.shape[1] != 4:
        raise ValueError("quaternions must have shape (num_samples, 4)")
    if len(quaternion_array) != len(sequence.positions):
        raise ValueError("quaternion and position lengths do not match")
    if not np.isfinite(quaternion_array).all():
        raise ValueError("quaternions contain NaN or infinite values")

    X, _ = create_temporal_windows(sequence, window_size=window_size, stride=stride)
    positions = sequence.positions.to_numpy(dtype=np.float64, copy=False)
    starts = range(0, len(positions) - window_size + 1, stride)
    local_targets = []
    for start_index in starts:
        end_index = start_index + window_size - 1
        delta_global = positions[end_index] - positions[start_index]
        rotation_start = quaternion_to_rotation_matrix(quaternion_array[start_index])
        local_targets.append(rotation_start.T @ delta_global)

    y = np.stack(local_targets, axis=0)
    if len(X) != len(y):
        raise RuntimeError("local window and target counts do not match")
    if not np.isfinite(y).all():
        raise ValueError("local targets contain NaN or infinite values")
    return X, y
