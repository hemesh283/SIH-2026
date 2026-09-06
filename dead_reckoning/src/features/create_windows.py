"""Temporal window generation for a single synchronized OxIOD sequence.

Each input window contains only its own raw IMU samples. Its label is the
translation from the Vicon position at the first sample to the Vicon position
at the last sample in that same window. This removes dependence on the global
position origin but, for this initial implementation, leaves the displacement
components expressed in the Vicon global axes.

The resulting target is therefore a *translation-relative, global-axis*
displacement--not the final body/local-frame target. ``construct_displacement_target``
accepts an optional initial-frame rotation so a future step can express the
same endpoint displacement in a local frame using orientation at the window
start. That rotation must be derived from the initial orientation only.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Support direct execution from the project root as well as module execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.create_features import SequenceFeatures, load_and_extract_sequence


def construct_displacement_target(
    positions: pd.DataFrame | np.ndarray,
    start_index: int,
    end_index: int,
    initial_frame_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Construct one endpoint-only relative displacement target.

    The unrotated target is ``position[end_index] - position[start_index]``.
    It is relative to the window start but its components remain in the Vicon
    global coordinate axes. Supplying a 3-by-3 ``initial_frame_rotation``
    applies a caller-provided global-to-local rotation after subtraction. A
    future orientation pipeline may compute that matrix from the orientation
    at ``start_index``; no later orientation or position samples are used here.
    """
    position_array = _as_finite_array(positions, "positions")
    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError("positions must have shape (num_samples, 3)")
    if not 0 <= start_index <= end_index < len(position_array):
        raise IndexError("window endpoint indices are outside the position sequence")

    displacement = position_array[end_index] - position_array[start_index]
    if initial_frame_rotation is None:
        return displacement

    rotation = _as_finite_array(initial_frame_rotation, "initial_frame_rotation")
    if rotation.shape != (3, 3):
        raise ValueError("initial_frame_rotation must have shape (3, 3)")
    return rotation @ displacement


def create_temporal_windows(
    sequence: SequenceFeatures,
    window_size: int = 200,
    stride: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Create raw IMU windows and endpoint-only relative displacement labels.

    Args:
        sequence: Row-aligned output of ``extract_sequence_features``.
        window_size: Number of IMU samples in every window. Must be positive.
        stride: Number of samples between consecutive window starts. Must be
            positive; the default produces non-overlapping windows.

    Returns:
        ``X`` with shape ``(num_windows, window_size, 6)`` and ``y`` with
        shape ``(num_windows, 3)``. A window starting at ``s`` ends at
        ``s + window_size - 1`` and uses only positions at those two endpoints
        for its target.

    Raises:
        ValueError: If dimensions, values, window settings, or alignment are
            invalid, or if there are insufficient samples for one full window.
    """
    if not isinstance(sequence, SequenceFeatures):
        raise TypeError("sequence must be a SequenceFeatures instance")
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    features = _as_finite_array(sequence.features, "features")
    positions = _as_finite_array(sequence.positions, "positions")
    if features.ndim != 2 or features.shape[1] != 6:
        raise ValueError("features must have shape (num_samples, 6)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (num_samples, 3)")
    if len(features) != len(positions):
        raise ValueError(
            "feature and position lengths do not match: "
            f"{len(features)} vs {len(positions)}"
        )
    if len(features) < window_size:
        raise ValueError(
            f"not enough samples ({len(features)}) for window_size={window_size}"
        )

    starts = range(0, len(features) - window_size + 1, stride)
    windows = []
    targets = []
    for start_index in starts:
        end_index = start_index + window_size - 1
        windows.append(features[start_index : end_index + 1])
        targets.append(construct_displacement_target(positions, start_index, end_index))

    X = np.stack(windows, axis=0)
    y = np.stack(targets, axis=0)
    if len(X) != len(y):
        raise RuntimeError("window and target counts do not match")
    return X, y


def _as_finite_array(values: pd.DataFrame | np.ndarray, name: str) -> np.ndarray:
    """Convert numeric values to float64 and reject NaN, infinity, and text."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values") from error
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return array


if __name__ == "__main__":
    example_path = Path(
        "data/raw/"
        "Oxford Inertial Odometry Dataset_2.0/"
        "Oxford Inertial Odometry Dataset/"
        "handheld/data1/syn"
    )
    sequence = load_and_extract_sequence(example_path)
    X, y = create_temporal_windows(sequence)

    print("Number of windows:", len(X))
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First X sample shape:", X[0].shape)
    print("First y target:", y[0])
