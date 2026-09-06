"""Feature extraction for one synchronized Oxford Inertial Odometry sequence.

This module intentionally performs no filtering, resampling, normalization, or
temporal windowing.  It keeps the raw six-axis IMU measurements, timestamps,
and Vicon positions aligned by row so that a later experiment can define a
supervised target explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import sys

import pandas as pd

# Support both ``python src/features/create_features.py`` and
# ``python -m src.features.create_features`` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.load_data import load_sequence


FEATURE_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
POSITION_COLUMNS = ["pos_x", "pos_y", "pos_z"]
TIMESTAMP_COLUMN = "time"


@dataclass(frozen=True)
class SequenceFeatures:
    """Row-aligned data retained from one synchronized OxIOD sequence.

    Attributes:
        imu_timestamps: IMU timestamps, one per feature row.
        vi_timestamps: Vicon timestamps, retained for future alignment checks.
        features: The six requested raw IMU measurements.
        positions: Raw Vicon positions. These are ground truth, but are not yet
            declared as a model target: a future experiment must choose whether
            to predict position, displacement, velocity, or another quantity.
    """

    imu_timestamps: pd.Series
    vi_timestamps: pd.Series
    features: pd.DataFrame
    positions: pd.DataFrame


def validate_synchronized_sequence(imu: pd.DataFrame, vi: pd.DataFrame) -> None:
    """Validate that a loaded OxIOD IMU/VI pair is usable without altering it.

    Raises:
        ValueError: If either sequence is empty, lengths differ, required
            columns are missing, or retained fields contain missing/non-numeric
            values.
        TypeError: If either input is not a pandas DataFrame.
    """
    if not isinstance(imu, pd.DataFrame) or not isinstance(vi, pd.DataFrame):
        raise TypeError("imu and vi must both be pandas DataFrames")
    if imu.empty or vi.empty:
        raise ValueError("IMU and VI sequences must both contain at least one row")
    if len(imu) != len(vi):
        raise ValueError(
            f"IMU and VI lengths do not match: {len(imu)} vs {len(vi)}"
        )

    required_imu = [TIMESTAMP_COLUMN, *FEATURE_COLUMNS]
    required_vi = [TIMESTAMP_COLUMN, *POSITION_COLUMNS]
    _validate_columns_and_values(imu, required_imu, "IMU")
    _validate_columns_and_values(vi, required_vi, "VI")


def _validate_columns_and_values(
    frame: pd.DataFrame, required_columns: list[str], frame_name: str
) -> None:
    """Raise a clear error when retained fields are absent, null, or non-numeric."""
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")

    retained = frame[required_columns]
    missing_values = retained.columns[retained.isna().any()].tolist()
    if missing_values:
        raise ValueError(f"{frame_name} has missing values in: {missing_values}")

    non_numeric = [
        column
        for column in required_columns
        if pd.to_numeric(retained[column], errors="coerce").isna().any()
    ]
    if non_numeric:
        raise ValueError(f"{frame_name} has non-numeric values in: {non_numeric}")


def extract_sequence_features(imu: pd.DataFrame, vi: pd.DataFrame) -> SequenceFeatures:
    """Return validated, row-aligned raw IMU features and Vicon positions.

    The returned DataFrames are copies, preventing later manipulation of the
    result from modifying the loaded source frames. Values are deliberately not
    normalized or otherwise transformed.
    """
    validate_synchronized_sequence(imu, vi)
    return SequenceFeatures(
        imu_timestamps=imu[TIMESTAMP_COLUMN].copy(),
        vi_timestamps=vi[TIMESTAMP_COLUMN].copy(),
        features=imu[FEATURE_COLUMNS].copy(),
        positions=vi[POSITION_COLUMNS].copy(),
    )


def load_and_extract_sequence(sequence_path: str | Path) -> SequenceFeatures:
    """Load one synchronized OxIOD directory and extract its retained fields."""
    imu, vi = load_sequence(sequence_path)
    return extract_sequence_features(imu, vi)


def as_supervised_components(
    sequence: SequenceFeatures,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Expose features, positions, and timestamps without choosing a target.

    This convenience function is intentionally descriptive rather than a
    training-data builder. A later temporal-sampling design must define the
    prediction target and horizon before using ``positions`` as labels.
    """
    return (
        sequence.features,
        sequence.positions,
        sequence.imu_timestamps,
        sequence.vi_timestamps,
    )


if __name__ == "__main__":
    example_path = Path(
        "data/raw/"
        "Oxford Inertial Odometry Dataset_2.0/"
        "Oxford Inertial Odometry Dataset/"
        "handheld/data1/syn"
    )

    imu, vi = load_sequence(example_path)
    sequence = extract_sequence_features(imu, vi)

    print("Input shape:", imu.shape)
    print("Selected feature shape:", sequence.features.shape)
    print("Target/position shape:", sequence.positions.shape)
    print("IMU timestamp shape:", sequence.imu_timestamps.shape)
    print("VI timestamp shape:", sequence.vi_timestamps.shape)
    print("\nFirst few selected features:")
    print(sequence.features.head())
    print("\nFirst few positions:")
    print(sequence.positions.head())
