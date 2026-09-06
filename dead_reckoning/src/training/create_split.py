"""Create a reproducible, sequence-level train/validation/test split for OxIOD.

This script splits synchronized IMU/VI *files*, not temporal windows. All
windows generated later from a given ``sequence_id`` must use the split written
here, preventing windows from the same recording from crossing split boundaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SEQUENCE_COUNT = 24
SPLIT_COUNTS = {"train": 17, "validation": 3, "test": 4}
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDHELD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Oxford Inertial Odometry Dataset_2.0"
    / "Oxford Inertial Odometry Dataset"
    / "handheld"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "splits" / "handheld_sequence_split.csv"


def discover_handheld_sequences(
    handheld_root: str | Path,
    expected_count: int = EXPECTED_SEQUENCE_COUNT,
) -> pd.DataFrame:
    """Discover and validate matching synchronized IMU/VI file pairs.

    File records use paths relative to ``handheld_root`` so the split CSV stays
    portable within a checkout. Each sequence ID has the form ``dataN/imuM``.
    The function fails if an IMU or VI counterpart is missing, an ID is not
    unique, or the expected number of pairs is not discovered.
    """
    handheld_root = Path(handheld_root)
    if not handheld_root.is_dir():
        raise FileNotFoundError(f"handheld dataset directory does not exist: {handheld_root}")

    imu_paths = sorted(handheld_root.glob("data*/syn/imu*.csv"))
    vi_paths = sorted(handheld_root.glob("data*/syn/vi*.csv"))
    if not imu_paths and not vi_paths:
        raise ValueError("no synchronized IMU or VI CSV files were found")

    imu_by_id = {
        _sequence_id_from_file(path, "imu"): path
        for path in imu_paths
    }
    vi_by_id = {
        _sequence_id_from_file(path, "vi"): path
        for path in vi_paths
    }
    if len(imu_by_id) != len(imu_paths) or len(vi_by_id) != len(vi_paths):
        raise ValueError("duplicate synchronized IMU or VI sequence IDs were found")

    missing_vi = sorted(set(imu_by_id) - set(vi_by_id))
    missing_imu = sorted(set(vi_by_id) - set(imu_by_id))
    if missing_vi or missing_imu:
        message_parts = []
        if missing_vi:
            message_parts.append(f"IMU files without matching VI files: {missing_vi}")
        if missing_imu:
            message_parts.append(f"VI files without matching IMU files: {missing_imu}")
        raise ValueError("; ".join(message_parts))

    sequence_ids = sorted(imu_by_id)
    if len(sequence_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} synchronized handheld sequences, "
            f"but found {len(sequence_ids)}"
        )

    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "sequence": [sequence_id.split("/", maxsplit=1)[0] for sequence_id in sequence_ids],
            "imu_file": [imu_by_id[sequence_id].relative_to(handheld_root).as_posix() for sequence_id in sequence_ids],
            "vi_file": [vi_by_id[sequence_id].relative_to(handheld_root).as_posix() for sequence_id in sequence_ids],
        }
    )


def create_sequence_split(
    sequences: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Assign each discovered sequence exactly once to train, validation, or test.

    A NumPy generator seeded with ``random_state`` shuffles only sequence row
    indices. No IMU samples or temporal windows are read or shuffled here.
    """
    _validate_discovered_sequences(sequences)
    if len(sequences) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SEQUENCE_COUNT} sequences before splitting, "
            f"but received {len(sequences)}"
        )

    shuffled_indices = np.random.default_rng(random_state).permutation(len(sequences))
    split_labels = np.empty(len(sequences), dtype=object)
    train_end = SPLIT_COUNTS["train"]
    validation_end = train_end + SPLIT_COUNTS["validation"]
    split_labels[shuffled_indices[:train_end]] = "train"
    split_labels[shuffled_indices[train_end:validation_end]] = "validation"
    split_labels[shuffled_indices[validation_end:]] = "test"

    split = sequences.copy()
    split["split"] = split_labels
    split = split[["sequence_id", "split", "sequence", "imu_file", "vi_file"]]
    validate_sequence_split(split)
    return split.sort_values("sequence_id", kind="stable").reset_index(drop=True)


def _validate_discovered_sequences(sequences: pd.DataFrame) -> None:
    """Validate the file-pair table before assigning split labels."""
    required_columns = {"sequence_id", "sequence", "imu_file", "vi_file"}
    missing_columns = required_columns - set(sequences.columns)
    if missing_columns:
        raise ValueError(
            f"discovered sequences are missing required columns: {sorted(missing_columns)}"
        )
    if sequences.empty:
        raise ValueError("no sequences are available to split")
    if sequences[list(required_columns)].isna().any().any():
        raise ValueError("discovered sequence metadata contains missing values")
    if sequences["sequence_id"].duplicated().any():
        raise ValueError("discovered sequence IDs must be unique")


def validate_sequence_split(split: pd.DataFrame) -> None:
    """Validate sequence uniqueness and the exact required split counts."""
    required_columns = {"sequence_id", "split", "sequence", "imu_file", "vi_file"}
    missing_columns = required_columns - set(split.columns)
    if missing_columns:
        raise ValueError(f"split is missing required columns: {sorted(missing_columns)}")
    if len(split) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(f"split must contain exactly {EXPECTED_SEQUENCE_COUNT} sequences")
    if split["sequence_id"].isna().any() or split["sequence_id"].duplicated().any():
        raise ValueError("every sequence must have one unique, non-empty sequence_id")
    if split["split"].isna().any():
        raise ValueError("every sequence must belong to exactly one split")

    counts = split["split"].value_counts().to_dict()
    if counts != SPLIT_COUNTS:
        raise ValueError(f"unexpected split counts: {counts}; expected {SPLIT_COUNTS}")


def save_sequence_split(split: pd.DataFrame, output_path: str | Path) -> None:
    """Validate and write the sequence-level split CSV without an index column."""
    validate_sequence_split(split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, index=False)


def _sequence_id_from_file(path: Path, prefix: str) -> str:
    """Return a shared ID for a synchronized IMU or VI filename."""
    if not path.stem.startswith(prefix):
        raise ValueError(f"unexpected {prefix} filename: {path.name}")
    suffix = path.stem.removeprefix(prefix)
    if not suffix:
        raise ValueError(f"missing sequence number in filename: {path.name}")
    return f"{path.parent.parent.name}/imu{suffix}"


if __name__ == "__main__":
    sequences = discover_handheld_sequences(HANDHELD_ROOT)
    split = create_sequence_split(sequences, random_state=RANDOM_STATE)
    save_sequence_split(split, OUTPUT_PATH)

    counts = split["split"].value_counts()
    print(f"Total sequences: {len(split)}")
    print(f"Train: {counts['train']}")
    print(f"Validation: {counts['validation']}")
    print(f"Test: {counts['test']}")
    print(f"Saved split: {OUTPUT_PATH}")
