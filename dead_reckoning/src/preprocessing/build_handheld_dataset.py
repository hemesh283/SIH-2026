"""Build sequence-level-split window datasets from synchronized handheld OxIOD data.

The split CSV is the sole authority for train/validation/test membership. This
module processes every listed synchronized IMU/VI pair separately, then appends
only that sequence's windows to its assigned split. It never re-splits or
shuffles individual windows.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow ``python src/preprocessing/build_handheld_dataset.py`` from project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.create_features import extract_sequence_features
from src.features.create_windows import create_temporal_windows
from src.preprocessing.load_data import load_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDHELD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Oxford Inertial Odometry Dataset_2.0"
    / "Oxford Inertial Odometry Dataset"
    / "handheld"
)
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "handheld_sequence_split.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOW_SIZE = 200
STRIDE = 200
EXPECTED_SPLIT_COUNTS = {"train": 17, "validation": 3, "test": 4}
REQUIRED_SPLIT_COLUMNS = {"sequence_id", "split", "sequence", "imu_file", "vi_file"}


def load_sequence_split(split_path: str | Path) -> pd.DataFrame:
    """Load and validate the fixed 24-sequence handheld split assignment."""
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"sequence split CSV does not exist: {split_path}")
    split = pd.read_csv(split_path)
    missing_columns = REQUIRED_SPLIT_COLUMNS - set(split.columns)
    if missing_columns:
        raise ValueError(f"split CSV is missing columns: {sorted(missing_columns)}")
    if len(split) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise ValueError(f"split CSV must contain 24 rows, found {len(split)}")
    if split[list(REQUIRED_SPLIT_COLUMNS)].isna().any().any():
        raise ValueError("split CSV contains missing required values")
    if split["sequence_id"].duplicated().any():
        duplicates = split.loc[split["sequence_id"].duplicated(), "sequence_id"].tolist()
        raise ValueError(f"a sequence occurs more than once: {duplicates}")
    if (split.groupby("sequence_id")["split"].nunique() > 1).any():
        raise ValueError("a sequence occurs in more than one split")

    counts = split["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unexpected split counts: {counts}; expected {EXPECTED_SPLIT_COUNTS}")
    return split.sort_values("sequence_id", kind="stable").reset_index(drop=True)


def resolve_synchronized_pair(row: pd.Series, handheld_root: str | Path) -> tuple[Path, str, str]:
    """Resolve and validate the exact ``syn`` directory and pair named by a split row."""
    handheld_root = Path(handheld_root)
    sequence_id = str(row["sequence_id"])
    sequence_name = str(row["sequence"])
    imu_filename = Path(str(row["imu_file"])).name
    vi_filename = Path(str(row["vi_file"])).name
    syn_dir = handheld_root / sequence_name / "syn"

    expected_prefix = f"{sequence_name}/syn/"
    if not str(row["imu_file"]).replace("\\", "/").startswith(expected_prefix):
        raise ValueError(f"{sequence_id}: imu_file is not in its synchronized directory")
    if not str(row["vi_file"]).replace("\\", "/").startswith(expected_prefix):
        raise ValueError(f"{sequence_id}: vi_file is not in its synchronized directory")
    if sequence_id != f"{sequence_name}/{Path(imu_filename).stem}":
        raise ValueError(f"{sequence_id}: sequence_id does not match sequence and IMU filename")

    expected_vi_filename = f"vi{Path(imu_filename).stem.removeprefix('imu')}.csv"
    if not imu_filename.startswith("imu") or vi_filename != expected_vi_filename:
        raise ValueError(f"{sequence_id}: IMU/VI filenames are not a matching pair")
    imu_path = syn_dir / imu_filename
    vi_path = syn_dir / vi_filename
    if not imu_path.is_file():
        raise FileNotFoundError(f"{sequence_id}: IMU file does not exist: {imu_path}")
    if not vi_path.is_file():
        raise FileNotFoundError(f"{sequence_id}: VI file does not exist: {vi_path}")
    return syn_dir, imu_filename, vi_filename


def validate_window_arrays(X: np.ndarray, y: np.ndarray, sequence_id: str) -> None:
    """Validate one sequence's raw windows and endpoint displacement targets."""
    if X.ndim != 3 or X.shape[1:] != (WINDOW_SIZE, 6):
        raise ValueError(f"{sequence_id}: X must have shape (n, {WINDOW_SIZE}, 6), got {X.shape}")
    if y.ndim != 2 or y.shape[1:] != (3,):
        raise ValueError(f"{sequence_id}: y must have shape (n, 3), got {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"{sequence_id}: X/y window counts differ: {len(X)} vs {len(y)}")
    if len(X) == 0:
        raise ValueError(f"{sequence_id}: no complete windows were generated")
    if not np.isfinite(X).all():
        raise ValueError(f"{sequence_id}: X contains NaN or infinite values")
    if not np.isfinite(y).all():
        raise ValueError(f"{sequence_id}: y contains NaN or infinite values")


def build_handheld_dataset(
    split: pd.DataFrame,
    handheld_root: str | Path = HANDHELD_ROOT,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    """Process every split row into separately concatenated train/validation/test arrays.

    Uses the existing loader, feature extractor, and temporal window generator
    with ``window_size=200`` and ``stride=200``. A failure identifies its exact
    sequence rather than skipping it.
    """
    # Reuse all split validation even when this function is called programmatically.
    _validate_split_frame(split)
    split_parts = {name: {"X": [], "y": [], "sequence_ids": []} for name in EXPECTED_SPLIT_COUNTS}
    metadata_rows: list[dict[str, object]] = []

    for _, row in split.iterrows():
        sequence_id = str(row["sequence_id"])
        split_name = str(row["split"])
        try:
            syn_dir, imu_filename, vi_filename = resolve_synchronized_pair(row, handheld_root)
            imu, vi = load_sequence(syn_dir, imu_file=imu_filename, vi_file=vi_filename)
            sequence = extract_sequence_features(imu, vi)
            X, y = create_temporal_windows(
                sequence, window_size=WINDOW_SIZE, stride=STRIDE
            )
            validate_window_arrays(X, y, sequence_id)
        except Exception as error:
            raise RuntimeError(f"failed to process {sequence_id}: {error}") from error

        split_parts[split_name]["X"].append(X)
        split_parts[split_name]["y"].append(y)
        split_parts[split_name]["sequence_ids"].extend([sequence_id] * len(X))
        metadata_rows.extend(
            {
                "split": split_name,
                "sequence_id": sequence_id,
                "window_index": window_index,
            }
            for window_index in range(len(X))
        )
        print(
            f"Sequence: {sequence_id}\n"
            f"Split: {split_name}\n"
            f"Samples: {len(imu)}\n"
            f"Windows: {len(X)}"
        )

    datasets = {
        split_name: {
            "X": np.concatenate(parts["X"], axis=0),
            "y": np.concatenate(parts["y"], axis=0),
            "sequence_ids": np.asarray(parts["sequence_ids"], dtype=str),
        }
        for split_name, parts in split_parts.items()
    }
    metadata = pd.DataFrame(metadata_rows, columns=["split", "sequence_id", "window_index"])
    validate_built_dataset(datasets, metadata, split)
    return datasets, metadata


def validate_built_dataset(
    datasets: dict[str, dict[str, np.ndarray]], metadata: pd.DataFrame, split: pd.DataFrame
) -> None:
    """Validate all split arrays, provenance, and metadata after processing."""
    _validate_split_frame(split)
    total_windows = 0
    for split_name in EXPECTED_SPLIT_COUNTS:
        if split_name not in datasets:
            raise ValueError(f"missing processed dataset for {split_name}")
        data = datasets[split_name]
        if set(data) != {"X", "y", "sequence_ids"}:
            raise ValueError(f"{split_name}: dataset keys must be X, y, sequence_ids")
        validate_window_arrays(data["X"], data["y"], split_name)
        if len(data["sequence_ids"]) != len(data["X"]):
            raise ValueError(f"{split_name}: sequence ID count does not match window count")
        assigned_ids = set(split.loc[split["split"] == split_name, "sequence_id"])
        observed_ids = set(data["sequence_ids"])
        if observed_ids != assigned_ids:
            raise ValueError(f"{split_name}: windows do not map exactly to assigned sequences")
        total_windows += len(data["X"])

    if len(metadata) != total_windows:
        raise ValueError("metadata row count does not match generated window count")
    if metadata[["split", "sequence_id", "window_index"]].isna().any().any():
        raise ValueError("metadata contains missing values")
    if set(metadata["sequence_id"]) != set(split["sequence_id"]):
        raise ValueError("metadata does not contain every processed sequence")
    if (metadata.groupby("sequence_id")["split"].nunique() > 1).any():
        raise ValueError("metadata shows a sequence in more than one split")


def save_processed_datasets(
    datasets: dict[str, dict[str, np.ndarray]], metadata: pd.DataFrame, output_dir: str | Path
) -> None:
    """Save compressed NPZ split datasets and provenance metadata, then reload them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {"train": "handheld_train.npz", "validation": "handheld_val.npz", "test": "handheld_test.npz"}
    for split_name, filename in output_names.items():
        output_path = output_dir / filename
        np.savez_compressed(output_path, **datasets[split_name])
        _validate_saved_npz(output_path, datasets[split_name])

    metadata_path = output_dir / "handheld_window_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    loaded_metadata = pd.read_csv(metadata_path)
    if len(loaded_metadata) != len(metadata):
        raise ValueError("saved metadata CSV row count does not match in-memory metadata")


def _validate_split_frame(split: pd.DataFrame) -> None:
    """Apply the same invariant checks to an in-memory split table."""
    missing_columns = REQUIRED_SPLIT_COLUMNS - set(split.columns)
    if missing_columns:
        raise ValueError(f"split is missing columns: {sorted(missing_columns)}")
    if len(split) != 24 or split["sequence_id"].duplicated().any():
        raise ValueError("split must contain exactly 24 unique sequence IDs")
    if split.groupby("sequence_id")["split"].nunique().gt(1).any():
        raise ValueError("a sequence occurs in multiple splits")
    if split["split"].value_counts().to_dict() != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"split does not have required counts: {EXPECTED_SPLIT_COUNTS}")


def _validate_saved_npz(output_path: Path, expected: dict[str, np.ndarray]) -> None:
    """Confirm an NPZ can be loaded and preserves required array shapes and values."""
    if not output_path.is_file():
        raise FileNotFoundError(f"processed NPZ was not created: {output_path}")
    with np.load(output_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"X", "y", "sequence_ids"}:
            raise ValueError(f"{output_path}: unexpected NPZ keys: {loaded.files}")
        for key in ("X", "y", "sequence_ids"):
            if loaded[key].shape != expected[key].shape:
                raise ValueError(f"{output_path}: saved {key} shape does not match")
        validate_window_arrays(loaded["X"], loaded["y"], str(output_path))


if __name__ == "__main__":
    split = load_sequence_split(SPLIT_PATH)
    datasets, metadata = build_handheld_dataset(split)
    save_processed_datasets(datasets, metadata, PROCESSED_DIR)

    print("\nFINAL SUMMARY")
    for split_name, sequence_count in EXPECTED_SPLIT_COUNTS.items():
        print(f"{split_name.title()} sequences: {sequence_count}")
        print(f"{split_name.title()} windows: {len(datasets[split_name]['X'])}")
    print(f"Total windows: {len(metadata)}")
