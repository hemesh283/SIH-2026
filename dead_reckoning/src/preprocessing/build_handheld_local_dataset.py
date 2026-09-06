"""Build a parallel OxIOD handheld dataset with initial-frame local targets.

Global-target NPZ files are never modified. The fixed sequence split is reused
unchanged; no window is shuffled or transferred between splits.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.create_features import extract_sequence_features
from src.features.create_windows_local import QUATERNION_COLUMNS, create_temporal_windows_local
from src.preprocessing.build_handheld_dataset import (
    EXPECTED_SPLIT_COUNTS,
    HANDHELD_ROOT,
    PROCESSED_DIR,
    SPLIT_PATH,
    WINDOW_SIZE,
    STRIDE,
    load_sequence_split,
    resolve_synchronized_pair,
    validate_built_dataset,
    validate_window_arrays,
)
from src.preprocessing.load_data import load_sequence


LOCAL_OUTPUT_NAMES = {
    "train": "handheld_train_local.npz",
    "validation": "handheld_val_local.npz",
    "test": "handheld_test_local.npz",
}
LOCAL_METADATA_PATH = PROCESSED_DIR / "handheld_window_metadata_local.csv"
KNOWN_FIRST_TARGET = np.array([0.00081389, -0.00220183, 0.00169102])


def build_handheld_local_dataset(
    split: pd.DataFrame, handheld_root: str | Path = HANDHELD_ROOT
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    """Create local-target windows for every sequence in the fixed split CSV."""
    parts = {name: {"X": [], "y": [], "sequence_ids": []} for name in EXPECTED_SPLIT_COUNTS}
    metadata_rows: list[dict[str, object]] = []

    for _, row in split.iterrows():
        sequence_id, split_name = str(row["sequence_id"]), str(row["split"])
        try:
            syn_dir, imu_filename, vi_filename = resolve_synchronized_pair(row, handheld_root)
            imu, vi = load_sequence(syn_dir, imu_file=imu_filename, vi_file=vi_filename)
            sequence = extract_sequence_features(imu, vi)
            X, y = create_temporal_windows_local(
                sequence, vi[QUATERNION_COLUMNS], window_size=WINDOW_SIZE, stride=STRIDE
            )
            validate_window_arrays(X, y, sequence_id)
        except Exception as error:
            raise RuntimeError(f"failed to process {sequence_id}: {error}") from error

        parts[split_name]["X"].append(X)
        parts[split_name]["y"].append(y)
        parts[split_name]["sequence_ids"].extend([sequence_id] * len(X))
        metadata_rows.extend(
            {
                "split": split_name,
                "sequence_id": sequence_id,
                "window_index": index,
                "start_index": index * STRIDE,
                "end_index": index * STRIDE + WINDOW_SIZE - 1,
            }
            for index in range(len(X))
        )
        print(f"Sequence: {sequence_id}\nSplit: {split_name}\nSamples: {len(imu)}\nWindows: {len(X)}")

    datasets = {
        name: {
            "X": np.concatenate(part["X"]),
            "y": np.concatenate(part["y"]),
            "sequence_ids": np.asarray(part["sequence_ids"], dtype=str),
        }
        for name, part in parts.items()
    }
    metadata = pd.DataFrame(
        metadata_rows, columns=["split", "sequence_id", "window_index", "start_index", "end_index"]
    )
    validate_built_dataset(datasets, metadata, split)
    return datasets, metadata


def save_local_datasets(datasets: dict[str, dict[str, np.ndarray]], metadata: pd.DataFrame) -> None:
    """Write local-target NPZ files and reload them to verify persistence."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, filename in LOCAL_OUTPUT_NAMES.items():
        output_path = PROCESSED_DIR / filename
        np.savez_compressed(output_path, **datasets[split_name])
        with np.load(output_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"X", "y", "sequence_ids"}:
                raise ValueError(f"{output_path}: unexpected NPZ fields")
            validate_window_arrays(loaded["X"], loaded["y"], str(output_path))
            if loaded["X"].shape != datasets[split_name]["X"].shape or loaded["y"].shape != datasets[split_name]["y"].shape:
                raise ValueError(f"{output_path}: saved shapes do not match")
    metadata.to_csv(LOCAL_METADATA_PATH, index=False)
    if len(pd.read_csv(LOCAL_METADATA_PATH)) != len(metadata):
        raise ValueError("saved local metadata row count does not match")


def _print_target_statistics(datasets: dict[str, dict[str, np.ndarray]]) -> None:
    """Print train/validation local target distribution summaries."""
    for split_name in ("train", "validation"):
        targets = datasets[split_name]["y"]
        print(f"\n{split_name.title()} local target statistics")
        print("Mean:", targets.mean(axis=0))
        print("Std:", targets.std(axis=0))
        print("Min:", targets.min(axis=0))
        print("Max:", targets.max(axis=0))


if __name__ == "__main__":
    split = load_sequence_split(SPLIT_PATH)
    datasets, metadata = build_handheld_local_dataset(split)
    save_local_datasets(datasets, metadata)
    first_target = datasets["train"]["y"][
        np.flatnonzero(datasets["train"]["sequence_ids"] == "data1/imu1")[0]
    ]
    if not np.allclose(first_target, KNOWN_FIRST_TARGET, atol=1e-7):
        raise ValueError(f"known first local target mismatch: {first_target}")
    print("\nKnown first local target:", first_target)
    print("\nFINAL SUMMARY")
    for name, count in EXPECTED_SPLIT_COUNTS.items():
        print(f"{name.title()} sequences: {count}")
        print(f"{name.title()} windows: {len(datasets[name]['X'])}")
    print(f"Total windows: {len(metadata)}")
    _print_target_statistics(datasets)
