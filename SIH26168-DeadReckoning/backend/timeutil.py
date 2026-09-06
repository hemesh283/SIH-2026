"""Timestamp-unit handling shared between ingest and the mock pipeline.

Mirrors `physics-baseline/strapdown.py`'s `_normalize_time` auto-detection
(unix seconds vs. milliseconds vs. nanoseconds, by magnitude) so a trace
ingested here normalizes the same way the real baseline would.
"""
import numpy as np

SCALE = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}


def detect_time_unit(raw_timestamps: np.ndarray) -> str:
    magnitude = float(np.median(np.abs(raw_timestamps)))
    if magnitude > 1e17:
        return "ns"
    if magnitude > 1e12:
        return "ms"
    return "s"


def to_seconds_elapsed(raw_timestamps: np.ndarray, time_unit: str) -> np.ndarray:
    t = raw_timestamps * SCALE[time_unit]
    return t - t[0]


def to_raw(t_seconds_elapsed, start_timestamp_raw: float, time_unit: str):
    """Inverse of to_seconds_elapsed, anchored at start_timestamp_raw."""
    return start_timestamp_raw + np.asarray(t_seconds_elapsed) / SCALE[time_unit]
