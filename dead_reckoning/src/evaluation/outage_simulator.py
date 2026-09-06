"""Deterministic GNSS outage interval simulation and masking.

Provides reproducible GNSS-denied intervals across specified durations
(e.g., 10s, 30s, 60s, 120s) without ground-truth leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class OutageInterval:
    """Defines a deterministic GNSS blackout interval."""

    duration_s: float
    start_time_s: float
    end_time_s: float
    start_idx: int
    end_idx: int


class OutageSimulator:
    """Simulates controlled GNSS-denied environments."""

    def __init__(
        self,
        outage_durations_s: Sequence[float] = (10.0, 30.0, 60.0, 120.0),
        warmup_duration_s: float = 15.0,
        recovery_duration_s: float = 15.0,
        random_seed: int = 42,
    ) -> None:
        self.outage_durations_s = list(outage_durations_s)
        self.warmup_duration_s = warmup_duration_s
        self.recovery_duration_s = recovery_duration_s
        self.random_seed = random_seed

    def plan_outages(
        self,
        timestamps_s: np.ndarray,
        duration_s: float,
    ) -> List[OutageInterval]:
        """Generate deterministic outage intervals for a given duration.
        
        Args:
            timestamps_s: Monotonically increasing timestamps in seconds.
            duration_s: Outage duration in seconds (e.g. 10, 30, 60, 120).
            
        Returns:
            List of valid OutageInterval objects.
        """
        total_time = timestamps_s[-1] - timestamps_s[0]
        min_required = self.warmup_duration_s + duration_s + self.recovery_duration_s
        if total_time < min_required:
            # If sequence is shorter than warmup + outage + recovery,
            # clamp warmup to 5s or scale proportionally
            warmup = min(5.0, total_time * 0.1)
            recovery = min(5.0, total_time * 0.1)
            if total_time < duration_s + warmup + recovery:
                return []
        else:
            warmup = self.warmup_duration_s
            recovery = self.recovery_duration_s

        earliest_start = timestamps_s[0] + warmup
        latest_start = timestamps_s[-1] - recovery - duration_s

        if latest_start < earliest_start:
            return []

        # Deterministically place outage in the middle or at reproducible seed location
        rng = np.random.default_rng(self.random_seed + int(duration_s * 10))
        # Place outage at 40% into the valid window
        start_t = earliest_start + 0.3 * (latest_start - earliest_start)
        end_t = start_t + duration_s

        start_idx = int(np.searchsorted(timestamps_s, start_t))
        end_idx = int(np.searchsorted(timestamps_s, end_t))

        return [
            OutageInterval(
                duration_s=duration_s,
                start_time_s=start_t,
                end_time_s=end_t,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        ]

    def create_gnss_mask(self, timestamps_s: np.ndarray, interval: OutageInterval) -> np.ndarray:
        """Create a boolean mask where True = GNSS available, False = GNSS blackout."""
        mask = np.ones(len(timestamps_s), dtype=bool)
        mask[interval.start_idx : interval.end_idx] = False
        return mask
