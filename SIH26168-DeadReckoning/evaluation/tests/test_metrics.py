"""Unit tests for evaluation/metrics.py, against small synthetic tracks with
known-by-construction answers. This is code-correctness testing only -- it
is not, and must never be read as, a real accuracy evaluation (see
evaluation/AUDIT_Phase1-4_Report.md Section 1's caveat about synthetic data).

Run: `pytest evaluation/tests/test_metrics.py` from the repo root, or
`python tests/test_metrics.py` from evaluation/.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics


def test_perfect_match_is_zero_error():
    t = np.linspace(0, 100, 101)
    x = t * 1.5
    y = t * 0.5

    result = metrics.evaluate_trajectory(t, x, y, x, y, rte_window_s=20.0)

    assert result["ate_m"] == 0.0
    assert result["rte_m"] == 0.0
    assert result["cep50_m"] == 0.0
    assert result["cep90_m"] == 0.0
    assert result["drift_rate_pct"] == 0.0
    assert result["n_samples"] == 101


def test_constant_offset_gives_exact_ate_and_cep():
    """A fixed 3m offset (3-4-5 triangle) applied uniformly: every sample's
    error is exactly 5m, so ATE/CEP50/CEP90 must all equal 5.0 exactly, and
    RTE (which measures drift within a window, not a constant bias) must be
    ~0 since the offset never changes."""
    t = np.linspace(0, 100, 101)
    truth_x = t
    truth_y = np.zeros_like(t)
    est_x = truth_x + 3.0
    est_y = truth_y + 4.0

    result = metrics.evaluate_trajectory(t, est_x, est_y, truth_x, truth_y, rte_window_s=20.0)

    assert np.isclose(result["ate_m"], 5.0)
    assert np.isclose(result["cep50_m"], 5.0)
    assert np.isclose(result["cep90_m"], 5.0)
    assert np.isclose(result["rte_m"], 0.0, atol=1e-9)


def test_rte_isolates_within_window_drift_from_constant_offset():
    """Same constant 5m offset as above, plus a linearly growing drift on
    top. RTE over a window should reflect only the drift accumulated within
    that window, not the constant part."""
    t = np.linspace(0, 100, 101)
    truth_x = t
    truth_y = np.zeros_like(t)
    drift_rate = 0.1  # meters of extra error per second, accumulating
    est_x = truth_x + 3.0 + drift_rate * t
    est_y = truth_y + 4.0

    result = metrics.evaluate_trajectory(t, est_x, est_y, truth_x, truth_y, rte_window_s=20.0)

    # Over a 20s window the drift term alone contributes drift_rate * 20 = 2.0m
    assert np.isclose(result["rte_m"], drift_rate * 20.0, atol=1e-6)
    # ATE folds in both the constant offset and the growing drift, so it's
    # well above the RTE-isolated per-window figure.
    assert result["ate_m"] > result["rte_m"]


def test_drift_rate_pct_matches_final_error_over_distance():
    t = np.linspace(0, 10, 11)
    truth_x = t * 10.0  # travels 100m total
    truth_y = np.zeros_like(t)
    est_x = truth_x.copy()
    est_y = truth_y.copy()
    est_y[-1] = 5.0  # 5m error only at the final sample

    result = metrics.evaluate_trajectory(t, est_x, est_y, truth_x, truth_y)

    assert np.isclose(result["distance_m"], 100.0)
    assert np.isclose(result["drift_rate_pct"], 5.0)  # 5m / 100m * 100


def test_drift_rate_pct_zero_when_no_distance_travelled():
    assert metrics.drift_rate_pct(final_error_m=5.0, distance_m=0.0) == 0.0


def test_evaluate_against_ground_truth_interpolates_estimate_onto_sparser_truth():
    """Ground truth (e.g. GNSS fixes) is typically sampled much sparser than
    the estimate (e.g. 50Hz IMU-derived positions) -- evaluate_against_ground_truth
    must interpolate the dense estimate onto the sparse truth timestamps,
    not require them to already match."""
    est_t = np.linspace(0, 10, 101)  # dense
    est_x = est_t * 2.0
    est_y = np.zeros_like(est_t)

    truth_t = np.array([0.0, 5.0, 10.0])  # sparse
    truth_x = truth_t * 2.0
    truth_y = np.zeros_like(truth_t)

    result = metrics.evaluate_against_ground_truth(est_t, est_x, est_y, truth_t, truth_x, truth_y)

    assert result["n_samples"] == 3
    assert np.isclose(result["ate_m"], 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("all metrics tests passed")
