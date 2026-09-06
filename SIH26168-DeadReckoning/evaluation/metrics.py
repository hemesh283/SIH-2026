"""ATE / RTE / CEP / drift-rate implementations for scoring an estimated
trajectory against ground truth (dossier headline metrics, see README.md).

These are the same metric definitions backend/services/pipeline.py's mock
generator already computes (ate_m, rte_m, cep50_m, cep90_m, drift_rate_pct)
-- written here as a real, tested, reusable implementation so a run_eval.py
result and a mock /evaluation response are directly comparable once
physics-baseline/ml-residual produce a real trajectory to feed this instead.

Nothing in this module is data-source-specific: it scores whatever two
position tracks (arrays of x/y meters in a shared local frame, at shared
timestamps) you hand it, real or synthetic. See tests/test_metrics.py for
synthetic sanity checks with known answers; see run_eval.py for the CLI
that loads real trace CSVs and calls these functions.
"""
import numpy as np

DEFAULT_RTE_WINDOW_S = 60.0


def interpolate_positions(t_query, t_source, pos_x, pos_y):
    """Linear-interpolate a (pos_x, pos_y) track sampled at t_source onto t_query."""
    return (
        np.interp(t_query, t_source, pos_x),
        np.interp(t_query, t_source, pos_y),
    )


def position_errors(est_x, est_y, truth_x, truth_y):
    """Per-sample Euclidean position error, meters."""
    return np.hypot(np.asarray(est_x) - np.asarray(truth_x), np.asarray(est_y) - np.asarray(truth_y))


def ate(errors):
    """Absolute Trajectory Error: RMS of per-sample position error, meters."""
    errors = np.asarray(errors)
    if len(errors) == 0:
        return 0.0
    return float(np.sqrt(np.mean(errors ** 2)))


def cep(errors, percentile):
    """Circular Error Probable: radius (meters) containing `percentile`% of position error."""
    errors = np.asarray(errors)
    if len(errors) == 0:
        return 0.0
    return float(np.percentile(errors, percentile))


def rte(t, est_x, est_y, truth_x, truth_y, window_s=DEFAULT_RTE_WINDOW_S):
    """Relative Trajectory Error: RMS, over fixed-length sub-windows, of how
    much the estimate's displacement across a window differs from the
    truth's displacement across the same window.

    This isolates drift accumulated *within* a window from any constant
    offset already present going into it (which ATE folds in as a flat
    penalty for the whole track) -- the standard relative-pose-error style
    drift metric (cf. TUM RGB-D benchmark, KITTI odometry).
    """
    t = np.asarray(t)
    est_x, est_y = np.asarray(est_x), np.asarray(est_y)
    truth_x, truth_y = np.asarray(truth_x), np.asarray(truth_y)
    if len(t) < 2 or t[-1] <= t[0]:
        return 0.0

    window_starts = np.arange(t[0], t[-1], window_s)
    window_errs = []
    for ws in window_starts:
        we = ws + window_s
        idx = np.where((t >= ws) & (t <= we))[0]
        if len(idx) < 2:
            continue
        dx_est = est_x[idx[-1]] - est_x[idx[0]]
        dy_est = est_y[idx[-1]] - est_y[idx[0]]
        dx_truth = truth_x[idx[-1]] - truth_x[idx[0]]
        dy_truth = truth_y[idx[-1]] - truth_y[idx[0]]
        window_errs.append(np.hypot(dx_est - dx_truth, dy_est - dy_truth))

    if not window_errs:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window_errs))))


def path_length(pos_x, pos_y):
    """Total distance travelled along a position track, meters."""
    pos_x, pos_y = np.asarray(pos_x), np.asarray(pos_y)
    if len(pos_x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(pos_x), np.diff(pos_y))))


def drift_rate_pct(final_error_m, distance_m):
    """Final position error as a percentage of distance travelled.

    Returns 0.0 (rather than a meaningless huge percentage from dividing by
    ~zero) when the track travelled effectively no distance.
    """
    if distance_m <= 1e-6:
        return 0.0
    return float(100.0 * final_error_m / distance_m)


def evaluate_trajectory(t, est_x, est_y, truth_x, truth_y, rte_window_s=DEFAULT_RTE_WINDOW_S):
    """Score one estimated trajectory against ground truth, already aligned
    onto a common time base `t`.

    Returns the same field set backend's mock /evaluation response uses:
    ate_m, rte_m, rte_window_s, cep50_m, cep90_m, drift_rate_pct, distance_m,
    plus n_samples.
    """
    est_x, est_y = np.asarray(est_x), np.asarray(est_y)
    truth_x, truth_y = np.asarray(truth_x), np.asarray(truth_y)

    errors = position_errors(est_x, est_y, truth_x, truth_y)
    distance_m = path_length(truth_x, truth_y)
    final_error_m = float(errors[-1]) if len(errors) else 0.0

    return {
        "ate_m": ate(errors),
        "rte_m": rte(t, est_x, est_y, truth_x, truth_y, rte_window_s),
        "rte_window_s": rte_window_s,
        "cep50_m": cep(errors, 50),
        "cep90_m": cep(errors, 90),
        "drift_rate_pct": drift_rate_pct(final_error_m, distance_m),
        "distance_m": distance_m,
        "n_samples": int(len(errors)),
    }


def evaluate_against_ground_truth(est_t, est_x, est_y, truth_t, truth_x, truth_y,
                                   rte_window_s=DEFAULT_RTE_WINDOW_S):
    """Convenience wrapper: interpolates the estimate onto the ground-truth
    track's own timestamps (ground truth is the sparser, authoritative
    clock -- e.g. GNSS fixes), then scores it with evaluate_trajectory."""
    aligned_x, aligned_y = interpolate_positions(truth_t, est_t, est_x, est_y)
    return evaluate_trajectory(truth_t, aligned_x, aligned_y, truth_x, truth_y, rte_window_s)
