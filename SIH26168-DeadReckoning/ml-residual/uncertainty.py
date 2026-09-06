"""Duration-bucketed uncertainty calibration (dossier Section 4.3 / jury Q8).

Deliberately NOT a per-point adaptive confidence interval: the reported
radius depends only on which blackout-duration bucket a prediction's
elapsed time falls into (0-30s / 30-90s / 90s+ by default), derived from
held-out cross-validation residual error within that bucket. This keeps the
uncertainty story honest and simple enough to state to a jury in one
sentence -- see Q8 in the dossier -- rather than implying more precision
than the method actually supports.

The radius is the CEP-`coverage` (default CEP-90) radial position error
within each bucket: sqrt(err_x^2 + err_y^2) at the `coverage` percentile.
Using CEP here, rather than inventing a separate uncertainty metric, keeps
this consistent with the CEP metric evaluation/ already reports (Section 8)
-- one way of stating "how wrong, how often" throughout the project.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

DEFAULT_BUCKETS = [(0.0, 30.0), (30.0, 90.0), (90.0, np.inf)]


def bucket_label(lo, hi):
    if np.isinf(hi):
        return f"{int(lo)}s+"
    return f"{int(lo)}-{int(hi)}s"


def assign_bucket(duration_s, buckets=DEFAULT_BUCKETS):
    for lo, hi in buckets:
        if lo <= duration_s < hi:
            return bucket_label(lo, hi)
    return bucket_label(*buckets[-1])


def calibrate_duration_buckets(feature_df, oof_predictions, duration_col="time_since_last_gnss_fix",
                                buckets=DEFAULT_BUCKETS, coverage=0.9,
                                pos_targets=("resid_pos_x", "resid_pos_y")):
    """Empirical CEP-`coverage` radius per duration bucket, from out-of-fold CV residuals.

    feature_df: the same frame passed to train_residual_models (has the
    true resid_pos_x/y targets and time_since_last_gnss_fix).
    oof_predictions: the out-of-fold predictions DataFrame train.py returned
    alongside the trained models -- must be row-aligned with feature_df.
    """
    err_x = feature_df[pos_targets[0]].to_numpy() - oof_predictions[pos_targets[0]].to_numpy()
    err_y = feature_df[pos_targets[1]].to_numpy() - oof_predictions[pos_targets[1]].to_numpy()
    radial_error = np.sqrt(err_x ** 2 + err_y ** 2)
    duration = feature_df[duration_col].to_numpy()

    calibration = {}
    for lo, hi in buckets:
        mask = (duration >= lo) & (duration < hi) & np.isfinite(radial_error)
        label = bucket_label(lo, hi)
        n = int(mask.sum())
        if n == 0:
            calibration[label] = {"radius_m": None, "n_samples": 0}
            continue
        calibration[label] = {
            "radius_m": float(np.percentile(radial_error[mask], coverage * 100)),
            "n_samples": n,
        }

    return calibration


def save_calibration(calibration, path):
    Path(path).write_text(json.dumps(calibration, indent=2))


def load_calibration(path):
    return json.loads(Path(path).read_text())


def confidence_radius(duration_s, calibration, buckets=DEFAULT_BUCKETS):
    """Look up the fixed radius for whichever bucket `duration_s` falls into.

    This is the whole interface: a duration in, a radius out. No per-point
    features are consulted here -- that is the point (Q8): the number
    widens with blackout duration and nothing else.
    """
    label = assign_bucket(duration_s, buckets)
    entry = calibration.get(label)
    if entry is None or entry["radius_m"] is None:
        raise ValueError(f"no calibrated radius for bucket '{label}' (0 held-out samples fell in it "
                          f"-- collect more traces with blackouts in that duration range)")
    return entry["radius_m"]
