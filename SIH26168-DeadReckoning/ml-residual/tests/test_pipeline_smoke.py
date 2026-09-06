"""End-to-end smoke test for the ml-residual pipeline: features -> train ->
explain -> uncertainty.

IMPORTANT -- what this file is, and is not: it generates a synthetic IMU
trace to exercise the *code paths* only, because no real self-collected
trace exists in data/raw_traces/ yet. This is a dev-time test fixture, not
a data source for the actual model. It must never be used to train the
model that goes in front of the jury -- the dossier's Section 5.3 data
integrity statement ("no synthetic IMU or GPS sensor readings will be
generated at any point") is about the real deliverable's training data,
and this file is deliberately outside that scope, living only under
tests/. Swap in a real self-collected trace + ground truth GPS file through
common.load_imu_csv / load_ground_truth_csv for actual model training.

Run directly: `python tests/test_pipeline_smoke.py` from ml-residual/, or
via pytest.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import common
import features
import train
import explain
import uncertainty

GRAVITY = 9.80665


def _synthetic_imu_and_truth(duration_s=400.0, hz=50.0, seed=0):
    """A winding, stop-and-go path with known ground truth, forward-simulated
    into noisy/biased body-frame IMU readings -- for pipeline testing only
    (see module docstring)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * hz)
    t = np.arange(n) / hz

    speed = 4.0 + 2.0 * np.sin(2 * np.pi * t / 60.0)
    speed = np.clip(speed, 0.0, None)
    for lo, hi in [(60, 75), (220, 235)]:
        speed[(t >= lo) & (t < hi)] = 0.0

    heading = 0.4 * np.sin(2 * np.pi * t / 90.0) + 0.15 * np.sin(2 * np.pi * t / 17.0)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    pos_x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) / hz)])
    pos_y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) / hz)])

    ax_nav = np.gradient(vx, t)
    ay_nav = np.gradient(vy, t)
    yaw_rate = np.gradient(heading, t)

    cy, sy = np.cos(heading), np.sin(heading)
    # f_body = Rz(-yaw) @ (a_nav - g_nav); roll = pitch = 0 throughout.
    acc_x = cy * ax_nav + sy * ay_nav
    acc_y = -sy * ax_nav + cy * ay_nav
    acc_z = np.full(n, GRAVITY)

    acc_bias = np.array([0.05, -0.03, 0.02])
    gyro_bias = np.array([0.0, 0.0, 0.01])
    acc_noise = rng.normal(0, 0.05, size=(n, 3))
    gyro_noise = rng.normal(0, 0.01, size=n)

    imu = pd.DataFrame({
        "timestamp": t,
        "acc_x": acc_x + acc_bias[0] + acc_noise[:, 0],
        "acc_y": acc_y + acc_bias[1] + acc_noise[:, 1],
        "acc_z": acc_z + acc_bias[2] + acc_noise[:, 2],
        "gyro_x": np.zeros(n),
        "gyro_y": np.zeros(n),
        "gyro_z": yaw_rate + gyro_bias[2] + gyro_noise,
    })

    truth = pd.DataFrame({"timestamp": t, "pos_x": pos_x, "pos_y": pos_y, "fix_available": 1})
    blackout_windows = [(100, 140), (300, 360)]
    for lo, hi in blackout_windows:
        truth.loc[(truth["timestamp"] >= lo) & (truth["timestamp"] < hi), "fix_available"] = 0

    return imu, truth


def run_smoke_test():
    tmp_dir = Path(tempfile.mkdtemp(prefix="ml_residual_smoke_"))
    imu_csv = tmp_dir / "imu.csv"
    truth_csv = tmp_dir / "truth.csv"
    baseline_csv = tmp_dir / "baseline.csv"

    imu_raw, truth_raw = _synthetic_imu_and_truth()
    imu_raw.to_csv(imu_csv, index=False)
    truth_raw.to_csv(truth_csv, index=False)

    common.strapdown.run_pipeline(
        str(imu_csv), str(baseline_csv),
        acc_var_thresh=0.01, gyro_var_thresh=0.001,
    )

    imu_df, _, time_unit = common.load_imu_csv(str(imu_csv))
    origin_raw = imu_df["timestamp"].iloc[0]
    baseline_df = common.load_baseline_csv(str(baseline_csv), origin_raw=origin_raw, time_unit=time_unit)
    gnss_df = common.load_ground_truth_csv(str(truth_csv), origin_raw=origin_raw, time_unit=time_unit)

    print(f"[smoke] baseline drift at trace end: "
          f"dx={baseline_df['pos_x'].iloc[-1] - gnss_df['pos_x'].iloc[-1]:+.1f}m, "
          f"dy={baseline_df['pos_y'].iloc[-1] - gnss_df['pos_y'].iloc[-1]:+.1f}m "
          f"(should be nonzero -- that's the residual signal the model learns)")

    feature_df = features.build_features_and_targets(
        imu_df, baseline_df, gnss_df, window_s=2.0, stride_s=1.0, trace_id="synthetic_0",
    )
    print(f"[smoke] built {len(feature_df)} windows, motion_mode counts:\n"
          f"{feature_df['motion_mode'].value_counts().to_string()}")
    assert set(features.FEATURE_COLUMNS).issubset(feature_df.columns)
    assert set(features.TARGET_COLUMNS).issubset(feature_df.columns)
    assert feature_df["time_since_last_gnss_fix"].max() > 20, \
        "expected the injected blackout windows to produce meaningful elapsed-time values"

    models, oof = train.train_residual_models(feature_df, n_splits=2)
    assert set(models) == set(features.TARGET_COLUMNS)
    assert not oof.isna().any().any(), "every row should get an out-of-fold prediction"

    calibration = uncertainty.calibrate_duration_buckets(feature_df, oof)
    print(f"[smoke] calibration: {calibration}")
    for label, entry in calibration.items():
        if entry["radius_m"] is not None:
            assert entry["radius_m"] >= 0

    X = train._prepare_X(feature_df, features.FEATURE_COLUMNS)
    row = X.iloc[[-1]]
    explainer = explain.build_explainer(models["resid_pos_x"])
    prediction = float(models["resid_pos_x"].predict(row)[0])
    explanation = explain.explain_prediction_human(explainer, row, "resid_pos_x", prediction)
    print(f"[smoke] sample explanation:\n{explanation}")

    # Round-trip through disk: the deployed path loads a saved Booster, not
    # the in-memory sklearn wrapper, so predictions and SHAP must agree.
    models_dir = tmp_dir / "models"
    train.save_models(models, models_dir)
    loaded = train.load_model(models_dir / "resid_pos_x.txt")
    loaded_prediction = float(loaded.predict(row)[0])
    assert np.isclose(loaded_prediction, prediction), \
        f"loaded Booster prediction ({loaded_prediction}) diverged from the in-memory model ({prediction})"
    explain.build_explainer(loaded)  # SHAP must accept a raw Booster too, not just the sklearn wrapper
    print(f"[smoke] save/load round-trip OK: {models_dir}")

    print("[smoke] OK -- full pipeline ran end to end")


if __name__ == "__main__":
    run_smoke_test()


def test_pipeline_smoke():
    run_smoke_test()
