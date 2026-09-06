# ml-residual/

LightGBM regressor trained on the residual between the physics baseline's
predicted position/velocity error and ground truth, plus SHAP explainability.

Modules:
- `common.py` — shared IO: loads physics-baseline's `strapdown.py` by path (sibling dirs with hyphens aren't importable as packages) and normalizes IMU/baseline/ground-truth timestamps onto one shared origin/clock
- `features.py` — windowed IMU feature extraction (mean/variance of accel & angular rate, turn-rate signatures, vibration/noise, time-since-last-GNSS-fix, motion mode) + residual target construction
- `train.py` — train LightGBM on the residual target, group-aware CV by `trace_id`
- `explain.py` — SHAP values per prediction, rolled up into turn / vibration / blackout-duration / motion-mode groups
- `uncertainty.py` — duration-bucketed confidence radius (CEP-90 by default) from cross-validation error (0–30s, 30–90s, 90s+)
- `tests/test_pipeline_smoke.py` — end-to-end smoke test on a **synthetic** trace, dev-only (see its docstring — never used to train the real model, per the dossier's Section 5.3 data-integrity statement)
- `models/` — saved model artifacts (gitignore large binaries or use Git LFS)

## Data contract

Three CSVs per recorded trace, all sharing the same raw `timestamp` clock/unit:

| File | Columns | Produced by |
|---|---|---|
| IMU | `timestamp, acc_x/y/z, gyro_x/y/z[, mag_x/y/z]` | sensor-logging app |
| Baseline | `timestamp, pos_x/y/z, velocity, orientation` | `physics-baseline/strapdown.py` |
| Ground truth | `timestamp, pos_x, pos_y[, pos_z], fix_available` | real GPS + blackout-window masking (dossier 5.1) |

`fix_available` (0/1) marks which rows are treated as available GNSS fixes;
position must still be present (and real) on `fix_available=0` rows — those
are the withheld labels the residual model trains and is scored against.
Load all three with `common.load_imu_csv` / `load_baseline_csv` /
`load_ground_truth_csv`, which line up their timestamps onto the IMU file's
start time, then pass them to `features.build_features_and_targets`.

## Usage

```bash
pip install -r requirements.txt
python tests/test_pipeline_smoke.py   # validates the install + pipeline wiring
```

For real training, build one features DataFrame per trace with
`features.build_features_and_targets(imu_df, baseline_df, gnss_df, trace_id=...)`,
`pd.concat` them, then:

```bash
python train.py all_traces_features.csv --out-dir models/
```

`train.py` also writes `models/oof_predictions.csv` — feed that plus the
features CSV into `uncertainty.calibrate_duration_buckets(...)` and
`uncertainty.save_calibration(...)` to produce the duration-bucketed
confidence radii served alongside predictions.
