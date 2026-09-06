# data/

Two subfolders (see their own READMEs for what belongs in each):
- `raw_traces/` — self-collected sensor-logger exports, real, never synthetic.
- `public_datasets/` — OxIOD/RoNIN/TLIO/RIDI/etc., supplementary volume.

Plus the two scripts here, which turn a raw phone export into the exact
CSV contract `physics-baseline/strapdown.py` and `ml-residual/` expect
(see `ml-residual/common.py` and `ml-residual/README.md`'s "Data contract"
section for the authoritative column list — these scripts are written to
match it exactly and were verified end-to-end against the real
`strapdown.py`/`common.py`/`features.py`).

## `convert_trace.py` — raw Sensor Logger export → IMU + ground-truth CSVs

```bash
python data/convert_trace.py data/raw_traces/<export_folder> \
    --stationary-start 2.0 --stationary-end 2.0
```

Input: an unzipped [Sensor Logger](https://www.tszheichoi.com/sensorlogger)
export (free, same app on Android/iOS) containing `Accelerometer.csv`,
`Gyroscope.csv`, `Location.csv`, and optionally `Magnetometer.csv`. Record
with the phone held still for ~2-3s at the start and end of every trace —
required for the unit/axis sanity checks below and for ZUPT.

Does, in order:
1. **Unit sanity check** — stationary accel norm should be ~9.8 m/s² (catches
   G's-vs-m/s² and gravity-removed "LinearAcceleration" exports, which
   `strapdown.py` would otherwise double-subtract gravity from); stationary
   gyro should be near-zero rad/s.
2. **Axis-order sanity plot** (`*_sanity.png`) — accel/gyro x/y/z traces plus
   the GPS track in local ENU meters, so you can eyeball axis swaps/sign
   flips and confirm the GPS track resembles the real route.
3. **Timestamp alignment** — checks accel/gyro sampling regularity (dropped-
   sample gaps) and that both sensors share the same epoch clock, then
   merges gyro (and magnetometer, if present) onto the accelerometer's
   timestamp grid (nearest-match, 50 ms tolerance).
4. Writes `*_report.json` (pass/fail detail) and, if every check passes:

   **`*_imu.csv`** — feed this to `physics-baseline/strapdown.py`:

   | column | meaning |
   |---|---|
   | `timestamp` | raw epoch (ns), unchanged from the phone's clock |
   | `acc_x, acc_y, acc_z` | accelerometer, m/s², includes gravity |
   | `gyro_x, gyro_y, gyro_z` | gyroscope, rad/s |
   | `mag_x, mag_y, mag_z` | magnetometer, if `Magnetometer.csv` was exported |

   **`*_ground_truth.csv`** — feed this to `ml-residual/` (via
   `common.load_ground_truth_csv`):

   | column | meaning |
   |---|---|
   | `timestamp` | raw epoch (ns), same clock as the IMU file |
   | `pos_x, pos_y, pos_z` | **local ENU meters**, origin = the trace's first GPS fix (NOT lat/lon — `strapdown.py`'s baseline output is also local ENU meters, so the two are directly comparable/subtractable) |
   | `fix_available` | always 1 here; `blackout_mask.py` is what flips windows of this to 0 |
   | `speed_mps_reported`, `bearing_deg_reported`, `horizontal_accuracy_m` | informational extras from the phone's GPS, not required by any loader |

If a check fails, fix and re-record, or pass `--allow-fail` to write outputs
anyway (not recommended for anything used in training — Section 5.1
verification discipline).

## `blackout_mask.py` — simulate a GNSS blackout (Section 5.1 / 5.3)

```bash
python data/blackout_mask.py data/raw_traces/<trace>_ground_truth.csv \
    --out-prefix data/raw_traces/<trace>_blackout \
    --window 60 120
```

Flips `fix_available` to **0** for the given window(s) (repeat `--window`
for more than one, or use `--random-duration 90 --seed 42` to auto-pick a
valid window from a currently-continuous segment). **Positions are never
touched** — per `ml-residual/common.py`'s own docstring, `pos_x/pos_y/pos_z`
must stay present even where `fix_available=0`, because those are the
withheld true labels `ml-residual/features.py` scores the baseline+residual
estimate against during the simulated blackout. `strapdown.py` never reads
this file at all (it dead-reckons from the IMU file alone), so this mask
only changes what `ml-residual/` treats as an available fix.

No accelerometer/gyroscope/GPS value is ever fabricated — matching the
dossier's Section 5.3 Data Integrity Statement.

## End-to-end example

```bash
python data/convert_trace.py data/raw_traces/20260904_walk_hostel_pixel7
python data/blackout_mask.py data/raw_traces/20260904_walk_hostel_pixel7_ground_truth.csv \
    --out-prefix data/raw_traces/20260904_walk_hostel_pixel7_blackout --window 60 120
python physics-baseline/strapdown.py data/raw_traces/20260904_walk_hostel_pixel7_imu.csv \
    data/raw_traces/20260904_walk_hostel_pixel7_baseline.csv

cd ml-residual && python -c "
import common, features
imu_df, has_mag, time_unit = common.load_imu_csv('../data/raw_traces/20260904_walk_hostel_pixel7_imu.csv')
origin_raw = imu_df['timestamp'].iloc[0]
baseline_df = common.load_baseline_csv('../data/raw_traces/20260904_walk_hostel_pixel7_baseline.csv', origin_raw=origin_raw, time_unit=time_unit)
gnss_df = common.load_ground_truth_csv('../data/raw_traces/20260904_walk_hostel_pixel7_blackout_ground_truth.csv', origin_raw=origin_raw, time_unit=time_unit)
feat_df = features.build_features_and_targets(imu_df, baseline_df, gnss_df, trace_id='20260904_walk_hostel_pixel7')
feat_df.to_csv('../data/raw_traces/20260904_walk_hostel_pixel7_features.csv', index=False)
"
```
