# SIH26168 — Smartphone-Only GNSS-Blackout Dead Reckoning

Physics-baseline + ML-residual pipeline for estimating position during GPS/GNSS
blackouts (tunnels, underground structures, dense high-rises) using only
bare smartphone IMU sensors — no extra hardware, no CAN-bus data.

Architecture (see `docs/SIH26168_Technical_Dossier.docx` for full detail):

1. **Ingest** — raw accelerometer, gyroscope, magnetometer, and (when available) GPS.
2. **Physics baseline** (`physics-baseline/`) — strapdown mechanization + complementary
   filter/EKF for orientation and position, with Zero-Velocity Update (ZUPT)
   corrections at detected stationary points.
3. **ML residual** (`ml-residual/`) — LightGBM regressor trained on windowed IMU
   features to correct the baseline's systematic bias/drift, with SHAP explainability.
4. **Fusion** — baseline + residual correction = final position estimate, with an
   uncertainty radius that widens with blackout duration.
5. **Reacquisition** — smooth reconciliation with GPS when signal returns.
6. **Presentation** (`backend/`, `frontend-android/`) — FastAPI + DuckDB backend,
   dashboard/app showing live trajectory, naive vs. corrected path, and confidence radius.

## Repo layout

```
docs/                   technical dossier, PS screenshot, learning notes
data/
  raw_traces/            self-collected sensor-logger exports (real, never synthetic)
  public_datasets/        OxIOD, RoNIN, TLIO, RIDI, etc. (supplementary training volume)
physics-baseline/        strapdown + EKF + ZUPT (Python)
ml-residual/              LightGBM training + SHAP
backend/                  FastAPI + DuckDB
frontend-android/         friend's existing Android Studio project
evaluation/               ATE/RTE/CEP scripts, result plots
ppt/                      slide content, video, jury Q&A doc
```

## Current status (read before a demo or jury Q&A)

Architecture and code are ahead of data collection. As of the last audit
(`evaluation/AUDIT_Phase1-4_Report.md`):
- **No real trace has been recorded yet** — `data/raw_traces/` is empty
  except its own README; no public dataset (OxIOD/RoNIN/TLIO/RIDI/GSDC) has
  been downloaded either.
- **No model has been trained** — `ml-residual/train.py` has only ever run
  against one synthetic dev fixture (`ml-residual/tests/`), never real data.
  There is no saved model file anywhere in this repo.
- **The backend is mock** — `backend/services/pipeline.py` generates
  deterministic fake trajectories/uncertainty/evaluation numbers, disclosed
  via a `"source": "mock"` field on every response. See `backend/README.md`.
- **ZUPT is unvalidated** — `physics-baseline/strapdown.py`'s stationary-
  detection thresholds were only ever fit to synthetic noise; use
  `physics-baseline/calibrate_zupt.py` on a real recorded trace before
  trusting it.

None of the numbers currently produced anywhere in this repo (mock
`/evaluation`, the ml-residual smoke test's `[train]`/`[smoke]` prints, or
`evaluation/run_eval.py` against the synthetic fixture) are real accuracy
figures — don't present them as such. Read the audit's full "what you
should NOT claim to a judge yet" list before a demo.

## Data integrity

No synthetic IMU or GPS sensor readings are ever generated. The only simulated
element is the blackout-window boundary (masking known-good GPS labels from a
real trace to test blackout performance) — a standard, accepted dead-reckoning
evaluation technique, not synthetic data.

## Timeline (to 20 September 2026)

- Week 1 — data collection protocol + first real traces; public datasets integrated
- Week 2 — strapdown + EKF baseline implemented and validated; ZUPT tuned
- Week 3 — LightGBM residual model trained; SHAP wired in; uncertainty calibrated
- Week 4 — FastAPI + DuckDB backend + React/Recharts dashboard; end-to-end test
- Final days — live-demo rehearsal, jury Q&A rehearsal, docs/pitch polish
