# backend/

> **DEPRECATED — not part of the shipped product (as of 2026-09-05).** The
> shipped system (`gudumap/`) does ML inference, EKF fusion, and ZUPT entirely
> on-device on Android; a network backend has no role in a product whose
> whole point is working through a GNSS blackout with no connectivity. This
> was always mock data (see `services/pipeline.py`) and predates that
> decision. Kept only because `evaluation/metrics.py` is still genuinely used
> for offline scoring — not as a running service. See
> `../../docs/PROJECT_STATUS.md` for the current architecture. Do not present
> this as a live component to a judge.

FastAPI + DuckDB. Self-contained, no Docker/cloud dependency — fast to demo.

Layout:
- `main.py` — FastAPI app, mounts routers, CORS
- `db.py` — DuckDB connection/schema (traces metadata, raw IMU samples, evaluation metrics)
- `schemas.py` — Pydantic request/response models (the API contract)
- `timeutil.py` — timestamp-unit detection/conversion, shared with `physics-baseline/strapdown.py`'s convention
- `routers/`
  - `ingest.py` — `POST /traces`, `GET /traces`, `GET /traces/{id}`
  - `trajectory.py` — `GET /traces/{id}/baseline`, `/corrected`, `/uncertainty`
  - `evaluation.py` — `GET /traces/{id}/evaluation` (ATE/RTE/CEP/drift-rate)
- `services/pipeline.py` — **currently mock**. Every router calls this module,
  never numpy directly, so swapping in the real `physics-baseline/` +
  `ml-residual/` pipeline later is a drop-in replacement behind the same
  function signatures. See its docstring for the simulation model.
- `scripts/generate_openapi.py` — writes `openapi.json` from the live app
- `scripts/smoke_test.py` — manual end-to-end check (not pytest)

## Run

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Raw spec: http://127.0.0.1:8000/openapi.json

Re-generate the static spec after any endpoint/schema change:

```bash
python scripts/generate_openapi.py   # writes backend/openapi.json
```

Hand `openapi.json` to the Android teammate — it can be imported directly
into Postman/Insomnia or run through an OpenAPI code generator, no server
required.

## Mock data, by design

`baseline`, `corrected`, and `uncertainty` are **not** computed from
`physics-baseline/` or `ml-residual/` yet — they're deterministic mock data
generated per `trace_id` (see `services/pipeline.py`), so:
- the same trace always returns the same trajectory on repeated GETs
- the Android client can be built and tested against a stable contract
  while the real pipeline is still in development
- every trajectory/uncertainty response carries a `source` field
  (`"mock"` today) so client code can branch on it instead of assuming

`POST /traces` computes and stores ATE/RTE/CEP-50/CEP-90/drift-rate too —
but from the same mock trajectories, so **these numbers are not a real
accuracy measurement either**, just self-consistent with what `/corrected`
returns. What's real here is only the *plumbing*: same tables, same
endpoint, same response shape the real pipeline will populate once
`physics-baseline/`+`ml-residual/` replace the RNG. Do not quote
`/traces/{id}/evaluation`'s numbers as system accuracy to anyone — see
`evaluation/AUDIT_Phase1-4_Report.md` Section 4.

Swapping to the real pipeline later means rewriting `services/pipeline.py`'s
functions to call `physics-baseline/strapdown.py` and the trained
`ml-residual/` model instead of the RNG — no router or schema changes.

## Data model (DuckDB)

- `traces` — one row per ingested trip: device, label, time range, sample
  count, `has_mag`/`has_gps` flags
- `raw_imu_samples` — one row per ingested sample: accel/gyro(/mag),
  optional lat/lon/alt/accuracy, `fix_available`
- `evaluation_metrics` — one row per trace: `ate_m`, `rte_m` (+ its window
  size), `cep50_m`, `cep90_m`, `drift_rate_pct`, `distance_m`

DB file lives at `backend/data/traces.duckdb` (gitignored — delete it to
reset state during development).
