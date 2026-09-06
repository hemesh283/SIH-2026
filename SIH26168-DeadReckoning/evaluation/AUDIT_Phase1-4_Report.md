# SIH26168 — Phase 1–4 Audit (Skeptical, Evidence-Based)

**Scope:** `D:\Projects\SIH_2026\SIH26168-DeadReckoning`, checked against `docs/SIH26168_Technical_Dossier.docx`.
**Method:** Read every source file involved, then actually executed it — installed a fresh Python 3.10 environment, ran the ML smoke test, wrote and ran standalone scripts to interrogate ZUPT and trajectory output, started the FastAPI backend and hit every endpoint with curl. All commands and raw output are reproduced below.

**Headline finding, before the phase-by-phase detail:** the repository contains **zero bytes of data anywhere** — no self-collected traces, no downloaded public datasets, no trained models, no saved evaluation results, and it isn't even a git repo yet (no `.git/`). Everything that exists is code and documentation. This one fact drives most of the PASS/FAIL calls below.

---

## 1. DATA PROVENANCE (Section 5) — **FAIL**

```
$ find data/raw_traces -maxdepth 3        # → only README.md
$ find data/public_datasets -maxdepth 3   # → only README.md
$ find . -iname "*.csv" -o -iname "*.pkl" -o -iname "*.npz" -o -iname "*.model" \
    -not -path "*/.venv/*"                # → nothing (0 data files anywhere in the repo)
```

- `data/raw_traces/` contains **only its own README** — no self-collected sensor-logger export exists. Zero walk/drive tests have been recorded.
- `data/public_datasets/` contains **only its own README**, whose own checklist confirms it:
  ```
  - [ ] OxIOD requested
  - [ ] OxIOD downloaded → public_datasets/oxiod/
  - [ ] RoNIN
  - [ ] TLIO
  - [ ] RIDI
  - [ ] Google Smartphone Decimeter Challenge
  ```
  Every box is unchecked. None of OxIOD/RoNIN/TLIO/RIDI/GSDC has been requested or downloaded. The dossier's Section 3.3 / 5.2 tables describing these datasets are accurate descriptions of *public* datasets, but none of that data is in this project.
- No `.git` directory exists at all, so there's no history to check either — this has never been version-controlled.
- **Synthetic data:** there is exactly one place synthetic sensor data is generated — `ml-residual/tests/test_pipeline_smoke.py`. Its own docstring is explicit and correct about scope: *"generates a synthetic IMU trace to exercise the code paths only... must never be used to train the model that goes in front of the jury."* That line is honest and matches the dossier's Section 5.3 claim — but it also means the *only* pipeline execution that has ever happened, anywhere, used synthetic data, which is the opposite of what "the model" is supposed to have been trained on.
- **convert_trace.py sanity checks — could not re-run them "and show real output," because there is no raw Sensor-Logger export to feed it.** I read the implementation carefully instead (unit check via stationary `|a|` norm, gyro-zero check, sampling-gap check, accel/gyro clock-skew check — `data/convert_trace.py:103-150`) and it's a reasonable, correctly-implemented set of checks. But **no PASS printout exists anywhere to take on faith or verify** — the script has never been run once. Nothing has been recorded, so nothing has been sanity-checked.

**Verdict: FAIL.** Not "unverified" — there is no data to verify. Every claim in Section 5 about "traces will be cross-checked" and "no synthetic sensor readings" is aspirational/methodological, not yet demonstrated on a single real file.

---

## 2. PHYSICS BASELINE (Section 4.1) — **PARTIAL PASS on mechanics, FAIL on validation**

No real trace exists (see above), so I could not "re-run `strapdown.py` on a real trace" as asked — that's impossible right now, and I'm not going to paper over that. What I *did* do: ran it end-to-end on the project's own synthetic dev-fixture (`ml-residual/tests/test_pipeline_smoke.py`'s generator — reused verbatim, not a new fixture I invented) to check the code mechanically works, and specifically instrumented ZUPT.

```
$ source /tmp/audit_venv/bin/activate && python -m pytest ml-residual/tests/test_pipeline_smoke.py -v -s
[smoke] baseline drift at trace end: dx=-1447.5m, dy=-86.3m (should be nonzero...)
[smoke] built 398 windows, motion_mode counts: pedestrian 397, vehicle 1
...
PASSED — 1 passed in 6.56s
```

**Does it produce a trajectory, not NaNs/errors?** Yes.
```
pos_def:  any NaN=False, any Inf=False, final=[-59.9  20.5  221.9]
pos_loose: any NaN=False, any Inf=False, final=[0.24  0.42  0.09]
```
No crashes, no NaNs, on either threshold setting.

**Does ZUPT actually fire?** I wrote a standalone check (`detect_stationary` + `integrate_trajectory` called directly, comparing against the two known-stationary windows the synthetic generator injects at t=[60,75) and t=[220,235)). Result — **this is the most important finding in this section**:

| Threshold setting | Overall % flagged stationary | % flagged during real stop [60,75) | % flagged during real stop [220,235) | False-positive rate (flagged "stationary" while actually moving) |
|---|---|---|---|---|
| **Default** (`acc_var_thresh=0.0005, gyro_var_thresh=0.0003` — what you get running the CLI with no extra flags, exactly as the top-level README's own example shows) | 0.1% | **0.0%** | 0.1% | 0.12% |
| **Loosened** (`0.01 / 0.001` — what `test_pipeline_smoke.py` uses internally to make the pipeline exercise ZUPT at all) | 99.7% | 98.1% | 98.1% | **99.85%** |

With the **default settings a real user would actually run**, ZUPT essentially never fires — 0% detection during a modeled full stop. With the settings that do make it fire, it fires on 99.85% of samples that are *not* stationary, i.e. it's not gating anything — it's zeroing velocity almost everywhere, which is why the "loosened" trajectory collapses to near the origin (`pos_loose final = [0.24, 0.42, 0.09]`) while the true synthetic path travelled ~1450m. The 1450m of "residual" the smoke test prints isn't realistic MEMS drift for the ML model to learn — it's mostly an artifact of ZUPT effectively disabling the baseline's ability to move at all.

This exactly matches the code's own self-documented caveat (`strapdown.py:184-191`): *"The defaults below were fit against one synthetic noise profile only; plot acc/gyro variance for a real recorded stationary segment... before trusting this on a real trace."* That caveat is correct and this audit confirms it in both directions — under both threshold settings tried, ZUPT is not usable as-is.

A plot comparing ground truth vs. both threshold settings is attached (`AUDIT_synthetic_zupt_check.png` — also saved into `evaluation/` in your project folder). **Read the caption on it** — this is synthetic-fixture output for code-mechanics verification only, not a real-world accuracy claim, and the plot is titled that way on purpose so it can't be mistaken for one later.

**Verdict: PARTIAL PASS / FAIL.** The strapdown+complementary-filter+double-integration code runs correctly and produces a plausible trajectory shape. ZUPT is implemented but **unvalidated and, on the only data available, either never fires or fires almost everywhere** depending on which threshold you use — neither is correct. There is no evidence it works on real phone accelerometer/gyro noise, because none has ever been recorded.

---

## 3. ML RESIDUAL MODEL (Section 4.2) — **FAIL (no real model has ever been trained)**

- **How many traces was this trained on?** Zero real traces. `train.py` has never been run against real data — there is no `models/` directory, no saved `.txt` LightGBM boosters, no `oof_predictions.csv` anywhere in the repo. The only time `train_residual_models()` has ever executed is inside the synthetic smoke test, on **one** synthetic trace. You are right that one trace isn't enough for a meaningful split — the code agrees with you and says so out loud when it happens:
  ```
  [train] only 1 trace group(s) available -- falling back to plain KFold(n_splits=2).
  Group-aware CV needs >= 2 distinct trace_ids; add more traces before trusting
  these out-of-fold numbers as a real accuracy estimate.
  ```
- **Leakage check.** I read `train.py` line by line for this specifically. It does the right thing: `GroupKFold` keyed on `trace_id` (`train.py:45-54`), specifically so that overlapping-stride windows from the same recording can't land on both sides of a split. It only falls back to plain `KFold` — and prints the warning above — when fewer trace groups exist than requested splits. **This is correctly engineered to prevent the leakage you're worried about**, conditional on there eventually being ≥5 (or `--n-splits`) distinct real traces to group by. Right now there is exactly one trace ever used (synthetic), so the fallback path is what actually ran, and the code's own warning is the honest disclosure of that.
- **Re-run evaluation from scratch — real numbers obtained (on the synthetic fixture, since no real data exists):**
  ```
  [train] resid_pos_x: out-of-fold RMSE = 427.4366
  [train] resid_pos_y: out-of-fold RMSE = 14.1570
  [train] resid_speed: out-of-fold RMSE = 0.8737
  [smoke] calibration: {'0-30s': {'radius_m': 695.0, 'n_samples': 358},
                        '30-90s': {'radius_m': 604.5, 'n_samples': 40},
                        '90s+': {'radius_m': None, 'n_samples': 0}}
  ```
  A 427m out-of-fold RMSE and a 695m calibrated confidence radius, on a trace that only travels ~1.5km total, are **not remotely plausible** dead-reckoning numbers by any standard — but that's expected and consistent with finding #2 above: the ZUPT miscalibration means the baseline barely moves, so the "residual" the model is trying to learn is dominated by that bug, not by realistic MEMS bias/drift. This is the opposite failure mode from what you asked me to watch for (suspiciously *perfect* numbers from leakage) — these numbers are suspiciously *catastrophic*, for a different, identifiable reason.
- **Comparison against Tier 2 benchmarks (OxIOD/RoNIN/TLIO, dossier Section 3.3):** not meaningful yet. TLIO reports <3m error for 90% of a 3–7 minute trial; the smoke-test numbers above are 100–200x worse, but on synthetic data with a known ZUPT bug, so this isn't a fair or informative comparison in either direction. **There is currently no real evaluation number to compare against the literature at all.**
- **SHAP — confirmed running on the actual trained model, not a placeholder:**
  ```
  [smoke] sample explanation:
  Predicted resid_pos_x correction: +1291.73m.
    - raw motion level: pushed the correction up by +278.019m
    - turning: pushed the correction up by +178.201m
    - vibration road noise: pushed the correction up by +177.884m
    - blackout duration: pulled the correction down by -50.847m
  ```
  This is `shap.TreeExplainer` called on the real fitted `LGBMRegressor` (`explain.py:39-52`), and the smoke test additionally verifies SHAP works on a Booster loaded back from disk, not just the in-memory sklearn wrapper (`test_pipeline_smoke.py:145-149`) — a real, non-trivial check (booster round-trip is a common source of silent breakage). SHAP is genuinely wired up. The *values* are enormous only because the underlying prediction is enormous, for the ZUPT reason above — the mechanism is real, the inputs feeding it aren't yet.
- **Uncertainty radius — confirmed derived from actual cross-validation error, not hardcoded:** `uncertainty.calibrate_duration_buckets()` computes CEP-90 directly from `feature_df` targets minus `oof_predictions` (`uncertainty.py:51-53`) — real arithmetic on real (if synthetic-fixture) residuals, and it correctly returns `None`/0-samples for the `90s+` bucket rather than fabricating a number when no data fell in it. **Separately and importantly**: the backend's `services/pipeline.py` has its *own*, completely different, hand-picked `UNCERTAINTY_BUCKETS = [3.5m, 9m, 22m]` (`pipeline.py:41-45`) which is explicitly labeled mock and has no connection to `ml-residual/uncertainty.py` at all — see Section 4 below.

**Verdict: FAIL as a trained system, PASS on code correctness.** No real model has ever been trained — there's nothing deployed. The training/CV/SHAP/uncertainty machinery is well-engineered and does the right things (group-aware CV, real SHAP, real calibration arithmetic), but every number it has ever produced came from one synthetic trace corrupted by the ZUPT issue in Section 2. There is currently no meaningful accuracy claim you could make about this model to anyone.

---

## 4. BACKEND (Section 6–7) — **FAIL — confirmed mocked, but transparently so**

This is disclosed in the code, not hidden — I want to be precise about that distinction before the verdict, because "the code lies about this" and "the code is honest that this isn't done yet" are very different problems and it's the second one here.

`backend/main.py`'s own FastAPI description string says, verbatim:
> *"Baseline, corrected, and uncertainty are currently mock data — deterministic per trace, stable across repeated calls, generated by `backend/services/pipeline.py` — while `physics-baseline/` and `ml-residual/` are still in development."*

`services/pipeline.py`'s docstring: *"Mock stand-in for the real pipeline... This is the ONLY module that will need to change once the real strapdown mechanization and the trained LightGBM residual model are ready."*

I verified this isn't stale documentation by reading the actual router code and then hitting a live server:

```python
# routers/trajectory.py — every trajectory endpoint calls pipeline.generate_*(), never strapdown.py or a saved model
points = pipeline.generate_baseline(trace_id, trace["duration_s"], ...)
return TrajectoryResponse(..., source="mock", points=points)
```

**Live request/response evidence** (server started with `uvicorn main:app`, hit with `curl`):

```
$ curl -X POST http://127.0.0.1:8123/traces -d @payload.json
{"trace_id":"e0cf7276-8547-4b24-9386-f8bf91b6ab56", ..., "status":"ready"}   # 201 Created

$ curl http://127.0.0.1:8123/traces/e0cf7276.../baseline
source= mock
n_points= 21
first point= {'t': 0.0, 'pos_x': 0.0, 'pos_y': 0.0, 'velocity': 0.86, 'orientation': 0.0}
last point=  {'t': 3.98, 'pos_x': 1.43, 'pos_y': -0.09, 'velocity': 0.45, 'orientation': 349.1}

$ curl http://127.0.0.1:8123/traces/e0cf7276.../corrected
source= mock

$ curl http://127.0.0.1:8123/traces/e0cf7276.../uncertainty
source= mock
buckets= [{'label': '0-30s', 'radius_m': 3.5}, {'label': '30-90s', 'radius_m': 9.0}, {'label': '90s+', 'radius_m': 22.0}]

$ curl http://127.0.0.1:8123/traces/e0cf7276.../evaluation
{"ate_m":0.044,"rte_m":0.030,"cep50_m":0.040,"cep90_m":0.064,"drift_rate_pct":2.06,"source":"mock"}

$ curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8123/traces/does-not-exist/baseline
404
```

Every payload carries `"source": "mock"` — every field, every response, no exceptions — and the fake "evaluation" numbers above (ATE 0.04m, drift rate 2%) are *not* computed from anything real; `generate_evaluation_metrics()` scores a fabricated "corrected" path against a fabricated "true" path, both generated from the same seeded RNG (`pipeline.py:160-181, 249-288`). **Those numbers happen to look great specifically because the mock generator makes them look great — they say nothing about system accuracy and must not be quoted as if they do.**

What *is* real and working: request validation, DuckDB persistence (traces + samples + evaluation-metrics tables), the ingest→list→get flow, 404 handling, and the API contract shape (`TrajectoryResponse`, `UncertaintyResponse`, etc.) — genuinely functional plumbing. It is only the actual dead-reckoning computation, on all three trajectory-related endpoints plus stored evaluation metrics, that's a stand-in.

One more concrete gap while I was in here: `evaluation/` (the directory the top-level README and dossier point to for "ATE/RTE/CEP scripts, result plots") **contains only a README** — `metrics.py` and `run_eval.py` are listed as "planned modules," not implemented. The only ATE/RTE/CEP/drift-rate code that exists anywhere in the repo is the mock version inside `backend/services/pipeline.py`.

**Verdict: FAIL** against "not mocked." The good news: the mocking is self-labeled at every layer (docstrings, FastAPI description, and a `source` field on every single API response), so there's no risk of *accidentally* presenting this as real — the risk is entirely about what gets said out loud in a demo or a slide, not what the code claims.

---

## 5. FINAL VERDICT

**What's genuinely working:**
- The strapdown mechanization math (orientation via complementary filter, body→nav rotation, gravity removal, trapezoidal double integration) runs cleanly, produces a finite trajectory, and its shortcomings (no-mag yaw drift, small-angle approximation) are correctly self-documented in the code, not hidden.
- The ML side's *engineering* is genuinely good: group-aware cross-validation that actually prevents the leakage failure mode you asked about, real SHAP on real fitted/reloaded models, and uncertainty calibration that computes real percentiles from real (if currently synthetic) held-out residuals rather than hardcoding a number.
- The FastAPI + DuckDB backend plumbing (ingest, storage, listing, 404s, response contracts) works and is honestly self-labeled as mock where it's mock.
- Data-integrity *discipline as written* (blackout masking never touches real positions/IMU values, convert_trace.py's sanity-check design) is sound in design.

**What's fragile or unverified:**
- ZUPT's stationary-detection thresholds are unvalidated in both directions tested (never fires / fires almost always) and explicitly self-flagged in the code as fit to a made-up noise profile.
- The entire ML pipeline has only ever executed once, on one synthetic trace, and every number it has ever produced is downstream of the ZUPT problem above.
- `evaluation/`'s promised metrics scripts don't exist yet outside the mock backend code.

**What you should NOT claim to a judge yet, because the evidence doesn't support it:**
1. Do not say "we collected real walk/drive data" or "we integrated OxIOD/RoNIN" — zero bytes of either exist in this repo right now. If asked directly ("show me a real trace"), there is currently nothing to show.
2. Do not show or quote the `/evaluation` endpoint's numbers, or anything from it, as system accuracy — they're a self-consistent simulation, not a measurement. The `AUDIT_synthetic_zupt_check.png` plot and `[train]`/`[smoke]` numbers above are also not real-world accuracy figures — they're a synthetic-fixture diagnostic and would be an equally bad thing to present as real results.
3. Do not claim ZUPT is "tuned" or "working" — it has two known failure modes on the only test available and has never touched real accelerometer/gyro noise.
4. Do not claim the LightGBM residual model is "trained" in any deployable sense — no saved model file exists.
5. If a judge asks to see the live dashboard/trajectory, be ready for the honest answer that baseline/corrected/uncertainty are current placeholders wired for the Android client's contract, not the physics/ML pipeline output yet — the code already tells you to say exactly this, almost verbatim, in its own docstrings.

**What would embarrass you fastest under a pointed follow-up:** a technically literate ISRO judge asking "can I see one real recorded trace and its sanity-check report" or "show me the model file and how many traces it trained on" would immediately surface that neither exists. Given the dossier's Section 5.3 and jury Q&A (Q5/Q6) are written entirely around "we're honest about what's simulated," the actual gap right now is upstream of that promise — there's no real data yet to be honest *about*. The engineering scaffolding is unusually solid for a pre-data-collection hackathon codebase (the self-documented caveats throughout are a real asset — most teams don't write "these thresholds are unvalidated" into their own source), but as of today this is a well-designed pipeline that has never been run on anything real, sitting in front of a backend that everyone already knows is mocked. The honest framing for where you are: **architecture and code quality — strong and ahead of schedule; data and trained model — not started.**
