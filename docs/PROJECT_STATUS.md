# SIH26168 — Project Status

**Written:** 2026-09-05. Replaces the original technical dossier, which was deleted from `docs/` before this document existed. Everything below was independently re-verified against the current state of the repo — commands and output are shown, not just claimed. Where something couldn't be verified, it's marked **UNVERIFIED** rather than assumed.

**Root:** `D:\Projects\SIH_2026\` (note: this is the real path — no nested `Projects\SIH_2026\Projects\SIH_2026\` folder exists, despite earlier references to one).

---

## 1. Architecture — what this project actually is

**Primary system: `gudumap/`** — an Android Studio app (Kotlin, Jetpack Compose, min SDK 24 / target SDK 37) doing on-device AI/ML dead reckoning to bridge GNSS blackouts (tunnels, underground parking, urban canyons) using only phone-grade accelerometer/gyroscope.

Live wiring, traced import-by-import:
```
MainActivity → ui/screens/NavigationScreen.kt → viewmodel/NavigationViewModel.kt
  → navigation/NavigationEngine.kt
    → navigation/DeadReckoningEngine.kt (EKF, ZuptDetector, NHC, IMUBuffer, CoordinateTransformer,
        ml/ModelRunner, ml/InputNormalizer, ml/ModelMetadata, sensor/DiagnosticRecorder,
        tracking/TrajectoryIntegrator)
    → sensors/{SensorManager, SensorFusionManager, LocationManager}
    → map/{MapMatcher, OfflineMapManager}
```

Two ONNX models ship in `app/src/main/assets/models/`:
- `gru_local.onnx` — trained on OxIOD (per its metadata)
- `gru_io_vnbd.onnx` — trained on IO-VNBD (the dataset ISRO linked on the official SIH portal)

**⚠️ Verified fact, not previously documented:** only `gru_io_vnbd.onnx` is actually wired into the running app. `ml/ModelRunner.kt` hardcodes `ModelMetadata.MODEL_FILE_NAME = "gru_io_vnbd.onnx"` as its load target, and the shared `ModelMetadata` object hardcodes a single global window contract (`WINDOW_SIZE = 20`, 10 Hz) that matches only `gru_io_vnbd`. A repo-wide grep for `gru_local` finds it **only** in the metadata JSON's own documentation field — zero references anywhere in Kotlin source. `gru_local.onnx` sits in `assets/` completely unused by the live app.

**There is no backend in the shipped product.** Everything — ML inference, EKF fusion, ZUPT, map matching — runs on-device inside `gudumap/`. This is intentional: the entire point of the system is working through a GNSS blackout with no connectivity, so a network service would be irrelevant to the demo and the product.

**Secondary/support system: `SIH26168-DeadReckoning/`** — an earlier Python/FastAPI prototype (physics `strapdown.py` baseline + LightGBM residual correction design), **deprecated as of 2026-09-05** (see §6). A prior audit (`SIH26168-DeadReckoning/evaluation/AUDIT_Phase1-4_Report.md`, re-read and spot-checked here) found: zero real data ever collected, no LightGBM model ever trained on real traces, ZUPT thresholds unvalidated in both directions tried, and the FastAPI backend fully mocked (self-labeled `"source":"mock"` on every response). That audit's own claim that `evaluation/metrics.py` didn't exist yet is now **stale** — the file exists and is real (see §3). Nothing else in that audit's findings was contradicted by this pass. This project is not what ships; its ATE/RTE/CEP/drift-rate evaluation code (`evaluation/metrics.py`) is the one part still actively kept, reused for scoring the Android models honestly.

A fourth folder mentioned in earlier notes, `Mobile_app/`, does not exist in the current tree — consistent with it having been deleted and superseded by `gudumap/`.

---

## 2. Task 1 — dead code consolidation (DONE)

Traced every import in the live path above, including same-package references that don't need an `import` statement in Kotlin. Two of the three pairs you originally suspected as duplicates were investigated and turned out **not** to be duplicates at all; five additional dead files were found that weren't in the original suspect list.

### Resolved

| Suspected pair | Verdict |
|---|---|
| `navigation/MapMatcher.kt` vs `map/MapMatcher.kt` | **Not a duplicate.** `navigation/MapMatcher.kt` is an interface (+ `PassThroughMapMatcher`, `OsmRoadNetworkMapMatcher` adapter) consumed by `DeadReckoningEngine`. `map/MapMatcher.kt` is the concrete OSM-backed implementation, wrapped by the adapter in `NavigationEngine`. **Both live, both kept.** |
| `sensor/SensorManager.kt` vs `sensors/SensorManager.kt` | `sensors/` (plural) is live, imported by `NavigationEngine.kt`. `sensor/` (singular) had zero references anywhere. **Deleted.** |
| `sensors/DeadReckoningEngine.kt` vs `navigation/DeadReckoningEngine.kt` | `navigation/` is live. `sensors/` was dead in production but directly instantiated in `DeadReckoningEngineIntegrationTest.kt::testBackwardCompatibilityAdapter`. **File deleted; that one test method deleted with it** (the test file's other 3 tests, which cover the live engine, were kept). |

### Additional dead code found while tracing (not in the original suspect list)

| File | Why dead | Disposition |
|---|---|---|
| `ml/MLModelManager.kt` | Wraps `MLInputProcessor`/`MLOutputProcessor`; not used by the live `DeadReckoningEngine` (which uses `ml/InputNormalizer` + `ml/ModelRunner` instead); had no test at all | Deleted |
| `navigation/ExtendedKalmanFilter.kt` | Live engine uses `EKF`, not this; only referenced by its own test | Deleted, with `ExtendedKalmanFilterTest.kt` |
| `sensors/MotionDetector.kt` | Live engine uses `ZuptDetector` for motion state, not this; only referenced by its own test | Deleted, with `MotionDetectorTest.kt` |
| `ml/MLInputProcessor.kt` | Not used by live engine; only referenced by its own test | Deleted, with `MLInputProcessorTest.kt` |
| `ml/MLOutputProcessor.kt` | Not used by live engine; only referenced by its own test | Deleted, with `MLOutputProcessorTest.kt` |

**Total: 7 dead source files removed, 4 test files removed (each verified to test *only* the dead class before deletion), 1 test method surgically removed from an otherwise-live test file.**

**Post-deletion check:** manually traced every `import com.example.gudumap.*` across all remaining 51 `.kt` files — every import resolves to a file that still exists. No dangling references. (Gradle itself was not run — no network access to Google's Maven repo from this environment — so this is a manual static check, not a compiler-verified one.)

---

## 3. Task 2 — real accuracy numbers (BLOCKED — no real data present)

### Verified: no training data or GRU training scripts exist locally

Searched the entire `D:\Projects\SIH_2026\` tree:
```
$ find . -iname "*oxiod*" -o -iname "*io_vnbd*" -o -iname "*.pt" -o -iname "*.pth" -o -iname "*.ckpt" -o -iname "train*.py"
→ SIH26168-DeadReckoning/ml-residual/train.py   (trains LightGBM residuals — unrelated to the GRU models)
```
- `SIH26168-DeadReckoning/data/public_datasets/README.md`'s own checklist: OxIOD/RoNIN/TLIO/RIDI/GSDC all unchecked, nothing downloaded.
- `SIH26168-DeadReckoning/data/raw_traces/` contains only its README — no self-collected traces.
- **No training script, notebook, or checkpoint for `gru_local` or `gru_io_vnbd` exists anywhere in this tree.** Only the deployed `.onnx` files and a metadata JSON survive.

**Action needed from you before real numbers can exist:**
- OxIOD: `http://deepio.cs.ox.ac.uk/` → "Dataset (1.01G)" link → Google Form (name/email/intended use) → download link emailed → unzip into `SIH26168-DeadReckoning/data/public_datasets/oxiod/`.
- IO-VNBD: `https://github.com/onyekpeu/IO-VNBD` → direct clone/download, no form.

Until either exists locally, any ATE/RTE/CEP/drift-rate number for these two models would be fabricated. None is reported here.

### Verified: `evaluation/metrics.py`'s formulas are correct and genuinely tested

Checked against standard inertial-odometry definitions (not trusted just because the code exists):

| Metric | Formula in `metrics.py` | Standard? |
|---|---|---|
| ATE | RMS of per-sample Euclidean position error | ✓ (no trajectory alignment step, but valid here since both tracks share the same anchor/frame) |
| CEP-50 / CEP-90 | 50th / 90th percentile of position error | ✓ |
| RTE | RMS, over fixed windows, of (estimate displacement − truth displacement) within each window | ✓ matches TUM RGB-D / KITTI-style relative pose error |
| drift-rate % | final position error ÷ distance travelled × 100 | ✓ |

Ran its test suite directly:
```
$ python -m pytest SIH26168-DeadReckoning/evaluation/tests -v
6 passed in 0.19s
```
All 6 are synthetic known-answer cases (perfect match → zero error, constant offset → exact ATE/CEP, etc.). This module is trustworthy and ready to score real trajectories once real data exists. (Note: a prior audit of this repo claimed `metrics.py` didn't exist yet — that claim is now stale; the file was added after that audit ran.)

### Found and corrected: a windowing mismatch that would have silently corrupted results

The two ONNX models do **not** share the same input contract. Verified directly against the ONNX graphs (`onnxruntime.InferenceSession(...).get_inputs()`), not documentation:

| Model | Actual input shape |
|---|---|
| `gru_local.onnx` | `[batch, 200, 6]` (200 samples @ 100 Hz — matches its own `model_metadata.json`) |
| `gru_io_vnbd.onnx` | `[batch, 20, 6]` (20 samples @ 10 Hz) |

`gru_local.onnx` is a **200-sample** window, not 20. Feeding it a 20×6 window (the shape the app's shared `ModelMetadata` object hardcodes globally) would run without erroring and produce meaningless output. Any future evaluation script must use each model's own real input contract, not a single shared constant.

### UNVERIFIED: train/val/test split integrity, leakage risk

`model_metadata.json` records `best_epoch: 19`, `total_epochs: 20`, `best_val_loss_normalized: 0.084` — implying a train/val split existed. There is no evidence anywhere in this tree of the training script or split methodology that produced these two `.onnx` files. **Confidence on leakage: low — cannot confirm or rule out.** The artifact that would answer this (the training code) is not present locally. Do not present `best_val_loss_normalized: 0.084` as an accuracy figure to a judge — it is a normalized loss, not an interpretable distance error, and it says nothing about held-out generalization without knowing the split.

### Comparison to published benchmarks — not yet possible

TLIO/RoNIN/OxIOD publish real ATE/drift figures (e.g. TLIO: <3 m error for 90% of a 3–7 min trial). No comparison is made here because there is currently no real evaluation number on this project's side to compare.

---

## 4. What you should NOT claim yet

1. Real accuracy in meters for either GRU model — no held-out test evaluation has been run.
2. "Two fused ONNX models" as a production claim — only `gru_io_vnbd.onnx` is wired into the live app; `gru_local.onnx` is an unused bundled asset.
3. `best_val_loss_normalized: 0.084` as an accuracy number — it's a normalized training-loss value, not ATE/RTE/CEP/drift-rate.
4. Anything about the Python/`SIH26168-DeadReckoning` side being trained or non-mocked — per the existing audit, re-confirmed here: zero real data, backend mock-labeled at every layer.

## 5. Next steps (superseded in part by §6 below — gru_local's fate is now decided)

1. You download OxIOD and/or IO-VNBD (instructions above).
2. Confirm the official train/test split each dataset publishes (do not invent a random split).
3. Write and run the ONNX evaluation script per each model's **real** input contract (200×6 for `gru_local`, 20×6 for `gru_io_vnbd` — not a shared constant).
4. Score against ground truth with the now-verified `metrics.py`.

---

## 6. 2026-09-05 (same day, continued session) — Tasks A–D

### Task A — re-checked dataset download status: still BLOCKED

Re-ran the exact same search §3 documents:
```
$ find . -iname "*oxiod*" -o -iname "*io_vnbd*" -o -iname "*.pt" -o -iname "*.ckpt" -o -iname "train*.py"
→ same single hit as before: SIH26168-DeadReckoning/ml-residual/train.py (unrelated LightGBM script)
$ find data/public_datasets, data/raw_traces → still only READMEs
```
No change. Task 2b (real ONNX evaluation) remains blocked on you downloading OxIOD and/or IO-VNBD. Not re-blocking the rest of this session's work on it, per instruction — see Tasks B–D below.

### Task B — gru_local.onnx: cut from shipped assets (default call executed)

Re-verified independently (second pass, since this changes what ships): `grep -rn "gru_local" gudumap/ --include="*.kt" --include="*.kts" --include="*.xml" --include="*.pro" --include="*.json"` found it only in metadata JSON text — zero Kotlin references, confirming the earlier finding still holds.

While verifying, found the removal scope was **larger than just `gru_local.onnx`**: `app/src/main/assets/models/` was a byte-identical duplicate of the top-level asset files (confirmed via matching file sizes) plus `gru_local.onnx`, and **none of it was ever read by the app** — `ModelRunner.kt`/`ModelMetadata.kt` call `context.assets.open(...)` with bare filenames (`"gru_io_vnbd.onnx"`, `"io_vnbd_normalization.json"`), which Android's `AssetManager` resolves against the assets root, never `models/`. The top-level `model_metadata.json` and `normalization.json` were also unread (no code ever opens `METADATA_FILE_NAME` or bare `"normalization.json"` — verified by grepping every `assets.open(` call site in `main/`). `model_metadata.json` additionally only documented `gru_local`'s OxIOD training details, so keeping it after cutting `gru_local` would have been actively misleading.

**Removed:** the entire `assets/models/` subfolder (`gru_io_vnbd.onnx` dup, `gru_local.onnx`, `io_vnbd_normalization.json` dup, `model_metadata.json` dup, `normalization.json` dup) + top-level `assets/model_metadata.json` + top-level `assets/normalization.json`. **Kept:** `assets/gru_io_vnbd.onnx` and `assets/io_vnbd_normalization.json` — the only two files the app actually reads — plus `assets/maps/`. Post-removal grep for `"models/`, `gru_local`, or any reference to the deleted files: zero hits anywhere in `app/src`.

`gru_local.onnx` existed, was trained on OxIOD, and was cut for scope ahead of the 20 Sept deadline rather than spending remaining time building a two-model ensemble that was never wired in to begin with. It was not hidden — it's documented here as a deliberate descope.

### Task C — backend deprecated, not deleted

Sanity-checked the default before executing: there's no live-backend use case for a product whose entire value proposition is working through a GNSS blackout with no connectivity (a fleet-tracking pivot would be a different product, out of scope with 2 weeks left). Proceeded with the default.

- Added a deprecation notice to the top of `SIH26168-DeadReckoning/backend/main.py`'s module docstring and to the top of `backend/README.md` — both state plainly: not part of the shipped product, kept only because `evaluation/metrics.py` is still used, real system is the on-device Android pipeline.
- `backend/` folder itself was **not** deleted — its evaluation code is genuinely still in use (see §3).
- §1 (Architecture) above updated to state plainly: no backend in the shipped product, everything runs on-device.

### Task D — dashboard gap analysis + implementation

**What existed before this session, verified by reading the actual Compose/Kotlin code (not assumed):**
- `NavigationScreen.kt` showed the current corrected position (lat/lon numbers), a single vehicle marker on the map (`MapView.kt`), and — only while the phone's real GPS continues running in the background during a *simulated* blackout — a "Max Error"/"DR Distance" numeric comparison against that live GNSS ground truth (explicitly labeled "Evaluation only - NOT used for navigation").
- `EKF.kt` already maintained a real 6×6 covariance matrix `P` with a proper Joseph-form update (`navigation/EKF.kt:42,189-214`) — genuine uncertainty growth was being computed internally the whole time.
- **Missing, confirmed by grep and by reading `NavigationEngineState`/`NavigationState`:** no naive/uncorrected trajectory existed anywhere (only one fused `TrajectoryIntegrator` path); the EKF's own covariance was never read by anything outside `EKF.kt` itself — not exposed to `NavigationEngineState`, `NavigationState`, or the UI; the map rendered a single point marker with no trail for either path and no uncertainty visualization at all.
- Distinction that matters: the existing "Max Error" numeric readout is an *evaluation crutch* that only works because the demo device still has real GPS reception during a "simulated" blackout — it would show nothing in an actual tunnel. A genuine uncertainty estimate has to come from the EKF's own covariance, independent of secretly-available ground truth. That's what was missing and is the core gap this task closes.

**Implemented:**
1. `navigation/NaiveIntegrator.kt` (new) — pure double-integration of world-frame accelerometer samples, deliberately with no ZUPT/ML/EKF, to show what raw phone-grade dead reckoning looks like on its own.
2. Wired into `DeadReckoningEngine.kt`: fed the same rotated accelerometer samples the corrected pipeline already computes (`addSensorSample`), reset alongside the other integrators on `initialize()`/`reset()`, and exposed as `naiveLatitude`/`naiveLongitude` on `NavigationEngineState`.
3. Added `uncertaintyRadiusMeters` to `NavigationEngineState`, computed as `sqrt(P[0][0] + P[1][1])` from the EKF's own covariance in `getState()` — a real 1-sigma circular position uncertainty, not derived from ground truth, that grows on its own during blackout since no GNSS position update shrinks it.
4. Threaded both new fields through `NavigationEngine.kt` → `NavigationState.kt` (UI-facing state).
5. `MapView.kt`: added a blue trail polyline for the corrected path (there was no trail at all before — just a point marker) and a red trail polyline for the naive path, both capped at 2000 points; both trails reset automatically when a new blackout starts. Added a translucent circle overlay (`org.osmdroid.views.overlay.Polygon.pointsAsCircle`) around the current position sized to `uncertaintyRadiusMeters`, shown only during an active blackout.
6. `NavigationScreen.kt`: wired the new state fields into `MapView(...)`, and added a "Confidence Radius" metric tile (±meters) next to the existing "DR Distance"/"Max Error"/"ML Latency" tiles — a number that works without secret GNSS ground truth, unlike the existing ones.
7. Verified no dangling references: manually re-traced every `import com.example.gudumap.*` across all 52 `.kt` files (51 + the new `NaiveIntegrator.kt`) post-change — all resolve. Not compiler-verified (still no Gradle/network access here) — Android Studio should be used to do a real build check before the next demo run.

**Not done / explicitly out of scope for this task:** did not touch anything related to `gru_local.onnx` (per instruction, handled separately in Task B). Did not attempt to run the app (no emulator/device available in this environment) — the visual result (colors, polyline behavior, circle sizing) has not been eyeballed on an actual screen and should be checked in Android Studio before relying on it for a demo.

**Aside, found while reading this code, not acted on:** `ui/components/MetricCard.kt` and `ui/components/StatusCard.kt` appear to be dead — `NavigationScreen.kt` defines and uses its own private `MetricTile`/`StatusRow` composables instead of calling either of these. This wasn't caught in Task 1's trace (which focused on navigation/sensor/ml, not ui/components) and wasn't part of Task D's ask, so it's flagged here rather than acted on — worth a follow-up cleanup pass if you want it.

---

## 7. 2026-09-06 — Audit of `dead_reckoning/` (teammate-provided folder): §3's "no training data" finding is SUPERSEDED

A new folder, `dead_reckoning/`, arrived from the teammate. §3 above stated no training script or checkpoint for either shipped model existed anywhere in `D:\Projects\SIH_2026\` — that was true *at the time*, for what existed in this tree. `dead_reckoning/` is new evidence that changes the picture. Audited with the same skepticism as the original Phase 1–4 audit of the Python prototype — verified claims independently rather than trusting the folder's own (extensive) self-documentation.

### Task 1 — Provenance: is this really the source of the shipped models?

**Yes, confirmed for `gru_io_vnbd.onnx` — exact hash match, not just same filename:**
```
$ md5sum dead_reckoning/models/gru_io_vnbd.onnx gudumap/app/src/main/assets/gru_io_vnbd.onnx
0ed1af0362cd10154c2547094a58812a  dead_reckoning/models/gru_io_vnbd.onnx
0ed1af0362cd10154c2547094a58812a  gudumap/app/src/main/assets/gru_io_vnbd.onnx
```
Also matches `dead_reckoning/models/frozen_io_vnbd/gru_io_vnbd.onnx` (a separate "frozen" backup copy — see below). This is the actual bit-for-bit file shipped in the app, not a lookalike.

`gru_local.onnx` is also present (`dead_reckoning/models/gru_local.onnx`, hash `6aa416e7b318f8e2f7f7e7acd4248ee0`) but **cannot be hash-compared against the shipped copy** — that copy was deleted from `gudumap/assets/` in §6 Task B, before this folder arrived. Given the exact match on the sibling model and everything else below, there's no reason to doubt this is the same file, but it's not a verified bitwise match like the io_vnbd one.

`README.md` at the repo root is **empty** — a real gap, flagged rather than glossed over. The actual documentation lives in `docs/` instead (7 substantive markdown files: `ML_VALIDATION_REPORT.md`, `ML_FINAL_CONTRACT.md`, `ML_DEPLOYMENT_CONTRACT.md`, `ML_FREEZE_CHECKLIST.md`, plus two `NAVIGATION_ENGINE_*` docs). `ML_VALIDATION_REPORT.md` describes training and evaluating `gru_local` (OxIOD) and `gru_io_vnbd` (IO-VNBD) specifically — not something else.

**`scripts/`/`src/` is genuinely runnable training code, not scaffolding** (a real, meaningful contrast to how the earlier `SIH26168-DeadReckoning` Python prototype turned out):
- `src/models/gru_model.py` defines `GRUDeadReckoning`: 2-layer GRU, hidden=64, dropout=0.2, input=6, output=3 — matches the documented architecture exactly, and matches what's already reverse-engineered into `gudumap`'s own `ModelMetadata.kt`.
- `src/training/train_io_vnbd.py` (211 lines), `src/evaluation/run_all_test_sequences_benchmark.py` (652 lines, implements the 7-baseline "B1 Pure INS" .. "B7 ML+INS+EKF+NHC+ZUPT" ladder the report describes) — both substantial, specific, not stub files.
- `src/navigation/{ekf.py, nhc.py, zupt.py, coordinate_frames.py}` — a Python reference implementation of the same EKF/NHC/ZUPT architecture that's in the Kotlin app, which is good corroborating structure (the Android port has a real reference to have been ported from).
- **`__pycache__/*.pyc` files exist throughout `src/`** — this code has actually been *executed* on this machine, not merely authored and left unrun.

### Task 2 — Auditing `results/` before trusting any number in it

`results/io_vnbd/` (37 files: CSVs, JSONs, PNGs, and its own markdown audit docs — `final_leakage_audit.md`, `ground_truth_definition.md`, `sensor_schema.md`, `FINAL_AUDIT_SUMMARY.md`) is where the real numbers live.

**Split methodology, checked, not assumed:**
- `results/io_vnbd/split_manifest.csv` assigns each of 72 sequences to exactly one of train (53) / val (10) / test (9), keyed by whole physical driving sequence (not by window) — real per-sequence data, not a placeholder.
- `tests/test_native_io_vnbd.py::test_05_train_val_test_leakage_absence` asserts the three split sets are pairwise disjoint *and* asserts the exact counts (53/10/9) — a real, specific assertion, not a vacuous stub.
- `tests/test_native_io_vnbd.py::test_06_normalization_leakage_check` recomputes mean/std directly from `io_vnbd_train_local.npz` and asserts it matches `models/io_vnbd_normalization.json` to `1e-4` — i.e., normalization stats are independently re-derivable from the train split alone, not fabricated or leaked from test data.
- `final_leakage_audit.md` walks through 7 specific leakage criteria with cited source lines. Six pass. **The 7th is disclosed as a real, material limitation, not swept under the rug**: the benchmark evaluates displacement dead reckoning *under reference heading* — it uses the vehicle's own CAN-bus/dual-antenna heading (`gt_hdg`) to rotate body-frame quantities into the navigation frame, not an open-loop integrated heading. An audit that reports one real weakness alongside six passes is a stronger signal of genuine rigor than an audit that reports zero problems.

**Independent spot-check I ran myself** (not just reading their CSV): loaded a real IO-VNBD smartphone CSV (`S-M.csv`, Driver B, 105,974 rows, confirmed genuine AndroSensor-format columns — `GYROSCOPE Pitch/Yaw/Roll`, `ACCELEROMETER X/Y/Z`, `GRAVITY X/Y/Z`, matching the report's documented schema exactly), built a real 20×6 window using the documented feature contract (gravity-subtracted body acceleration ÷ 9.80665, gyro pitch→x/roll→y/yaw→z remap), applied `io_vnbd_normalization.json`, and ran it through the actual shipped `gru_io_vnbd.onnx` via `onnxruntime` myself:
```
raw normalized output: [-0.923, 3.069, 0.253]
denormalized displacement [dx,dy,dz] meters: [10.82, 4.38, 0.25]
```
Finite, physically-scaled (meters, not NaN or absurd), on real sensor data I extracted and windowed independently of their pipeline. This confirms the ONNX file is a genuine functioning trained model responding sensibly to the documented contract — not a corrupted file or random weights.

**What I could NOT independently reproduce, and why:** I attempted a second check — comparing the model's predicted displacement against real GPS-derived ground truth for a "fast" window — and hit a genuine complication: the phone's own raw GPS (`S-M.csv`) showed a single-fix jump of ~198m in 2 seconds while its own reported speed was only 22 km/h (~12m expected) — a classic GPS multipath/reacquisition glitch, not real vehicle motion. Reading `src/preprocessing/io_vnbd_loader.py` confirmed the pipeline is aware of this: it deliberately sources `gt_lat`/`gt_lon` from the **vehicle's own CAN-bus/ECU file** (`V-*.csv`), not the phone's noisier GPS — the right design choice, and the reason my own crude proxy didn't line up. Reproducing their exact 11.28m mean-error figure end-to-end would require pulling in the vehicle file and their full alignment code, which I did not do. **The headline number (11.28m mean Euclidean error / 54.45% reduction vs. OxIOD, on 27,964 held-out test windows) is well-documented and plausible given everything else checked out, but was not independently re-derived by me from raw data end-to-end — treat it as "audited and spot-checked," not "independently reproduced."**

**Hygiene issue found, not fabrication:** `results/synthetic/` (16 files) is a byte-identical subset of an *earlier* snapshot of `results/io_vnbd/` (37 files) — every file present in both is identical, and `io_vnbd/` simply has 21 additional `real_*`-prefixed files that `synthetic/` lacks. Despite the name, `results/synthetic/` does not appear to hold independently-computed synthetic-fixture output — it looks like a stale copy left over from before the real benchmark files were added, never cleaned up or renamed. Worth asking the teammate to confirm and delete if so; it isn't evidence of fabricated numbers (the real numbers in `io_vnbd/` are the ones addressed above), just a misleading folder name sitting around.

### Task 3 — OxIOD zip and `data/`

- Zip listed without full extraction: `Oxford Inertial Odometry Dataset_2.0.zip`, 937 entries — matches OxIOD's published structure exactly (`large scale/floor4/tango/*.csv`, `handheld/`, `handbag/`, `pocket/`, `running/`, `slow walking/`, `multi devices/`, `multi users/`), plus `__MACOSX/` junk entries consistent with a genuine macOS-sourced download (not fabricated).
- **The zip has already been extracted** — `data/raw/Oxford Inertial Odometry Dataset_2.0/` exists on disk, 2.7GB, matching the same folder structure. Nothing further needs extracting for OxIOD.
- **IO-VNBD raw data is genuinely present**: `data/raw/IO-VNBD/`, 825MB, 360 files, real per-driver/route folder structure (`M (Driver B)`, `S (Driver A)/S1..S4`, `Vf (Driver E)/V-Vfa01..02`, `Vta (Driver E)/Vta01a..`) with synchronized `S-*.csv`/`V-*.csv` pairs — confirmed by directly opening and reading one, not just listing filenames.
- `data/processed/` has the actual `.npz` split files (`io_vnbd_{train,val,test}_local.npz`, `handheld_*.npz`) the training/eval scripts reference.

**This directly closes the gap §3 flagged: real OxIOD and real IO-VNBD data now exist locally, in this new folder, already extracted and already used.**

### Task 4 — `FE/` — confirmed to be "Frontend," and it's stale

`FE/gudumap.zip` + `FE/gudumap/gudumap/` (an extracted Android Studio project) — this is a snapshot of the `gudumap` frontend the teammate bundled alongside the ML work, not a mysterious unrelated folder.

**Diffed against the current live `gudumap/`: this snapshot substantially predates it**, and predates *both* of this session's and the previous session's cleanup:
- Missing entirely: `viewmodel/`, `map/` package, `NavigationEngine.kt`, `NavigationState.kt`, `NaiveIntegrator.kt`, most of the current test suite (`DeadReckoningEngineTest.kt`, `GnssBlackoutWorkflowTest.kt`, `MapMatcherTest.kt`, `NavigationViewModelTest.kt`, `OfflineMapManagerTest.kt`, `PipelineDiagnosticTest.kt`, and more).
- Still contains the dead code §2 removed: `sensor/SensorManager.kt`, `sensors/DeadReckoningEngine.kt`.
- Still contains the assets §6 Task B removed: `gru_local.onnx`, `model_metadata.json`, `normalization.json` sitting directly in `assets/`.

**Conclusion: `FE/` is an old reference copy, not the current or authoritative frontend.** It should not be used as a basis for anything going forward — the live `gudumap/` at the project root is still the one and only frontend.

### Final verdict

**`dead_reckoning/` is a real, substantially-executed training pipeline with real results — a genuine, meaningful step up from the earlier `SIH26168-DeadReckoning` Python prototype, which was scaffolding with zero real data.** Evidence: exact hash match on the shipped `gru_io_vnbd.onnx`; both OxIOD (2.7GB) and IO-VNBD (825MB) genuinely present and already extracted, matching their known published structures; real, substantial, previously-executed training/evaluation code; a leakage audit that discloses one genuine limitation instead of claiming perfection; and my own independent onnxruntime spot-check producing sane, physically-plausible output from real sensor data.

**Caveats to carry forward, not overclaim past:**
1. The empty root `README.md` should be filled in (the content clearly exists in `docs/` — it just isn't surfaced at the entry point).
2. `results/synthetic/` looks like a stale duplicate folder and should be clarified/cleaned up with the teammate.
3. The reported 54.45% error reduction / 11.28m mean error is well-documented and passed my spot-check, but was not independently reproduced end-to-end by me — present it as "audited, real, held-out" rather than claiming a from-scratch third-party reproduction.
4. **The heading-reference dependency is real and should not be dropped when presenting these numbers**: the benchmark assumes access to accurate vehicle heading during the outage (from CAN bus/dual-antenna GPS), not open-loop smartphone gyroscope integration. Unassisted deployment — a phone with no heading reference, exactly gudumap's actual deployment scenario — would likely see heading drift degrade both the displacement rotation and the NHC constraint over 60–120s outages. This is the single most important thing to disclose alongside the headline numbers, and the source report already says so explicitly.
5. Edge integration is still pending: `gudumap`'s current input contract is `[1, 20, 6]` for `gru_io_vnbd` (matches), but the report itself notes the Android side hasn't yet been updated to reflect anything from this newer pipeline beyond the model files already shipped — treat `dead_reckoning/`'s docs about Android integration as aspirational/planned, not already done.

---

## 8. 2026-09-06 (continued) — Heading-reference dependency: how bad is the gap, really?

§7's final caveat needed a real answer, not a guess: the 11.28m/54.45% figure was computed using the vehicle's own CAN-bus/dual-antenna heading during the blackout window. `gudumap` has no CAN-bus. What does it actually use?

### Task 1 — Verified facts, with file/line citations

**`navigation/EKF.kt` has no heading state at all.** The state vector is `[p_N, p_E, p_D, v_N, v_E, v_D]` — six elements, all position/velocity (`EKF.kt:8-14`). The class docstring explicitly discloses this: *"Does NOT explicitly estimate accelerometer biases, gyroscope biases, or attitude errors"* (`EKF.kt:16-19`). There is no magnetometer measurement update anywhere in the EKF — heading is never touched by any `update*` method. This exactly mirrors the Python reference (`dead_reckoning/src/navigation/ekf.py:1-23`, same disclosed limitation, same 6-state design, same absence of a heading state) — **the EKF port is faithful, not diverged.**

**`navigation/NHC.kt` does not correct heading — confirmed, not assumed.** `applyConstraint()` (`NHC.kt:29-61`) takes `headingDeg` as an **input** parameter and uses it only to build the measurement matrix `H` that constrains lateral/vertical *velocity* to zero (`H` rows touch only `v_N`/`v_E`/`v_D`, `NHC.kt:46-51`). Heading itself is never in `z`, `H`, or the updated state — NHC assumes heading is already correct and uses it as a fixed rotation angle. Again, this is an exact match to the Python reference (`dead_reckoning/src/navigation/ekf.py:172-191`, identical `H` matrix structure: `H[0,3]=-sin(ψ), H[0,4]=cos(ψ)` for lateral, `H[1,5]=1` for vertical).

**`navigation/ZuptDetector.kt` confirmed velocity-only, as suspected.** `update()` only classifies motion state (`STATIONARY`/`ROTATING_IN_PLACE`/`MOVING`) from accelerometer/gyroscope magnitude+variance (`ZuptDetector.kt:62-124`); `EKF.updateZupt()` only ever constrains `v_N`/`v_E`/`v_D` to zero. No heading coupling anywhere.

**So where does heading actually come from? `sensors/SensorFusionManager.kt` + Android's `TYPE_ROTATION_VECTOR` sensor — and this is the one place the two systems genuinely diverge, not in the filter math but in the heading *input source*.**

- `sensors/SensorManager.kt:92-93` registers `sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)` — **not** `TYPE_GAME_ROTATION_VECTOR`. This distinction matters: per Android's own sensor contract, `TYPE_ROTATION_VECTOR` is defined as fusing accelerometer + gyroscope + **magnetometer**, while `TYPE_GAME_ROTATION_VECTOR` deliberately excludes the magnetometer (gyro+accel only, drift-free of magnetic interference but with no absolute heading reference). gudumap uses the magnetometer-inclusive one.
- `NavigationEngine.kt:91-93` and `:119-121` confirm both are wired up: `startRotationVector` feeds `sensorFusionManager.updateRotationVector(...)`, and `startMagnetometer` feeds `sensorFusionManager.updateMagnetometer(...)`.
- Inside `SensorFusionManager.kt`, there are genuinely **two** heading paths: a manual complementary filter that explicitly fuses gyro-integrated azimuth with magnetometer+accelerometer-derived orientation (`updateGyroscope()`, `SensorFusionManager.kt:68-106`, complementary filter at line 100) — **but this path only runs `if (!hasHardwareRotation)` (line 84)**. Since `updateRotationVector()` sets `hasHardwareRotation = true` on the very first rotation-vector sample (`SensorFusionManager.kt:111-120`) and rotation-vector data arrives continuously from app start, **the manual complementary-filter path is dead in practice on any real device that has this sensor** — which is effectively all Android phones. The heading gudumap actually uses is whatever Android's own `TYPE_ROTATION_VECTOR` fusion produces, taken as-is (`SensorFusionManager.kt:111-120`, `getOrientation()` at line 157-166).
- **Verified separately: sensor accuracy is read from Android but silently discarded.** `SensorManager.kt:301-303`: `override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { // Not used }`. Android reports a real accuracy/reliability level for the magnetometer and rotation-vector sensor (this is exactly the signal that would tell you when magnetic interference is degrading heading quality) — gudumap receives this callback and throws it away.
- **Crucially, none of this is disabled during GNSS blackout.** The rotation-vector sensor keeps running regardless of GNSS state — heading isn't cut off when GNSS is, it's just never independently corrected by anything GNSS-related either way (heading was never wired to GNSS in the first place).

### Task 2 — Honest assessment

**Classification: (c), closer to (a) than (b), but with a real caveat that matters specifically for this project's deployment scenario.** Heading is not open-loop, uncorrected gyro integration — `TYPE_ROTATION_VECTOR` does fuse the magnetometer continuously, independent of GNSS state, so it is not the unbounded-drift worst case. But magnetometer-based heading correction is well known to degrade under magnetic interference, and the two places this system is deployed (**inside a moving vehicle's steel chassis**, and **inside tunnels/underground parking with dense rebar/steel**) are close to a worst case for magnetometer reliability — and gudumap currently has no way to detect or respond to that degradation (the accuracy callback that would tell it is discarded, per above). So the honest answer is: heading correction exists and is probably *usually* decent, but its reliability specifically *during* a blackout is genuinely uncertain, in an environment somewhat adversarial to the sensor it depends on, with no fallback or even detection wired up.

**Back-of-envelope estimate — explicitly NOT equivalent in rigor to the 11.28m benchmark figure, a rough order-of-magnitude illustration only:**

Using the report's own mean vehicle speed (45.08 km/h = 12.52 m/s) and three illustrative MEMS-gyro heading-drift-rate scenarios spanning published smartphone-grade ranges (well short of tactical-grade INS, which is 100-1000x better):

| Scenario | Drift rate | Peak heading error @ 60s | Est. lateral position error @ 60s | Peak heading error @ 120s | Est. lateral position error @ 120s |
|---|---:|---:|---:|---:|---:|
| LOW (well-calibrated) | ~1°/min | 1.0° | **~6.6 m** | 2.0° | **~26.2 m** |
| MEDIUM | ~3°/min | 3.0° | **~19.7 m** | 6.0° | **~78.6 m** |
| HIGH (poorly-calibrated / interfered) | ~10°/min | 10.0° | **~65.5 m** | 20.0° | **~260.9 m** |

(Method: assumes linear heading-error growth from an uncorrected drift rate — a pessimistic upper bound, since it ignores whatever partial correction the magnetometer is actually still providing — then computes cross-track lateral displacement as `distance_travelled × sin(average_heading_error_over_the_interval)`. This is illustrative arithmetic, not a simulation of gudumap's actual filter.)

**Why this matters, concretely:** the report's own 120s moving-sequence median RMSE (with full heading reference) is ~250m (`ML+INS`: 252.10m median, `ML Only`: 251.22m median, from §7 / `real_benchmark_aggregate_moving.csv`). Even the LOW scenario above (~26m) is a non-trivial fraction of that budget on its own; the MEDIUM scenario (~79m) is nearly a third of it; the HIGH scenario (~261m) would roughly **double** the reported error if it stacked linearly with the existing error sources (it wouldn't stack quite that cleanly in practice, since heading error also feeds back into the NHC constraint and ML rotation in a coupled, not purely additive, way — but the order of magnitude is the point). **This is a genuine, materially-sized unknown, not a rounding error.**

### Verdict: how to present this

**Do not present 11.28m / 54.45% as gudumap's expected real-world blackout accuracy as-is.** It is a real, honestly-audited, held-out result — but it measures **displacement-model accuracy given accurate heading**, not end-to-end system accuracy on a standalone phone. The gap between those two things is plausibly anywhere from "small" (if the phone's magnetometer fusion holds up fine in-vehicle) to "large enough to roughly double the error" (if it degrades the way tunnels/vehicle interiors are known to challenge magnetometers) — and gudumap currently has no instrumentation to tell you which case you're in during an actual blackout.

**Responsible framing for the PPT/demo:** present the 11.28m figure explicitly as *"ML displacement-model accuracy, evaluated with reference heading"*, and separately and explicitly disclose that end-to-end on-device accuracy depends on phone heading quality, which has not yet been isolated or measured. This is a more defensible, judge-proof framing than presenting 11.28m unqualified — and it's already exactly what `dead_reckoning/`'s own leakage audit says in Criterion 7 (§7 above), so this isn't a new admission, it's carrying forward a caveat the source material already made explicit.

### Proposed fix (not implemented — diagnosis only, per instruction)

**Low-effort, real value: stop discarding sensor accuracy, and surface it.** `SensorManager.kt:301-303`'s `onAccuracyChanged` is currently a no-op. Wiring it up to track `TYPE_MAGNETIC_FIELD` and `TYPE_ROTATION_VECTOR` accuracy (Android reports `SENSOR_STATUS_UNRELIABLE` / `LOW` / `MEDIUM` / `HIGH`) and exposing a "heading confidence: LOW" flag through `NavigationState` to the UI during blackout would cost very little and directly addresses the core problem: right now, if heading degrades during a blackout, gudumap has no way to know, and neither does anyone watching the demo. This doesn't fix the heading accuracy itself, but it turns an invisible failure mode into a disclosed one — which, given everything above, is exactly the property this project's own engineering culture (self-disclosed limitations everywhere in both codebases) already values.

A second, more involved option worth considering later (not low-effort, flagged only): register `TYPE_GAME_ROTATION_VECTOR` in parallel with `TYPE_ROTATION_VECTOR` and compare them — since `GAME_ROTATION_VECTOR` is magnetometer-free, a large and growing divergence between the two during a blackout would itself be a strong, real-time signal that magnetic interference is corrupting the primary heading estimate, without needing to trust Android's own accuracy flag alone.

---

## 9. 2026-09-06 (continued) — Implemented: heading-confidence indicator

Implemented the low-effort fix from §8 (item 1 of that section only — the `TYPE_GAME_ROTATION_VECTOR` comparison was explicitly deferred, per instruction, and was not touched).

**What changed, file by file:**

1. **`sensors/SensorManager.kt`** — `onAccuracyChanged` was a no-op (`// Not used`). Now tracks `magnetometerAccuracy` and `rotationVectorAccuracy` as public read-only `Int` properties (Android's raw `SENSOR_STATUS_*` values), updated whenever the OS calls back for `TYPE_MAGNETIC_FIELD` or `TYPE_ROTATION_VECTOR` specifically. Both reset to `SENSOR_STATUS_UNRELIABLE` when their sensor is stopped, matching the existing reset pattern for the other status flags in this class.

2. **`sensors/SensorFusionManager.kt`** — added a `HeadingConfidence` enum (`HIGH`/`MEDIUM`/`LOW`/`UNRELIABLE`) and two setters (`updateMagnetometerAccuracy`, `updateRotationVectorAccuracy`) plus a computed `headingConfidence` property that takes the **worse** of the two raw accuracy values — deliberately pessimistic, since either sensor being unreliable makes the fused heading it feeds into suspect. Reset alongside the file's other state in `reset()`.

3. **`navigation/NavigationEngine.kt`** — in `emitThrottledState()`, forwards `sensorManager.magnetometerAccuracy`/`rotationVectorAccuracy` into `sensorFusionManager` every tick (~12 fps, same cadence as every other polled sensor status in this method) and reads back `sensorFusionManager.headingConfidence.name` into the emitted state.

4. **`navigation/NavigationState.kt`** — added `headingConfidence: String = "UNRELIABLE"` (string, not the enum type, matching the existing convention in this file where every other status field — `mlStatus`, `ekfStatus`, `gnssStatus`, `motionState` — is a plain string).

5. **`ui/screens/NavigationScreen.kt`** — added a new `HeadingConfidenceTile` composable (colored dot + colored text: green=HIGH, amber=MEDIUM, red=LOW/UNRELIABLE) placed directly beside the existing "Confidence Radius" tile in the blackout status card. Since that row only renders `if (navState.blackoutMode)`, this indicator is automatically blackout-only, matching the "especially during an active blackout" ask without a separate conditional.

**One deliberate deviation from the requested wiring path, flagged for transparency:** the task described the chain as "`SensorFusionManager → NavigationEngine → NavigationEngineState → NavigationState → NavigationScreen`" (mirroring Task D's `naiveIntegrator`/`uncertaintyRadiusMeters` path). `NavigationEngineState` is `DeadReckoningEngine`'s own internal state class — heading confidence isn't a `DeadReckoningEngine` concern (it's computed from `SensorFusionManager`, which is a direct sibling field of `NavigationEngine`, not something inside `DeadReckoningEngine`), so routing it through `NavigationEngineState` would have been an unnecessary detour. Instead it goes `SensorFusionManager → NavigationEngine → NavigationState` directly, which is how `NavigationEngine` already reads other sibling-component status (e.g. `sensorManager.isAccelerometerActive`) without going through `DeadReckoningEngine` at all. Functionally identical outcome, one fewer hop.

**Verification performed:** re-traced every `import com.example.gudumap.*` across all 52 `.kt` files (no new files added this session) — all resolve. Grepped for every new identifier (`headingConfidence`, `HeadingConfidence`, `magnetometerAccuracy`, `rotationVectorAccuracy`) — consistent across exactly the 5 touched files, no orphaned references. No test file references `SensorFusionManager` or `SensorManager`'s accuracy fields, so nothing in the existing test suite could have broken.

**Not build-tested — same caveat as Task D last session, now compounding across three sessions of changes:** no Gradle/network access in this environment. This is a real risk at this point, not a formality — Task 1 (dead code deletion), Task B (asset removal), Task D (naive trail/uncertainty circle), and this fix have *all* gone in without ever running a compiler. **Android Studio build verification is the single most important thing to do before touching the PPT.**

### Two things worth flagging honestly before shifting to demo prep

1. **A real Android quirk that could make this fix a no-op on some devices:** `onAccuracyChanged` is well-documented to be inconsistently called across OEMs for some sensor types — a subset of real devices simply never fire it for `TYPE_ROTATION_VECTOR`, leaving `rotationVectorAccuracy` stuck at the default `SENSOR_STATUS_UNRELIABLE` forever (which, via the pessimistic worst-of-two logic, would pin the whole indicator at `UNRELIABLE` even when heading is actually fine). This can't be checked without a real device, and isn't something to fix speculatively — but it means: **test this on the actual demo phone before the demo**, don't assume it works from reading the code.
2. **This is a good moment to build the actual APK.** Not just for this fix — three sessions of Kotlin changes have accumulated with zero compiler verification. Recommend: open the project in Android Studio, resolve any real build errors (imports/Gradle sync will need network access this environment doesn't have), and do one real on-device or emulator run through a simulated blackout before locking anything in for the demo. If something in this session's or Task D's changes doesn't compile cleanly, better to find out now than during the demo.

---

## 10. 2026-09-06 (continued) — Attempted a real Gradle build here; could not get one to run

Per instruction, attempted to actually build the project (`./gradlew assembleDebug`) rather than just re-stating the "no Gradle access" caveat from earlier sessions. Confirmed network access to Google's Maven repo works from this environment (`dl.google.com` responded), so a real build attempt was worth trying.

**Result: could not get Gradle's daemon to start, in four independent attempts:**
1. `./gradlew assembleDebug --stacktrace` (default daemon, Bash)
2. `./gradlew assembleDebug --no-daemon --stacktrace` (Bash)
3. `.\gradlew.bat assembleDebug --no-daemon --stacktrace` (PowerShell, in case of a shell-specific issue)
4. Same as #2, with this session's own sandbox network restriction explicitly lifted (in case that restriction was the cause)

**All four failed identically**, before Gradle ever reads a build file:
```
FAILURE: Build failed with an exception.
* What went wrong:
java.io.IOException: Unable to establish loopback connection
...
Caused by: java.io.IOException: Unable to establish loopback connection
	at java.base/sun.nio.ch.WEPollSelectorImpl.<init>(WEPollSelectorImpl.java:78)
	at java.base/sun.nio.ch.WEPollSelectorProvider.openSelector(WEPollSelectorProvider.java:33)
	at java.base/java.nio.channels.Selector.open(Selector.java:295)
Caused by: java.net.SocketException: Invalid argument: connect
	at java.base/sun.nio.ch.UnixDomainSockets.connect0(Native Method)
```

**What this means, and what it doesn't:** the JVM (Temurin 17.0.17, this machine's `JAVA_HOME`) is failing to create a local loopback socket that Gradle's launcher needs purely to talk to its own daemon process — this has nothing to do with the app's source code, and happens identically regardless of shell (Bash/PowerShell) or whether this session's own sandbox restriction is active. Since disabling the sandbox made no difference, this is a genuine limitation of this specific machine's JDK/Windows networking stack (common causes: antivirus/EDR software blocking Java from binding local sockets, or a disabled/misconfigured Windows loopback interface) — **not evidence that the code is broken, and not something fixable by retrying with different Gradle flags.**

Also checked for a workaround: no `kotlinc` on this machine to type-check independently of Gradle, and while Android Studio's settings folder exists (`AppData/Local/Google/AndroidStudio2026.1.4`, confirming it's installed), its program/JBR install directory wasn't found under the usual `Program Files` locations in the time available to search.

**Bottom line: the build status is still genuinely unknown — neither confirmed working nor confirmed broken.** The right next step is exactly what was recommended before, but now for a specific, concrete reason: **open the project in the real Android Studio GUI on this machine and let it sync/build there.** Android Studio manages its own bundled JBR JDK and Gradle invocation path, which may well not hit this same loopback issue (it's frequently a per-process antivirus/EDR allowlisting quirk, not a system-wide block) — that's a materially different code path than this session's terminal `gradlew` invocation, so it's a genuinely separate test, not a repeat of what already failed here.

**Update: resolved by the user independently.** A real on-device 90-second GNSS-blackout walk test was run (see §11) — meaning a working build does exist on the user's own machine/Android Studio setup. The build question above is moot for whichever commit was actually installed on the test device; it does not by itself confirm every change described in §9/§10 was included in that build.

---

## 11. 2026-09-06 (continued) — Diagnosis of two on-device test findings (no fixes applied yet, per instruction)

A real 90-second GNSS-blackout walk test surfaced two issues. Both are diagnosed below with exact file/line citations; **nothing was changed in the code for this section** — fixes are proposed at the end of each finding, pending confirmation of which explanation matches what actually happened operationally.

### Finding 1 — ML Gate rejected 100% of windows, yet DR Distance grew to 630.3m while Motion showed STATIONARY

**1. Where the gating logic lives:** `navigation/DeadReckoningEngine.kt`, `processWindowInference()` (lines 345–464). Every ~1-second window ends in exactly one of three outcomes, each incrementing its own counter (`acceptedPredictionCount`/`clampedPredictionCount`/`rejectedPredictionCount`, lines 426–430):

- **REJECTED** (line 386–390) — forced whenever `zuptDetector.isNavStationary` is true (motion state is `STATIONARY` or `ROTATING_IN_PLACE`), *before* the ML model is even consulted. Also forced (line 405–409) when the model *is* consulted but its prediction looks kinematically implausible: `maxHorizAcc < 0.35 m/s² AND baseSpeed < 0.30 m/s AND rawMag > maxPlausibleDist`. Either path sets `localDisplacement = [0,0,0]`.
- **CLAMPED** (line 410–415) — model consulted, prediction exceeds the kinematic bound but conditions above didn't classify it as implausible-at-rest; displacement is scaled down to `kinematicDist + 0.15m` and kept (not zeroed).
- **ACCEPTED** (line 416–419) — model consulted, prediction within bound, used as-is.

**2. Is a speed/motion precondition present, and is 100% rejection correct for a walking test?** Yes — and **the ML model itself is entirely out of its trained domain for this test.** Per §7, the only model actually wired into the app is `gru_io_vnbd.onnx`, trained exclusively on **vehicle** motion (IO-VNBD: mean speed 45 km/h, target displacement mean 26.47m per 2-second window). A 90-second *walking* test (or standing still) is nowhere near that distribution. Given that, **0 ACCEPTED windows and mostly-REJECTED is arguably textbook-correct behavior for this specific model on this specific kind of test** — it's not obviously a miscalibrated threshold so much as a scope mismatch: the shipped ML correction only applies to vehicle-speed motion, and won't fire during human walking/standing regardless of tuning. This is an important finding in its own right, independent of the distance bug below: **if the real demo scenario is a person walking (not a vehicle), the ML half of "GRU+EKF+ZUPT" contributes nothing** — the system runs on EKF+ZUPT+NHC alone in that case, unassisted by the trained model.

**3. Does ZUPT actually zero velocity when STATIONARY is displayed? Traced, not assumed:**

- `navigation/ZuptDetector.kt` and the UI's "Motion" field share one source of truth: `NavigationEngine.emitThrottledState()` sets `motionState = drState.motionState`, and `DeadReckoningEngine.getState()` sets that from `zuptDetector.motionState.name` — the same live property `processWindowInference()` reads as `isNavStationary` at line 352. No separate/stale copy exists; there's no display-vs-filter desync in the wiring itself.
- When `isNavStationary` (or `REJECTED`) is true, lines 437–443 correctly **hard-zero** velocity: `ekf.predict([0,0,0], dt)` then `ekf.state[3]=state[4]=state[5]=0.0` — not just a soft Kalman pull, an explicit reset. `tracking/TrajectoryIntegrator.kt:49` separately gates distance accumulation on `point.isStationary` (`if (!point.isStationary && ...) totalDistanceTravelled += dist`), and that flag is set from `isNavStationary` at the point of construction (`DeadReckoningEngine.kt`, `rawPoint = TrajectoryPoint(..., isStationary = isNavStationary)`) and survives `mapMatcher.match()` unchanged (`navigation/MapMatcher.kt:54`, `OsmRoadNetworkMapMatcher` uses `point.copy(latitude=..., longitude=...)`, which preserves every other field including `isStationary`). **So the STATIONARY-gating mechanism, as written, is real and correctly wired end-to-end — it isn't a no-op.**
- **But the reported counters prove `isNavStationary` was not continuously true for the whole 90 seconds.** `CLAMPED` only happens inside the `else if (modelRunner.ready)` branch (line 391) — i.e., only on windows where `isNavStationary` evaluated **false**. The reported deltas (`C: 146→154`, i.e. 8 clamped windows) mean at least 8 of the ~90 one-second windows during this test were classified as *not* stationary at the exact instant of window processing — even though the continuously-updated UI "Motion" label, sampled independently at ~12 fps, apparently read STATIONARY throughout. This is a real, confirmed gap between the momentary window-processing classification and what a human watching the screen would have seen — plausible if the phone had brief hand-tremor/adjustment moments too short to visibly flip the displayed label but long enough to catch a 1 Hz window boundary.

**4. Where the 630m most likely comes from — the real lead, structurally confirmed in the code (not yet numerically simulated):** Look at how the CLAMPED ceiling is computed (lines 372–413):
```kotlin
val vBefore = sqrt(ekf.state[3]² + ekf.state[4]²)          // filter's OWN prior velocity
val gnssRef = if (isBlackoutMode) 0f else ...               // forced 0 during blackout
val baseSpeed = max(vBefore, gnssRef)                        // => baseSpeed = vBefore during blackout
val kinematicDist = baseSpeed * dt + 0.5 * effectiveAcc * dt²
val maxPlausibleDist = kinematicDist + 1.2f                  // reject-vs-clamp boundary
val maxClampedDist = kinematicDist + 0.15f                   // clamp ceiling
```
**During blackout, the kinematic gate's own ceiling is fed by the filter's own previous velocity estimate — not an independent reference.** This is self-referential: if one window's CLAMPED (nonzero) output raises `ekf.state[3]/[4]`, that raised velocity becomes next window's `baseSpeed`, which raises the ceiling for the *next* window's clamp/reject decision too. Worse, the REJECT condition at line 405 explicitly requires `baseSpeed < 0.30 m/s` — **once one clamped window pushes velocity above that threshold, the implausibility-reject path can no longer trigger at all**, leaving only `isNavStationary` (a separate, accel/gyro-based check) able to force a reset. Between whichever windows `isNavStationary` doesn't happen to catch, this is a structurally real feedback loop: each un-reset CLAMPED window can compound on the last. Combined with finding 3.6's point that the model is being asked to extrapolate on input entirely outside its training distribution (raw predictions on out-of-domain input are uncalibrated and could be large), a short run of consecutive un-reset CLAMPED windows compounding geometrically is a plausible, code-grounded mechanism for reaching hundreds of meters from just ~8 clamped events — far more plausible than 8 independent bounded ~0.25m corrections (which alone would only total ~2m).

**Recommended way to actually confirm this, rather than continue reasoning from counters alone:** `processWindowInference()` already logs every window's `rawMag, maxPlausibleDist, gateAction, motionState, maxHorizAcc, maxGyro, vBefore, speed(after), distIncr, dist` to logcat under tag `GUDUMAP_DIAG` (`DeadReckoningEngine.kt:497-504`). **Pulling the logcat from this exact test run and looking at `vBefore` and `dDist` across consecutive CLAMPED windows would directly confirm or rule out the compounding-velocity hypothesis** — this is a five-minute check against real data rather than more speculation.

**4b. IMUBuffer.kt / CoordinateTransformer.kt — checked for an independent units/frame bug, per your request:** No unit inconsistency found. `IMUBuffer.kt:70-76` correctly divides accelerometer input by `GRAVITY_MPS2` (m/s² → g, matching the model's documented g-unit contract) and leaves gyro in rad/s untouched. `CoordinateTransformer`'s rotation and geodetic-conversion formulas are dimensionally consistent (meters in, meters/degrees out, standard WGS-84 approximations). **Nothing in these two files independently explains the drift** — the leading explanation remains the self-referential kinematic-gate ceiling above.

**4c. One more contributing factor worth flagging, not yet quantified:** `NHC.applyConstraint()` (line 461–463) runs on every non-stationary window and assumes vehicle-style motion (lateral body-frame velocity ≈ 0 — valid for a car constrained to its longitudinal axis, **not necessarily valid for a walking human**, who can turn, sidestep, or shift the phone independent of travel direction). This wouldn't independently *generate* runaway magnitude, but it could misdirect whatever velocity the CLAMPED windows already introduced along the wrong axis, compounding the same underlying issue rather than correcting it.

**Proposed fixes — NOT implemented, need your confirmation on the intended deployment mode first (vehicle-mounted vs. handheld pedestrian), since that changes which parts are "working as designed" vs. "a bug":**
1. Stop feeding the kinematic gate's ceiling from the filter's own prior velocity during blackout; use a decaying/bounded reference instead, or require N consecutive non-stationary windows before trusting any nonzero CLAMPED output.
2. Add an independent, non-self-referential backstop cap on displacement-per-window during blackout (e.g., a small constant ceiling for a pedestrian profile), regardless of what `baseSpeed` claims.
3. If the real deployment is pedestrian/handheld (not vehicle-mounted): reconsider whether `NHC` should be enabled at all for this profile — its core assumption doesn't hold for a walking person.
4. Separately: if the real demo scenario is pedestrian, note plainly that the currently-shipped ML model (vehicle-trained) will not meaningfully assist in that scenario per finding 2 above — that's a scope decision, not a bug, but worth knowing going into a demo.

### Finding 2 — Position defaulted to Coimbatore when the tester was not physically there

**1–3. Traced exactly where the initial/anchor position comes from — it is explanation (b), a hardcoded fallback, confirmed with certainty:**

- `NavigationEngine.kt:82` (`init` block, runs at app/engine construction, before `start()` ever requests a location): `deadReckoningEngine.initialize(11.0168, 76.9558)` — hardcoded Coimbatore coordinates, with the comment *"Initialize DeadReckoningEngine with Coimbatore default"* directly above it (line 81). This runs unconditionally, synchronously, before any GPS fix could possibly have arrived.
- `latestRawGnssLocation` (line 56) is the *only* thing that would override this with a real fix, and it is set **only** inside `onGnssLocationChanged()`, which fires **only** on a genuine `LocationListener.onLocationChanged` callback from `sensors/LocationManager.kt` — i.e., only after GPS/Network provider actually delivers a real fix.
- `setBlackoutMode(true)` (lines 218–251) — the moment blackout starts, it anchors the whole session with: `blackoutStartLat = latestRawGnssLocation?.latitude ?: drCurrent.latitude` (line 227, same pattern line 228 for longitude). **If blackout is toggled on before the first real GPS fix arrives, `latestRawGnssLocation` is still null, so this falls back to `drCurrent.latitude` — which, since nothing has corrected the engine yet, is still exactly the hardcoded 11.0168/76.9558 from the `init` block.** The entire 90-second dead-reckoning trajectory would then be computed relative to a Coimbatore anchor with no connection to the tester's real location.
- This is consistent with a real GNSS-blackout test being started quickly (cold GPS lock can take 10–30+ seconds outdoors, longer indoors/urban) — if "START GNSS BLACKOUT" was tapped before a location fix had actually landed, this fallback is exactly what would fire.

**2. Where, clearly, for the record:** `navigation/NavigationEngine.kt:82` (unconditional hardcoded init) and `:227-228` (silent fallback to that same uncorrected default when no fix has arrived at blackout start). Both need addressing before any demo where actual location matters — this is not a hypothetical, it's a directly-traced code path that produces exactly the symptom reported.

**3. `sensors/LocationManager.kt` checked separately:** no issue found there — it correctly requests real updates from `GPS_PROVIDER` (falling back to `NETWORK_PROVIDER`) and only invokes the location callback on genuine fixes (lines 47–52). One unrelated but real observation: `isGnssAvailable` is set `true` as soon as the provider is merely *enabled* (line 82: `if (locationManager.isProviderEnabled(...)) { ...; isGnssAvailable = true }`), before any actual fix has been received — meaning "GNSS: AVAILABLE" could display briefly even before a real position exists. Not the cause of Finding 2 (that's driven by `initialize()`'s hardcoded default, not this flag), but worth knowing.

**Proposed fixes — NOT implemented, need your confirmation on desired behavior:**
1. Don't allow `setBlackoutMode(true)` to proceed if `latestRawGnssLocation` is still null — either block the "START GNSS BLACKOUT" action with a "waiting for GPS lock" state, or clearly surface in the UI that the anchor is a placeholder, not a real position, whenever this fallback path is taken.
2. Consider removing the hardcoded Coimbatore `initialize()` call from `NavigationEngine`'s `init` block entirely, and instead leave the engine uninitialized until either a real GPS fix arrives or the offline Coimbatore map is explicitly selected as the operating context — the current code conflates "default map assets happen to be Coimbatore" with "assume the phone is in Coimbatore," which are different things.

---

## 12. 2026-09-06 (continued) — Both fixes implemented, per your two decisions

**Decisions locked in:** (1) vehicle-only going forward — no pedestrian-mode detection built, `gru_local` stays cut. (2) blackout entry is now refused outright (not defaulted) if no real GPS fix has ever been obtained, with a visible "Waiting for GPS fix..." state.

**Logs from the actual test run were not available** — no `adb`, no exported log files, and no connected device from this environment (checked, none found). Per your fallback instruction, both fixes proceed on the code-level diagnosis from §11, which was already a structural trace of the actual code paths, not a guess.

### Fix 1 — Kinematic gate feedback loop

**Reasoning, before implementing:** The bad dependency was `baseSpeed` (used both as the kinematic ceiling's v₀ term and as the reject-condition's own threshold check) being sourced from `ekf.state[3]/[4]` — the filter's own live velocity — during blackout. An accepted/clamped correction raises that velocity, which raises the ceiling for the *next* window, with nothing external bounding the loop. The fix has to replace that live, self-referential value with something that (a) is fixed the moment blackout starts, or grows only in a bounded, externally-verifiable way, and (b) still lets a genuinely-accelerating vehicle be gated permissively, matching decision 1 (vehicle-only). A physically-motivated speed **envelope** — last known real speed at blackout entry, widening only with elapsed blackout time at a fixed plausible acceleration bound, capped at an absolute sanity ceiling — satisfies both: it can never be influenced by what the ML model or EKF produced in between, and it still permits genuine acceleration from a real starting speed.

**Implemented (`navigation/DeadReckoningEngine.kt`):**
- `companion object` (lines 72–79): two new constants, `MAX_SPEED_CHANGE_MPS2 = 4.0f` (m/s², a moderate plausible vehicle accel/decel rate) and `MAX_PLAUSIBLE_SPEED_MPS = 50.0f` (~180 km/h absolute ceiling).
- Two new fields (lines 105–106): `blackoutEntrySpeedMps` and `blackoutEntryTimestampNs`, both reset to sentinel values outside blackout.
- `setBlackoutMode(true)` (lines ~697–701): captures `blackoutEntrySpeedMps = latestGnssSpeed ?: 0f` **before** `latestGnssSpeed` is nulled to isolate GNSS ground truth (unchanged, pre-existing line) — this is the last real speed the filter ever saw. `blackoutEntryTimestampNs` is reset to `0L` (unset) rather than stamped with `System.nanoTime()`, specifically to avoid mixing that clock with the sensor-timestamp clock `processWindowInference` actually runs on.
- `processWindowInference()` (lines 403–411): `baseSpeed` during blackout is now `min(blackoutEntrySpeedMps + MAX_SPEED_CHANGE_MPS2 × elapsedBlackoutSeconds, MAX_PLAUSIBLE_SPEED_MPS)` — where `elapsedBlackoutSeconds` is computed from `blackoutEntryTimestampNs`, itself lazily captured from the **first window's own `timestampNs` parameter** the moment blackout begins (line 404–406), so the elapsed-time calculation never mixes clocks. Outside blackout, `baseSpeed` is unchanged (`max(vBefore, latestGnssSpeed ?: 0f)`) — GNSS keeps correcting the EKF in that regime, so the old self-reference risk doesn't apply there.
- `setBlackoutMode(false)` and `reset()`: both new fields reset to `0f`/`0L`.

**Re-verification — walked through the logic again explicitly, not just asserted:**

*Scenario: blackout starts with the vehicle at rest (`blackoutEntrySpeedMps ≈ 0`), and `isNavStationary` happens to miss a few consecutive windows despite genuine stillness (the same gap §11 identified).*

- **Window at t=0s** (first window after blackout starts): `blackoutEntryTimestampNs` gets set to this window's own timestamp, so `elapsedBlackoutSeconds = 0`. `baseSpeed = min(0 + 4.0×0, 50) = 0`. `kinematicDist ≈ 0.1m`, `maxPlausibleDist ≈ 1.3m`. If `isNavStationary` is false but the raw ML prediction is large (plausible — the model is being asked to extrapolate on out-of-domain input), the reject condition (`maxHorizAcc<0.35 && baseSpeed<0.30 && rawMag>maxPlausibleDist`) still fires correctly, since `baseSpeed=0 < 0.30`. **Rejected, zero displacement, exactly as intended for a stationary start.**
- **Window at t=1s**: if still not caught as stationary, `baseSpeed = min(0 + 4.0×1, 50) = 4.0 m/s`. Now `baseSpeed < 0.30` is false, so the reject-via-implausibility path can no longer trigger — only `isNavStationary` can still force a reset at this point. If not caught, this window falls to `CLAMPED`, capped at `kinematicDist + 0.15 ≈ 4.25m`. **This is intentional, not a residual bug**: the whole reason `baseSpeed` factors into the reject condition at all is to let a genuinely-accelerating vehicle (e.g., pulling away from a stop when blackout begins) through — 0→4 m/s in 1 second is a mild, physically real acceleration. The key difference from the old design: this 4.25m ceiling is now driven **only** by fixed elapsed time and the entry speed — not by what the previous window's own (possibly bad) output was.
- **Structural bound on the worst case:** the envelope is capped at `MAX_PLAUSIBLE_SPEED_MPS` regardless of how many consecutive windows get clamped, and it grows **linearly** with elapsed time — not multiplicatively/compounding the way the old EKF-state-fed version could. Even in a pathological worst case (every single window clamped at the maximum for the whole blackout), the per-window ceiling is bounded at `50 × 1 + 0.35 ≈ 50.35m`, a fixed number independent of prior windows' outputs — a fundamentally different (bounded, predictable) failure mode than before (unbounded, compounding, driven by the model's own possibly-corrupted output feeding back into itself).
- **What this fix does NOT claim to guarantee:** if `isNavStationary` itself fails to detect genuine stillness for an extended period (a separate concern from Fix 1 — that's ZUPT's own detection quality, untouched here), the kinematic gate alone will still permit *some* nonzero drift, bounded by the envelope above, rather than exactly zero. The primary safety net against a genuinely stationary phone remains `isNavStationary`'s hard velocity reset (unchanged, `DeadReckoningEngine.kt` lines ~437–443 in the version traced in §11) — Fix 1's job was specifically to stop the *kinematic gate itself* from being able to snowball via self-reference, which it now demonstrably cannot.

### Fix 2 — Hardcoded Coimbatore default

**Implemented, file by file:**
1. **`navigation/NavigationEngine.kt` `init{}`** — the unconditional `deadReckoningEngine.initialize(11.0168, 76.9558)` call is removed entirely. The engine now stays uninitialized (`isInitialized == false`) until a real fix arrives via `correctWithGnss()`'s existing self-init path, or blackout entry is explicitly refused (below).
2. **`navigation/DeadReckoningEngine.kt`** — the class's own field defaults (`originLat`/`originLon`/`currentLat`/`currentLon`, previously `11.0168`/`76.9558`) and `reset()` both changed to `0.0`/`0.0` — a deliberate "no real fix yet" sentinel, not a real-looking coordinate. `NavigationEngineState`'s default params (`latitude`/`longitude`/`naiveLatitude`/`naiveLongitude`) updated to match for consistency (these are effectively decorative in the live path since `getState()` always passes explicit values, but were left inconsistent otherwise).
3. **`navigation/NavigationState.kt`** — same defaults updated to `0.0`/`0.0` (this one **is** live-path-relevant: `NavigationEngine`'s `_state = MutableStateFlow(NavigationState())` uses these bare defaults as the very first UI-visible state before any real tick). Added `val hasGpsFix: Boolean = false`.
4. **`navigation/NavigationEngine.kt` `setBlackoutMode(true)`** — new guard at the top: if `latestRawGnssLocation == null`, logs a warning and returns without touching any state (no fallback to any default, silent or otherwise). Past that guard, `latestRawGnssLocation` is now guaranteed non-null, so the old `?: drCurrent.latitude`-style fallbacks for lat/lon/heading/speed were removed and replaced with a direct non-null read (`gnssAtEntry = latestRawGnssLocation!!`) — the dead fallback branch is gone, not just unreachable. `emitThrottledState()` now also populates `hasGpsFix = (latestRawGnssLocation != null)` every tick.
5. **`ui/screens/NavigationScreen.kt`** — new top-priority branch in the blackout control button's `when` block: `!navState.hasGpsFix` shows a disabled "WAITING FOR GPS FIX..." button instead of the normal GNSS/blackout controls, so the block is visible and proactive (shown before the user even taps anything), not just a silent refusal if they do tap.
6. **`ui/components/MapView.kt`** — the vehicle marker is now only created/updated when `latitude > 1.0 && longitude > 1.0` (a real fix); if no real fix exists, any existing marker is removed from the map's overlays. The map's own viewport still falls back to the bundled Coimbatore tileset center when there's no real fix (unchanged, pre-existing behavior) — that's a generic "nothing better to look at" camera default, not a claimed position, and is fine per your instruction; only the marker (which *would* read as a position claim) is now suppressed.

**Confirmed the offline map still shows something reasonable pre-fix, per your instruction to check:** yes — with no real fix, the map still renders its normal offline tiles centered on the bundled Coimbatore viewport (unavoidable, it's the only tileset shipped), but shows no vehicle marker and no trail, and the blackout control area shows "WAITING FOR GPS FIX..." — nothing on screen claims a specific position before one is real.

**Test suite kept consistent with these changes (found by grepping for every `11.0168`/`76.9558` occurrence in `app/src/test`, not just assumed clean):**
- `NavigationViewModelTest.kt` — `testNavigationStateDefaults()` asserted the old hardcoded default (`11.0168`/`76.9558`) for a bare `NavigationState()`; updated to assert `0.0`/`0.0` and `hasGpsFix == false`, since that's now the deliberately-correct behavior, not a regression.
- `DeadReckoningEngineTest.kt` — `testDeadReckoningDisplacement()` asserted the same old default on a bare (never-`initialize()`-called) `DeadReckoningEngine()`; updated to `0.0`/`0.0`. The rest of its relative-displacement assertions (`updated.latitude > initial.latitude`) hold regardless of the absolute anchor value, so nothing else in that test needed changing.
- Every other `11.0168`/`76.9558` occurrence across the test suite (`GnssBlackoutWorkflowTest.kt`, `PipelineDiagnosticTest.kt`, `MapMatcherTest.kt`) is inside an explicit `engine.initialize(11.0168, 76.9558)` call or a hardcoded test fixture coordinate — those test "does the engine behave correctly once given a real position," which is unaffected by changing what the *default* is before `initialize()` is ever called. `OfflineMapManagerTest.kt`'s `11.0168`/`76.9558` assertions are against `OfflineMapManager.COIMBATORE_DEFAULT_LAT/LON`, a map-tileset-location constant untouched by this fix — also unaffected.

**Not touched, and confirmed still correct:** `NavigationEngine.kt`'s blackout-*exit* path (`setBlackoutMode(false)`) still has an `?: drState.latitude`-style fallback for the recovery GNSS location — this is now safe by construction rather than by luck, since blackout can no longer be *entered* without a real fix, so `drState` at exit time is always a real, dead-reckoned-from-a-real-start position, never the old Coimbatore sentinel. No change was needed there, matching your scoping of Fix 2 to blackout *entry* specifically.

### Neither fix has been build-tested

Per the environment limitation in §10, a real Gradle build still could not be run from this session. **Both fixes above are code-level changes only, verified by manual re-tracing (imports, call sites, and every test assertion that could be affected by the changed defaults) — not by compiling or running them.** Before trusting either fix for a demo: sync in Android Studio, resolve any real compiler errors, and re-run the same 90-second GNSS-blackout walk test on-device to confirm (a) the drift no longer runs away the way it did, and (b) blackout entry is correctly refused with the new UI state when attempted before a GPS fix lands.

---

## 13. 2026-09-06 (continued) — hasGpsFix never becomes true on-device: diagnosed as PRE-EXISTING, not a Fix 2 regression

Real-device comparison: Google Maps gets a fix in the same spot; gudumap's GNSS status stays UNAVAILABLE indefinitely, `hasGpsFix` never flips true, blackout stays permanently blocked. Traced end-to-end per your instruction to check Fix 2 first before assuming a pre-existing bug — **the evidence points the other way: this is a pre-existing bug in code Fix 2 never touched, which Fix 2 made visible/blocking instead of silently papering over.**

### Task 1 — End-to-end trace

**1. Which location API, which providers:** `sensors/LocationManager.kt` uses the raw platform `android.location.LocationManager` with `GPS_PROVIDER` (line 88) and `NETWORK_PROVIDER` (line 101) — **not** `FusedLocationProviderClient`, despite `com.google.android.gms:play-services-location:21.3.0` being a declared Gradle dependency (`app/build.gradle.kts:96`) that is **never actually referenced anywhere in the Kotlin source** (confirmed by grep — zero hits for `FusedLocationProviderClient`/`LocationServices` in `app/src`). This matters: Google Maps uses fused location (blending GPS + Wi-Fi + cell), which acquires fixes indoors far faster and more reliably than raw `GPS_PROVIDER` alone — a real, plausible contributor to the "Maps succeeds, gudumap doesn't, same spot" symptom, independent of anything else below.

**2. Filtering/validation of incoming fixes:** none. `onLocationChanged(location: Location)` (`LocationManager.kt:58-65`, post-logging) accepts **every** callback unconditionally — no accuracy threshold, no minimum-fix-quality check. This rules out "Android delivers fixes, gudumap filters them out" as a possible explanation entirely — there is no filter to reject anything.

**3. The actual `hasGpsFix` wiring, re-checked line by line — correct on its own terms:**
- `NavigationEngine.kt:174`: `latestRawGnssLocation = location`, set unconditionally inside `onGnssLocationChanged()`, itself only ever called from the `onLocationChanged` callback registered in `start()` (`NavigationEngine.kt:130-132`).
- `NavigationEngine.kt` `emitThrottledState()`: `hasGpsFix = (latestRawGnssLocation != null)` — reads the same variable, no misnaming, no wrong-object bug in what §12 added.
- **This wiring is not the bug.** `hasGpsFix` will correctly become `true` the instant `onGnssLocationChanged` ever fires. The problem is upstream: it may never fire at all.

**4. The actual root cause — a permission-registration-timing gap, confirmed by tracing the full call chain, not assumed:**
- `NavigationViewModel.kt:71`: `navigationEngine.start()` is called exactly once, synchronously, inside the ViewModel's `init {}` block — which runs at the Composable's first composition, i.e. essentially at app launch.
- `NavigationEngine.kt`'s `start()` calls `locationManager.startLocationUpdates(...)` (line ~129) at that same moment.
- `LocationManager.kt:48-54` (pre-existing, unchanged by Fix 2): `if (!hasLocationPermission()) { ...; return }` — **if runtime location permission has not yet been granted at this exact instant, `requestLocationUpdates()` is never called, for either provider, for the rest of the app's life.**
- `NavigationScreen.kt:64-68`: the permission request launcher's callback does exactly one thing: `permissionGranted = granted` — a **local Compose UI variable only**. Grepped every reference to `permissionGranted`/`permissionLauncher` in this file (lines 55, 64-68, 583, 590, 593) — **nothing anywhere calls back into `navViewModel` or `navigationEngine` to retry location registration after the user grants permission.**
- **Net effect: if the runtime location permission is not already granted at the exact moment the app launches and the ViewModel is constructed, location updates are never requested — even if the user grants permission ten seconds later by tapping "Allow."** This exactly matches "GNSS stays UNAVAILABLE... indefinitely": `LocationManager.kt:51` sets `isGnssAvailable = false` on that early return, and nothing downstream ever revisits it.

**Is this a Fix 2 regression? No — checked precisely, not assumed:** `NavigationViewModel.kt`, `LocationManager.kt`, and `NavigationScreen.kt`'s permission-launcher wiring were **not modified** by Fix 2 (§12) at all. Fix 2 only touched `NavigationEngine.kt`'s `init{}`/`setBlackoutMode()`, `DeadReckoningEngine.kt`'s position defaults, `NavigationState.kt`, `NavigationScreen.kt`'s *button when-block* (a different section of that file than the permission launcher), and `MapView.kt`'s marker logic. **This permission-timing gap already existed before Fix 2** — the difference is what happened when it was hit: before Fix 2, `setBlackoutMode(true)` would silently fall back to the hardcoded Coimbatore default and proceed anyway, masking the fact that location was never actually being tracked. After Fix 2, the same missing-fix condition is (correctly) refused instead of masked — which is exactly why it's visible now and wasn't before. **Fix 2 exposed a pre-existing bug; it did not introduce one.**

### Task 2 — Diagnostic logging added (temporary, not a fix)

Since the static trace above already identifies the exact branch point, but confirming it needs to see what actually happens on this specific device, added:
- `LocationManager.kt:48-50`: logs a warning at the precise permission-check branch, stating plainly that `requestLocationUpdates` will never be called if this path is hit.
- `LocationManager.kt:55`: logs when registration *does* proceed.
- `LocationManager.kt:58-61`: logs `provider`, `lat`, `lon`, `accuracy`, `hasSpeed()` on **every** raw `onLocationChanged` callback.
- `LocationManager.kt:88-89, 101-102`: logs whether `GPS_PROVIDER`/`NETWORK_PROVIDER` were reported enabled at registration time.
- `NavigationEngine.kt` `onGnssLocationChanged()`: logs once, specifically on the **first** real fix received, exactly where `hasGpsFix` would flip true.

**What the next test run's logcat will show, and what each outcome means:**
- If `"permission NOT granted at this moment"` appears → confirms the permission-timing gap above is what happened this run.
- If `"permission granted, registering listeners"` appears but no `onLocationChanged` line ever follows → confirms registration succeeded but Android itself never delivered a fix (points at provider reliability — raw GPS vs. fused — as the actual bottleneck instead).
- If `onLocationChanged` lines *do* appear but `"FIRST real GPS fix received"` never does → would mean the bug is between `LocationManager` and `NavigationEngine`, but per the code trace above this path has no logic gap, so this outcome would be a surprise worth re-investigating on its own.

### Task 3 — Definitive answer

**Gudumap is not receiving location callbacks at all — this is a registration/permission-handling bug, not a filtering/logic bug.** There is no code path anywhere that inspects and rejects a delivered location; the entire failure mode traces to whether `requestLocationUpdates()` is ever called in the first place, which depends on permission already being granted at the single moment `start()` runs, with no retry mechanism if it wasn't. Confirmed pre-existing (present before Fix 2, in files Fix 2 never touched) and now correctly visible instead of silently masked.

### Proposed fixes — NOT implemented, awaiting confirmation

1. **Primary, directly closes the gap:** add a way to retry location registration after permission is granted late. Concretely: expose `NavigationEngine.retryLocationUpdates()` (re-invokes just the location-registration portion of `start()`) and a matching `NavigationViewModel` passthrough; call it from `NavigationScreen.kt`'s `permissionLauncher` callback when `granted == true`. This directly fixes "permission granted after the engine already started."
2. **Secondary, more involved, flagged for a separate decision:** the app declares `play-services-location` but never uses it — switching from raw `LocationManager` to `FusedLocationProviderClient` would likely improve indoor fix speed/reliability to match what Google Maps demonstrably achieves in the same spot, but is a larger change (different API, Play Services availability handling) than fix #1 and shouldn't be bundled into it without a separate go-ahead.

---

## 14. 2026-09-06 (continued) — Fix #1 from §13 implemented (retry gap only; FusedLocationProviderClient explicitly deferred, not touched)

### Implemented, file by file

1. **`sensors/LocationManager.kt`** — added `fun isListening(): Boolean = listener != null`, so callers can check whether a retry is even useful before calling again. Made `startLocationUpdates()` itself safely re-callable: if a listener is already registered, it now tears it down via `stopLocationUpdates()` first before re-registering, rather than silently accumulating a second listener alongside the first (the old code created a fresh anonymous `LocationListener` on every call and only ever unregistered it in `stopLocationUpdates()` — calling `startLocationUpdates()` twice without that guard would have leaked a duplicate registration). Diagnostic logging from §13 kept in place, untouched.

2. **`navigation/NavigationEngine.kt`** — extracted the location-registration block from `start()` into a new private `startLocationListening()`, so it can be invoked again later without re-running the rest of `start()` (which also registers all the IMU sensor listeners — re-running those on every retry would have been unnecessary and untested territory). Added a new public method:
   ```kotlin
   fun retryLocationUpdatesIfNeeded() {
       if (locationManager.isListening()) return
       if (!locationManager.hasLocationPermission()) { ...; return }
       startLocationListening()
   }
   ```
   Safe to call unconditionally as often as needed (every resume, every permission-grant callback) — it's a no-op if already listening or still denied.

3. **`viewmodel/NavigationViewModel.kt`** — added a simple passthrough `retryLocationUpdatesIfNeeded()`, matching the existing pattern used by `setBlackoutMode`/`toggleBlackout`.

4. **`ui/screens/NavigationScreen.kt`** — two call sites, covering both scenarios you asked for:
   - The `permissionLauncher` callback (in-app "Allow" dialog) now calls `navViewModel.retryLocationUpdatesIfNeeded()` when `granted == true`, in addition to the pre-existing local `permissionGranted` UI-flag update.
   - A new `DisposableEffect` attaches a `LifecycleEventObserver` to `LocalLifecycleOwner.current` for the Composable's lifetime; on every `Lifecycle.Event.ON_RESUME` it re-checks `ContextCompat.checkSelfPermission(...)` (covers permission granted via system Settings while the app was backgrounded — the "wait, grant it again" mid-demo scenario) and calls `retryLocationUpdatesIfNeeded()` regardless. Used `rememberUpdatedState(navViewModel)` inside the observer closure as standard practice for a long-lived effect callback, and the observer is properly removed in `onDispose`.

### Explicit trace-through, as asked — not just asserted

*Scenario: app launches, location permission is denied; user grants it via the in-app dialog 10 seconds later.*

1. `NavigationViewModel.init{}` → `navigationEngine.start()` → `startLocationListening()` → `LocationManager.startLocationUpdates()`. `listener` is `null` (first call), so the new "already listening" branch is skipped. `hasLocationPermission()` is `false` → logs the warning from §13, `isGnssAvailable=false`, returns. **Critically, `listener` is never assigned in this path** — the early return happens before that line.
2. User taps "GRANT LOCATION PERMISSION" → OS dialog → 10s later, taps "Allow". The `permissionLauncher` callback fires with `granted=true`: `permissionGranted=true`, then `navViewModel.retryLocationUpdatesIfNeeded()` → `navigationEngine.retryLocationUpdatesIfNeeded()`.
3. Inside that: `locationManager.isListening()` → `listener` is still `null` from step 1 → `false`, so it does **not** return early here. `locationManager.hasLocationPermission()` → **now true** → does not return early either. Logs "permission now granted, registering" → calls `startLocationListening()` again.
4. `LocationManager.startLocationUpdates()` runs a second time: `listener` is still `null` (nothing to tear down) → skips the teardown branch → `hasLocationPermission()` now passes → creates a new `LocationListener`, assigns it to `listener`, logs `GPS_PROVIDER`/`NETWORK_PROVIDER` enabled state, and calls `requestLocationUpdates()` for whichever is enabled.
5. Once Android's location subsystem delivers a fix (now a pure hardware/environment question — no app-level gap left to block it), `listener.onLocationChanged()` fires: logs the diagnostic line from §13, sets `isGnssAvailable=true`, and invokes the callback chain down to `NavigationEngine.onGnssLocationChanged(location)`.
6. `onGnssLocationChanged`: `latestRawGnssLocation` was `null`, so this is recognized as the first fix, logs accordingly, sets `latestRawGnssLocation = location`, and (since `gnssNavMode` starts as `"GNSS_AVAILABLE"`) calls `deadReckoningEngine.correctWithGnss(location)` — which self-initializes the engine at this **real** position (Fix 2, §12, working as designed: no default was ever substituted).
7. On the very next `emitThrottledState()` tick (driven continuously by IMU sensor callbacks, independent of location status, so this is at most ~80ms later): `hasGpsFix = (latestRawGnssLocation != null)` evaluates `true` for the first time → pushed into `NavigationState` → UI recomposes → the "WAITING FOR GPS FIX..." button (§12) disappears, replaced by the normal GNSS/blackout controls.

**Confirmed: yes, a real fix now arrives and `hasGpsFix` correctly flips to `true`, via the exact mechanism traced above — not asserted, walked through step by step.** One additional note caught during the trace: the `DisposableEffect`'s `ON_RESUME` observer will also very likely fire once immediately on first composition (Android's lifecycle dispatches the current state to a newly-attached observer), redundantly calling `retryLocationUpdatesIfNeeded()` right alongside `start()`'s own attempt — this is harmless by construction, since the method is a no-op whether or not it has anything to do.

### What was deliberately NOT done

Per your instruction, `FusedLocationProviderClient` was not introduced — `LocationManager.kt` still uses raw `GPS_PROVIDER`/`NETWORK_PROVIDER`. That remains a separate, deferred decision (§13's secondary proposal). The §13 diagnostic logging (permission-check branch, every raw callback, provider-enabled checks, first-fix log) is untouched and still in place.

### Still not build-tested

Same caveat as every prior session — no working Gradle build in this environment (§10). This fix is a code-level change, verified by manual re-tracing of every call site and the full permission-grant/resume scenario above, **not by compiling or running it.** Before the next demo: Android Studio sync, resolve any real compiler errors, and re-run the on-device test — specifically, try denying permission at launch and granting it later (both via the in-app dialog and via system Settings while backgrounded) to confirm `hasGpsFix` actually flips to `true` and a real fix appears in the location log lines from §13.

**Confirmed working on-device (reported by the user, not this session):** the retry fix above resolved the permission-timing gap — GPS now locks correctly, which is what surfaced §15/§16 below.

---

## 15. 2026-09-06 (continued) — Speed-sanitization fix implemented (Task 1)

Confirmed diagnosis from the prior session (one shared root cause — raw, unfiltered `Location.getSpeed()` feeding both the displayed-speed path and Fix 1's envelope baseline) approved and implemented exactly as proposed.

**Implemented, `navigation/DeadReckoningEngine.kt`:**
- Two new fields: `lastSanitizedGnssSpeed: Float = 0f`, `lastGnssSpeedTimestampNs: Long = 0L`.
- New private `sanitizeGnssSpeed(rawSpeedMps: Float?, timestampNs: Long): Float?` — bounds how much a reported speed may change since the last fix, using elapsed time (`location.elapsedRealtimeNanos`, a monotonic clock, not wall-clock `location.time`) and the **same** `MAX_SPEED_CHANGE_MPS2` constant Fix 1 already defined (reused, not duplicated, for one consistent "plausible vehicle accel/decel" definition). First-ever reading (no prior reference) passes through unchanged — a disclosed, narrow edge case, not a gap in the general fix.
- `correctWithGnss(location: Location)` now calls `sanitizeGnssSpeed()` once, and passes the **sanitized** result into `correctWithGnss(lat, lon, speed, bearing)` — the single downstream function that both sets `latestGnssSpeed` (→ `blackoutEntrySpeedMps`) and calls `ekf.updateGnssVelocity()`. Both consumers now draw from the same sanitized value, as designed.
- Both new fields reset in `reset()`.

**Explicit trace-through, as asked — not just asserted:**

*Scenario: tester genuinely near-stationary (~0.5 m/s), a spurious raw reading reports 27.8 m/s (100 km/h), 1 second after the last real fix.*

1. `sanitizeGnssSpeed(27.8f, T)` called. Not the first reading (`lastGnssSpeedTimestampNs != 0`), so the delta check runs.
2. `dt = 1.0s`. `maxDelta = MAX_SPEED_CHANGE_MPS2 × dt = 4.0 × 1.0 = 4.0 m/s`.
3. `delta = 27.8 − 0.5 = 27.3 m/s`. `abs(delta) = 27.3 > maxDelta = 4.0` → sanitization triggers.
4. `sanitized = 0.5 + 4.0 × sign(27.3) = 4.5 m/s` (≈16.2 km/h) — **not** 27.8. This is what gets returned and used everywhere downstream.
5. `latestGnssSpeed = 4.5` (not 27.8) → `blackoutEntrySpeedMps`, if captured at this exact moment, is bounded to 4.5, not poisoned by the raw spike.
6. `ekf.updateGnssVelocity(vN, vE, 0)` receives a measurement built from 4.5 m/s, not 27.8 — even with a high Kalman gain, the worst-case displayed-speed contribution from this one bad reading is now ~16 km/h for that tick, not 100+.
7. If the next fix returns to a plausible value, the very next call's `delta` will be small and pass through unchanged immediately — the filter doesn't lag behind real speed changes, it only clips implausible single-step jumps.

**Disclosed limitation, not silently omitted:** if the spurious reading happens to be the **very first** GNSS fix of a session (no prior sanitized value to compare against), it passes through unfiltered — a delta-based filter has nothing to bound against on the first sample. Narrow case (only the first fix ever), doesn't apply to the reported scenario (mid-session, after real prior fixes existed).

**A related, NOT-yet-fixed gap found during verification, flagged rather than silently expanded into scope:** `NavigationEngine.kt`'s `setBlackoutMode(true)` captures `blackoutStartSpeed = gnssAtEntry.speed` directly from its own separately-tracked `latestRawGnssLocation` — **raw, not sanitized** — and passes it to `deadReckoningEngine.initialize(speedMps = blackoutStartSpeed, ...)`, which seeds the EKF's velocity state at the exact moment blackout begins. This bypasses `correctWithGnss()`/`sanitizeGnssSpeed()` entirely, since `initialize()` doesn't route through it. If the specific fix captured at blackout entry is itself the spurious one, this path could still seed a corrupted initial velocity — arguably the most consequential moment for Finding B's "at or near blackout entry" framing. **Not fixed in this pass** (wasn't part of the approved design) — flagging for a decision on whether to address it too.

**Verification:** re-grepped for every new identifier (`sanitizeGnssSpeed`, `lastSanitizedGnssSpeed`, `lastGnssSpeedTimestampNs`) — confined to `DeadReckoningEngine.kt` as designed, no new files, no dangling references. Not build-tested (same standing caveat).

---

## 16. 2026-09-06 (continued) — Offline map coverage expansion: investigation only (Task 2), no data pipeline work done

### 1. How were the existing assets generated?

**No documentation exists anywhere in the repo.** Searched `dead_reckoning/docs/`, `dead_reckoning/scripts/`, `gudumap`'s own docs, and grepped every directory for "coimbatore" case-insensitively — the only hits outside the asset files themselves and this status document are the source comments listed below (§16.3). There is no README, generation script, or note describing how `coimbatore.mbtiles` or `coimbatore_roads.json` were produced (which tool, which OSM extract, which rendering style, what tolerance/simplification was applied to the road geometry). **If this needs to be regenerated or extended, it will have to be reverse-engineered from the output files' structure, or redone from scratch with a fresh, documented pipeline** — there's nothing to build on.

One small provenance clue that does exist: `assets/maps/coimbatore.map` (a 171-byte plain-text manifest, not the tile data itself) records `BBOX=10.98,76.92,11.05,77.02`, `ROADS_COUNT=813`, `CENTER=11.0168,76.9558` — consistent with, but not proof of, the same bounding box `OfflineMapManager.kt`'s `COIMBATORE_BOUNDS` constant declares in code. This file isn't read by any code path (grepped — nothing opens `coimbatore.map`); it looks like a leftover manifest from whatever process generated the real assets, not something the app depends on.

### 2. Actual file sizes — real baseline, not estimated

| Asset | Size |
|---|---|
| `coimbatore.mbtiles` | **13.86 MB** (13,864,960 bytes) |
| `coimbatore_roads.json` | **894 KB** (915,593 bytes) — duplicated byte-for-byte in both `assets/maps/` and `assets/maps/coimbatore/` (pre-existing duplication, not addressed here — out of this task's scope) |
| `coimbatore.map` | 171 bytes (manifest only, unread by code) |

Coverage: `MIN_ZOOM=11`, `MAX_ZOOM=17` (`OfflineMapManager.kt:37-38`), bounding box **23.3 km × 20.7 km ≈ 482 km²** (per the code's own comment, `OfflineMapManager.kt:40`), **813 roads**.

### 3. How `OfflineMapManager.kt` loads these — hardcoded, not parameterized

Every reference is a literal string, not a variable pulled from a region config:
- `OfflineMapManager.kt:79`: `File(context.filesDir, "maps/coimbatore")`
- `OfflineMapManager.kt:82`: `File(mapsDir, "coimbatore.mbtiles")`
- `OfflineMapManager.kt:86-91`: `context.assets.open("maps/coimbatore/coimbatore.mbtiles")` (with a flat-path fallback to `"maps/coimbatore.mbtiles"`)
- `OfflineMapManager.kt:185`: tile source literally named `"CoimbatoreOffline"`
- `OfflineMapManager.kt:34-42`: `COIMBATORE_DEFAULT_LAT/LON`, `COIMBATORE_BOUNDS`, `COIMBATORE_CENTER` — named constants, but Coimbatore-specific by name and value, not a swappable region parameter.
- `map/MapMatcher.kt:51,53`: same hardcoded-filename pattern for `coimbatore_roads.json`, plus a hardcoded default road name `"Coimbatore Road"` (line 63) used whenever a road entry's own `name` field is missing.

Grepped every "coimbatore" occurrence across `app/src/main/java` (20 hits, 4 files) to make sure this list is complete, not a sample.

### 4a. Rough size estimate for full Tamil Nadu coverage — clearly labeled as rough, not a measurement

Using the Coimbatore baseline as the only real data point, scaled by area ratio (Tamil Nadu ≈ 130,058 km² ÷ Coimbatore's 482 km² bounding box ≈ **270×**):

| Asset | Coimbatore (482 km²) | Naive linear scaling → Tamil Nadu (130,058 km²) |
|---|---:|---:|
| `.mbtiles` (map tiles) | 13.86 MB | **≈ 3.5 GB** |
| `roads.json` | 894 KB | **≈ 0.2 GB** (≈219,000 roads) |

**Why this is a rough estimate, not a real prediction, stated plainly:**
- Linear-by-area scaling assumes uniform tile/data density across the whole state at the same zoom range (11–17). Coimbatore is a dense **urban** area; the real Tamil Nadu average (large rural stretches, forests, less-mapped roads) would likely render **smaller**, not larger, per km² at the same zoom levels — so 3.5 GB is plausibly a ceiling, not a central estimate, for tile size specifically. Road density, conversely, could be uneven in the other direction (some rural OSM coverage is sparser than Coimbatore's, some highway corridors denser) — no way to know without pulling a real state-wide OSM extract and checking.
- **A ~3.5 GB bundled asset is a real practicality problem independent of the estimate's precision**: Android app bundles / Play Store distribution have practical size ceilings that a multi-gigabyte raw asset bundled directly in the APK would strain or exceed, and even if technically possible via Play Asset Delivery/on-demand modules, **no on-demand-download code path exists anywhere in this app today** — `OfflineMapManager.kt` only ever copies from a bundled APK asset, there is no download/fetch mechanism. Going statewide as a bundled asset the way Coimbatore is bundled now is very unlikely to be viable without adding a real download pipeline — a materially bigger scope than "swap the region."

### 4b. Would a region swap be easy, or need real refactoring? — needs real refactoring, not a simple swap

**Not a simple config change.** Two separate kinds of work would be needed:

1. **Parameterization** (mechanical but touches multiple files): city name, bounding box, center point, and all asset file paths are hardcoded string/constant literals spread across `OfflineMapManager.kt` and `map/MapMatcher.kt` (20 hardcoded references found, listed in §16.3), not read from one swappable config. A real region swap needs a `MapRegion`-style parameter object (name, bounds, center, asset filenames) threaded through both classes, replacing the literals — a contained but genuine refactor, not a rename.

2. **Algorithmic scaling concern in `map/MapMatcher.kt`, found while reading it — a real, separate issue from the file-path hardcoding:** `match()` (`MapMatcher.kt:89-172`) does a **brute-force linear scan** over every road segment's every point-pair, on **every call** — and it's called roughly 12 times/second from `NavigationEngine.emitThrottledState()`. There is no spatial index (grid, quadtree, or similar) — just a flat `ArrayList<RoadSegment>` scanned start to finish each tick. This is fine at Coimbatore's scale (813 roads); at the naive ~270× road-count scaling for a full state (≈219,000 roads), this loop would very plausibly become a real per-frame performance bottleneck on a phone, not just a bigger file to load. **A statewide (or even large-region) rollout would need a real spatial-indexing addition to this matcher, not just bigger input data** — this is a second, independent piece of required refactoring beyond parameterizing file paths.

### Not done, per instruction

No map data was downloaded, generated, or modified this session — investigation and estimation only, as asked. Decision on scope (targeted region vs. full state) is yours to make once you've seen these real numbers.

---

## 17. 2026-09-06 (continued) — Closed the remaining bypass: blackout-entry velocity now sanitized too

Closed the gap flagged at the end of §15: `NavigationEngine.kt`'s `setBlackoutMode(true)` was still seeding the EKF's blackout-entry velocity from `gnssAtEntry.speed` completely raw, bypassing `sanitizeGnssSpeed()` since `initialize()` never routes through `correctWithGnss()`.

### Implemented

**`navigation/DeadReckoningEngine.kt:349-351`** — new public method, reusing the existing private logic rather than duplicating it:
```kotlin
fun sanitizeExternalGnssSpeed(rawSpeedMps: Float?, timestampNs: Long): Float? {
    return sanitizeGnssSpeed(rawSpeedMps, timestampNs)
}
```

**`navigation/NavigationEngine.kt:286-290`** — `blackoutStartSpeed` (previously `gnssAtEntry.speed`, raw) now:
```kotlin
blackoutStartSpeed = deadReckoningEngine.sanitizeExternalGnssSpeed(
    gnssAtEntry.speed,
    gnssAtEntry.elapsedRealtimeNanos
) ?: 0f
```
`blackoutStartSpeed` then feeds `deadReckoningEngine.initialize(speedMps = blackoutStartSpeed, ...)` (line ~307, unchanged) — which seeds `ekf.initialize(vNorth, vEast, ...)`, the EKF's blackout-entry velocity state — now sanitized at the source instead of downstream.

### Task 2 — scope/lifecycle trace of shared sanitizer state, as asked, not assumed

The concern: `sanitizeGnssSpeed()`'s internal state (`lastSanitizedGnssSpeed`, `lastGnssSpeedTimestampNs`) is shared between two now-active call sites — `correctWithGnss(location: Location)`'s own internal call, and this new external one. Traced explicitly:

- `gnssAtEntry` (`NavigationEngine`'s `latestRawGnssLocation`) is set inside `onGnssLocationChanged()`, which — in both `"GNSS_AVAILABLE"` and `"GNSS_RECOVERY"` modes — always calls `deadReckoningEngine.correctWithGnss(location)` for that same fix. Blackout can only be *entered* from one of those two modes (confirmed: `setBlackoutMode` requires `blackoutActive == false`, i.e. not already in blackout, and the app starts in `"GNSS_AVAILABLE"`). **So by the time `setBlackoutMode(true)` runs, the exact `Location` object in `gnssAtEntry` has, in every reachable case, already been processed once by `correctWithGnss()` — meaning `sanitizeGnssSpeed()` has already seen this same `(rawSpeedMps, timestampNs)` pair once.**
- Traced what happens calling it a *second* time with the identical pair: `lastGnssSpeedTimestampNs` was just set to this same `timestampNs` by the first call, so `dt = max(0.001, (timestampNs - lastGnssSpeedTimestampNs)/1e9)` collapses to the floor value `0.001s` (not zero — the existing `max(0.001, ...)` guard, already in the code, prevents a divide-by-zero-shaped issue here). `maxDelta = MAX_SPEED_CHANGE_MPS2 × 0.001 = 0.004 m/s` — a deliberately tiny allowance, appropriate since essentially no real time has passed between the two calls. If the first call already accepted the raw value as plausible, the second call sees `delta ≈ 0` and returns the same value unchanged. If the first call *already clamped* a spurious reading, the second call sees a large `delta` against the *already-clamped* baseline, and clamps again to within `0.004 m/s` of it — i.e. it stays clamped, doesn't let the raw spike back in through the second call. **Idempotent and safe either way** — confirmed by tracing the arithmetic, not assumed from the method existing.
- Checked whether anything could clear this state between the two calls and make the *second* one wrongly take the "first reading ever" branch (which accepts unsanitized): grepped `NavigationEngine.kt` for `.reset()` — **`deadReckoningEngine.reset()` is never called anywhere in the live app**, only `initialize()` (which does not touch `lastSanitizedGnssSpeed`/`lastGnssSpeedTimestampNs`). No lifecycle hazard found.

### Task 3 — full re-verification: any third bypass path?

Grepped every `.speed`/`speedMps`/`latestGnssSpeed` occurrence in both `NavigationEngine.kt` and `DeadReckoningEngine.kt` (28 hits total) and traced each one:

- **`ekf.updateGnssVelocity()`** (one call site, inside the `(lat,lon,speedMps,bearingDeg)` overload) — only ever reached via delegation from `correctWithGnss(location: Location)`, which sanitizes first. No direct external caller found.
- **`ekf.initialize()`** (two call sites: the blackout-entry path just fixed, and the `!isEngineInitialized` self-init fallback inside `correctWithGnss(lat,lon,speedMps,bearingDeg)`) — the self-init fallback receives `speedMps` from the *same* already-sanitized parameter the enclosing `correctWithGnss(location: Location)` computed. Both now sanitized.
- **`latestGnssSpeed`/`blackoutEntrySpeedMps`** — already fixed in §15 (both draw from the same sanitized field).
- **Displayed Speed** (`drState.speed * 3.6f`) and the three other `speedMps = speed` sites (`TrajectoryPoint` construction) — all read the EKF's own derived velocity state, not a raw GNSS value; not a bypass, just downstream of whatever the EKF already holds.
- **A separate `initialize(location: Location)` overload exists** (`DeadReckoningEngine.kt`, extracts `location.speed` unsanitized) **but is never called from anywhere in `app/src/main`** — confirmed by grep. Dead code, not a live bypass. Not touched (out of scope for this fix; flagging only for completeness since the ask was to confirm no *reachable* third path exists).

**Conclusion: Finding B's "bad reading right at blackout entry" scenario is now fully closed, with no remaining reachable bypass.** Every path that can feed a raw GNSS speed value into either the EKF's velocity state or the blackout-entry/kinematic-gate baseline now passes through `sanitizeGnssSpeed()` first, via either its direct internal use in `correctWithGnss()` or the new `sanitizeExternalGnssSpeed()` wrapper.

### Still not build-tested

Same standing caveat — no working Gradle build in this environment. Verified by manual re-tracing of every call site and the shared-state lifecycle above, not by compiling or running it. Needs an Android Studio sync and an on-device retest (ideally reproducing the same near-stationary-indoors scenario) before trusting it for a demo.

---

## 18. 2026-09-06 (continued) — .gitignore written for the first push, real sizes audited

Before the user's first push to a shared repo. Sizes were measured (`du`, `ls -la`), not guessed; a couple of the larger scans genuinely timed out on this Windows/Git-Bash filesystem (huge `.venv` file counts) but that only affects precision, not the exclude/include decision — a `.venv` gets excluded regardless of whether it's exactly 1.5GB or 1.6GB.

### Real sizes found

| Path | Size | Verdict |
|---|---:|---|
| `/dead_reckoning.zip` (repo root) | **2.83 GB** | Exclude — largest file in the entire tree by far, ~28x GitHub's 100MB hard block. Appears to be a full backup/transfer archive of the whole `dead_reckoning/` folder. **Not caught by the first draft of the .gitignore** — found only during the dry-run verification pass, then fixed. |
| `dead_reckoning/data/raw/` | 3.8 GB (OxIOD 2.9GB + IO-VNBD 865MB) | Exclude — published, separately-downloadable datasets |
| `dead_reckoning/Oxford Inertial Odometry Dataset_2.0.zip` | 955 MB | Exclude — single file over GitHub's 100MB hard limit |
| `dead_reckoning/.venv/` | ~1.5 GB | Exclude — Python virtual environment |
| `SIH26168-DeadReckoning/backend/.venv/` | 208 MB (incl. one 37.4MB `.pyd`) | Exclude — Python virtual environment |
| `gudumap/app/build/` | 272 MB (incl. a 47.5MB `app-debug.apk`) | Exclude — regenerable Gradle output |
| `dead_reckoning/FE/gudumap/gudumap/app/build/` | incl. a 78.8MB `app-debug.apk` | Exclude — same reason, inside the stale nested Android project copy |
| `dead_reckoning/models/` | <1 MB across 14 files | **Keep** — small, needed for provenance |
| `dead_reckoning/data/processed/` (*.npz splits) | ~80 MB across 12 files, largest 19.8MB | **Keep** — a judgment call, flagged below |
| `dead_reckoning/results/` | 7.2 MB | Keep |
| `SIH26168-DeadReckoning/data/public_datasets/`, `raw_traces/` | a few KB (README only) | Keep the READMEs, pre-emptively ignore anything dropped in later |

**Found during the audit, not asked for but relevant:** three nested `.gitignore` files already exist — `gudumap/.gitignore` and `SIH26168-DeadReckoning/.gitignore` (both pre-existing, sensible, left untouched) and `dead_reckoning/.gitignore` (present but empty). The root `.gitignore` below works alongside these, not instead of them — git layers them. `SIH26168-DeadReckoning/.gitignore` already excludes `backend/data/*.duckdb`/`.duckdb.wal` (small demo-mock DB files, 12KB+118KB) — the dry-run below correctly accounts for this pre-existing rule too.

### `.gitignore` written to `D:\Projects\SIH_2026\.gitignore`

Sections: Android/Gradle build artifacts (unanchored `build/`, `.gradle/`, `.kotlin/`, `.idea/`, `local.properties`, `*.apk`/`*.aab` — catches both `gudumap/` and the nested `FE/` copy in one pattern each), Python (`.venv/`, `__pycache__/`, `*.pyc`, etc.), the large-file exclusions from the table above (each with a comment explaining size and why), and OS junk. Full content is in the file itself — every exclusion is commented with the real measured size and reason, not a generic boilerplate ignore file.

### Task 3 — nothing needed to build gudumap is excluded

Explicitly verified (not assumed): `gudumap/app/src/main/assets/gru_io_vnbd.onnx`, `io_vnbd_normalization.json`, `maps/coimbatore.mbtiles` (13.2MB), and `maps/coimbatore_roads.json` all appear in the final "would be staged" list. None of the exclusion patterns (`build/`, `*.apk`, `.venv/`, etc.) touch `app/src/`. **One judgment call, not a silent exclusion, flagged for you to confirm:** `dead_reckoning/data/processed/*.npz` (~80MB total, largest single file 19.8MB) and `dead_reckoning/models/*` (under 1MB total) were left trackable rather than excluded — these aren't needed to build gudumap, but they represent real training-split/model artifacts that would otherwise require re-downloading and reprocessing the multi-GB raw datasets to reproduce. Reasonable people could call this either way; override by adding `dead_reckoning/data/processed/` to the .gitignore if you'd rather keep the repo leaner.

### Task 4 — dry run, without running `git init`

Since no `.git` exists yet and the instruction was explicitly not to create one, simulated `git add -A` in Python instead — walked the full tree, loaded and correctly layered **all** `.gitignore` files found (root + the two pre-existing nested ones), and applied their patterns (including `!` negation) file by file. Caught and fixed one real bug in the simulator itself along the way (a leading `/` in `/dead_reckoning.zip` wasn't being handled correctly, which is exactly what surfaced the fact that the *first draft* of the actual .gitignore didn't cover the root-level zip either — the same dry-run pass that was supposed to just verify the .gitignore ended up finding a real gap in it).

**Result: 353 files would be staged** (~68,364 excluded across `.pytest_cache` (4), `SIH26168-DeadReckoning/` (7,813), `dead_reckoning/` (59,737), the root `dead_reckoning.zip` (1), and `gudumap/` (810)). Full list sent to the user directly (`kept_files_full_list.txt`) for their own review before running any real git command. Manually scanned it for anything sensitive — no `.env`, credentials, or key files present. Also reran a final sanity check on the "kept" list specifically for any file over 5MB that slipped through uncaught: only the already-reviewed `data/processed/*.npz` files and the required `coimbatore.mbtiles` — nothing unexpected.

**Aside, not part of this task's scope, flagged for awareness:** the "kept" list includes all 79 files of `dead_reckoning/FE/gudumap/gudumap/` — the stale pre-cleanup frontend snapshot identified in §7/§16, including its own copies of `gru_local.onnx`, `model_metadata.json`, and `normalization.json` (files deliberately removed from the live `gudumap/` in §6 Task B). Not a size problem, so it wasn't excluded — but committing it means a teammate cloning fresh will see two `gudumap` Android projects and some already-removed files reappearing in the stale one. Purely a content/clarity call, not something this .gitignore pass was asked to resolve.

### Not done

No `git init`, `git add`, `git commit`, or `git push` was run, per instruction — the user runs those manually.
