# evaluation/

Scripts and result plots for the headline metrics used to judge trajectory quality:

| Metric | What it measures |
|---|---|
| ATE (Absolute Trajectory Error) | RMS deviation between estimated and ground-truth trajectory over a full segment |
| RTE (Relative Trajectory Error) | Drift over fixed-length sub-windows (e.g. every 60s) |
| CEP (Circular Error Probable) | Radius within which true position falls with given probability (CEP-50, CEP-90) |
| Drift rate | Position error as % of distance travelled |

- `metrics.py` — ATE / RTE / CEP / drift-rate implementations. Pure functions
  scoring an estimated (x, y, t) track against ground truth; see
  `tests/test_metrics.py` for synthetic known-answer checks. Same field
  names/definitions as backend's mock `/evaluation` response
  (`ate_m`, `rte_m`, `cep50_m`, `cep90_m`, `drift_rate_pct`) so a real result
  from here is directly comparable to (and meant to replace) that mock.
- `run_eval.py` — CLI: scores a baseline trajectory CSV against a
  ground-truth CSV, and optionally a `--corrected` (baseline+residual) CSV
  for side-by-side comparison. Run `python run_eval.py --help` for usage.
  **Does not itself produce those CSVs** — it scores whatever
  physics-baseline/ml-residual output you point it at. As of
  `evaluation/AUDIT_Phase1-4_Report.md`, no real trace/baseline/corrected
  CSV exists yet anywhere in this repo, so this has only ever been run
  against the ml-residual smoke test's synthetic fixture (code-mechanics
  check only — see that audit's Section 1 for why those numbers can't be
  trusted as real accuracy).

Planned:
- `plots/` — generated comparison plots (naive vs. corrected path, error-over-time, confidence radius)
- `results/` — saved metric tables per trace/dataset
