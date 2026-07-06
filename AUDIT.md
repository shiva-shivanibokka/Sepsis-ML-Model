# Repo Audit Report — Sepsis-ML-Model

**Date:** 2026-07-06
**Stack detected:** Python 3.9+ — pandas / numpy / scikit-learn / XGBoost / imbalanced-learn / scikit-optimize (ML); FastAPI / Uvicorn / Pydantic (serving); pytest, Docker, Fly.io/Cloud Run, GitHub Actions (infra). Four Jupyter notebooks (teaching).
**Scope:** `src/sepsis_icu/`, `train.py`, `tests/`, the four notebooks, and infra/config files. The 153 MB raw dataset and generated CSVs are out of scope (data, not code).

## Summary

- Total findings: **7** (0 Critical, 0 Major, 4 Minor, 3 Notes)
- Auto-fixed (trivial-safe): **1**
- Needs review (see `PLAN.md`): **3**
- Notes (documented, no action required): **3**

This is a small, recently-built, tested codebase (18 passing tests, verified end-to-end on the real 40,336-patient dataset). No correctness-critical, security, or data-loss issues were found. The findings below are genuine but minor; nothing blocks use or a portfolio showcase.

## Production-readiness scorecard

| Category | Status | Notes |
|---|---|---|
| Correctness | ⚠️ | One subtle residual leak in *threshold* calibration (does not affect the reported test metrics). |
| Silent failures | ✅ | No bare excepts, no swallowed errors, no masking fallbacks. |
| Security | ⚠️ | No secrets, no injection, trusted-only deserialization. `/predict` has no rate limiting (only relevant once publicly deployed). |
| Concurrency | ⚠️ | Benign check-then-set race on the cached model bundle (worst case: loaded twice, both valid). |
| Performance | ✅ | Stateless inference, model cached in-process, sub-ms predictions. |
| Architecture | ✅ | Clean layering (data → features → models → evaluate → serve); no circular deps; notebooks and CLI share one implementation. |
| Production-readiness | ⚠️ | Structured logging + `/health` present; rate limiting absent; not yet deployed. |
| Test coverage | ⚠️ | Modules + API well covered; `train.py` orchestration and the `RandomizedSearchCV` fallback are only manually verified. |

## Auto-fixed (trivial-safe)

- **`train.py:35`** — removed `import pandas as pd`; it was imported but never used anywhere in the file (confirmed: zero `pd.` references). Dead import, no behavior change.

> Note on other pyflakes hits that were **not** removed (they are intentional, not dead): `config.py:51 import google.colab` is the Colab-probe inside `_running_in_colab()` (`# noqa: F401`), and the `__init__.py` module imports are the package's public re-export surface (`# noqa: F401`). Both are deliberate.

## Findings requiring review

### Correctness — pass 6 (cross-file contract / leakage)

- **`train.py` — `main()` + `_finalize()` (threshold calibration)**
  - **Severity:** Minor
  - **What's wrong:** The decision threshold is chosen on `X_val`, a split held out of training. But the RFE-selected feature set passed to the threshold model was chosen by the search's `best_estimator_`, which `GridSearchCV`/`BayesSearchCV` refit on the **full** `X_train` — including `X_val`. So `X_val` influenced feature *selection*, even though it was excluded from the threshold model's *training*.
  - **Why it matters in production:** The chosen threshold is marginally optimistic — `X_val` isn't fully unseen with respect to feature selection, so the F1-optimal threshold found there can be slightly off the true optimum. **Impact is small and the reported test metrics are unaffected** (the test set is held out from tuning, selection, deployable-model fitting, and threshold choice). Worth fixing for a fully-clean story, or documenting as a known residual.
  - **Suggested fix:** Either (a) document it as a known minor limitation, or (b) do all tuning + selection on `X_tr2`, pick the threshold on `X_val`, then refit the deployable model on the full train with those features/params. See PLAN.md Task 2.

### Test coverage — pass 13

- **`train.py` orchestration + `models.py` fallback**
  - **Severity:** Minor
  - **What's wrong:** `train.py`'s `_finalize`/`_build_demo`/`main` and the `RandomizedSearchCV` fallback in `tune_xgboost` (used when scikit-optimize is absent) have no automated test. The end-to-end training path was verified manually on synthetic and real data but isn't guarded by CI.
  - **Why it matters in production:** A refactor could silently break artifact generation or the demo JSON shape and CI wouldn't catch it. See PLAN.md Task 1 (a fast synthetic smoke test) and Task 3.

### Production-readiness / Security — passes 8, 12

- **`serve.py` — `/predict`**
  - **Severity:** Minor
  - **What's wrong:** No rate limiting on the public prediction endpoint.
  - **Why it matters in production:** Once deployed publicly, `/predict` runs model inference per request and is unauthenticated (appropriate for a demo, but abusable). A simple per-IP limit or a platform-level limit would bound cost/abuse. See PLAN.md Task 4.

## Notes (documented, no fix required)

- **`serve.py` — `_BUNDLE` lazy load** — the `if _BUNDLE is None: _BUNDLE = joblib.load(...)` is a check-then-set race under FastAPI's sync-endpoint threadpool. Benign: the worst case is loading the (identical) model twice on the first concurrent requests; last write wins and every copy is valid. Not worth a lock.
- **`train.py:114` uses `data.train_test_split`** — a transitive re-export of sklearn's `train_test_split` through the `data` module rather than a direct import. Works; mildly fragile (breaks if `data.py` stops importing it). Cosmetic.
- **`evaluate.roc_points` / `evaluate.comparison_frame`** — zero call sites in the repo. They are intended as helpers for the teaching notebooks, but the notebooks don't import the package, so they're currently unused. Left in place as intentional public API (not auto-removed, since deleting plausibly-intended API is higher-risk than a dead one-liner).

## Clean areas

- **`data.py`, `features.py`, `evaluate.py`** — control/data flow, boundary conditions, and cross-file contracts all check out. Label-free reduction, the leakage-free pipeline ordering, and metric computation are correct and unit-tested.
- **Silent failures** — none. No bare/broad excepts, no ignored returns, no masking `.get(...)` fallbacks.
- **Security** — no hardcoded secrets, no `eval`/`exec`/`os.system`/`subprocess`, no untrusted deserialization (the only `joblib.load` reads a self-produced artifact).
- **`serve.py` request handling** — missing-feature → `422`, non-numeric → Pydantic `422`, named-DataFrame construction avoids the sklearn feature-name warning (regression-tested).
- **Notebooks** — the threshold-on-validation and NB2-summary fixes from the prior audit are consistent; the replaced cells reference only variables defined earlier in each notebook.
