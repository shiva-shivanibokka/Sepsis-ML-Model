# Fix Plan — Sepsis-ML-Model

Generated from repo-bug-audit on 2026-07-06. 4 tasks. All findings are **Minor** — none block use. Ordered so the test comes before the correctness fix it guards (TDD), then the two independent hardening items.

---

## Task 1: Add a fast end-to-end smoke test for `train.py` orchestration

- **File:** `tests/test_train.py` (new)
- **Category:** Test coverage (pass 13)
- **Severity:** Minor
- **Finding:** `train.py`'s `_finalize` / `_build_demo` and the artifact-writing path have no automated test; a refactor could silently break `model.joblib` / `examples.json` generation.
- **Why it matters:** CI would not catch a broken training run or a changed demo-JSON shape (which the frontend depends on).
- **Proposed change:** Add a test that runs the real orchestration functions on a tiny synthetic imbalanced dataset (no `main()`/CLI, no real CSV), asserting the bundle + demo dict shapes. Reuse the pattern already proven in `tests/test_features.py::test_deployable_model_trains_and_predicts_on_imbalanced_data`.
  ```python
  # tests/test_train.py
  import numpy as np, pandas as pd
  from sklearn.ensemble import RandomForestClassifier
  from sepsis_icu import features, models, evaluate
  import train

  def _tiny():
      rng = np.random.default_rng(0)
      n_neg, n_pos = 160, 24
      X = pd.DataFrame(np.vstack([rng.normal(0,1,(n_neg,6)), rng.normal(2.5,1,(n_pos,6))]),
                       columns=[f"f{i}" for i in range(6)])
      y = pd.Series([0]*n_neg + [1]*n_pos)
      return X, y

  def test_finalize_produces_valid_model_and_metrics():
      X, y = _tiny()
      from sklearn.model_selection import train_test_split
      Xtr2, Xval, ytr2, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
      pipe = features.build_model_pipeline(
          RandomForestClassifier(n_estimators=25, random_state=0)
      ).set_params(rfe__n_features_to_select=4)
      pipe.fit(X, y)
      class _S:  # minimal fake search
          best_estimator_ = pipe
          best_params_ = {"clf__n_estimators": 25, "rfe__n_features_to_select": 4}
      model, feats, thr, metrics = train._finalize(_S(), X, y, Xtr2, ytr2, Xval, yval, X, y)
      assert len(feats) == 4
      assert 0.05 <= thr <= 0.5
      assert set(("f1","roc_auc","precision","recall","confusion")).issubset(metrics)

  def test_build_demo_shape():
      X, y = _tiny()
      pipe = features.build_model_pipeline(
          RandomForestClassifier(n_estimators=25, random_state=0)
      ).set_params(rfe__n_features_to_select=4)
      pipe.fit(X, y)
      from sklearn.pipeline import Pipeline
      from sklearn.preprocessing import StandardScaler
      dep = Pipeline([("scaler", StandardScaler()), ("clf", pipe.named_steps["clf"])])
      feats = list(features.selected_feature_names(pipe, X.columns))
      dep.fit(X[feats], y)
      prob = evaluate.predict_proba_pos(dep, X[feats])
      m = evaluate.evaluate_at_threshold(y, prob, 0.5)
      demo = train._build_demo(dep, feats, 0.5, m, "Random Forest", X, X, y, 6, 10)
      assert demo["features"] == feats
      assert demo["meta"]["confusion"] == m["confusion"]
      assert all("weight" in t for t in demo["top_features"])
  ```
- **Verification:** `pytest tests/test_train.py -q` (should pass; run full `pytest -q` to confirm nothing else breaks).
- **Depends on:** none.

---

## Task 2: Remove the residual threshold-calibration leak

- **File:** `train.py` (`main()` and `_finalize()`)
- **Category:** Correctness / cross-file contract (pass 6)
- **Severity:** Minor
- **Finding:** The threshold is chosen on `X_val`, but the RFE feature set it uses was selected by the search refit on the full `X_train` (which includes `X_val`). `X_val` is not fully unseen with respect to feature selection.
- **Why it matters:** The chosen threshold is marginally optimistic. Reported **test** metrics are unaffected, so impact is low — but a portfolio "no leakage anywhere" claim should be exact.
- **Proposed change (rigorous option):** Tune + select on `X_tr2` only; pick the threshold on `X_val`; refit the deployable model on the full `X_train` with the selected features/params. Concretely, in `main()` run `tune_random_forest`/`tune_xgboost` on `(X_tr2, y_tr2)` instead of `(X_train, y_train)`, keep `_finalize`'s deployable refit on the full `X_train`, and drop the separate `thr_model` (the search's own `best_estimator_` on `X_tr2` already excludes `X_val`, so `predict_proba` on `X_val` is clean). Trade-off: features are selected on 80% of train rather than 100% — negligible at 32k rows.
  - **Cheap alternative:** keep the code and add one sentence to the README/`docs/architecture.md` noting the threshold validation set overlaps feature selection (a minor residual, test metrics unaffected).
- **Verification:** `python train.py --xgb-iters 25` completes; `artifacts/metrics.json` still shows XGBoost F1 ≈ 0.67 (the number may shift by ≤0.01 and the threshold may change slightly — that is expected and correct). Re-run costs ~13 min.
- **Depends on:** Task 1 (have the smoke test in place first so the refactor is guarded).

---

## Task 3: Test the `RandomizedSearchCV` fallback path in `tune_xgboost`

- **File:** `tests/test_models.py` (new)
- **Category:** Test coverage (pass 13)
- **Severity:** Minor
- **Finding:** `tune_xgboost` falls back to `RandomizedSearchCV` when scikit-optimize is missing; that branch is never exercised.
- **Why it matters:** Deploy/CI environments without scikit-optimize would hit untested code; a param-space typo would surface only there.
- **Proposed change:** Monkeypatch `skopt` import to force the fallback, run `tune_xgboost` with `n_iter=2` on the tiny synthetic set, assert it returns a fitted search with `best_estimator_`.
  ```python
  # tests/test_models.py
  import builtins, numpy as np, pandas as pd, pytest
  from sepsis_icu import models

  def test_xgboost_randomizedsearch_fallback(monkeypatch):
      real_import = builtins.__import__
      def fake_import(name, *a, **k):
          if name.startswith("skopt"):
              raise ImportError("forced")
          return real_import(name, *a, **k)
      monkeypatch.setattr(builtins, "__import__", fake_import)
      rng = np.random.default_rng(0)
      X = pd.DataFrame(np.vstack([rng.normal(0,1,(120,6)), rng.normal(2.5,1,(18,6))]),
                       columns=[f"f{i}" for i in range(6)])
      y = pd.Series([0]*120 + [1]*18)
      search = models.tune_xgboost(X, y, n_iter=2)
      assert hasattr(search, "best_estimator_")
  ```
- **Verification:** `pytest tests/test_models.py -q`.
- **Depends on:** none.

---

## Task 4: Add basic rate limiting to `/predict` (only before a public deploy)

- **File:** `src/sepsis_icu/serve.py`, `pyproject.toml`
- **Category:** Security / production-readiness (passes 8, 12)
- **Severity:** Minor
- **Finding:** The public `/predict` endpoint has no rate limiting.
- **Why it matters:** Once deployed unauthenticated, it can be hammered (each call runs model inference). Fine locally; a gap for a real public URL.
- **Proposed change:** Add `slowapi` (a lightweight FastAPI limiter) and cap `/predict`, e.g. `@limiter.limit("60/minute")`. Alternatively rely on the platform (Cloud Run concurrency / a WAF) and just document that — no code change. Recommend documenting for now, wiring `slowapi` only when deploying.
  ```python
  # sketch, in serve.py
  from slowapi import Limiter
  from slowapi.util import get_remote_address
  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter
  # @app.post("/predict") ... add: @limiter.limit("60/minute")
  ```
- **Verification:** hit `/predict` 61×/min in a test and expect a `429`; confirm `/health` is unlimited.
- **Depends on:** none. Defer until deployment.
