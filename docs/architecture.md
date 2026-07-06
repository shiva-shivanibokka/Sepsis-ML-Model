# Architecture & Design Decisions

How the project fits together, plus the decisions that aren't obvious from the
code. Doubles as a lightweight set of Architecture Decision Records (ADRs).

## System overview

```
                    ┌──────────────────────────────────────────────┐
                    │            src/sepsis_icu/ (package)           │
                    │                                                │
 Sepsis_Dataset ──► │  data.py ──────► features.py ──► models.py     │
 (hourly records)   │   load           label-free      leakage-free  │
                    │   aggregate      reduction        tuning        │
                    │   split               │              │          │
                    │                       ▼              ▼          │
                    │                 evaluate.py    train.py (CLI)   │
                    │                 (metrics +          │           │
                    │                  threshold)         │           │
                    └─────────────────────────────────────┼──────────┘
                                                           ▼
                        artifacts/model.joblib + metrics.json + examples.json
                                                           │
                              ┌────────────────────────────┼──────────┐
                              │                             ▼          │
   01–04 notebooks ──────────┤ import           serve.py (FastAPI)    │
   (teaching narrative,      │ the same         /health /model        │
    import from the package) │ modules          /predict  /           │
                              └─────────────────────────────────────────┘
                                             │
                                    Dockerfile (container)
```

The four notebooks and the training CLI import the **same** modules, so there is
one implementation of every step. Notebooks are the readable narrative; the
package is the source of truth.

## Data flow

1. **Load & aggregate** (`data.py`) — read the hourly PhysioNet CSV, collapse each
   patient's stay into 173 summary features (min/max/mean/std/range per vital &
   lab, plus Age, Gender, ICULOS_max) and an "ever septic" target. Stratified
   80/20 train/test split with a fixed seed.
2. **Label-free reduction** (`features.py`) — zero-variance filter, high-null
   filter + median impute, and a variance filter on the raw imputed values. None
   use the target, so they run once on the training set without leaking anything.
3. **Leakage-free supervised selection + tuning** (`models.py`) — an imbalanced-learn
   `Pipeline` (`scale → RFE → SMOTE → classifier`) is tuned with `GridSearchCV`
   (RF) / `BayesSearchCV` (XGB). Because selection *and* SMOTE live inside the
   pipeline, both are re-fit within every CV fold, and the number of selected
   features is tuned as a hyperparameter.
4. **Threshold calibration** (`evaluate.py`) — the decision threshold is chosen on
   a validation split carved out of training, never on the test set.
5. **Deploy** — the features RFE chose are used to fit a compact
   `StandardScaler → classifier` on the SMOTE-balanced training set. This is what's
   saved and served, so the API takes ~50 feature values instead of 173.
6. **Serve** (`serve.py`) — FastAPI loads the bundle and answers `/predict`.

---

## ADR-001: Move supervised selection and SMOTE inside the CV pipeline

**Context.** The original notebooks ran RFE once on the whole training set and
applied SMOTE once to the whole training set, then cross-validated. Every CV
fold's model was therefore built on features chosen with — and synthetic samples
interpolated from — data that included the fold's own validation rows. That is
selection/resampling leakage, which inflates CV scores relative to the honest
test score.

**Decision.** Wrap `scale → RFE → SMOTE → classifier` in an
`imblearn.pipeline.Pipeline` and tune it with the search cross-validator, so
selection and oversampling are re-fit per fold (SMOTE resamples only each fold's
training portion). The number of selected features
(`rfe__n_features_to_select`) becomes a tuned hyperparameter.

**Consequences.** CV scores become honest and track the test score. Tuning is
slower (RFE + SMOTE run inside every fold); mitigated by a small RFE candidate
set and a lightweight RF ranker.

## ADR-002: Calibrate the decision threshold on a validation split, not the test set

**Context.** The notebooks originally swept the decision threshold directly on the
test set and reported the F1 at the best one — the same class of leak fixed for
RFE in Notebook 2, reintroduced for the threshold. The reported best-threshold
F1 was optimistic.

**Decision.** `evaluate.choose_threshold` runs on a validation split held out of
training. `train.py` fits a threshold-selection model on that reduced training
subset, picks the F1-optimal threshold on the held-out validation rows, bakes it
into the model bundle, and measures the test set **once** at that threshold. A
test (`test_predict_uses_bundle_threshold_for_label`) locks in that the served
label uses the calibrated threshold, not a hardcoded 0.5.

## ADR-003: Serve a compact model on selected features, not the full pipeline

**Context.** The tuned pipeline expects the full 173-feature matrix. An API
requiring 173 values per request is impractical, and most are redundant.

**Decision.** After tuning identifies the best features and hyperparameters, fit a
small `StandardScaler → classifier` on just those features (SMOTE applied once)
and save that. The served contract is ~50 feature values. Reported metrics are
for the model that is actually deployed — you report what you ship.

## ADR-004: Config-driven Colab/local portability

**Context.** The original notebooks hard-coded a Google Drive path and imported
`google.colab`, so they ran only on Colab — and the README's documented path
didn't even match the code.

**Decision.** `config.py` auto-detects Colab vs local, resolves the dataset from
the repo root locally (or Drive on Colab), and allows env-var overrides
(`SEPSIS_DATA_DIR`, `SEPSIS_ARTIFACTS_DIR`). `mount_drive_if_colab()` is a no-op
off Colab.

---

## Known limitation: retrospective, not real-time

Features aggregate a patient's **entire** ICU stay, including hours after sepsis
onset, and the target is "ever septic". So this is a *retrospective whole-stay
classifier*, not a real-time early-warning system. A genuine early predictor
would aggregate only the measurements available up to a fixed prediction time
(e.g. the first 6–12 ICU hours) and re-run the whole pipeline. This is the single
most valuable extension. See the README "Limitations" section.

## What happens at 10× load?

The service is stateless — the model is loaded once and cached in-process, and a
prediction is one scaled forward pass over a few hundred shallow trees, i.e.
sub-millisecond. Scaling out is horizontal: run N replicas behind a load
balancer; `/health` gives the orchestrator its readiness signal. The model
artifact is baked into the image at build time, so a retrain is `python train.py`
+ a source redeploy; for fast local iteration you can mount over the baked-in
file (`-v "$PWD/artifacts:/app/artifacts"`). The realistic bottleneck at high
load is per-request JSON (de)serialization, not inference.

## Testing strategy

- **`test_data.py`** — the hourly→per-patient aggregation: one row per patient,
  "ever septic" target, range and ICULOS_max features.
- **`test_features.py`** — label-free filters, leakage-free pipeline ordering, and
  an end-to-end smoke that trains the deployable path on imbalanced synthetic data.
- **`test_api.py`** — the `/health`, `/model`, `/predict` contracts (including the
  bundle-threshold and no-feature-name-warning regressions) against a tiny
  synthetic model, so the suite runs without the dataset or a real train.

Run with `pytest`.
