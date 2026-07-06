"""End-to-end smoke tests for train.py orchestration (_finalize, _build_demo).

Runs the real orchestration functions on a tiny synthetic imbalanced dataset —
no CLI, no real CSV — so a refactor that breaks artifact/demo generation is
caught by CI. Kept fast by using a small pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train
from sepsis_icu import evaluate, features


def _tiny():
    rng = np.random.default_rng(0)
    n_neg, n_pos = 160, 24  # ~13% positive, like the sepsis imbalance
    X = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (n_neg, 6)), rng.normal(2.5, 1, (n_pos, 6))]),
        columns=[f"f{i}" for i in range(6)],
    )
    y = pd.Series([0] * n_neg + [1] * n_pos)
    return X, y


def _fitted_fake_search(X, y, k=4):
    pipe = features.build_model_pipeline(
        RandomForestClassifier(n_estimators=25, random_state=0)
    ).set_params(rfe__n_features_to_select=k)
    pipe.fit(X, y)

    class _Search:
        best_estimator_ = pipe
        best_params_ = {"clf__n_estimators": 25, "rfe__n_features_to_select": k}

    return _Search()


def test_finalize_produces_valid_model_and_metrics():
    X, y = _tiny()
    # Search fitted on tr2 only, exactly as main() now does.
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    search = _fitted_fake_search(X_tr2, y_tr2, k=4)

    model, feats, thr, metrics = train._finalize(
        search, X, y, X_val, y_val, X, y
    )
    assert len(feats) == 4
    assert 0.05 <= thr <= 0.5
    assert {"f1", "roc_auc", "precision", "recall", "confusion"}.issubset(metrics)
    # deployable model predicts calibrated probabilities on the selected features
    prob = evaluate.predict_proba_pos(model, X[feats])
    assert ((prob >= 0) & (prob <= 1)).all()


def test_build_demo_shape():
    X, y = _tiny()
    search = _fitted_fake_search(X, y, k=4)
    feats = list(features.selected_feature_names(search.best_estimator_, X.columns))

    deployable = Pipeline(
        [("scaler", StandardScaler()), ("clf", search.best_estimator_.named_steps["clf"])]
    )
    deployable.fit(X[feats], y)
    prob = evaluate.predict_proba_pos(deployable, X[feats])
    metrics = evaluate.evaluate_at_threshold(y, prob, 0.5)

    demo = train._build_demo(
        deployable, feats, 0.5, metrics, "Random Forest", X, X, y, n_cand=6, n_samples=10
    )
    assert demo["features"] == feats
    assert demo["meta"]["confusion"] == metrics["confusion"]
    assert demo["meta"]["threshold"] == 0.5
    assert all("weight" in t and "feature" in t for t in demo["top_features"])
    # every demo sample carries all selected features and a valid label
    assert demo["samples"], "expected at least one demo sample"
    for s in demo["samples"]:
        assert set(s["features"]) == set(feats)
        assert s["label"] in {"Sepsis", "No sepsis"}
