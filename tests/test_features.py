"""Tests for feature reduction and the leakage-free model pipeline."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sepsis_icu import evaluate, features, models


def _toy_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "informative": rng.normal(0, 5, n),      # high variance
            "tiny_var": rng.normal(0, 0.001, n),     # near-constant
            "constant": np.ones(n),                  # zero variance
            "has_nulls": rng.normal(0, 1, n),
        }
    )


def test_drop_zero_variance_removes_constant():
    X = _toy_frame()
    kept_tr, kept_te, cols = features.drop_zero_variance(X, X.copy())
    assert "constant" in cols
    assert "constant" not in kept_tr.columns
    assert list(kept_tr.columns) == list(kept_te.columns)


def test_high_null_dropped_and_remaining_imputed():
    X = _toy_frame()
    X.loc[: int(len(X) * 0.95), "has_nulls"] = np.nan  # >90% missing -> dropped
    X.loc[0, "informative"] = np.nan                    # a stray null -> imputed
    kept_tr, kept_te, high_null = features.drop_high_null_and_impute(X, X.copy())
    assert "has_nulls" in high_null
    assert not kept_tr.isnull().values.any()  # remaining nulls filled


def test_variance_filter_removes_low_variance_on_raw():
    X = _toy_frame().drop(columns=["constant"])
    kept, _, cols = features.variance_filter(X, X.copy(), threshold=0.05)
    assert "informative" in kept.columns
    assert "tiny_var" not in kept.columns   # variance well below 0.05
    assert set(cols) == set(kept.columns)


def test_model_pipeline_order_is_leakage_free():
    """scale -> select -> resample -> classify: selection & SMOTE precede the
    classifier so they are re-fit inside each CV fold."""
    pipe = features.build_model_pipeline(RandomForestClassifier())
    names = [name for name, _ in pipe.steps]
    assert names == ["scaler", "rfe", "smote", "clf"]


def test_deployable_model_trains_and_predicts_on_imbalanced_data():
    """End-to-end smoke: an imbalanced synthetic set flows through the deployable
    fit path (SMOTE + scaler + classifier) and yields calibratable probabilities."""
    rng = np.random.default_rng(1)
    n_neg, n_pos = 180, 20  # 10% positive, like the real sepsis imbalance
    X = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (n_neg, 6)), rng.normal(2.5, 1, (n_pos, 6))]),
        columns=[f"f{i}" for i in range(6)],
    )
    y = pd.Series([0] * n_neg + [1] * n_pos)

    # Minimal fake "search" carrying a fitted pipeline + best params.
    pipe = features.build_model_pipeline(
        RandomForestClassifier(n_estimators=30, random_state=0)
    ).set_params(rfe__n_features_to_select=4)
    pipe.fit(X, y)

    class _Search:
        best_estimator_ = pipe
        best_params_ = {"clf__n_estimators": 30, "rfe__n_features_to_select": 4}

    selected = features.selected_feature_names(pipe, X.columns)
    assert len(selected) == 4

    model = models.deployable_model_from_search(_Search(), X, y, selected)
    prob = evaluate.predict_proba_pos(model, X[list(selected)])
    assert prob.shape == (len(X),)
    assert ((prob >= 0) & (prob <= 1)).all()

    thr, rows = evaluate.choose_threshold(y, prob)
    assert 0.05 <= thr <= 0.5
    assert len(rows) > 0
