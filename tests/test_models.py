"""Tests for the tuning entry points, including the scikit-optimize-absent path."""

import builtins

import numpy as np
import pandas as pd

from sepsis_icu import models


def _tiny_imbalanced():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (120, 6)), rng.normal(2.5, 1, (18, 6))]),
        columns=[f"f{i}" for i in range(6)],
    )
    y = pd.Series([0] * 120 + [1] * 18)
    return X, y


def test_xgboost_randomizedsearch_fallback(monkeypatch):
    """When scikit-optimize is unavailable, tune_xgboost must fall back to
    RandomizedSearchCV and still return a fitted search. This branch is otherwise
    never exercised (skopt is installed in dev/CI)."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("skopt"):
            raise ImportError("forced: skopt unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    X, y = _tiny_imbalanced()
    search = models.tune_xgboost(X, y, n_iter=2)

    from sklearn.model_selection import RandomizedSearchCV

    assert isinstance(search, RandomizedSearchCV)
    assert hasattr(search, "best_estimator_")
    assert 0.0 <= search.best_score_ <= 1.0
