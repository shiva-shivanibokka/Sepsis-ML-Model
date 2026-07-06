"""API contract tests. A tiny synthetic model bundle is written to a temp path
so these run in milliseconds without the real dataset or a trained model.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sepsis_icu import config, serve

FEATURES = ["ICULOS_max", "Lactate_max", "Temp_range"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Train a trivial but real pipeline on separable synthetic data. Fit on a
    # *named* DataFrame so the pipeline stores feature_names_in_, exactly like
    # the production model — this is what makes the feature-name warning test
    # meaningful (a bare-array fit would never trigger it).
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (n, 3)), rng.normal(4, 1, (n, 3))]),
        columns=FEATURES,
    )
    y = np.array([0] * n + [1] * n)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", XGBClassifier(n_estimators=20, eval_metric="logloss"))])
    model.fit(X, y)

    bundle_path = tmp_path / "model.joblib"
    import joblib

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": 0.45,
            "model_type": "XGBoost",
            "class_pos": config.CLASS_POS,
            "class_neg": config.CLASS_NEG,
        },
        bundle_path,
    )

    examples_path = tmp_path / "examples.json"
    examples_path.write_text(
        json.dumps({"samples": [], "features": FEATURES, "stats": {},
                    "meta": {"threshold": 0.45}, "model_type": "XGBoost"})
    )

    monkeypatch.setattr(config, "MODEL_PATH", bundle_path)
    monkeypatch.setattr(config, "EXAMPLES_PATH", examples_path)
    monkeypatch.setattr(serve, "_BUNDLE", None)
    return TestClient(serve.app)


def test_index_serves_landing_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "ICU Sepsis Classifier" in r.text


def test_health_reports_model_available(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_available"] is True


def test_model_info_lists_features_and_threshold(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["features"] == FEATURES
    assert body["threshold"] == 0.45


def test_predict_returns_label_and_probability(client):
    r = client.post("/predict", json={"features": {f: 5.0 for f in FEATURES}})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {config.CLASS_POS, config.CLASS_NEG}
    assert 0.0 <= body["probability_sepsis"] <= 1.0
    assert body["threshold"] == 0.45


def test_predict_uses_bundle_threshold_for_label(client):
    """Label must be decided by the bundle's threshold, not a hardcoded 0.5."""
    r = client.post("/predict", json={"features": {f: 5.0 for f in FEATURES}})
    body = r.json()
    expected = config.CLASS_POS if body["probability_sepsis"] >= 0.45 else config.CLASS_NEG
    assert body["prediction"] == expected


def test_predict_rejects_missing_features(client):
    r = client.post("/predict", json={"features": {"ICULOS_max": 1.0}})
    assert r.status_code == 422
    assert "Missing" in r.json()["detail"]


def test_predict_emits_no_feature_name_warning(client):
    """Regression: /predict must not emit sklearn's 'X does not have valid
    feature names' UserWarning, which would pollute the structured JSON logs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = client.post("/predict", json={"features": {f: 5.0 for f in FEATURES}})
    assert r.status_code == 200
    offending = [str(w.message) for w in caught if "feature names" in str(w.message)]
    assert not offending, offending


def test_index_handles_missing_examples(tmp_path, monkeypatch):
    """The landing page must still render (200) when no demo artifacts exist."""
    monkeypatch.setattr(config, "EXAMPLES_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(serve, "_BUNDLE", None)
    r = TestClient(serve.app).get("/")
    assert r.status_code == 200
    assert "ICU Sepsis Classifier" in r.text
