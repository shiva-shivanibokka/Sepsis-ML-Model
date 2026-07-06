"""Evaluation metrics and threshold calibration, shared by notebooks and train.py.

Keeping this logic in one place means Notebooks 3-4 and ``train.py`` compute
metrics identically, so their numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def predict_proba_pos(fitted_estimator, X) -> np.ndarray:
    """Return P(sepsis) for a pipeline trained on 1/0 labels (1 = sepsis)."""
    pos_idx = list(fitted_estimator.classes_).index(1)
    return fitted_estimator.predict_proba(X)[:, pos_idx]


def evaluate_at_threshold(y_true, y_prob_pos, threshold: float = 0.5) -> dict:
    """Metric bundle for the positive (sepsis) class at a given decision threshold.

    Inputs are binary 1/0 (1 = sepsis). Returns a JSON-serializable dict.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob_pos) >= threshold).astype(int)

    # labels=[1, 0] => rows/cols are [sepsis, no-sepsis]; ravel is [TP, FN, FP, TN].
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = (int(v) for v in cm.ravel())

    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_pos)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def choose_threshold(
    y_true, y_prob_pos, grid: list[float] | None = None
) -> tuple[float, list[dict]]:
    """Pick the F1-maximising decision threshold on a VALIDATION set.

    Never call this on the test set: choosing a threshold on the same data you
    then report on is a subtle leak. Returns ``(best_threshold, sweep_rows)``.
    """
    grid = grid or [i / 100 for i in range(5, 55, 5)]
    y_true = np.asarray(y_true)
    rows = []
    for t in grid:
        y_pred = (np.asarray(y_prob_pos) >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    best = max(rows, key=lambda r: r["f1"])["threshold"]
    return best, rows


def roc_points(y_true, y_prob_pos):
    """FPR/TPR arrays for plotting an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos, pos_label=1)
    return fpr, tpr


def comparison_frame(rf_metrics: dict, xgb_metrics: dict) -> pd.DataFrame:
    """Side-by-side metrics table for the two models."""
    rows = {"Random Forest": rf_metrics, "XGBoost": xgb_metrics}
    return pd.DataFrame(
        {
            name: {
                "F1 Score": round(m["f1"], 4),
                "ROC-AUC": round(m["roc_auc"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
            }
            for name, m in rows.items()
        }
    ).T


def winner_by_f1(rf_metrics: dict, xgb_metrics: dict) -> str:
    return "XGBoost" if xgb_metrics["f1"] >= rf_metrics["f1"] else "Random Forest"
