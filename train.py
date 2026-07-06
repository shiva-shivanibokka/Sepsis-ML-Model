#!/usr/bin/env python
"""End-to-end training CLI.

Runs the whole pipeline the notebooks describe, but headless and reproducible:

    raw hourly CSV -> aggregate to one row per patient -> stratified split
                   -> label-free reduction (zero-var, high-null+impute, variance)
                   -> leakage-free tuning of Random Forest and XGBoost
                      (RFE + SMOTE inside every CV fold)
                   -> threshold calibrated on a validation split (never the test set)
                   -> evaluate both on the held-out test set
                   -> save the deployable model + selected features + metrics.json

The deployed model is a compact ``StandardScaler -> classifier`` trained on only
the features RFE selected, so the serving API takes a few dozen values rather
than all 173 aggregated features.

Usage:
    python train.py                  # full run, saves artifacts/
    python train.py --xgb-iters 10   # faster Bayesian search for a quick smoke test
    python train.py --demo-samples 80  # how many held-out stays to bake into the UI
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sepsis_icu import config, data, evaluate, features, models

# Silence only known-harmless deprecation noise from the sklearn/skopt/xgboost
# stack so genuine signals (e.g. ConvergenceWarning) still surface in the log.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def _finalize(search, X_train_red, y_train, X_tr2, y_tr2, X_val, y_val, X_test_red, y_test):
    """From a fitted search: recover features, fit the deployable model, calibrate
    the threshold on the validation split, and score on the held-out test set."""
    selected = features.selected_feature_names(search.best_estimator_, X_train_red.columns)

    # Deployable model: refit on the FULL training set (SMOTE-balanced).
    deployable = models.deployable_model_from_search(
        search, X_train_red, y_train, selected
    )
    # Threshold model: refit on tr2 only, so val is genuinely held out for
    # threshold selection (choosing the threshold on the test set would leak).
    thr_model = models.deployable_model_from_search(search, X_tr2, y_tr2, selected)
    val_prob = evaluate.predict_proba_pos(thr_model, X_val[list(selected)])
    threshold, _ = evaluate.choose_threshold(y_val, val_prob)

    test_prob = evaluate.predict_proba_pos(deployable, X_test_red[list(selected)])
    metrics = evaluate.evaluate_at_threshold(y_test, test_prob, threshold)
    return deployable, list(selected), threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ICU sepsis classifier.")
    parser.add_argument("--xgb-iters", type=int, default=25,
                        help="Bayesian search iterations for XGBoost.")
    parser.add_argument("--variance-threshold", type=float, default=0.01,
                        help="Variance filter threshold on raw imputed features.")
    parser.add_argument("--demo-samples", type=int, default=80,
                        help="Held-out stays (stratified) baked into the demo UI.")
    args = parser.parse_args()

    t0 = time.time()
    print(config.describe())
    config.ensure_artifacts_dir()

    # --- Load + aggregate to one row per patient -----------------------------
    _log("loading raw hourly dataset ...")
    raw = data.load_raw()
    patient_df = data.aggregate_patients(raw)
    patient_df.to_csv(config.PATIENT_CSV)
    n_cand = patient_df.shape[1] - 1
    _log(f"aggregated: {patient_df.shape[0]:,} patients x {n_cand} candidate features "
         f"(sepsis rate {patient_df[config.TARGET_COL].mean()*100:.1f}%)")

    # --- Split ---------------------------------------------------------------
    X, y = data.split_features_target(patient_df)
    X_train, X_test, y_train, y_test = data.make_split(X, y)
    _log(f"split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")

    # --- Label-free reduction (no target used) -------------------------------
    X_train, X_test, zero_cols = features.drop_zero_variance(X_train, X_test)
    X_train, X_test, null_cols = features.drop_high_null_and_impute(X_train, X_test)
    X_train, X_test, kept = features.variance_filter(
        X_train, X_test, threshold=args.variance_threshold
    )
    _log(f"label-free reduction: {n_cand} -> {X_train.shape[1]} features "
         f"(zero-var -{len(zero_cols)}, high-null -{len(null_cols)}, "
         f"low-var -{n_cand - len(zero_cols) - len(null_cols) - X_train.shape[1]})")

    # Persist the reduced split so the notebooks can reload it identically.
    X_train.to_csv(config.X_TRAIN_CSV)
    X_test.to_csv(config.X_TEST_CSV)
    y_train.to_csv(config.Y_TRAIN_CSV)
    y_test.to_csv(config.Y_TEST_CSV)

    # Validation split (out of training) for threshold calibration only.
    X_tr2, X_val, y_tr2, y_val = data.train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train,
        random_state=config.RANDOM_SEED,
    )

    # --- Tune + finalize each model, one at a time. Finalizing (and freeing)
    # each search before starting the next keeps only one large search object in
    # memory at a time — fitting both final models while both searches were still
    # alive was a peak-memory spike that could get the process killed.
    _log("tuning Random Forest (GridSearchCV) ...")
    rf_search = models.tune_random_forest(X_train, y_train)
    _log(f"  best CV F1={rf_search.best_score_:.4f}  params={rf_search.best_params_}")
    rf_cv_f1, rf_best_params = float(rf_search.best_score_), dict(rf_search.best_params_)
    rf_model, rf_feats, rf_thr, rf_metrics = _finalize(
        rf_search, X_train, y_train, X_tr2, y_tr2, X_val, y_val, X_test, y_test
    )
    del rf_search
    gc.collect()

    _log(f"tuning XGBoost (BayesSearchCV, {args.xgb_iters} iters) ...")
    xgb_search = models.tune_xgboost(X_train, y_train, n_iter=args.xgb_iters)
    _log(f"  best CV F1={xgb_search.best_score_:.4f}  params={dict(xgb_search.best_params_)}")
    xgb_cv_f1, xgb_best_params = float(xgb_search.best_score_), dict(xgb_search.best_params_)
    xgb_model, xgb_feats, xgb_thr, xgb_metrics = _finalize(
        xgb_search, X_train, y_train, X_tr2, y_tr2, X_val, y_val, X_test, y_test
    )
    del xgb_search
    gc.collect()

    winner = evaluate.winner_by_f1(rf_metrics, xgb_metrics)
    if winner == "XGBoost":
        best_model, best_feats, best_thr, best_metrics = (
            xgb_model, xgb_feats, xgb_thr, xgb_metrics)
    else:
        best_model, best_feats, best_thr, best_metrics = (
            rf_model, rf_feats, rf_thr, rf_metrics)

    # --- Demo data for the UI (baked into the serving image) -----------------
    demo = _build_demo(
        best_model, best_feats, best_thr, best_metrics, winner,
        X_train, X_test, y_test, n_cand, args.demo_samples,
    )
    config.EXAMPLES_PATH.write_text(json.dumps(demo))

    # --- Persist artifacts ---------------------------------------------------
    joblib.dump(
        {
            "model": best_model,
            "features": list(best_feats),
            "threshold": float(best_thr),
            "model_type": winner,
            "class_pos": config.CLASS_POS,
            "class_neg": config.CLASS_NEG,
        },
        config.MODEL_PATH,
    )

    metrics = {
        "random_forest": {**rf_metrics, "cv_f1": rf_cv_f1,
                          "best_params": _jsonable_params(rf_best_params),
                          "n_features": len(rf_feats), "threshold": float(rf_thr)},
        "xgboost": {**xgb_metrics, "cv_f1": xgb_cv_f1,
                    "best_params": _jsonable_params(xgb_best_params),
                    "n_features": len(xgb_feats), "threshold": float(xgb_thr)},
        "winner": winner,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_candidates": int(n_cand),
        "n_features_after_label_free_reduction": int(X_train.shape[1]),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    _log(f"saved model -> {config.MODEL_PATH} ({winner}, {len(best_feats)} features, "
         f"threshold {best_thr:.2f})")
    _log(f"saved metrics -> {config.METRICS_PATH}")
    _print_summary(rf_metrics, xgb_metrics, winner)
    _log(f"done in {time.time() - t0:.0f}s")


def _build_demo(model, feats, thr, metrics, winner, X_train, X_test, y_test,
                n_cand, n_samples) -> dict:
    """Assemble the JSON the landing page reads: importances, per-feature stats,
    and a stratified subset of real held-out stays for the picker."""
    clf = model.named_steps["clf"]
    importances = getattr(clf, "feature_importances_", np.zeros(len(feats)))
    top = sorted(
        ({"feature": f, "weight": round(float(w), 4)} for f, w in zip(feats, importances)),
        key=lambda d: abs(d["weight"]), reverse=True,
    )[:8]

    cm = metrics["confusion"]
    accuracy = (cm["tp"] + cm["tn"]) / max(1, sum(cm.values()))

    # Stratified subset of test stays for the demo picker (the full test set can
    # be thousands of rows — too large to bake into the image or ship to the UI).
    y_test = y_test.reset_index(drop=True)
    X_sel = X_test[feats].reset_index(drop=True)
    per_class = max(1, n_samples // 2)
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = []
    for cls in (1, 0):
        pool = y_test.index[y_test == cls].to_numpy()
        take = min(per_class, len(pool))
        idx.extend(rng.choice(pool, size=take, replace=False).tolist())
    rng.shuffle(idx)

    samples = [
        {"label": config.CLASS_POS if y_test[i] == 1 else config.CLASS_NEG,
         "features": {f: round(float(X_sel.iloc[i][f]), 4) for f in feats}}
        for i in idx
    ]

    return {
        "features": list(feats),
        "model_type": winner,
        "top_features": top,
        "meta": {
            "roc_auc": round(metrics["roc_auc"], 4),
            "f1": round(metrics["f1"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "accuracy": round(accuracy, 4),
            "threshold": round(float(thr), 4),
            "confusion": cm,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_candidates": int(n_cand),
            "n_demo_samples": len(samples),
        },
        "stats": {
            f: {"mean": round(float(X_train[f].mean()), 4),
                "std": round(float(X_train[f].std()) or 1.0, 4)}
            for f in feats
        },
        "samples": samples,
    }


def _jsonable_params(params) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(params).items()}


def _print_summary(rf_metrics, xgb_metrics, winner) -> None:
    print("\n" + "=" * 52)
    print(f"{'Metric':<12}{'RandForest':>12}{'XGBoost':>14}")
    print("-" * 52)
    for key, label in [("f1", "F1"), ("roc_auc", "ROC-AUC"),
                       ("precision", "Precision"), ("recall", "Recall")]:
        print(f"{label:<12}{rf_metrics[key]:>12.4f}{xgb_metrics[key]:>14.4f}")
    print("-" * 52)
    print(f"Winner by F1: {winner}")
    print("=" * 52)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
