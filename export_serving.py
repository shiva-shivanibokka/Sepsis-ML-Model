#!/usr/bin/env python
"""Export the trained pipeline into a form the deployed service can serve.

`train.py` saves a pickled scikit-learn Pipeline (StandardScaler -> XGBClassifier).
Loading that pickle in production drags in scikit-learn, pandas, scipy, joblib and
imbalanced-learn, and — worse — ties the running service to the exact library
versions that wrote it. The pinned training version of XGBoost is not even on
PyPI, so a `pip install` deployment cannot reproduce it.

So the serving copy is written twice over, in formats with no pickle in them:

    artifacts/model.ubj    XGBoost's own binary format, explicitly stable
                           across library versions
    artifacts/serving.json the scaler's mean_ / scale_ arrays, the feature
                           order, the decision threshold, the class names

Standardising is then two numpy operations, so the deployed service needs
xgboost and numpy and nothing else.

This is only safe if it scores identically. The script re-scores every baked
demo patient plus a block of random rows through both the original pipeline and
the exported pair, and refuses to write anything unless they agree.

    python export_serving.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from sepsis_icu import config  # noqa: E402

# The exported path does the same arithmetic in the same float32 booster, so the
# bar is exact agreement, not "close enough". Anything above zero means the
# export changed the model and needs explaining, not tolerating.
PROB_TOL = 0.0
N_RANDOM_ROWS = 2000


def _score_exported(booster, mean, scale, X: np.ndarray) -> np.ndarray:
    """P(sepsis) the way the deployed service will compute it."""
    z = ((X - mean) / scale).astype(np.float32)
    return np.asarray(booster.inplace_predict(z), dtype=np.float64)


def _sample_rows(features: list[str], stats: dict) -> np.ndarray:
    """Random rows spread across each feature's own training range.

    Sampling around zero would leave most of the 196 trees' split points
    untouched and the comparison would prove very little.
    """
    rng = np.random.default_rng(0)
    mu = np.array([stats.get(f, {}).get("mean", 0.0) for f in features])
    sd = np.array([max(stats.get(f, {}).get("std", 1.0), 1e-6) for f in features])
    return rng.normal(mu, sd * 2.5, size=(N_RANDOM_ROWS, len(features)))


def verify(pipe, features, booster, mean, scale, threshold: float) -> bool:
    demo = json.loads(config.EXAMPLES_PATH.read_text()) if config.EXAMPLES_PATH.exists() else {}
    demo_rows = [[float(s["features"][f]) for f in features] for s in demo.get("samples", [])]

    X = np.vstack([
        np.asarray(demo_rows, dtype=np.float64).reshape(-1, len(features)),
        _sample_rows(features, demo.get("stats", {})),
    ])

    ref = pipe.predict_proba(X)[:, 1]
    got = _score_exported(booster, mean, scale, X)

    delta = np.abs(ref - got)
    mismatches = int(((ref >= threshold) != (got >= threshold)).sum())

    print(f"  compared {len(demo_rows)} demo patients + {N_RANDOM_ROWS} random rows")
    print(f"  max |delta P(sepsis)| : {delta.max():.3e}")
    print(f"  label flips at {threshold:.2f}   : {mismatches}")

    ok = delta.max() <= PROB_TOL and mismatches == 0
    print("  RESULT                : " + ("PASS" if ok else "FAIL"))
    return ok


def main() -> int:
    if not config.MODEL_PATH.exists():
        print(f"No model at {config.MODEL_PATH}. Run `python train.py` first.", file=sys.stderr)
        return 1

    bundle = joblib.load(config.MODEL_PATH)
    pipe = bundle["model"]
    features = list(bundle["features"])
    threshold = float(bundle.get("threshold", 0.5))

    scaler = pipe.named_steps["scaler"]
    booster = pipe.named_steps["clf"].get_booster()
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)

    print(f"loaded {bundle['model_type']} pipeline, {len(features)} features, "
          f"threshold {threshold}")

    tmp = config.MODEL_UBJ_PATH.with_suffix(".ubj.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(tmp))

    reloaded = xgb.Booster()
    reloaded.load_model(str(tmp))

    # Verify against the *reloaded* booster, not the in-memory one: the point of
    # the check is that the round-trip through the file preserves the model.
    if not verify(pipe, features, reloaded, mean, scale, threshold):
        tmp.unlink(missing_ok=True)
        print("\nExported model disagrees with the pipeline — nothing written.", file=sys.stderr)
        return 1

    tmp.replace(config.MODEL_UBJ_PATH)
    config.SERVING_META_PATH.write_text(json.dumps({
        "features": features,
        "scaler_mean": [float(v) for v in mean],
        "scaler_scale": [float(v) for v in scale],
        "threshold": threshold,
        "model_type": bundle["model_type"],
        "class_pos": bundle["class_pos"],
        "class_neg": bundle["class_neg"],
    }, indent=2))

    print(f"\nwrote {config.MODEL_UBJ_PATH.name}      "
          f"{config.MODEL_UBJ_PATH.stat().st_size / 1e6:.2f} MB "
          f"(joblib was {config.MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {config.SERVING_META_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
