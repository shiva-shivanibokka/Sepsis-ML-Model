"""FastAPI serving layer for the trained sepsis classifier.

Exposes:

    GET  /         -> interactive demo landing page (an ICU observation sheet)
    GET  /health   -> liveness/readiness probe
    GET  /model    -> metadata: model type, the features it expects, threshold
    POST /predict  -> {feature: value, ...} -> predicted class + P(sepsis)

The model is loaded from the pair `export_serving.py` writes — XGBoost's own
``model.ubj`` plus the scaler's arrays in ``serving.json`` — so the running
service needs xgboost and numpy and no pickle at all. It falls back to the
training pickle when that pair is absent, which is what happens in a fresh
checkout before ``export_serving.py`` has run.

Every prediction is logged as a structured (JSON) line so the service is
observable in production log aggregators without extra tooling.

Run locally:
    uvicorn sepsis_icu.serve:app --reload
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config

# --- Structured logging ------------------------------------------------------
logger = logging.getLogger("sepsis_icu.serve")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}))


# --- Model loading (lazy, cached) --------------------------------------------
# Deliberately lazy: the landing page is static HTML plus a JSON blob and does
# not need the model at all. Importing xgboost costs most of a second, and on a
# cold serverless instance that would be paid by whoever loads the page rather
# than by the first prediction.
_BUNDLE: dict | None = None


class _ExportedModel:
    """The scaler and booster `export_serving.py` wrote, without the pickle.

    Reproduces ``Pipeline([StandardScaler(), XGBClassifier()]).predict_proba``
    exactly — `export_serving.py` refuses to write these files otherwise.
    """

    def __init__(self, ubj_path, meta: dict) -> None:
        import numpy as np
        import xgboost as xgb

        self._booster = xgb.Booster()
        self._booster.load_model(str(ubj_path))
        self._mean = np.asarray(meta["scaler_mean"], dtype=np.float64)
        self._scale = np.asarray(meta["scaler_scale"], dtype=np.float64)

    def predict_proba_pos(self, rows: list[list[float]]) -> list[float]:
        import numpy as np

        z = ((np.asarray(rows, dtype=np.float64) - self._mean) / self._scale).astype(np.float32)
        return [float(p) for p in self._booster.inplace_predict(z)]


class _PickledPipeline:
    """Fallback: the scikit-learn pipeline straight out of `train.py`."""

    def __init__(self, pipe) -> None:
        self._pipe = pipe
        self._pos = list(pipe.classes_).index(1)
        # The pipeline was fitted on a named frame, so feeding it a bare array
        # makes sklearn warn "X does not have valid feature names" on every
        # request, which would land in the middle of the structured JSON logs.
        names = getattr(pipe, "feature_names_in_", None)
        self._names = [] if names is None else [str(n) for n in names]

    def predict_proba_pos(self, rows: list[list[float]]) -> list[float]:
        import numpy as np

        X: Any = np.asarray(rows, dtype=np.float64)
        if self._names:
            import pandas as pd

            X = pd.DataFrame(X, columns=self._names)
        return [float(p) for p in self._pipe.predict_proba(X)[:, self._pos]]


def model_available() -> bool:
    return (config.MODEL_UBJ_PATH.exists() and config.SERVING_META_PATH.exists()) or (
        config.MODEL_PATH.exists()
    )


def load_bundle() -> dict:
    """Load and cache whichever model form is present, exported first."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    if config.MODEL_UBJ_PATH.exists() and config.SERVING_META_PATH.exists():
        meta = json.loads(config.SERVING_META_PATH.read_text())
        _BUNDLE = {
            "model": _ExportedModel(config.MODEL_UBJ_PATH, meta),
            "features": list(meta["features"]),
            "threshold": float(meta.get("threshold", 0.5)),
            "model_type": meta["model_type"],
            "class_pos": meta["class_pos"],
            "class_neg": meta["class_neg"],
            "source": "exported",
        }
    elif config.MODEL_PATH.exists():
        import joblib

        raw = joblib.load(config.MODEL_PATH)
        _BUNDLE = {
            "model": _PickledPipeline(raw["model"]),
            "features": list(raw["features"]),
            "threshold": float(raw.get("threshold", 0.5)),
            "model_type": raw["model_type"],
            "class_pos": raw["class_pos"],
            "class_neg": raw["class_neg"],
            "source": "joblib",
        }
    else:
        raise FileNotFoundError(
            f"No model at {config.MODEL_UBJ_PATH} or {config.MODEL_PATH}. "
            "Run `python train.py` then `python export_serving.py`."
        )

    _log_event(
        "model_loaded",
        model_type=_BUNDLE["model_type"],
        source=_BUNDLE["source"],
        n_features=len(_BUNDLE["features"]),
        threshold=_BUNDLE["threshold"],
    )
    return _BUNDLE


# --- Presentation metadata ---------------------------------------------------
# What each of the 50 columns is in clinical terms. The model neither knows nor
# needs any of this; it exists so the demo sheet can be read by a person.
#
# base signal -> (display name, unit, organ-system key)
_SIGNALS: dict[str, tuple[str, str, str]] = {
    "HR": ("Heart rate", "bpm", "circ"),
    "Temp": ("Temperature", "°C", "circ"),
    "SBP": ("Systolic pressure", "mmHg", "circ"),
    "DBP": ("Diastolic pressure", "mmHg", "circ"),
    "MAP": ("Mean arterial pressure", "mmHg", "circ"),
    "Resp": ("Respiration rate", "/min", "gas"),
    "FiO2": ("Inspired oxygen", "fraction", "gas"),
    "PaCO2": ("Arterial CO₂", "mmHg", "gas"),
    "BaseExcess": ("Base excess", "mmol/L", "gas"),
    "BUN": ("Urea nitrogen", "mg/dL", "renal"),
    "Creatinine": ("Creatinine", "mg/dL", "renal"),
    "Calcium": ("Calcium", "mg/dL", "renal"),
    "Phosphate": ("Phosphate", "mg/dL", "renal"),
    "Platelets": ("Platelets", "×10³/µL", "haem"),
    "WBC": ("White cells", "×10³/µL", "haem"),
    "Lactate": ("Lactate", "mmol/L", "metab"),
    "AST": ("AST", "U/L", "metab"),
    "Bilirubin_total": ("Bilirubin", "mg/dL", "metab"),
    "ICULOS": ("Time in ICU", "hours", "stay"),
}

# What the aggregation suffix means, in words rather than in pandas.
_AGGREGATIONS: dict[str, str] = {
    "min": "lowest",
    "max": "peak",
    "mean": "average",
    "std": "variability",
    "range": "swing",
}

_SYSTEMS: dict[str, str] = {
    "circ": "circulation",
    "gas": "gas exchange",
    "renal": "renal",
    "haem": "haematology",
    "metab": "metabolic",
    "stay": "ICU stay",
}


def describe_features(features: list[str]) -> dict[str, dict[str, str]]:
    """Split each ``Signal_agg`` column into something readable."""
    out: dict[str, dict[str, str]] = {}
    for f in features:
        base, _, agg = f.rpartition("_")
        name, unit, system = _SIGNALS.get(base, (base.replace("_", " "), "", "stay"))
        # ICULOS is a running counter, so its max is the length of the stay
        # rather than a "peak" of anything.
        label = "total" if base == "ICULOS" else _AGGREGATIONS.get(agg, agg)
        out[f] = {"name": name, "agg": label, "unit": unit, "system": system}
    return out


def load_examples() -> dict:
    """Real held-out patients + stats baked in at build time for the demo UI."""
    if config.EXAMPLES_PATH.exists():
        return json.loads(config.EXAMPLES_PATH.read_text())
    return {"samples": [], "features": [], "stats": {}, "meta": {}, "model_type": ""}


# --- Request / response schemas ----------------------------------------------
class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Mapping of feature name -> value. Must include every feature "
        "listed at GET /model.",
        examples=[{"ICULOS_max": 48.0, "Lactate_max": 3.1, "Temp_range": 2.4}],
    )


class PredictResponse(BaseModel):
    # model_type would collide with pydantic's protected "model_" namespace.
    model_config = {"protected_namespaces": ()}

    prediction: str
    probability_sepsis: float
    threshold: float
    model_type: str


app = FastAPI(
    title="ICU Sepsis Classifier",
    description="Flags ICU patients who developed sepsis from aggregated "
    "vital-sign and lab summaries (PhysioNet/CinC 2019).",
    version="1.1.0",
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Interactive demo landing page — an ICU observation sheet."""
    ex = load_examples()
    if not ex.get("model_type"):
        try:
            ex["model_type"] = load_bundle()["model_type"]
        except Exception:
            ex["model_type"] = ""
    ex["describe"] = describe_features(list(ex.get("features", [])))
    ex["systems"] = _SYSTEMS
    return _LANDING_PAGE.replace("__DATA__", json.dumps(ex))


@app.get("/health")
def health() -> dict:
    """Liveness probe. Reports whether a model is available, without loading it."""
    return {"status": "ok", "model_available": model_available()}


@app.get("/model")
def model_info() -> dict:
    """Metadata about the loaded model, including the exact features it expects."""
    bundle = load_bundle()
    return {
        "model_type": bundle["model_type"],
        "classes": {"positive": bundle["class_pos"], "negative": bundle["class_neg"]},
        "threshold": bundle["threshold"],
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
        "loaded_from": bundle["source"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict sepsis vs no-sepsis for one patient's aggregated features.

    Unauthenticated and unthrottled by design (open demo, no sensitive data). Add
    rate limiting before public exposure — see docs/deploy.md "Hardening".
    """
    bundle = load_bundle()
    features = bundle["features"]
    threshold = bundle["threshold"]

    missing = [f for f in features if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing {len(missing)} required feature(s), e.g. {missing[:5]}",
        )

    row = [[float(req.features[f]) for f in features]]

    t0 = time.time()
    prob = bundle["model"].predict_proba_pos(row)[0]
    label = bundle["class_pos"] if prob >= threshold else bundle["class_neg"]

    _log_event(
        "prediction",
        prediction=label,
        probability_sepsis=round(prob, 4),
        threshold=threshold,
        latency_ms=round((time.time() - t0) * 1000, 2),
    )
    return PredictResponse(
        prediction=label,
        probability_sepsis=prob,
        threshold=threshold,
        model_type=bundle["model_type"],
    )


# --- Landing page ------------------------------------------------------------
# One self-contained page, no external requests. `__DATA__` is replaced at
# request time with the demo JSON: every held-out patient, per-feature training
# quantiles, all 50 feature importances, and the confusion counts at each of 101
# decision thresholds (which is what makes the alarm line draggable for real).
_LANDING_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ICU Sepsis Classifier - observation sheet</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='88'>&#129656;</text></svg>">
<style>
  :root{
    --paper:#FFFFFF; --ink:#141E27; --soft:#5A6B79; --faint:#93A3AF;
    --rule:#E3E9ED; --edge:#141E27; --band:#E6ECF0;
    --flag:#B33A2B; --ok:#2A7A69; --warn:#8E6A1E;
    --circ:#A83E2E; --gas:#3E6E96; --renal:#8E6A1E; --haem:#6B5A96;
    --metab:#2A7A69; --stay:#5A6B79;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;
    --mono:ui-monospace,"Cascadia Mono",Consolas,"Liberation Mono",monospace;
  }
  *{box-sizing:border-box}
  html{font-size:17px}
  body{margin:0;background:#FFFFFF;color:var(--ink);font-family:var(--sans);
    line-height:1.55;-webkit-font-smoothing:antialiased}

  .sheet{max-width:1180px;margin:2.2rem auto 3rem;background:var(--paper);
    border:1.5px solid var(--edge);container-type:inline-size;
    box-shadow:0 2px 4px rgba(20,30,39,.04),0 24px 56px -28px rgba(20,30,39,.28)}
  @media(max-width:1240px){.sheet{margin:0 auto;border-left:none;border-right:none}}

  /* ---- form bar ---- */
  .formbar{display:flex;align-items:baseline;gap:.9rem;flex-wrap:wrap;
    padding:.55rem 1.6rem;border-bottom:1px solid var(--edge);background:#FAFCFD;
    font:600 .66rem/1.6 var(--mono);letter-spacing:.19em;text-transform:uppercase;
    color:var(--soft)}
  .formbar .f{color:var(--flag)}
  .formbar .r{margin-left:auto;letter-spacing:.13em;display:flex;align-items:center;gap:.45rem}
  .live{width:7px;height:7px;border-radius:50%;background:var(--ok);flex:0 0 auto}
  @media(prefers-reduced-motion:no-preference){
    .live{animation:beat 1.6s ease-in-out infinite}
    @keyframes beat{0%,100%{opacity:1}50%{opacity:.3}}
  }

  /* ---- header ---- */
  .hdr{padding:1.7rem 1.6rem 1.4rem;border-bottom:1.5px solid var(--edge)}
  /* sized against the sheet, not the viewport, so the line fills the measure */
  h1{margin:0 0 .55rem;font-family:var(--serif);font-weight:600;
    font-size:clamp(1.7rem,4.6cqi,3rem);line-height:1.08;letter-spacing:-.012em}
  h1 em{font-style:italic;color:var(--flag)}
  .lede{margin:0;color:var(--soft);font-size:1.02rem;text-wrap:pretty}

  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    border-top:1px solid var(--rule);margin-top:1.3rem}
  .st{padding:.75rem 1rem .8rem;border-right:1px solid var(--rule)}
  .st:last-child{border-right:none}
  .st b{display:block;font:600 1.5rem/1.1 var(--mono);font-variant-numeric:tabular-nums}
  .st .k{display:flex;align-items:center;gap:.4rem;margin-top:.2rem;
    font:600 .63rem/1.4 var(--mono);letter-spacing:.16em;text-transform:uppercase;
    color:var(--soft)}
  @media(max-width:640px){.stats{grid-template-columns:1fr 1fr}
    .st{border-bottom:1px solid var(--rule)}}

  /* ---- rhythm strip ---- */
  .stripwrap{padding:1.2rem 1.6rem 0}
  .striphead{display:flex;align-items:center;gap:.45rem;margin-bottom:.5rem;
    font:600 .63rem/1.4 var(--mono);letter-spacing:.16em;text-transform:uppercase;
    color:var(--soft)}
  .striphead .rate{margin-left:auto;letter-spacing:.1em;text-transform:none;
    font-weight:400;color:var(--faint)}
  .strip{border:1px solid var(--rule);background:#FFFBFA;
    background-image:linear-gradient(#F6DED8 1px,transparent 1px),
                     linear-gradient(90deg,#F6DED8 1px,transparent 1px);
    background-size:100% 13px,13px 100%}
  .strip svg{display:block;width:100%;height:76px}

  /* ---- record index ---- */
  .idx{padding:1.5rem 1.6rem 0}
  .idxhead{display:flex;align-items:center;gap:.45rem;margin-bottom:.6rem;
    font:600 .63rem/1.4 var(--mono);letter-spacing:.16em;text-transform:uppercase;
    color:var(--soft)}
  /* A card-index drawer: one row you scroll along, rather than eight rows of
     tabs shouldering the sheet out of the way. */
  .tabs{display:flex;gap:.3rem;overflow-x:auto;padding-bottom:2px;
    scrollbar-width:thin;border-bottom:1px solid var(--edge)}
  .tab{border:1px solid var(--rule);border-bottom:none;background:#F7FAFB;
    padding:.38rem .7rem .42rem;cursor:pointer;font:.7rem/1.3 var(--mono);
    color:var(--soft);text-align:left;border-radius:0;flex:0 0 auto;white-space:nowrap}
  .tab b{display:block;color:var(--ink);font-weight:600;font-size:.75rem}
  .tab:hover{background:#EDF3F6;color:var(--ink)}
  .tab:focus-visible{outline:2px solid var(--flag);outline-offset:1px}
  .tab.on{background:var(--paper);border-color:var(--edge);
    color:var(--ink);box-shadow:inset 0 3px 0 var(--flag)}
  .draws{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.55rem}
  .draw{border:1px solid var(--edge);background:var(--paper);cursor:pointer;
    padding:.42rem .8rem;font:600 .72rem/1.3 var(--mono);color:var(--ink)}
  .draw:hover{background:#F2F7F9}
  .draw:focus-visible{outline:2px solid var(--flag);outline-offset:1px}
  .draw.sep{color:var(--flag);border-color:var(--flag)}
  .draw.no{color:var(--ok);border-color:var(--ok)}

  /* ---- flowsheet ---- */
  .cols,.row{display:grid;
    grid-template-columns:6px minmax(11rem,15rem) minmax(0,1fr) 6.2rem 5.4rem;
    align-items:center;gap:0}
  .cols{margin-top:1.4rem;padding:.5rem 1.6rem;border-top:1.5px solid var(--edge);
    border-bottom:1px solid var(--edge);
    font:600 .62rem/1.4 var(--mono);letter-spacing:.15em;text-transform:uppercase;
    color:var(--soft);position:sticky;top:0;background:var(--paper);z-index:5}
  .cols .c3{margin:0 .9rem;display:flex;align-items:center;gap:.4rem}
  .cols .c4,.cols .c5{display:flex;align-items:center;gap:.4rem;justify-content:flex-end}
  .row{padding:.34rem 1.6rem;border-bottom:1px solid var(--rule)}
  .row:nth-child(even){background:#FAFCFD}
  .row.driver{background:#FFF8F7}
  .row.driver:nth-child(even){background:#FEF4F2}
  .sys{width:5px;height:26px}
  .s-circ{background:var(--circ)}.s-gas{background:var(--gas)}
  .s-renal{background:var(--renal)}.s-haem{background:var(--haem)}
  .s-metab{background:var(--metab)}.s-stay{background:var(--stay)}
  .nm{padding-left:.65rem;font-size:.87rem;line-height:1.25;min-width:0}
  .nm i{font-style:normal;color:var(--soft)}
  .nm small{display:block;color:var(--faint);font:.63rem/1.5 var(--mono);
    letter-spacing:.05em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

  .rail{position:relative;height:26px;margin:0 .9rem}
  .rail .axis{position:absolute;left:0;right:0;top:17px;border-bottom:1px solid var(--rule)}
  /* min-width matters: for the many "swing" columns three quarters of the
     training cohort recorded exactly the same value, so the band is genuinely
     zero-wide and would otherwise vanish. */
  .rail .band{position:absolute;top:14px;height:7px;background:var(--band);
    border-left:1px solid #C3D0D8;border-right:1px solid #C3D0D8;min-width:5px}
  .pin{position:absolute;top:9px;width:2px;height:17px;background:var(--ink)}
  .pin.hi,.pin.lo{background:var(--flag);width:3px}
  .pin b{position:absolute;left:50%;transform:translateX(-50%);top:-11px;
    font:.6rem/1 var(--mono);white-space:nowrap;color:inherit;font-weight:500}
  .rail.empty{opacity:.55}

  .wt{display:flex;align-items:center;gap:.35rem;justify-content:flex-end}
  .wt .track{width:3rem;height:6px;background:#EEF3F6;border:1px solid var(--rule);flex:0 0 auto}
  .wt .fill{display:block;height:100%;background:var(--flag)}
  .wt span{font:.6rem/1 var(--mono);color:var(--faint);width:1.6rem;text-align:right}
  .val{text-align:right;font:600 .84rem/1 var(--mono);font-variant-numeric:tabular-nums;
    white-space:nowrap}
  .val.hi{color:var(--flag)} .val.lo{color:var(--flag)}
  .val .ar{font-size:.58rem;vertical-align:.16em;margin-left:.1rem}

  /* Below ~660px the five columns cannot all hold their minimum width, so each
     observation becomes three stacked bands instead of one row. Nothing is
     dropped — the rail is the point of the sheet and the help buttons have to
     stay reachable. */
  @media(max-width:660px){
    .cols{display:flex;flex-wrap:wrap;gap:.45rem 1.1rem;padding:.5rem 1rem}
    .cols>span:first-child{display:none}
    .cols .c3,.cols .c4,.cols .c5{margin:0;justify-content:flex-start}
    .row{grid-template-columns:5px minmax(0,1fr) auto;
      grid-template-areas:"s n v" "s r r" "s w w";padding:.5rem 1rem;row-gap:.15rem}
    .sys{grid-area:s;height:auto;align-self:stretch}
    .nm{grid-area:n}
    .val{grid-area:v;align-self:start}
    .rail{grid-area:r;margin:.1rem 0 0 .65rem}
    .wt{grid-area:w;justify-content:flex-start;margin:0 0 0 .65rem}
    .key{padding:.7rem 1rem}
    .idx,.stripwrap{padding-left:1rem;padding-right:1rem}
    .hdr,.formbar{padding-left:1rem;padding-right:1rem}
    .nomo,.tally{padding-left:1rem;padding-right:1rem}
    .foothint{padding-left:1rem;padding-right:1rem}
  }

  .key{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;
    padding:.7rem 1.6rem;border-bottom:1.5px solid var(--edge);
    font:.68rem/1.6 var(--mono);color:var(--soft)}
  .key i{display:inline-block;width:5px;height:11px;margin-right:.35rem;
    vertical-align:-.1em}
  .key .n{margin-left:auto;color:var(--faint)}

  /* ---- footer: nomogram + tally ---- */
  .foot{display:grid;grid-template-columns:1.5fr 1fr}
  @media(max-width:820px){.foot{grid-template-columns:1fr}
    .nomo{border-right:none!important;border-bottom:1px solid var(--edge)}}
  .nomo{padding:1.25rem 1.6rem 1.6rem;border-right:1px solid var(--edge)}
  .tally{padding:1.25rem 1.6rem 1.6rem}
  .k{display:flex;align-items:center;gap:.45rem;
    font:600 .63rem/1.4 var(--mono);letter-spacing:.17em;text-transform:uppercase;
    color:var(--soft)}

  .ruler{position:relative;height:34px;border:1px solid var(--edge);background:#F7FAFB;
    margin-top:3.5rem;touch-action:none}
  .ruler .ticks{position:absolute;inset:0;
    background:repeating-linear-gradient(90deg,transparent 0 9px,rgba(20,30,39,.11) 9px 10px)}
  .ruler .hot{position:absolute;top:0;bottom:0;right:0;background:rgba(179,58,43,.09)}
  .ruler .tk{position:absolute;bottom:-1.25rem;transform:translateX(-50%);
    font:.6rem/1 var(--mono);color:var(--faint)}
  .thr{position:absolute;top:-.7rem;bottom:-.3rem;width:2px;background:var(--edge);
    cursor:ew-resize}
  .thr b{position:absolute;left:50%;transform:translateX(-50%);top:-1.15rem;
    white-space:nowrap;font:600 .61rem/1 var(--mono);letter-spacing:.12em}
  .thr i{position:absolute;left:50%;transform:translateX(-50%);bottom:-.72rem;
    width:15px;height:11px;background:var(--edge);
    clip-path:polygon(50% 0,100% 100%,0 100%)}
  .grab{position:absolute;top:-1.4rem;bottom:-1.4rem;width:34px;
    transform:translateX(-50%);cursor:ew-resize;touch-action:none}
  .grab:focus-visible{outline:2px solid var(--flag)}
  .needle{position:absolute;top:-2.75rem;transform:translateX(-50%);text-align:center;
    transition:left .45s cubic-bezier(.2,.7,.3,1)}
  .needle .n{display:inline-block;font:600 1.1rem/1 var(--mono);
    border:2px solid var(--flag);color:var(--flag);background:var(--paper);
    padding:.16rem .42rem;font-variant-numeric:tabular-nums}
  /* display:block, or the <i> stays inline: it lands beside the plate instead
     of under it and paints as a solid block the width of its font. */
  .needle .st{display:block;width:2px;height:2.75rem;background:var(--flag);margin:0 auto}
  .needle.none{display:none}

  .verdict{display:flex;align-items:flex-start;gap:1rem;margin-top:2.7rem;min-height:4.4rem}
  .stamp{border:2.5px solid var(--faint);color:var(--faint);padding:.45rem .9rem;
    transform:rotate(-2deg);font:700 1rem/1.15 var(--mono);letter-spacing:.13em;
    text-transform:uppercase;flex:0 0 auto}
  .stamp small{display:block;font-weight:500;letter-spacing:.04em;font-size:.6rem;
    text-transform:none;margin-top:.2rem;opacity:.9}
  .stamp.sep{border-color:var(--flag);color:var(--flag)}
  .stamp.no{border-color:var(--ok);color:var(--ok)}
  .verdict p{margin:.15rem 0 0;color:var(--soft);font-size:.85rem}
  .verdict p b{color:var(--ink)}

  .tallybar{display:flex;height:22px;border:1px solid var(--edge);margin-top:1.2rem;
    font:600 .57rem/20px var(--mono);color:#fff;text-align:center;overflow:hidden}
  .tallybar span{display:block;overflow:hidden;min-width:0}
  .tallyrows{margin-top:.85rem;font:.72rem/1.95 var(--mono)}
  .tallyrows div{display:flex;justify-content:space-between;gap:1rem;
    border-bottom:1px dotted var(--rule)}
  .tallyrows b{font-weight:600;font-variant-numeric:tabular-nums}
  .tallyrows i{font-style:normal;display:inline-block;width:9px;height:9px;
    margin-right:.45rem}
  .note{margin:.9rem 0 0;color:var(--soft);font-size:.78rem;line-height:1.5}
  .note b{color:var(--ink);font-variant-numeric:tabular-nums}

  /* ---- help buttons + tooltip ---- */
  .q{width:17px;height:17px;border-radius:50%;border:1px solid var(--faint);
    background:var(--paper);color:var(--soft);font:.66rem/15px var(--mono);
    text-align:center;cursor:help;padding:0;flex:0 0 auto;
    transition:color .15s,border-color .15s}
  .q:hover,.q:focus-visible{color:var(--flag);border-color:var(--flag);outline:none}
  .tip{position:fixed;z-index:60;max-width:23rem;display:none;
    background:var(--ink);color:#F2F6F9;padding:.6rem .75rem;border-radius:3px;
    font-size:.79rem;line-height:1.5;
    box-shadow:0 10px 30px -10px rgba(20,30,39,.6)}
  .tip.on{display:block}
  .tip b{color:#fff}
  .tip kbd{font:.72rem/1 var(--mono);background:rgba(255,255,255,.14);
    padding:.12rem .3rem;border-radius:2px}

  .foothint{padding:.7rem 1.6rem 1rem;border-top:1px solid var(--rule);
    font:.7rem/1.6 var(--mono);color:var(--faint)}
  .foothint a{color:var(--soft)}
</style></head>
<body>
<div class="sheet">
  <div class="formbar">
    <span class="f">Form 7B</span><span>ICU observation record</span>
    <span class="r"><i class="live"></i><span id="src">model live</span></span>
  </div>

  <div class="hdr">
    <h1>Read the chart the way the <em>model</em> reads it.</h1>
    <p class="lede">A gradient-boosted classifier scores one ICU stay from 50 summary values —
      vitals, blood gas, organ chemistry. No notes, no diagnosis, no clinician's judgement.
      Pull a patient it never saw and work down the sheet.</p>

    <div class="stats">
      <div class="st"><b id="s-auc">&mdash;</b>
        <div class="k">ROC-AUC <button class="q" data-tip="auc">?</button></div></div>
      <div class="st"><b id="s-n">&mdash;</b>
        <div class="k">Patients scored <button class="q" data-tip="held">?</button></div></div>
      <div class="st"><b id="s-f">&mdash;</b>
        <div class="k">Values read <button class="q" data-tip="feats">?</button></div></div>
      <div class="st"><b id="s-model">&mdash;</b>
        <div class="k">Model <button class="q" data-tip="model">?</button></div></div>
    </div>
  </div>

  <div class="stripwrap">
    <div class="striphead">Rhythm strip <button class="q" data-tip="strip">?</button>
      <span class="rate" id="striprate">&mdash;</span></div>
    <div class="strip"><svg id="ecg" viewBox="0 0 1140 76" preserveAspectRatio="none"
      aria-hidden="true"></svg></div>
  </div>

  <div class="idx">
    <div class="idxhead">Records on the desk <button class="q" data-tip="index">?</button></div>
    <div class="tabs" id="tabs"></div>
    <div class="draws">
      <button class="draw sep" id="draw-sep">&#8635; draw one who developed sepsis</button>
      <button class="draw no" id="draw-no">&#8635; draw one who did not</button>
      <button class="draw" id="draw-any">&#8635; draw any of the 80</button>
      <span style="display:flex;align-items:center"><button class="q" data-tip="draw">?</button></span>
    </div>
  </div>

  <div class="cols">
    <span></span><span>Observation</span>
    <span class="c3">Cohort band &middot; this patient
      <button class="q" data-tip="band">?</button></span>
    <span class="c4">Weight <button class="q" data-tip="weight">?</button></span>
    <span class="c5">Value <button class="q" data-tip="value">?</button></span>
  </div>
  <div id="rows"></div>
  <div class="key" id="key"></div>

  <div class="foot">
    <div class="nomo">
      <div class="k">Decision nomogram &middot; P(sepsis)
        <button class="q" data-tip="nomo">?</button></div>
      <div class="ruler" id="ruler">
        <i class="ticks"></i><i class="hot" id="hot"></i>
        <i class="thr" id="thr"><b id="thrlab">ALARM 0.40</b><i></i></i>
        <div class="grab" id="grab" role="slider" tabindex="0"
             aria-label="Alarm threshold" aria-valuemin="0" aria-valuemax="1"
             aria-valuenow="0.4"></div>
        <i class="needle none" id="needle"><span class="n" id="needlev">&mdash;</span>
          <i class="st"></i></i>
        <i class="tk" style="left:0">0.0</i><i class="tk" style="left:25%">0.25</i>
        <i class="tk" style="left:50%">0.50</i><i class="tk" style="left:75%">0.75</i>
        <i class="tk" style="left:100%">1.0</i>
      </div>
      <div class="verdict">
        <div class="stamp" id="stamp">No record<small>pick one above</small></div>
        <p id="verdicttext">Choose a record and the sheet fills in, the needle lands, and the
          tally on the right recounts at whatever alarm line you set.</p>
      </div>
    </div>

    <div class="tally">
      <div class="k">Tally at this alarm line <button class="q" data-tip="tally">?</button></div>
      <div class="tallybar" id="tallybar"></div>
      <div class="tallyrows">
        <div><span><i style="background:var(--ok)"></i>Correct all-clear</span>
          <b id="t-tn">&mdash;</b></div>
        <div><span><i style="background:var(--flag)"></i>Sepsis caught</span>
          <b id="t-tp">&mdash;</b></div>
        <div><span><i style="background:var(--warn)"></i>Sepsis missed</span>
          <b id="t-fn">&mdash;</b></div>
        <div><span><i style="background:var(--faint)"></i>False alarms</span>
          <b id="t-fp">&mdash;</b></div>
      </div>
      <p class="note">Recall <b id="t-rec">&mdash;</b> &middot; precision <b id="t-pre">&mdash;</b>
        at this line. <span id="t-cmp"></span></p>
    </div>
  </div>

  <div class="foothint">PhysioNet/CinC 2019 &middot; every value on this sheet is real and
    held out of training &middot; <a href="/docs">API docs</a> &middot;
    <a href="/model">model metadata</a></div>
</div>

<script>
const DATA = __DATA__;
const g = id => document.getElementById(id);

const FEATS   = DATA.features || [];
const DESC    = DATA.describe || {};
const SYSTEMS = DATA.systems || {};
const STATS   = DATA.stats || {};
const SAMPLES = DATA.samples || [];
const META    = DATA.meta || {};
const SWEEP   = DATA.sweep || null;
const WEIGHTS = {};
(DATA.top_features || []).forEach(d => { WEIGHTS[d.feature] = d.weight; });
const MAXW = Math.max(1e-9, ...Object.values(WEIGHTS));

const BASE_THR = META.threshold != null ? META.threshold : 0.4;
let thr = BASE_THR;
let current = null;      // {sample, prob}

/* ---------- helpers ---------- */
const fmt = (v, d) => {
  if (v == null || !isFinite(v)) return '\\u2014';
  const a = Math.abs(v);
  const dp = d != null ? d : (a >= 100 ? 1 : a >= 10 ? 1 : a >= 1 ? 2 : 3);
  return v.toFixed(dp);
};
const pct = v => (100 * v).toFixed(1) + '%';
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* Index numbers, not patient numbers. The source data is de-identified and
   carries no identifiers, so inventing plausible-looking ones would be a small
   lie on a page whose whole point is that everything else on it is real. */
const recId = i => 'No. ' + String(i + 1).padStart(2, '0');

/* ---------- masthead ---------- */
g('s-auc').textContent = META.roc_auc != null ? META.roc_auc.toFixed(3) : '\\u2014';
g('s-n').textContent   = (META.n_test || 0).toLocaleString();
g('s-f').textContent   = FEATS.length;
g('s-model').textContent = DATA.model_type || '\\u2014';

/* ---------- rhythm strip ---------- */
/* Beat spacing comes from the patient's own recorded mean heart rate; the beat
   shape is drawn, because the Challenge data holds hourly summaries and not
   waveforms. Explained on the strip's own help button. */
function drawEcg(bpm){
  const W = 1140, H = 76, mid = H * 0.58;
  const beats = clamp(bpm ? bpm * 6 / 60 : 8, 3, 22);   // a 6-second strip
  const step = W / beats;
  let d = 'M0 ' + mid;
  for (let i = 0; i < beats + 1; i++){
    const x = i * step;
    d += ' L' + (x + step * .18).toFixed(1) + ' ' + mid;
    d += ' l' + (step * .05).toFixed(1) + ' -3 l' + (step * .05).toFixed(1) + ' 5';   // P
    d += ' L' + (x + step * .42).toFixed(1) + ' ' + mid;
    d += ' l' + (step * .04).toFixed(1) + ' 6 l' + (step * .05).toFixed(1) + ' -34';  // QRS
    d += ' l' + (step * .05).toFixed(1) + ' 44 l' + (step * .04).toFixed(1) + ' -16';
    d += ' L' + (x + step * .70).toFixed(1) + ' ' + mid;
    d += ' l' + (step * .07).toFixed(1) + ' -6 l' + (step * .08).toFixed(1) + ' 6';   // T
  }
  g('ecg').innerHTML = '<path fill="none" stroke="#A83E2E" stroke-width="1.5" d="' + d + '"/>';
  g('striprate').textContent = bpm
    ? 'drawn at this patient\\u2019s mean rate, ' + Math.round(bpm) + ' bpm \\u00b7 6 s'
    : 'idle \\u00b7 pick a record';
}
drawEcg(0);

/* ---------- record index ---------- */
const tabs = g('tabs');
SAMPLES.forEach((s, i) => {
  const b = document.createElement('button');
  b.className = 'tab';
  b.dataset.i = i;
  const los = s.features['ICULOS_max'];
  b.innerHTML = '<b>' + recId(i) + '</b>' + (los != null ? Math.round(los) + ' h in ICU' : '');
  b.addEventListener('click', () => pick(i));
  tabs.appendChild(b);
});

let lastDrawn = -1;
function drawFrom(pool){
  if (!pool.length) return;
  let k = Math.floor(Math.random() * pool.length);
  if (pool.length > 1 && pool[k] === lastDrawn)
    k = (k + 1 + Math.floor(Math.random() * (pool.length - 1))) % pool.length;
  pick(pool[k]);
}
const idxWhere = want => SAMPLES.map((s, i) => [s, i])
  .filter(p => (p[0].label === 'Sepsis') === want).map(p => p[1]);
g('draw-sep').addEventListener('click', () => drawFrom(idxWhere(true)));
g('draw-no').addEventListener('click',  () => drawFrom(idxWhere(false)));
g('draw-any').addEventListener('click', () => drawFrom(SAMPLES.map((_, i) => i)));

/* ---------- the sheet ---------- */
const ORDER = FEATS.slice().sort((a, b) => (WEIGHTS[b] || 0) - (WEIGHTS[a] || 0));

function railGeometry(f, v){
  const s = STATS[f] || {};
  let lo = s.q01, hi = s.q99;
  if (lo == null || hi == null || !(hi > lo)) {
    const m = s.mean || 0, sd = s.std || 1;
    lo = m - 2 * sd; hi = m + 2 * sd;
  }
  // Always keep the patient on the rail, with a little air either side.
  if (v != null && v < lo) lo = v - (hi - lo) * 0.08;
  if (v != null && v > hi) hi = v + (hi - lo) * 0.08;
  const span = (hi - lo) || 1;
  const at = x => clamp((x - lo) / span, 0, 1) * 100;
  const b0 = s.q25 != null ? at(s.q25) : null;
  const b1 = s.q75 != null ? at(s.q75) : null;
  return {at, b0, b1, q25: s.q25, q75: s.q75};
}

function buildRows(sample){
  const rows = g('rows');
  rows.innerHTML = '';
  ORDER.forEach((f, rank) => {
    const d = DESC[f] || {name: f, agg: '', unit: '', system: 'stay'};
    const v = sample ? sample.features[f] : null;
    const s = STATS[f] || {};
    // The band is cohort data, so it is drawn even before a record is chosen —
    // an empty sheet still tells you where the training patients sat.
    const geo = railGeometry(f, v);

    const out = v != null && s.q25 != null && s.q75 != null
      ? (v > s.q75 ? 'hi' : v < s.q25 ? 'lo' : '') : '';

    const row = document.createElement('div');
    row.className = 'row' + (rank < 3 ? ' driver' : '');

    const w = (WEIGHTS[f] || 0) / MAXW;
    const railHtml = '<i class="axis"></i>' +
      (geo.b0 != null
        ? '<i class="band" style="left:' + geo.b0 + '%;width:' +
          Math.max(0.4, geo.b1 - geo.b0) + '%"></i>' : '') +
      (v != null
        ? '<i class="pin ' + out + '" style="left:' + geo.at(v) + '%"><b>' + fmt(v) +
          '</b></i>' : '');

    row.innerHTML =
      '<i class="sys s-' + d.system + '"></i>' +
      '<div class="nm">' + d.name + ' <i>&middot; ' + d.agg + '</i>' +
        '<small>' + f.toUpperCase() + (d.unit ? ' \\u00b7 ' + d.unit : '') + '</small></div>' +
      '<div class="rail' + (v == null ? ' empty' : '') + '">' + railHtml + '</div>' +
      '<div class="wt"><i class="track"><i class="fill" style="width:' +
        (w * 100).toFixed(1) + '%"></i></i><span>' +
        (WEIGHTS[f] || 0).toFixed(3).slice(1) + '</span></div>' +
      '<div class="val ' + out + '">' + (v != null ? fmt(v) : '\\u2014') +
        (out === 'hi' ? '<span class="ar">\\u25b2</span>'
         : out === 'lo' ? '<span class="ar">\\u25bc</span>' : '') + '</div>';
    rows.appendChild(row);
  });
}

function buildKey(){
  const el = g('key');
  el.innerHTML = Object.keys(SYSTEMS).map(k =>
    '<span><i style="background:var(--' + k + ')"></i>' + SYSTEMS[k] + '</span>').join('') +
    '<span class="n">' + FEATS.length + ' observations, ordered by model weight</span>';
}
buildKey();
buildRows(null);

/* ---------- nomogram + tally ---------- */
function sweepAt(t){
  if (!SWEEP) return null;
  const i = clamp(Math.round(t / SWEEP.step), 0, SWEEP.tp.length - 1);
  const tp = SWEEP.tp[i], fp = SWEEP.fp[i];
  return {tp, fp, fn: SWEEP.n_pos - tp, tn: SWEEP.n_neg - fp};
}

function renderTally(){
  const c = sweepAt(thr);
  if (!c) return;
  g('tallybar').innerHTML =
    '<span style="background:var(--ok);flex:' + c.tn + '">' + c.tn + '</span>' +
    '<span style="background:var(--flag);flex:' + c.tp + '"></span>' +
    '<span style="background:var(--warn);flex:' + c.fn + '"></span>' +
    '<span style="background:var(--faint);flex:' + c.fp + '"></span>';
  g('t-tn').textContent = c.tn.toLocaleString();
  g('t-tp').textContent = c.tp.toLocaleString();
  g('t-fn').textContent = c.fn.toLocaleString();
  g('t-fp').textContent = c.fp.toLocaleString();
  // At a line of 1.00 nothing is flagged at all, so precision is 0/0 — undefined
  // rather than zero, and saying "0.0%" there would be a different claim.
  g('t-rec').textContent = c.tp + c.fn ? pct(c.tp / (c.tp + c.fn)) : '\\u2014';
  g('t-pre').textContent = c.tp + c.fp ? pct(c.tp / (c.tp + c.fp)) : 'undefined (no alarms)';

  const base = sweepAt(BASE_THR);
  const signed = n => (n >= 0 ? '+' : '\\u2212') + Math.abs(n);
  g('t-cmp').innerHTML = Math.abs(thr - BASE_THR) < 1e-9
    ? 'This is the tuned line the model shipped with.'
    : 'Against the tuned ' + BASE_THR.toFixed(2) + ' line: <b>' + signed(base.fn - c.fn) +
      '</b> caught, <b>' + signed(c.fp - base.fp) + '</b> false alarms.';
}

function renderThreshold(){
  const p = (thr * 100).toFixed(2) + '%';
  g('thr').style.left = p;
  g('hot').style.left = p;
  g('grab').style.left = p;
  g('thrlab').textContent = 'ALARM ' + thr.toFixed(2);
  g('grab').setAttribute('aria-valuenow', thr.toFixed(2));
  renderTally();
  renderVerdict();
}

function renderVerdict(){
  const stamp = g('stamp'), text = g('verdicttext');
  if (!current) {
    stamp.className = 'stamp';
    stamp.innerHTML = 'No record<small>pick one above</small>';
    text.textContent = 'Choose a record and the sheet fills in, the needle lands, and the ' +
      'tally on the right recounts at whatever alarm line you set.';
    g('needle').classList.add('none');
    return;
  }
  const p = current.prob;
  const alarm = p >= thr;
  const truth = current.sample.label === 'Sepsis';
  stamp.className = 'stamp ' + (alarm ? 'sep' : 'no');
  stamp.innerHTML = (alarm ? 'Sepsis' : 'No sepsis') +
    '<small>p = ' + p.toFixed(3) + ' at line ' + thr.toFixed(2) + '</small>';

  const right = alarm === truth;
  const outcome = truth ? 'did develop sepsis' : 'did not develop sepsis';
  text.innerHTML = '<b>' + (right ? 'Correct' : 'Wrong') + '.</b> This patient ' + outcome +
    '. ' + (right
      ? (alarm ? 'The alarm would have fired in time.'
               : 'No alarm, and none was needed.')
      : (alarm ? 'A false alarm \\u2014 the team would have been called for nothing.'
               : 'A miss \\u2014 the alarm stayed silent.'));
}

function setProb(p){
  const n = g('needle');
  n.classList.remove('none');
  n.style.left = (p * 100).toFixed(2) + '%';
  g('needlev').textContent = p.toFixed(2);
}

/* dragging the alarm line */
(function(){
  const ruler = g('ruler'), grab = g('grab');
  let dragging = false;
  const step = SWEEP ? SWEEP.step : 0.01;
  const fromX = x => {
    const r = ruler.getBoundingClientRect();
    return clamp(Math.round((x - r.left) / r.width / step) * step, 0, 1);
  };
  const move = e => { if (dragging) { thr = fromX(e.clientX); renderThreshold(); } };
  grab.addEventListener('pointerdown', e => {
    dragging = true; grab.setPointerCapture(e.pointerId); e.preventDefault();
  });
  grab.addEventListener('pointermove', move);
  grab.addEventListener('pointerup', () => { dragging = false; });
  grab.addEventListener('pointercancel', () => { dragging = false; });
  ruler.addEventListener('pointerdown', e => {
    if (e.target === grab) return;
    thr = fromX(e.clientX); renderThreshold();
  });
  grab.addEventListener('keydown', e => {
    const d = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step
            : e.key === 'Home' ? -1 : e.key === 'End' ? 1 : 0;
    if (!d) return;
    e.preventDefault();
    thr = Math.abs(d) === 1 ? (d < 0 ? 0 : 1) : clamp(+(thr + d).toFixed(2), 0, 1);
    renderThreshold();
  });
})();

/* ---------- prediction ---------- */
let busy = false;
async function pick(i){
  if (busy) return;
  busy = true;
  try { await run(i); } finally { busy = false; }
}

async function run(i){
  const sample = SAMPLES[i];
  if (!sample) return;
  lastDrawn = i;
  [...tabs.children].forEach(t => t.classList.toggle('on', +t.dataset.i === i));

  buildRows(sample);
  drawEcg(sample.features['HR_mean']);
  g('stamp').className = 'stamp';
  g('stamp').innerHTML = 'Scoring&hellip;<small>sending 50 values</small>';

  let prob = null;
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({features: sample.features}),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    prob = (await r.json()).probability_sepsis;
  } catch (err) {
    g('stamp').className = 'stamp';
    g('stamp').innerHTML = 'Unavailable<small>' + err.message + '</small>';
    g('verdicttext').textContent =
      'The model did not answer. The sheet above is still this patient\\u2019s real record.';
    return;
  }
  current = {sample, prob};
  setProb(prob);
  renderVerdict();
}

/* ---------- help tooltips ---------- */
const TIPS = {
  auc: '<b>Area under the ROC curve.</b> The chance the model gives a random patient who ' +
       'became septic a higher score than a random patient who did not. 0.5 is a coin flip, ' +
       '1.0 is perfect. It is measured across every threshold at once, so it does not move ' +
       'when you drag the alarm line.',
  held: 'Patients in the held-out test set. They were split off before any tuning and the ' +
        'model never saw them during training, feature selection, or threshold calibration \\u2014 ' +
        'so their scores are an honest estimate of new-patient performance.',
  feats: 'The 50 columns the model reads. They were reduced from 173 candidates by recursive ' +
         'feature elimination run <i>inside</i> cross-validation, so the selection could not ' +
         'peek at the test set. Each is one summary of one signal over one whole ICU stay.',
  model: 'Gradient-boosted decision trees (XGBoost), 196 of them, depth 7. It beat a tuned ' +
         'random forest on cross-validated F1, so it is the one that shipped. Served here from ' +
         'XGBoost\\u2019s own model format \\u2014 not a pickle \\u2014 so a library upgrade ' +
         'cannot quietly change a prediction.',
  strip: '<b>Illustrative, and the one thing on this sheet that is.</b> The beat spacing is ' +
         'this patient\\u2019s recorded mean heart rate, so a fast strip really is a fast ' +
         'patient. The beat <i>shape</i> is drawn: the Challenge data holds hourly summaries, ' +
         'never waveforms, so no real trace exists to plot.',
  index: 'Eighty real held-out stays, forty of each outcome. <b>Scroll the drawer sideways</b> ' +
         'and click any card to send its 50 values to the model. The numbers are index ' +
         'positions, not patient IDs \\u2014 the source data is de-identified and carries none.',
  draw: '<b>Random draw.</b> Each button picks a patient at random from the 80, and skips ' +
        'whichever one you are already looking at. The outcome buttons tell you what you asked ' +
        'for but not what the model will say \\u2014 that is still the model\\u2019s call, and ' +
        'it gets it wrong often enough to be worth trying.',
  band: 'The grey block is where the <b>middle half</b> of the training patients sat for that ' +
        'observation \\u2014 the 25th to 75th percentile. It is a cohort range, not a clinical ' +
        'reference range. The rail spans the 1st to 99th percentile. The mark is this patient; ' +
        'red means outside the middle half. Where the block is a thin sliver, three quarters of ' +
        'the cohort recorded the same value \\u2014 common for the <i>swing</i> columns, since ' +
        'most stays never move that lab at all.',
  weight: 'How much the trained model leans on that observation, from XGBoost\\u2019s gain ' +
          'importance, scaled against the strongest. The sheet is sorted by it, so the top ' +
          'rows are what actually moves the score. The three shaded rows are the top three.',
  value: 'This patient\\u2019s value, exactly as the model received it. \\u25b2 marks above the ' +
         'cohort\\u2019s middle half, \\u25bc below. Many columns are a <i>swing</i> or a ' +
         '<i>variability</i>: how much a signal moved across the stay, which for lactate turns ' +
         'out to matter more than any single reading.',
  nomo: '<b>Drag the black marker.</b> It sets the probability above which the model calls ' +
        'sepsis, and everything recounts live \\u2014 the verdict stamp, and all four numbers ' +
        'in the tally. You can also click anywhere on the ruler, or focus the marker and use ' +
        '<kbd>\\u2190</kbd> <kbd>\\u2192</kbd>. The red needle is this patient\\u2019s score ' +
        'and does not move; only the line does. 0.40 is where cross-validation put it.',
  tally: 'Where all 8,068 held-out patients land at the line you have set. These are counted, ' +
         'not modelled: the model scored the whole test set once at build time and the counts ' +
         'at all 101 thresholds are baked into this page. Drag the line left and you catch more ' +
         'sepsis and raise more false alarms; drag it right and the trade reverses.',
};

(function(){
  const tip = document.createElement('div');
  tip.className = 'tip';
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  let tipFor = null;

  function place(q){
    const r = q.getBoundingClientRect(), m = 10;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = clamp(r.left + r.width / 2 - w / 2, m, window.innerWidth - w - m);
    let top = r.bottom + 8;
    if (top + h > window.innerHeight - m) top = Math.max(m, r.top - h - 8);
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function show(q){
    const t = TIPS[q.dataset.tip];
    if (!t) return;
    tip.innerHTML = t;
    tip.classList.add('on');
    tipFor = q;
    place(q);
  }
  function hide(){ tip.classList.remove('on'); tipFor = null; }

  document.querySelectorAll('.q').forEach(q => {
    q.setAttribute('aria-label', 'What this means');
    q.addEventListener('pointerenter', () => show(q));
    q.addEventListener('pointerleave', hide);
    q.addEventListener('focus', () => show(q));
    q.addEventListener('blur', hide);
    q.addEventListener('click', e => { e.preventDefault(); tipFor === q ? hide() : show(q); });
  });
  window.addEventListener('scroll', () => { if (tipFor) place(tipFor); }, {passive: true});
  window.addEventListener('resize', () => { if (tipFor) place(tipFor); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
})();

renderThreshold();
</script>
</body></html>
"""
