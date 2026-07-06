"""FastAPI serving layer for the trained sepsis classifier.

Loads the model saved by ``train.py`` and exposes:

    GET  /         -> interactive demo landing page (ICU signal readout)
    GET  /health   -> liveness/readiness probe (used by Docker / orchestrators)
    GET  /model    -> metadata: model type, the features it expects, threshold
    POST /predict  -> {feature: value, ...} -> predicted class + P(sepsis)

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

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config, evaluate

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
_BUNDLE: dict | None = None


def load_bundle() -> dict:
    """Load and cache the model bundle saved by train.py."""
    global _BUNDLE
    if _BUNDLE is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {config.MODEL_PATH}. Run `python train.py` first."
            )
        _BUNDLE = joblib.load(config.MODEL_PATH)
        _log_event(
            "model_loaded",
            model_type=_BUNDLE["model_type"],
            n_features=len(_BUNDLE["features"]),
            threshold=_BUNDLE.get("threshold", 0.5),
        )
    return _BUNDLE


def load_examples() -> dict:
    """Real held-out test samples + stats baked into the image for the demo UI."""
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
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Interactive demo landing page — a live ICU signal readout."""
    ex = load_examples()
    if not ex.get("model_type"):
        try:
            ex["model_type"] = load_bundle()["model_type"]
        except Exception:
            ex["model_type"] = ""
    return _LANDING_PAGE.replace("__DATA__", json.dumps(ex))


@app.get("/health")
def health() -> dict:
    """Liveness probe. Reports whether a trained model is available."""
    return {"status": "ok", "model_available": config.MODEL_PATH.exists()}


@app.get("/model")
def model_info() -> dict:
    """Metadata about the loaded model, including the exact features it expects."""
    bundle = load_bundle()
    return {
        "model_type": bundle["model_type"],
        "classes": {"positive": bundle["class_pos"], "negative": bundle["class_neg"]},
        "threshold": bundle.get("threshold", 0.5),
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict sepsis vs no-sepsis for one patient's aggregated features.

    Unauthenticated and unthrottled by design (open demo, no sensitive data). Add
    rate limiting before public exposure — see docs/deploy.md "Hardening".
    """
    bundle = load_bundle()
    features = bundle["features"]
    threshold = float(bundle.get("threshold", 0.5))

    missing = [f for f in features if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing {len(missing)} required feature(s), e.g. {missing[:5]}",
        )

    # Order features exactly as the model expects, in a named 1-row frame so the
    # fitted pipeline sees the column names it was trained on (a bare list
    # triggers sklearn's "X does not have valid feature names" UserWarning on
    # every request, polluting the structured JSON logs).
    row = pd.DataFrame([[float(req.features[f]) for f in features]], columns=features)

    t0 = time.time()
    model = bundle["model"]
    prob = float(evaluate.predict_proba_pos(model, row)[0])
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
# A single self-contained page. `__DATA__` is replaced at request time with the
# demo JSON (features, per-feature stats, all held-out patients, top feature
# importances, headline + confusion metrics).
_LANDING_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ICU Sepsis Classifier - live signal readout</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='88'>&#129656;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0A0E14; --panel:#111823; --panel2:#0D141D; --line:#1E2A38;
    --ink:#E7EEF6; --muted:#7C8CA0; --faint:#4C5B6E;
    --sep:#FF5D73; --well:#38C6E8;
    --sans:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:
      radial-gradient(1200px 520px at 50% -220px,#13223300,#0A0E14 72%),var(--bg);
    color:var(--ink);font-family:var(--sans);line-height:1.55;
    -webkit-font-smoothing:antialiased;}
  .wrap{max-width:800px;margin:0 auto;padding:clamp(2rem,6vw,4.5rem) 1.25rem 4rem;}
  .eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.28em;
    text-transform:uppercase;color:var(--well);margin:0 0 1rem;}
  h1{font-size:clamp(2rem,5.4vw,3.15rem);font-weight:600;line-height:1.04;
    letter-spacing:-.025em;margin:0 0 .9rem;}
  h1 .em{color:var(--sep);}
  .lede{color:var(--muted);font-size:1.06rem;max-width:56ch;margin:0 0 1.9rem;}
  .head{display:flex;align-items:center;gap:.5rem;margin:0 0 .55rem;}
  .k{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;
    text-transform:uppercase;color:var(--muted);margin:0;}
  .q{width:18px;height:18px;border-radius:50%;border:1px solid var(--line);
    background:transparent;color:var(--muted);font-family:var(--mono);font-size:.72rem;
    line-height:16px;text-align:center;cursor:pointer;padding:0;flex:0 0 auto;}
  .q:hover{color:var(--ink);border-color:var(--faint);}
  .explain{display:none;margin:0 0 1rem;color:var(--muted);font-size:.86rem;
    border-left:2px solid var(--line);padding-left:.75rem;line-height:1.5;}
  .explain.open{display:block;}
  .stats{display:flex;flex-wrap:wrap;border:1px solid var(--line);
    border-radius:12px;overflow:hidden;margin-bottom:2.4rem;}
  .stat{flex:1 1 30%;min-width:130px;padding:.85rem 1.1rem;border-right:1px solid var(--line);}
  .stat:last-child{border-right:0;}
  .stat b{font-family:var(--mono);font-size:1.35rem;font-weight:500;display:block;
    letter-spacing:-.02em;}
  .stat .head{margin:.15rem 0 0;}
  .stat .k{letter-spacing:.1em;font-size:.66rem;}
  .stat .explain{margin:.5rem 0 0;font-size:.78rem;}
  .panel{background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line);border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;}
  .chips{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;}
  .chip{cursor:pointer;text-align:left;padding:.8rem .95rem;border-radius:10px;
    border:1px solid var(--line);background:#0e1620;color:var(--ink);
    font-family:var(--sans);font-size:.95rem;
    transition:border-color .15s,background .15s,transform .06s;}
  .chip:hover{border-color:var(--faint);background:#101c28;}
  .chip:active{transform:translateY(1px);}
  .chip.on{border-color:currentColor;}
  .chip .dot{width:8px;height:8px;border-radius:50%;display:inline-block;
    margin-right:.5rem;vertical-align:middle;}
  .chip .lab{font-family:var(--mono);font-size:.72rem;color:var(--muted);
    display:block;margin-top:.15rem;letter-spacing:.03em;}
  .chip.sepc{color:var(--sep);} .chip.wellc{color:var(--well);}
  .rand{margin-top:.6rem;width:100%;cursor:pointer;padding:.72rem;border-radius:10px;
    border:1px dashed var(--line);background:transparent;color:var(--muted);
    font-family:var(--mono);font-size:.82rem;letter-spacing:.03em;transition:.15s;}
  .rand:hover{color:var(--ink);border-color:var(--faint);background:#0e1620;}
  .readout{margin-top:1.4rem;opacity:0;max-height:0;overflow:hidden;transition:opacity .45s ease;}
  .readout.show{opacity:1;max-height:none;}
  .rhead{font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;
    text-transform:uppercase;color:var(--muted);display:flex;align-items:center;
    gap:.5rem;margin:.2rem 0 .6rem;}
  .heat{display:flex;gap:3px;flex-wrap:wrap;}
  .cell{flex:1 1 14px;min-width:10px;height:38px;border-radius:3px;background:#16202c;
    opacity:0;transform:translateY(6px);
    transition:opacity .28s ease,transform .28s ease,background .28s ease;}
  .readout.show .cell{opacity:1;transform:none;}
  .heatlabels{display:flex;justify-content:space-between;font-family:var(--mono);
    font-size:.66rem;color:var(--muted);margin-top:.5rem;letter-spacing:.04em;}
  .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;
    vertical-align:middle;margin:0 .25rem;}
  .verdict{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin:1.3rem 0 .1rem;}
  .verdict .big{font-size:1.55rem;font-weight:600;letter-spacing:-.02em;}
  .verdict.sepv .big{color:var(--sep);} .verdict.wellv .big{color:var(--well);}
  .verdict .sci{font-family:var(--mono);font-size:.8rem;color:var(--muted);}
  .match{font-family:var(--mono);font-size:.74rem;padding:.15rem .5rem;border-radius:20px;
    border:1px solid var(--line);}
  .match.ok{color:var(--well);} .match.no{color:var(--sep);}
  .meter{margin-top:1rem;}
  .meter .track{position:relative;height:12px;border-radius:6px;
    background:linear-gradient(90deg,var(--well),#16202c 50%,var(--sep));}
  .meter .needle{position:absolute;top:-5px;width:3px;height:22px;border-radius:2px;
    background:var(--ink);left:0;transition:left .6s cubic-bezier(.2,.8,.2,1);
    box-shadow:0 0 0 2px var(--bg);}
  .meter .thr{position:absolute;top:-3px;width:2px;height:18px;background:var(--faint);}
  .meter .ends{display:flex;justify-content:space-between;font-family:var(--mono);
    font-size:.68rem;color:var(--muted);margin-top:.45rem;letter-spacing:.08em;}
  .meter .pval{font-family:var(--mono);font-size:.82rem;color:var(--ink);
    text-align:center;margin-top:.5rem;}
  .cmatrix{display:grid;grid-template-columns:6.5rem 1fr 1fr;gap:.4rem;align-items:stretch;}
  .cmatrix .colh,.cmatrix .rowh{font-family:var(--mono);font-size:.64rem;color:var(--muted);
    display:flex;align-items:center;letter-spacing:.04em;}
  .cmatrix .colh{justify-content:center;text-align:center;}
  .cmatrix .rowh{justify-content:flex-end;text-align:right;padding-right:.4rem;}
  .cmcell{border-radius:8px;padding:.55rem;text-align:center;border:1px solid var(--line);}
  .cmcell b{font-family:var(--mono);font-size:1.3rem;display:block;line-height:1.1;}
  .cmcell span{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .cmcell.good{background:rgba(56,198,232,.09);border-color:rgba(56,198,232,.35);}
  .cmcell.bad{background:rgba(255,93,115,.09);border-color:rgba(255,93,115,.35);}
  .cmcell.zero{opacity:.55;}
  .perf{font-family:var(--mono);font-size:.86rem;margin-top:1rem;color:var(--muted);}
  .perf b{color:var(--well);}
  .bars{display:grid;gap:.5rem;}
  .brow{display:grid;grid-template-columns:8.5rem 1fr;gap:.7rem;align-items:center;
    font-family:var(--mono);font-size:.72rem;}
  .brow .gid{color:var(--muted);text-align:right;overflow:hidden;text-overflow:ellipsis;}
  .btrack{position:relative;height:16px;background:#0e1620;border-radius:4px;border:1px solid var(--line);}
  .bfill{position:absolute;top:0;height:100%;border-radius:3px;left:0;}
  .steps{display:grid;margin:.4rem 0 0;}
  .step{display:flex;gap:.9rem;padding:.78rem 0;border-top:1px solid var(--line);}
  .step .n{font-family:var(--mono);font-size:.75rem;color:var(--well);
    min-width:2rem;padding-top:.15rem;letter-spacing:.05em;}
  .step p{margin:0;color:var(--muted);font-size:.95rem;}
  .step b{color:var(--ink);font-weight:600;}
  footer{border-top:1px solid var(--line);margin-top:2.2rem;padding-top:1.3rem;
    font-family:var(--mono);font-size:.8rem;color:var(--muted);
    display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;align-items:center;}
  footer a{color:var(--muted);text-decoration:none;border-bottom:1px solid var(--line);}
  footer a:hover{color:var(--ink);}
  .hint{font-size:.8rem;color:var(--faint);margin:1rem 0 0;font-family:var(--mono);}
  @media(max-width:520px){.chips{grid-template-columns:1fr}.cell{height:30px}}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<div class="wrap">
  <p class="eyebrow">ICU sepsis classifier &middot; live demo</p>
  <h1>Read <span class="em">sepsis</span> from the<br>signals an ICU stay leaves behind.</h1>
  <p class="lede">A model trained on 40,336 ICU stays (PhysioNet/CinC 2019) decides
     whether a patient developed sepsis &mdash; from a few dozen aggregated vital-sign
     and lab summaries. Pick a real ICU stay it never saw and watch it read the signals.</p>

  <div class="stats">
    <div class="stat"><b id="s-auc">&mdash;</b>
      <div class="head"><span class="k">Test ROC-AUC</span><button class="q">?</button></div>
      <p class="explain">How well the model separates sepsis from no-sepsis across every
         decision threshold. 1.0 is flawless; 0.5 is a coin flip.</p></div>
    <div class="stat"><b id="s-feat">&mdash;</b>
      <div class="head"><span class="k">Features used</span><button class="q">?</button></div>
      <p class="explain">The final model reads only this many summary features &mdash;
         selected from 173 &mdash; so each prediction is compact and interpretable.</p></div>
    <div class="stat"><b id="s-cand">&mdash;</b>
      <div class="head"><span class="k">Candidates screened</span><button class="q">?</button></div>
      <p class="explain">Each stay is summarised into 173 candidate features
         (min/max/mean/std/range per vital &amp; lab); selection keeps the informative few.</p></div>
  </div>

  <div class="panel">
    <div class="head"><p class="k">Pick an ICU stay</p><button class="q">?</button></div>
    <p class="explain">Real held-out ICU stays the model never trained on. Click a labelled
       one, or draw a random stay &mdash; the model reads its signals live.</p>
    <div class="chips" id="chips"></div>
    <button class="rand" id="rand">&#127922; Draw a random held-out ICU stay</button>

    <div class="readout" id="readout">
      <div class="rhead"><span>Signal readout</span><button class="q">?</button></div>
      <p class="explain">Each cell is one feature for this stay, standardised against the
         training average. Cyan = below average, red = above average.
         Hover a cell for the feature name and value.</p>
      <div class="heat" id="heat"></div>
      <div class="heatlabels">
        <span><span class="swatch" style="background:var(--well)"></span>below average</span>
        <span id="heat-count">signal readout</span>
        <span>above average<span class="swatch" style="background:var(--sep)"></span></span>
      </div>
      <div class="verdict" id="verdict"></div>
      <div class="rhead" style="margin-top:1rem"><span>Model confidence</span><button class="q">?</button></div>
      <p class="explain">The probability the model assigns to sepsis. Above the calibrated
         threshold (tick mark) it calls sepsis; the needle shows exactly how sure it is.</p>
      <div class="meter">
        <div class="track"><div class="thr" id="thr"></div><div class="needle" id="needle"></div></div>
        <div class="ends"><span>No sepsis</span><span>Sepsis</span></div>
        <div class="pval" id="pval"></div>
      </div>
      <p class="hint" id="hint"></p>
    </div>
  </div>

  <div class="panel">
    <div class="head"><p class="k">Held-out performance</p><button class="q">?</button></div>
    <p class="explain">How the model does across every ICU stay it never saw during
       training or tuning. Green cells are correct calls; red are mistakes.</p>
    <div class="cmatrix" id="cmatrix"></div>
    <p class="perf" id="perf"></p>
  </div>

  <div class="panel">
    <div class="head"><p class="k">What the model weights</p><button class="q">?</button></div>
    <p class="explain">The features with the largest influence on the decision, by the
       model's own importance scores.</p>
    <div class="bars" id="bars"></div>
  </div>

  <div class="head" style="margin-top:2rem"><p class="k">How it works</p><button class="q">?</button></div>
  <p class="explain">The pipeline behind every prediction, built to keep the test set
     untouched from feature selection through threshold calibration.</p>
  <div class="steps">
    <div class="step"><span class="n">01</span><p><b>Aggregate</b> each patient's hourly
       ICU record into 173 summary features &mdash; min/max/mean/std/range per vital &amp; lab.</p></div>
    <div class="step"><span class="n">02</span><p><b>Select</b> the most informative features
       with recursive elimination run <b>inside</b> cross-validation, with SMOTE for the
       ~7% sepsis rate &mdash; no selection or resampling leakage.</p></div>
    <div class="step"><span class="n">03</span><p><b>Classify &amp; calibrate</b> with a tuned
       Random Forest / XGBoost, the decision threshold chosen on a validation split, served
       here as a live API.</p></div>
  </div>

  <footer>
    <span id="f-model">model</span>
    <a href="/docs">API docs</a>
    <a href="/model">/model</a>
    <a href="/health">/health</a>
    <a href="https://github.com/shiva-shivanibokka/Sepsis-ML-Model">GitHub</a>
  </footer>
</div>

<script>
const DATA = __DATA__;
const SEPSIS='Sepsis', WELL='No sepsis';
const C_WELL=[56,198,232], C_NEUT=[24,34,48], C_SEP=[255,93,115];
function mix(a,b,t){return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')+')';}
function divColor(z){const L=1.6,c=Math.max(-L,Math.min(L,z)),t=(c+L)/(2*L);
  return t<0.5?mix(C_WELL,C_NEUT,t*2):mix(C_NEUT,C_SEP,(t-0.5)*2);}

const meta=DATA.meta||{}, samples=DATA.samples||[];
const FEATURES=DATA.features||[];
const THRESH=(meta.threshold!=null?meta.threshold:0.5);

// header stats
document.getElementById('s-auc').textContent=meta.roc_auc!=null?meta.roc_auc.toFixed(3):'--';
document.getElementById('s-feat').textContent=FEATURES.length||'--';
document.getElementById('s-cand').textContent=meta.n_candidates?meta.n_candidates.toLocaleString():'173';
document.getElementById('f-model').textContent=(DATA.model_type||'model')+' - '+(meta.n_test||'?')+' test stays';
document.getElementById('heat-count').textContent='signal readout - '+FEATURES.length+' features';
document.getElementById('thr').style.left=(THRESH*100).toFixed(1)+'%';

// "?" explainers: toggle the .explain that follows each header
document.querySelectorAll('.q').forEach(q=>q.addEventListener('click',()=>{
  const ex=q.closest('.head,.rhead').nextElementSibling;
  if(ex&&ex.classList.contains('explain'))ex.classList.toggle('open');
}));

// featured chips: first two of each class
const featured=[...samples.filter(s=>s.label===SEPSIS).slice(0,2),
                ...samples.filter(s=>s.label===WELL).slice(0,2)];
const chips=document.getElementById('chips');
featured.forEach(s=>{
  const b=document.createElement('button');
  b.className='chip '+(s.label===SEPSIS?'sepc':'wellc');
  b.innerHTML='<span class="dot" style="background:currentColor"></span>'+
    (s.label===SEPSIS?'Septic stay':'Non-septic stay')+
    '<span class="lab">confirmed: '+(s.label===SEPSIS?'sepsis':'no sepsis')+'</span>';
  b.addEventListener('click',()=>predict(s,b));
  chips.appendChild(b);
});
document.getElementById('rand').addEventListener('click',()=>{
  if(!samples.length)return;
  const s=samples[Math.floor(Math.random()*samples.length)];
  predict(s,null);
});
if(!samples.length){chips.innerHTML='<p class="hint">No demo samples baked in. Run <code>python train.py</code>.</p>';}

// confusion matrix
const cm=meta.confusion||{};
function cell(v,cls,lab){return '<div class="cmcell '+cls+(v===0?' zero':'')+'"><b>'+v+'</b><span>'+lab+'</span></div>';}
if(cm.tp!=null){
  document.getElementById('cmatrix').innerHTML=
    '<div></div><div class="colh">predicted sepsis</div><div class="colh">predicted none</div>'+
    '<div class="rowh">actual<br>sepsis</div>'+cell(cm.tp,'good','caught')+cell(cm.fn,'bad','missed')+
    '<div class="rowh">actual<br>none</div>'+cell(cm.fp,'bad','false alarm')+cell(cm.tn,'good','cleared');
  const total=cm.tp+cm.tn+cm.fp+cm.fn, correct=cm.tp+cm.tn;
  document.getElementById('perf').innerHTML=
    '<b>'+cm.tp+' / '+(cm.tp+cm.fn)+'</b> sepsis caught &nbsp;&middot;&nbsp; '+
    ((meta.accuracy!=null?meta.accuracy*100:correct/total*100).toFixed(1))+'% accuracy &nbsp;&middot;&nbsp; '+
    cm.fp+' false alarms';
}

// top feature importances
const tg=DATA.top_features||[];
const maxw=Math.max(1e-9,...tg.map(t=>Math.abs(t.weight)));
document.getElementById('bars').innerHTML=tg.map(t=>{
  const w=Math.abs(t.weight)/maxw*100;
  return '<div class="brow"><span class="gid" title="'+t.feature+'">'+t.feature+'</span>'+
    '<div class="btrack"><div class="bfill" style="width:'+w+'%;background:var(--sep)"></div></div></div>';
}).join('');

// heatmap
function buildHeat(features){
  const heat=document.getElementById('heat'); heat.innerHTML='';
  FEATURES.forEach((g,i)=>{
    const st=(DATA.stats||{})[g]||{mean:0,std:1};
    const z=(features[g]-st.mean)/(st.std||1);
    const cell=document.createElement('div');
    cell.className='cell'; cell.style.transitionDelay=(i*14)+'ms';
    requestAnimationFrame(()=>{cell.style.background=divColor(z);});
    cell.title=g+'  -  '+Number(features[g]).toFixed(2)+'  -  z '+z.toFixed(2);
    heat.appendChild(cell);
  });
}

let busy=false;
async function predict(sample,btn){
  if(busy)return; busy=true;
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));
  if(btn)btn.classList.add('on');
  const ro=document.getElementById('readout'); ro.classList.add('show');
  document.getElementById('hint').textContent='reading signals...';
  buildHeat(sample.features);
  try{
    const r=await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({features:sample.features})});
    const d=await r.json();
    const pct=d.probability_sepsis*100, isSep=d.prediction===SEPSIS, ok=d.prediction===sample.label;
    const v=document.getElementById('verdict');
    v.className='verdict '+(isSep?'sepv':'wellv');
    v.innerHTML='<span class="big">'+(isSep?'Sepsis':'No sepsis')+'</span>'+
      '<span class="sci">'+(isSep?'flagged high-risk':'cleared')+'</span>'+
      '<span class="match '+(ok?'ok':'no')+'">'+(ok?'matches outcome':'vs confirmed '+sample.label)+'</span>';
    document.getElementById('needle').style.left=pct.toFixed(1)+'%';
    document.getElementById('pval').textContent='P(sepsis) = '+pct.toFixed(1)+'%  (threshold '+(THRESH*100).toFixed(0)+'%)';
    document.getElementById('hint').textContent=FEATURES.length+' feature values -> live model -> prediction ('+
      (btn?'labelled stay':'random held-out stay')+')';
  }catch(e){
    document.getElementById('hint').textContent='Error: '+e;
  }
  busy=false;
}
</script>
</body></html>"""
