# Deploying the API

The FastAPI service deploys two ways: **Vercel** (primary — what the live demo
runs on) and a self-contained **Docker image** for any container platform, with
Google Cloud Run documented below.

**Live:** https://sepsis-icu-classifier.vercel.app (Vercel, serverless).

> Things that will break a freshly-deployed service if you skip them (all handled
> in this repo — noted here so a fork doesn't relearn them):
>
> 1. **Export the model before deploying.** The service loads `artifacts/model.ubj`
>    + `artifacts/serving.json`, written by `python export_serving.py`. Without them
>    it falls back to unpickling `model.joblib`, which needs scikit-learn, pandas and
>    joblib installed — none of which are in the serving dependency set.
> 2. **Keep the package `__init__` lazy.** `data`, `features` and `models` import
>    scikit-learn and imbalanced-learn. Importing them from the package root means
>    `import sepsis_icu.serve` pulls the whole training stack in, which the deployment
>    does not install. Submodules resolve on first access instead.
> 3. **Use `xgboost-cpu` on Linux.** The ordinary wheel bundles a CUDA runtime and
>    unpacks to 84 MB against 23 MB, which is the difference between a 525 MB bundle
>    and a 200 MB one. Vercel's function limit is 500 MB.
> 4. **Tell the container where the baked-in model is.** The package is pip-installed
>    into site-packages, so `config`'s `__file__`-relative default points at the Python
>    lib dir, not `/app/artifacts`. The Dockerfile sets
>    `ENV SEPSIS_ARTIFACTS_DIR=/app/artifacts`. Symptom if missing: `/health` still
>    returns 200 (`model_available: false`) but `/model` and `/predict` 500.

---

## Vercel (primary)

`api/index.py` exposes the FastAPI app as the ASGI entrypoint; `vercel.json` sets
which files travel with the function. Dependencies resolve from `pyproject.toml`,
so the serving set has to live in `[project] dependencies` rather than an extra.

```bash
python train.py                     # writes artifacts/model.joblib
python export_serving.py            # verifies, then writes model.ubj + serving.json
npx vercel deploy                   # preview
npx vercel deploy --prod            # production
```

Verify a deployment scores identically to the local pipeline before trusting it —
`/model` reports `loaded_from`, which should read `exported`.

## Prerequisites

1. A trained model exists: `python train.py` (creates `artifacts/model.joblib`
   and `artifacts/examples.json`), followed by `python export_serving.py` (creates
   `artifacts/model.ubj` and `artifacts/serving.json`, and refuses to write them
   unless they reproduce the pipeline exactly). Point training at your dataset
   first — see the README "How to Run".
2. The image respects `$PORT` (Cloud Run injects `8080`); see the `Dockerfile`.

---

## Local smoke test first

```bash
pip install -e ".[serve]"
uvicorn sepsis_icu.serve:app --reload
# open http://127.0.0.1:8000  (demo)  and  http://127.0.0.1:8000/docs  (API)
curl http://127.0.0.1:8000/health
```

Or the whole container:

```bash
docker build -t sepsis-icu .
docker run -p 8000:8000 sepsis-icu
```

---

## Google Cloud Run (primary)

Cloud Run builds from source via **Cloud Build** — no local Docker required — and
gives a permanent `*.run.app` URL on a generous perpetual free tier (2M
requests/month, scales to zero).

### One-time project setup

```bash
gcloud auth login
gcloud projects create sepsis-icu-portfolio --name "ICU Sepsis Classifier"
gcloud billing projects link sepsis-icu-portfolio --billing-account <YOUR_BILLING_ID>
gcloud config set project sepsis-icu-portfolio
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

(`gcloud billing accounts list` shows your billing account id.)

### Deploy

```bash
gcloud run deploy sepsis-icu \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1
```

`.dockerignore` keeps the raw CSV and notebooks out of the build context. The
command builds the Dockerfile, pushes to Artifact Registry, deploys, and prints
the service URL.

### Verify

```bash
URL=https://<your-service>.run.app
curl $URL/health
curl $URL/model                     # the features it expects + calibrated threshold
# interactive docs: $URL/docs
```

### Operations

- **Logs** (structured JSON prediction events): `gcloud run services logs read sepsis-icu --region us-central1`
- **Redeploy after retraining:** `python train.py && gcloud run deploy sepsis-icu --source . --region us-central1`
- **Cost:** Cloud Run defaults to scale-to-zero; the first request after idle
  cold-starts for a few seconds.
- **Roll back:** `gcloud run services update-traffic sepsis-icu --to-revisions <REV>=100 --region us-central1`

---

## A note on Fly.io

This service ran on Fly.io until the account's trial ended, at which point the app
was suspended and the demo went dark — TLS handshakes failing on a hostname that
still resolved. `fly.toml` has been removed. If you redeploy there, keep in mind
that a suspended app fails in a way that looks like a network problem rather than
a billing one.

---

## Hardening before public exposure

The demo is intentionally minimal. Before putting `/predict` on a public URL, add
abuse protection — it is unauthenticated and runs model inference per request:

- **Simplest:** rely on the platform. Cloud Run's `--max-instances` and per-instance
  `--concurrency` bound total load; a WAF / API-gateway rate limit covers per-IP abuse.
- **In-app (if you want per-IP limits in the code):** add [`slowapi`](https://github.com/laurentS/slowapi),
  register a `Limiter(key_func=get_remote_address)` on `app.state.limiter`, and decorate
  `/predict` with e.g. `@limiter.limit("60/minute")`. Left out of the base image on
  purpose so the demo stays dependency-light until it's actually deployed.

There is no auth on the endpoints by design: the model exposes no sensitive data and
the demo is meant to be openly clickable. Add auth only if you gate it behind a login.

## For the resume

The live URL + `$URL/docs` demonstrate: containerization, source-to-Cloud-Run
build via Cloud Build, a health-checked deployment, scale-to-zero cost awareness,
structured logging/observability, and a versioned model artifact — the
production-deployment gap most student portfolios miss.
