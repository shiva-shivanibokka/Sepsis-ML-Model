# Deploying the API

The FastAPI service ships as a self-contained Docker image (the trained model is
baked in), so it runs on any container platform. Two paths are documented:
**Google Cloud Run** (primary) and **Fly.io** (alternative).

> Not yet deployed to a public URL. After your first deploy, paste the live URL
> here and in the README so the demo is one click away.

## Prerequisites

1. A trained model exists: `python train.py` (creates `artifacts/model.joblib`
   and `artifacts/examples.json`). Point it at your dataset first — see the
   README "How to Run".
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

## Fly.io (alternative)

The repo includes a `fly.toml` (scale-to-zero, `/health` check). Note Fly's
current new-account trial is time-limited.

```bash
# Windows install: iwr https://fly.io/install.ps1 -useb | iex
fly auth login
fly apps create <unique-name>       # then set app = "<unique-name>" in fly.toml
fly deploy --remote-only            # remote build, no local Docker
```

Logs: `fly logs`.

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
