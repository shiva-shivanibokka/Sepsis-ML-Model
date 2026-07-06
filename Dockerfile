# Serving image for the trained ICU sepsis classifier.
# Build:  docker build -t sepsis-icu .
# Run:    docker run -p 8000:8000 sepsis-icu
#
# The trained model (artifacts/model.joblib) is baked into the image at build
# time, so the container is self-contained and deploys anywhere (Cloud Run,
# Fly.io, etc.) with no volume mounts. Produce the model first with
# `python train.py`, then build. To iterate on the model without rebuilding,
# mount over it locally:  -v "$PWD/artifacts:/app/artifacts".
FROM python:3.11-slim

WORKDIR /app

# Package source + metadata, then install with ONLY the serving deps. `.[serve]`
# pulls the lean core (pandas/numpy/scikit-learn/xgboost/imbalanced-learn/joblib)
# plus fastapi/uvicorn/pydantic — and deliberately NOT the training/notebook libs
# (matplotlib, seaborn, scikit-optimize), which the runtime never imports. Keeps
# the image small and cold starts fast.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

# Bake in the trained model + demo samples (small). Build fails here if you
# haven't run `python train.py` yet — the desired safety check.
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/examples.json ./artifacts/examples.json

EXPOSE 8000

# Listen on $PORT when the platform provides one (Cloud Run injects PORT=8080),
# else default to 8000 for local runs and Fly.
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn sepsis_icu.serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
