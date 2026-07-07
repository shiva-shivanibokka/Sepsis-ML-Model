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

# Install a PINNED serving lock, then the package itself with --no-deps. This is
# deliberate: the model artifact is a pickle, so the serving env must match the
# versions that created it (numpy 2.x cannot unpickle arrays written by 1.26, and
# sklearn pickles are version-sensitive). Using `.[serve]` here would let pip
# resolve the pyproject ranges to the latest releases and break model loading.
# requirements-serve.txt covers everything the runtime imports (including
# imbalanced-learn, pulled transitively when the package __init__ imports
# features/models) but NOT the train-only libs (matplotlib, seaborn, scikit-optimize).
COPY requirements-serve.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-serve.txt \
    && pip install --no-cache-dir --no-deps .

# Bake in the trained model + demo samples (small). Build fails here if you
# haven't run `python train.py` yet — the desired safety check.
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/examples.json ./artifacts/examples.json

EXPOSE 8000

# Point the app at the baked-in artifacts. Required because the package is
# pip-installed into site-packages, so config._default_data_dir() (which resolves
# relative to the package's __file__) would otherwise look in the Python lib dir
# instead of /app/artifacts where the model + demo data are COPYed above.
ENV SEPSIS_ARTIFACTS_DIR=/app/artifacts

# Listen on $PORT when the platform provides one (Cloud Run injects PORT=8080),
# else default to 8000 for local runs and Fly.
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn sepsis_icu.serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
