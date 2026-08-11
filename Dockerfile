# Serving image for the trained ICU sepsis classifier.
# Build:  docker build -t sepsis-icu .
# Run:    docker run -p 8000:8000 sepsis-icu
#
# The exported model (artifacts/model.ubj + serving.json) is baked into the
# image at build time, so the container is self-contained and deploys anywhere
# with no volume mounts. Produce them with `python train.py` followed by
# `python export_serving.py`, then build. To iterate on the model without
# rebuilding, mount over it locally:  -v "$PWD/artifacts:/app/artifacts".
FROM python:3.11-slim

WORKDIR /app

# Install the serving requirements, then the package with --no-deps. Nothing
# here is unpickled — the model travels in XGBoost's own cross-version format —
# so these can be ranges rather than the exact lock this file used to need.
COPY requirements-serve.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-serve.txt \
    && pip install --no-cache-dir --no-deps .

# Bake in the trained model + demo samples (small). Build fails here if you
# haven't run `python train.py` yet â€” the desired safety check.
COPY artifacts/model.ubj ./artifacts/model.ubj
COPY artifacts/serving.json ./artifacts/serving.json
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
