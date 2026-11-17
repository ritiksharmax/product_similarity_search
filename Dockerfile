# syntax=docker/dockerfile:1.7
#
# Two stages:
#   builder - installs the *build* toolchain (incl. torch/sentence-transformers when
#             ENCODER=minilm), unzips the dataset and precomputes the feature matrix +
#             HNSW graph into /app/artifacts.
#   runtime - slim image with only the serving dependencies (numpy, faiss, fastapi) and the
#             prebuilt artifacts. Queries are *by product id*, so the text model never runs
#             at serving time and torch never ships to production.
#
#   docker build -t product-similarity:1.0.0 .                       # accurate (MiniLM) - default
#   docker build --build-arg ENCODER=tfidf -t product-similarity:tfidf .  # efficient, no model download

ARG PYTHON_IMAGE=python:3.11-slim

# ----------------------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
ARG ENCODER=minilm
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/tmp/hf OMP_NUM_THREADS=1
WORKDIR /app

# 1) Serving venv (no torch), installed first so the layer caches across code changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Build-only venv: adds the MiniLM extra (CPU-only torch wheel, see pyproject [tool.uv]).
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$ENCODER" = "minilm" ]; then \
      UV_PROJECT_ENVIRONMENT=/opt/buildenv uv sync --frozen --no-dev --no-install-project --extra minilm; \
    else \
      UV_PROJECT_ENVIRONMENT=/opt/buildenv uv sync --frozen --no-dev --no-install-project; \
    fi

COPY src ./src
COPY scripts ./scripts
COPY data/archive.zip ./data/archive.zip
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable \
 && UV_PROJECT_ENVIRONMENT=/opt/buildenv uv sync --frozen --no-dev --no-editable $([ "$ENCODER" = "minilm" ] && echo --extra minilm)

# 3) Precompute artifacts with the build venv, then drop the raw data.
RUN python -m zipfile -e data/archive.zip data/ \
 && SIM_ENCODER=$ENCODER /opt/buildenv/bin/python scripts/build_artifacts.py \
 && rm -rf data

# ----------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
ARG ENCODER=minilm
LABEL org.opencontainers.image.title="product-similarity" \
      org.opencontainers.image.description="Amazon Fashion product similarity search (FastAPI + FAISS HNSW)"

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=10001:10001 /app/artifacts /app/artifacts
COPY app.py ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    NAME=ProductSimilarityApp \
    SIM_ARTIFACTS_DIR=/app/artifacts \
    SIM_ENCODER=${ENCODER} \
    SIM_BACKEND=hnsw \
    OMP_NUM_THREADS=1

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)" || exit 1

# One worker per container: scale horizontally with replicas/HPA rather than processes,
# so each pod's memory (index + vectors) stays predictable.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
