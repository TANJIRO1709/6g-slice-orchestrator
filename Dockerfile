# ─────────────────────────────────────────────────────────
# 6G Slice Orchestrator — single-container deployment
#
# Stage 1 compiles the Next.js dashboard to static files.
# Stage 2 installs the Python backend and serves those files
# from FastAPI, so the whole app lives behind one URL with
# no CORS and no cross-origin WebSocket.
#
# Build from the PROJECT ROOT (the folder holding backend/
# and frontend/), not from inside either one:
#     docker build -t 6g-orchestrator .
# ─────────────────────────────────────────────────────────

# ── Stage 1: build the dashboard ─────────────────────────
FROM node:20-slim AS frontend

WORKDIR /ui

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./

# Empty API URL => the dashboard uses relative paths and
# derives the WebSocket URL from window.location, so it
# works on whatever hostname it ends up deployed to.
ENV NEXT_PUBLIC_API_URL=""
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build


# ── Stage 2: backend + static dashboard ──────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache \
    DIAGNOSE_EVERY=20

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first. The default PyPI wheel pulls ~2GB of
# CUDA libraries that are useless on a CPU host; this keeps
# the image around 1.2GB instead of 4GB+.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# Copy the compiled dashboard in. FastAPI serves it.
COPY --from=frontend /ui/out ./static

# Build the RAG index at image-build time. This also pulls the
# ~80MB ONNX embedding model into the image layer, so the first
# request after deploy is fast instead of timing out while the
# model downloads.
RUN python -m rag.indexer --local

# Hugging Face Spaces expects 7860; Render/Railway inject $PORT.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
