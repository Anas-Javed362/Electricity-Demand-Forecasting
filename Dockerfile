# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools for C extensions (prophet, shap, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy ALL source code so the image is self-contained
COPY src/  ./src/
COPY api/  ./api/
COPY app/  ./app/

# Copy root-level dashboard (alternative entrypoint)
COPY app.py ./app.py

# Models, data and mlruns are mounted as volumes at runtime
# (kept out of the image to avoid bloating the layer)
VOLUME ["/app/models", "/app/data", "/app/mlruns"]

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
