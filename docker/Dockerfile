# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Bitcoin Time Series API (FastAPI)
# Multi-stage build : builder → runtime
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1 : builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Dépendances système pour compilation (numpy, scipy, prophet…)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances dans un venv isolé
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2 : runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Métadonnées
LABEL maintainer="Bitcoin TS Project" \
      version="1.0.0" \
      description="Bitcoin Time Series Forecasting API"

WORKDIR /app

# Dépendances runtime uniquement (pas de build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copie du venv depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copie du code source
COPY api/        ./api/
COPY models/     ./models/
COPY data/       ./data/

# Variables d'environnement
ENV PORT=8000 \
    WORKERS=2 \
    LOG_LEVEL=info \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck intégré
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Lancement avec Gunicorn + Uvicorn workers
CMD ["sh", "-c", \
     "uvicorn api.main:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers ${WORKERS} \
      --log-level ${LOG_LEVEL} \
      --access-log"]
