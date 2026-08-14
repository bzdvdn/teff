# Single-container build for the repair-ai-chat live demo.
#
# The image installs teff (with the extras the app needs) from PyPI, copies
# the app source + the bundled seed CSV catalog, and starts uvicorn on the
# FastAPI app.  Point TEFF_PROVIDER_BASE_URL at an Ollama instance (see
# docker-compose.yml, or set it to your Ollama/OpenAI-compatible endpoint).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The app needs teff's fastapi + observability extras plus the pydantic
# settings used by src/config/config.py.  python-multipart is required by
# the FastAPI Form endpoints (e.g. POST /api/catalog/load).
RUN pip install --no-cache-dir \
    "teff[fastapi,observability]" \
    "pydantic-settings>=2.0" \
    "python-multipart>=0.0.9"

# App source + bundled catalog/traces seed data (data/documents + catalog.db).
COPY main.py app.py cli.py ./
COPY src ./src
COPY web ./web
COPY data/documents ./data/documents

ENV TEFF_HOST=0.0.0.0 \
    TEFF_PORT=8000 \
    TEFF_PROVIDER=ollama \
    TEFF_MODEL=qwen2.5:7b \
    TEFF_CHECKPOINT_DIR=/data/checkpoints \
    TEFF_TRACES_DB=/data/traces.db

EXPOSE 8000

# Store durable checkpoints on the writeable /data volume (declared by compose).
# TEFF_PROVIDER_BASE_URL is intentionally NOT set by default: point it at any
# OpenAI-compatible endpoint (e.g. http://ollama:11434 in compose, or a hosted
# Groq/OpenRouter/Together endpoint for a RAM-free VPS deployment).

CMD ["uvicorn", "app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]