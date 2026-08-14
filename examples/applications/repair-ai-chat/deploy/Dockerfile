# syntax=docker/dockerfile:1
# repair-ai-chat — production image.
#
# The build context is the *repository root*: the app depends on the `teff`
# package, which is installed from the same tree. `compose.yaml` sets
# `build.context: ../../..` accordingly.

# ------------------------------------------------- build the dependencies ----
FROM python:3.11-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

# Only what `pip wheel` needs to resolve/isolate the project.
COPY pyproject.toml README.md ./
COPY teff ./teff

# Wheel the `teff` package plus its runtime deps.  The `fastapi` extra
# provides the server stack; `queue` provides Celery for the worker/beat
# background jobs.  The heavy `embedding` extras are intentionally not
# included to keep the image small.
RUN pip install --upgrade pip \
    && pip wheel -w /wheels '.[fastapi,queue]'

# ------------------------------------------------------------- runtime ------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the wheels built above, then `pydantic-settings` (+ its transitive
# deps) from the index — it is not a `teff` dependency. Drop the wheel cache.
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels 'teff[fastapi,queue]' \
    && pip install pydantic-settings python-multipart \
    && rm -rf /wheels

# The application itself (src + the seed CSVs for the RAG catalog).
COPY --chown=65534:65534 examples/applications/repair-ai-chat/app.py examples/applications/repair-ai-chat/main.py examples/applications/repair-ai-chat/cli.py ./
COPY --chown=65534:65534 examples/applications/repair-ai-chat/src ./src
COPY --chown=65534:65534 examples/applications/repair-ai-chat/data ./data

# Durable runtime directory (SQLite catalog + session store).  Declared and
# owned here so a mounted named volume inherits this ownership on first use
# and the non-root user can write to it.
RUN mkdir -p /data && chown 65534:65534 /data

# Non-root user (65534 = nobody) for the runtime worker.
USER 65534

# Durable data lives on /data (mounted from compose); seed CSVs stay baked
# into the image under /app/data/documents.
ENV TEFF_CATALOG_DB=/data/catalog.db \
    TEFF_CHECKPOINT_DB=/data/checkpoints.db

EXPOSE 8000
ENTRYPOINT ["python", "main.py"]