"""Typed application settings loaded from the environment / ``.env``.

Every knob the server needs lives here — the LLM ``provider``/``model``,
the optional ``api_key``, the bind address, app metadata and the storage
and RAG paths — so ``main.py``, ``app.py`` and ``cli.py`` share a single
source of truth.  Override any value with an environment variable prefixed
``TEFF_`` (e.g. ``TEFF_PORT=9000``) or a line in a local ``.env`` file.

The layout mirrors the ``teff new`` scaffold.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the ``repair-ai-chat`` service."""

    model_config = SettingsConfigDict(
        env_prefix="TEFF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: LLM provider (``"ollama"``, ``"openai"``, ...) and default model.
    provider: str = "ollama"
    model: str = "llama3.1:8b"

    #: Optional base URL override for the chat provider (e.g. the compose
    #: demo points the app at ``http://ollama:11434``).  ``None`` keeps the
    #: preset default (``http://localhost:11434`` for Ollama).
    provider_base_url: str | None = None

    #: When set, the chat/run routers require the ``X-API-Key`` header.
    api_key: str | None = None

    #: Bind address for ``main.py``.
    host: str = "127.0.0.1"
    port: int = 8000

    #: App metadata surfaced by FastAPI.
    app_title: str = "Production Repair AI"
    app_description: str = (
        "Supervisor flow for repair planning over room, budget and materials agents."
    )
    version: str = "0.1.0"

    #: Durable session storage and RAG data paths (None = project defaults).
    checkpoint_dir: str | None = None
    catalog_csv: str | None = None

    #: Trace-dashboard persistence (None = ``data/traces.db`` next to the app).
    #: The dashboard UI is mounted by ``app.py`` under ``traces_prefix``.
    traces_db: str | None = None
    traces_prefix: str = "/obs"

    #: SQLite persistence (shared file so API + workers read the same data).
    #: When set, ``checkpoint_db`` replaces the JSON-file checkpointer and
    #: ``catalog_db`` replaces the in-memory RAG vector store.
    checkpoint_db: str | None = None
    catalog_db: str | None = None

    #: PostgreSQL DSN (``postgres://...``).  When set, it wins over the
    #: SQLite paths above: both the RAG vector store (pgvector) and session
    #: checkpoints live in Postgres, shared by every process.
    database_url: str | None = None

    #: RAG embedder provider and top-k used by the catalog.
    rag_embedder: str = "ollama"
    rag_top_k: int = 3

    #: Redis broker URL for the Celery worker/beat tasks (e.g. a re-ingest
    #: scheduler).  ``None`` disables background jobs.
    redis_url: str | None = None

    #: Where the re-ingest beat task stores its change-detection marker.
    #: ``None`` = alongside the durable catalog DB.
    catalog_ingest_state: str | None = None

    #: Where Celery beat keeps its persistent schedule DB (writable dir —
    #: ``/app`` is read-only for the non-root runtime user).
    beat_schedule: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings (the environment is read once per process)."""
    return Settings()
