"""FastAPI application factory for the ``repair-ai-chat`` application.

Run (from the example root)::

    uv sync --extra api
    uv run python main.py              # or: uv run uvicorn app:create_app

Endpoint groups live in :mod:`src.api` and are assembled by
:mod:`src.api.router`; this module only builds the app and wires the
durable assets (graph, tools, checkpointer) onto ``app.state``.  Nothing
runs at import time, and the LLM provider/model come from
:class:`src.config.config.Settings` — no global defaults are mutated.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.router import api_router
from src.config.config import Settings, get_settings
from src.core.deps import build_deps
from src.graphs.build import build_flow
from src.graphs.state import STATE_REDUCERS, initial_state
from src.storage import TRANSIENT_KEYS, build_checkpointer

from teff import Assistant
from teff.observability import SQLiteExporter, topology_from_graph
from teff.observability.api import attach_dashboard


def create_app(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its graph, tools and checkpointer.

    Assets are built once and carried on ``app.state``.  Pass a
    ``Settings`` to override environment defaults (tests do this);
    ``checkpoint_dir`` is a convenience override for the storage location.
    """
    settings = settings or get_settings()
    if checkpoint_dir is not None:
        settings = settings.model_copy(update={"checkpoint_dir": checkpoint_dir})

    services, catalog = build_deps(
        provider=settings.provider, catalog_db=settings.catalog_db
    )
    flow, tools = build_flow(
        model=settings.model,
        provider=settings.provider,
        services=services,
        catalog=catalog,
        provider_base_url=settings.provider_base_url,
    )
    compiled = flow.compile()
    assistant = Assistant(
        compiled,
        tools,
        build_checkpointer(
            settings.checkpoint_dir, checkpoint_db=settings.checkpoint_db
        ),
        reducers=STATE_REDUCERS,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )

    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.version,
    )
    app.state.assistant = assistant
    app.state.catalog = catalog
    app.state.model = settings.model
    app.state.settings = settings
    app.include_router(api_router)

    # Trace dashboard: every chat turn is captured by a GraphObserver into a
    # local SQLite store and browsable at /obs/ui (prefix from settings).
    traces_path = settings.traces_db or (
        Path(__file__).resolve().parent / "data" / "traces.db"
    )
    traces_exporter = SQLiteExporter(str(traces_path))
    app.state.traces_exporter = traces_exporter
    app.state.trace_topology = topology_from_graph(compiled)
    attach_dashboard(app, traces_exporter, prefix=settings.traces_prefix)

    # Static web UI: a build-free chat page that talks to `/api/chat/stream`.
    # Mounted last, so `/api/*` and the trace dashboard win.
    _web = Path(__file__).resolve().parent / "web"
    if _web.is_dir():
        app.mount("/", StaticFiles(directory=str(_web), html=True), name="web")

    return app
