"""Chat endpoints — single-shot reply and an SSE token stream.

Handlers are thin: they read the :class:`~teff.assistant.Assistant` off
``request.app.state`` and delegate one turn to it.  Sessions are scoped
to a caller id (``X-User-Id`` header, enforced by
:func:`~src.api.auth.router.require_user_id`) and durable across requests
and process restarts.

Endpoints:
    POST   /api/chat         single-shot reply (runs the flow once)
    POST   /api/chat/stream  SSE event stream over ``graph.stream()``

The stream ends with a ``message`` event carrying the full assistant reply
(``{"session_id", "reply"}``), so clients never have to concatenate ``token``
events themselves — tool-using agents may not stream tokens at all.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from src.api.auth.router import require_api_key, require_user_id
from sse_starlette.sse import EventSourceResponse

from teff.observability import GraphObserver

router = APIRouter(dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = "Hi! Where should we start?"
    max_iterations: int = 80


def _session(request: Request, owner: str):
    """The durable session assets plus a fresh, server-generated session id.

    The session id is **always** generated here — never taken from the
    client — so callers cannot address or hijack another user's session by
    supplying an id.  Sessions live in the caller's own namespace (``owner``).
    """
    session_id = uuid.uuid4().hex
    return request.app.state.assistant, owner, session_id


def _observer(request: Request, owner: str, session_id: str) -> GraphObserver | None:
    """A fresh GraphObserver for one turn (None when tracing is disabled)."""
    exporter = request.app.state.traces_exporter
    if exporter is None:
        return None
    return GraphObserver(
        "chat",
        exporter=exporter,
        topology=request.app.state.trace_topology,
        owner=owner,
        checkpoint_id=session_id,
    )


def _finish(observer: GraphObserver | None) -> str | None:
    """Persist the captured run when an observer was active.

    The exporter itself is owned by the app (``app.state.traces_exporter``)
    and shared across requests, so only the run is written here — never
    closed.  Returns the persisted run id (``None`` when tracing is off).
    """
    if observer is None:
        return None
    return observer.export()


@router.post("")
async def chat(
    req: ChatRequest,
    request: Request,
    owner: str = Depends(require_user_id),
) -> dict:
    assistant, owner, session_id = _session(request, owner)
    observer = _observer(request, owner, session_id)
    try:
        result = await assistant.run(
            session_id,
            req.message,
            owner=owner,
            max_iterations=req.max_iterations,
            tracer=observer.tracer if observer else None,
            on_llm_payload=observer.on_llm_payload if observer else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="internal error") from exc
    finally:
        run_id = _finish(observer)
    return {
        "session_id": session_id,
        "reply": result.reply,
        "run_id": run_id,
        "waiting": result.waiting,
        "prompt": result.prompt if result.waiting else None,
        "question": result.prompt if result.waiting else None,
    }


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    owner: str = Depends(require_user_id),
) -> EventSourceResponse:
    assistant, owner, session_id = _session(request, owner)

    async def events():
        yield {"event": "chat_id", "data": json.dumps({"session_id": session_id})}
        observer = _observer(request, owner, session_id)
        try:
            async for event in assistant.stream(
                session_id,
                req.message,
                owner=owner,
                max_iterations=req.max_iterations,
                tracer=observer.tracer if observer else None,
                on_llm_payload=observer.on_llm_payload if observer else None,
            ):
                data = {"session_id": session_id}
                if event.node_id is not None:
                    data["node_id"] = event.node_id
                if event.node_type is not None:
                    data["node_type"] = event.node_type
                data.update(event.data)
                yield {"event": event.type, "data": json.dumps(data)}
        finally:
            run_id = _finish(observer)
        reply = await assistant.last_reply(session_id, owner=owner)
        yield {
            "event": "message",
            "data": json.dumps(
                {"session_id": session_id, "reply": reply, "run_id": run_id}
            ),
        }

    return EventSourceResponse(events())
