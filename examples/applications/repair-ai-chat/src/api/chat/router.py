"""Chat endpoints — single-shot reply and an SSE token stream.

Handlers are thin: they read the :class:`~teff.assistant.Assistant` off
``request.app.state`` and delegate one turn to it.  Every chat session runs
in its own owner namespace (the session id, or an ``X-User-Id`` header if
given), is durable across requests and process restarts, and never shares
state or traces with other sessions.

Endpoints:
    POST   /api/chat/stream  chat-style SSE stream (see below)

Stream events (chat-like by default; append ``?raw=1`` for the underlying
framework events):

* ``chat_id`` — the durable session id.
* ``status`` — the coordinator picked a tool (e.g. ``Составляю смету…``).
* ``content`` — streamed tokens when an agent streams them.
* ``message`` — the terminal event (``{"session_id", "reply", "run_id",
  "waiting"}``), a client can render it without concatenating ``content``
  events.  When the workflow paused asking the operator a question,
  ``reply`` carries that question and ``waiting`` is ``True`` — resume by
  posting the answer to the same session.  Otherwise ``reply`` is the full
  assistant answer and ``waiting`` is ``False``.

Human-in-the-loop: pause handling lives in the framework's
:class:`~teff.assistant.Assistant`.  Its ``turn``/``stream`` methods detect a
paused interrupt from the durable checkpoint and resume the run with the next
message — so this endpoint never sees a ``GraphInterrupt`` or a ``pending``
map.  It surfaces the question through the terminal ``message`` (``waiting``)
and the operator's answer resumes the run in the same session.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.api.auth.router import require_api_key
from teff.observability import GraphObserver

router = APIRouter(dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = "Помоги спланировать ремонт ванной комнаты, 5 м²."
    session_id: str | None = None
    max_iterations: int = 80


def _session(req: ChatRequest, request: Request, x_user_id: str | None):
    """The durable session assets plus the session id for one turn.

    Each session runs in its own owner namespace: when no ``X-User-Id`` is
    given, the owner is the session id itself, so sessions never share
    durable state or traces.
    """
    session_id = req.session_id or uuid.uuid4().hex
    owner = x_user_id or session_id
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


def _tracer_kwargs(observer: GraphObserver | None) -> dict:
    """The tracer/on_llm_payload kwargs for one turn."""
    return {
        "tracer": observer.tracer if observer else None,
        "on_llm_payload": observer.on_llm_payload if observer else None,
    }


@router.post("")
async def chat(
    req: ChatRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    assistant, owner, session_id = _session(req, request, x_user_id)
    observer = _observer(request, owner, session_id)
    try:
        result = await assistant.run(
            session_id,
            req.message,
            owner=owner,
            max_iterations=req.max_iterations,
            **_tracer_kwargs(observer),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        run_id = _finish(observer)
    return {
        "session_id": session_id,
        "message": result.reply if not result.waiting else result.prompt,
        "run_id": run_id,
    }


#: Human-friendly labels for the coordinator's tools, shown as ``status``
#: events so the stream reads like a working assistant rather than a graph.
_TOOL_LABELS = {
    "extract_project_info": "Распознаю параметры проекта…",
    "propose_plan": "Предлагаю план работ…",
    "select_materials": "Подбираю материалы…",
    "prepare_estimate": "Составляю смету…",
    "run_qa_check": "Проверяю проект…",
    "ask_human": "Жду подтверждения…",
    "reply_to_user": "Отвечаю…",
}


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> EventSourceResponse:
    assistant, owner, session_id = _session(req, request, x_user_id)
    raw = request.query_params.get("raw") == "1"

    async def events():
        yield {"event": "chat_id", "data": json.dumps({"session_id": session_id})}
        observer = _observer(request, owner, session_id)
        question: str | None = None
        try:
            async for event in assistant.stream(
                session_id,
                req.message,
                owner=owner,
                max_iterations=req.max_iterations,
                **_tracer_kwargs(observer),
            ):
                if event.type == "interrupt":
                    question = event.data.get("question") or event.data.get(
                        "prompt", ""
                    )
                    if raw:
                        data = {"session_id": session_id}
                        if event.node_id is not None:
                            data["node_id"] = event.node_id
                        if event.node_type is not None:
                            data["node_type"] = event.node_type
                        data.update(event.data)
                        yield {
                            "event": event.type,
                            "data": json.dumps(data),
                        }
                    break
                if raw:
                    data = {"session_id": session_id}
                    if event.node_id is not None:
                        data["node_id"] = event.node_id
                    if event.node_type is not None:
                        data["node_type"] = event.node_type
                    data.update(event.data)
                    yield {"event": event.type, "data": json.dumps(data)}
                elif event.type == "token":
                    yield {
                        "event": "content",
                        "data": json.dumps(
                            {
                                "session_id": session_id,
                                "typing": True,
                                "content": event.data.get("token", ""),
                            }
                        ),
                    }
                elif event.type == "tool_call":
                    name = event.data.get("name", "")
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "session_id": session_id,
                                "message": _TOOL_LABELS.get(name)
                                or f"Выполняю {name}…",
                            }
                        ),
                    }
        finally:
            run_id = _finish(observer)
        if question is not None:
            reply, waiting = question, True
        else:
            reply = await assistant.last_reply(session_id, owner=owner)
            waiting = False
        yield {
            "event": "message",
            "data": json.dumps(
                {
                    "session_id": session_id,
                    "reply": reply,
                    "run_id": run_id,
                    "waiting": waiting,
                }
            ),
        }

    return EventSourceResponse(events())
