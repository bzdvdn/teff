"""End-to-end wiring + API tests for the ``repair-ai-chat`` application.

Builds the agentic coordinator flow from ``examples/applications/repair-ai-chat``
and runs it against a mocked LLM transport — no network, no API keys — to
prove that a single coordinator ReAct agent drives the pipeline through
sub-agent tools, that ``ask_human`` approvals pause/resume the run, and that
the FastAPI server (chat, SSE, sessions) all work together.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "applications" / "repair-ai-chat"
)
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from src.graphs.build import build_flow  # noqa: E402
from src.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402


def _stub_embedder(client) -> None:
    """Swap the app catalog's embedder for a deterministic offline stub."""

    async def _embed_many(texts):
        return [
            list(__import__("numpy").random.default_rng(sum(map(ord, t))).random(4))
            for t in texts
        ]

    catalog = client.app.state.catalog
    catalog.embedder = type("_Stub", (), {"embed_many": staticmethod(_embed_many)})()
    catalog.store = __import__(
        "teff.rag.stores", fromlist=["InMemoryVectorStore"]
    ).InMemoryVectorStore(dim=4)
    catalog._ingested = 0


def test_catalog_load_and_update(tmp_path):
    """POST /api/catalog/load ingests a CSV in batches; update rebuilds."""
    pytest.importorskip("fastapi")
    from app import create_app
    from fastapi.testclient import TestClient

    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "Наименование,Цена,Ед\nКирпич М-150,24.2,₽/шт\nПлитка Керама-Белый,780,₽/м²\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(checkpoint_dir=str(tmp_path)))
    _stub_embedder(client)

    status = client.get("/api/catalog")
    assert status.status_code == 200
    queued0 = status.json()["queued"]
    assert status.json()["stored"] == 0  # stub cleared the store, nothing loaded

    loaded = client.post(
        "/api/catalog/load", data={"batch_size": "1", "path": str(csv_path)}
    )
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["queued_this_file"] == 2
    assert body["report"]["stored"] == queued0 + 2
    assert body["report"]["batches"] >= 1  # batched embed_many calls
    assert body["report"]["added"] == queued0 + 2

    # loading the same file again appends its rows to the store
    again = client.post(
        "/api/catalog/load", data={"batch_size": "10", "path": str(csv_path)}
    )
    assert again.json()["queued_this_file"] == 2
    assert again.json()["report"]["stored"] == queued0 + 4

    # update rebuilds the whole store from every queued document
    updated = client.post("/api/catalog/update", data={"batch_size": "10"})
    assert updated.status_code == 200
    assert updated.json()["report"]["stored"] == queued0 + 4


def _reply(content: str) -> dict:
    """A plain-text response for both OpenAI (``choices``) and Ollama (root
    ``message``) extraction paths used by the framework."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "message": {"role": "assistant", "content": content},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _tool_call(name: str, call_id: str, arguments: dict) -> dict:
    tc = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                }
            }
        ],
        "message": {"role": "assistant", "content": None, "tool_calls": [tc]},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


_PLAN = "1. Демонтаж. 2. Стены. 3. Пол. 4. Отделка."
_ESTIMATE = "Смета: стены 12000, пол 20000, отделка 30000."
_FINAL = f"План: {_PLAN}\n\nСмета: стены 12000, пол 20000, отделка 30000."

#: Coordinator step tokens in the order the mock emits them.  A resume of an
#: interrupted ``ask_human`` auto-re-capituates the same ask (the mock sees an
#: assistant tool-call awaiting its tool reply), so tests list each ask once.
_HAPPY_PATH = ["extract", "plan", "ask_plan", "estimate", "qa", "ask_estimate"]


async def _run_with_approval(
    graph, state, tools, *, answers=("да",), emit=None, checkpoint_dir="/tmp"
):
    """Run the agentic graph end to end, answering every ``ask_human`` pause.

    The coordinator pauses on ask_human interrupts; *answers* supplies the
    operator's replies in order (e.g. ``("нет", "да")`` to force one re-plan /
    re-estimate round).  Uses a per-run JSON-file checkpointer so resume works.
    """
    from teff.checkpoint import JSONFileCheckpointer
    from teff.node.interrupt import GraphInterrupt

    answers = list(answers)
    cp = JSONFileCheckpointer(str(checkpoint_dir))
    try:
        return await graph.run(
            state,
            tools=tools,
            reducers=STATE_REDUCERS,
            checkpointer=cp,
            checkpoint_id="run-1",
            max_iterations=80,
            emit=emit,
        )
    except GraphInterrupt:
        pass
    while True:
        answer = answers.pop(0) if answers else "да"
        try:
            return await graph.run(
                state,
                tools=tools,
                reducers=STATE_REDUCERS,
                checkpointer=cp,
                checkpoint_id="run-1",
                max_iterations=80,
                emit=emit,
                resume={"ask_human": answer},
            )
        except GraphInterrupt:
            continue


def _stream_lines(content: str) -> list[str]:
    """Split *content* into OpenAI-style SSE chunks."""
    content = content or ""
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]})
        for chunk in (content[i : i + 4] for i in range(0, len(content), 4))
    ]
    lines.append("data: [DONE]")
    return lines


class _MockTransport:
    """Serves a canned, system-prompt-aware reply per LLM call.

    Installed on both ``httpx.AsyncClient.post`` and ``.stream``.  Dispatches
    on the system-prompt text: the coordinator gets a scripted sequence of
    tool calls (``coordinator_steps``), while each sub-agent runs its canned
    content.
    """

    def __init__(
        self,
        *,
        coordinator_steps: list[str] | None = None,
        qa_verdicts: list[str] | None = None,
    ):
        self.calls: list[str] = []
        self.coordinator_calls = 0
        self.qa_calls = 0
        self.coordinator_steps = list(
            coordinator_steps if coordinator_steps is not None else _HAPPY_PATH
        )
        self.qa_verdicts = list(qa_verdicts or [])
        self._call_seq = 0

    def _coordinator_reply(self, messages: list[dict]) -> dict:
        self.coordinator_calls += 1
        self._call_seq += 1
        call_id = f"coord{self._call_seq}"
        # A resume returns the coordinator to a pending ask_human: the last
        # message is an assistant tool-call with no matching tool reply, so
        # re-emit the same ask for the executor to answer.
        last = messages[-1] if messages else None
        if last and last.get("role") == "assistant" and last.get("tool_calls"):
            fn = last["tool_calls"][0]["function"]
            return _tool_call(
                name=fn["name"], call_id=call_id, arguments=json.loads(fn["arguments"])
            )
        if not self.coordinator_steps:
            ran_tools = any(
                m.get("role") == "tool"
                or (m.get("role") == "assistant" and m.get("tool_calls"))
                for m in messages
            )
            if ran_tools:
                return _reply(_FINAL)
            return _reply("Здравствуйте! Помогу спланировать ремонт.")
        token = self.coordinator_steps.pop(0)
        if token == "extract":
            return _tool_call("extract_project_info", call_id, {})
        if token == "plan":
            return _tool_call("propose_plan", call_id, {})
        if token == "ask_plan":
            return _tool_call(
                "ask_human",
                call_id,
                {"question": "План готов. Одобряешь план? Ответь: да или нет."},
            )
        if token == "estimate":
            return _tool_call("prepare_estimate", call_id, {})
        if token == "qa":
            return _tool_call("run_qa_check", call_id, {})
        if token == "ask_estimate":
            return _tool_call(
                "ask_human",
                call_id,
                {"question": "Смета готова. Одобряешь смету? Ответь: да или нет."},
            )
        return _reply(_FINAL)

    def _content_for(self, body: dict) -> dict:
        system = "".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "system"
        )
        self.calls.append(system[:40])
        if "Координатор" in system:
            return self._coordinator_reply(body.get("messages", []) or [])
        if "извлекаешь" in system:
            return _reply(json.dumps({"room_type": "bathroom", "area": 5.0}))
        if "Planner" in system:
            return _reply(_PLAN)
        if "Estimator" in system:
            return _reply(_ESTIMATE)
        if "Materials Agent" in system:
            return _reply("Плитка Керама-Белый 780 ₽/м², ламинат Дуб-Прованс 890 ₽/м².")
        if "QA Agent" in system:
            if self.qa_verdicts:
                verdict = self.qa_verdicts[
                    min(self.qa_calls, len(self.qa_verdicts) - 1)
                ]
                self.qa_calls += 1
                return _reply(verdict)
            return _reply(json.dumps({"ok": True, "message": ""}))
        return _reply("Здравствуйте! Помогу спланировать ремонт.")

    def __call__(self, *args, **kwargs):
        data = self._content_for(kwargs.get("json") or {})

        if args and args[0] == "POST":  # streaming path

            class _StreamResp:
                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    content = (data.get("message") or {}).get("content", "")
                    for line in _stream_lines(content):
                        yield line

            class _StreamCM:
                async def __aenter__(self):
                    return _StreamResp()

                async def __aexit__(self, *exc):
                    return False

            return _StreamCM()

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return data

        async def _post():
            return _Resp()

        return _post()


@pytest.fixture
def transport(monkeypatch):
    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)
    return mock


def _state_with_request() -> dict:
    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Спланируй ремонт ванной 5 м²."}]
    return state


def _last_assistant(state: dict) -> dict:
    for message in reversed(state["messages"]):
        if message.get("role") == "assistant":
            return message
    return {}


@pytest.mark.asyncio
async def test_coordinator_runs_end_to_end(transport, tmp_path):
    """The coordinator drives extract -> plan -> (approval) -> estimate ->
    qa -> (approval) -> final answer through sub-agent tools."""
    flow, tools = build_flow()
    graph = flow.compile()

    events = []

    async def sink(ev):
        events.append(ev)

    result = await _run_with_approval(
        graph, _state_with_request(), tools, emit=sink, checkpoint_dir=str(tmp_path)
    )

    # the coordinator consulted the model once per tool round
    assert transport.coordinator_calls >= 7
    for section in ("plan", "estimate", "material_findings"):
        assert result[section] != ""
    assert result["qa_feedback"] == ""  # QA passed
    assert result["project_info"] == {"room_type": "bathroom", "area": 5.0}

    # only the final answer reaches the shared conversation
    assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
    assert len(assistant) == 1
    last = _last_assistant(result)
    assert last["content"].startswith("План:")
    assert "Смета:" in last["content"]

    # two ask_human pauses (plan + estimate approval), three runs total
    types = [ev.type for ev in events]
    assert types.count("run_start") == 3
    assert types.count("interrupt") == 2
    assert [ev.data["status"] for ev in events if ev.type == "run_end"] == [
        "interrupted",
        "interrupted",
        "ok",
    ]

    node_types = {ev.node_type for ev in events if ev.type == "node_start"}
    assert "react_agent" in node_types
    assert "tool_exec" in node_types
    assert "context_builder" in node_types


class _SyncTracerCtx:
    """Duck-typed exec context whose ``tracer.llm`` is a *sync* (None-returning)
    method — exactly what a real ``RunTracer`` provides.  The sub-agent harness
    must wrap it in an async ``on_llm`` hook rather than awaiting it directly."""

    class _Tracer:
        def llm(self, provider, model, prompt_tokens, completion_tokens, duration_ms):
            return None  # sync, as in teff.trace.RunTracer

    providers = None
    tracer = _Tracer()
    on_llm_payload = None


@pytest.mark.asyncio
async def test_subagent_tool_runs_with_sync_tracer(transport):
    """A sub-agent tool's internal LLM call must not ``await`` a sync tracer.

    Regression: ``AgentTool._harness`` wired ``harness.on_llm`` to a sync
    lambda calling ``tracer.llm``; the harness ``await``-s it, which raised
    ``TypeError: object NoneType can't be used in 'await' expression`` once a
    runtime ``RunTracer`` was attached.  The hook must be async.
    """
    from teff.harness.tools import _run_one_tool_call

    flow, tools = build_flow()
    by_name = {t.name: t for t in tools}

    result = await _run_one_tool_call(
        {
            "id": "c1",
            "name": "extract_project_info",
            "args": json.dumps({"message": "спланируй ремонт ванной"}),
        },
        by_name,
        "message",
        None,
        0,
        None,
        {},
        _SyncTracerCtx(),
    )
    assert "Error" not in result
    assert "bathroom" in result


@pytest.mark.asyncio
async def test_coordinator_replans_on_rejection(monkeypatch, tmp_path):
    """A "нет" plan answer makes the coordinator re-plan and ask again;
    "да" then completes."""
    mock = _MockTransport(
        coordinator_steps=[
            "extract",
            "plan",
            "ask_plan",
            "plan",
            "ask_plan",
            "estimate",
            "qa",
            "ask_estimate",
        ]
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    result = await _run_with_approval(
        graph,
        _state_with_request(),
        tools,
        answers=("нет", "да"),
        checkpoint_dir=str(tmp_path),
    )

    assert result["plan"] != ""
    assert result["estimate"] != ""
    # the planner sub-agent ran twice (initial + rework after "нет")
    planner_calls = sum(1 for c in mock.calls if "Planner" in c)
    assert planner_calls == 2
    assert _last_assistant(result)["content"].startswith("План:")
    assert "Смета:" in _last_assistant(result)["content"]


@pytest.mark.asyncio
async def test_unclear_answer_reasks_without_replanning(monkeypatch, tmp_path):
    """A gibberish reply makes the coordinator re-ask the same question —
    the planner sub-agent runs exactly once."""
    mock = _MockTransport(
        coordinator_steps=[
            "extract",
            "plan",
            "ask_plan",
            "ask_plan",
            "estimate",
            "qa",
            "ask_estimate",
        ]
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    result = await _run_with_approval(
        graph,
        _state_with_request(),
        tools,
        answers=("qhjrkjlkjsdgjdlksgj", "да", "да"),
        checkpoint_dir=str(tmp_path),
    )

    assert result["plan"] != ""
    assert result["estimate"] != ""
    planner_calls = sum(1 for c in mock.calls if "Planner" in c)
    assert planner_calls == 1


@pytest.mark.asyncio
async def test_estimate_rejection_recalculates_and_reasks(monkeypatch, tmp_path):
    """A "нет" on the estimate makes the coordinator re-run prepare_estimate
    (and QA) before asking again; the second "да" completes."""
    mock = _MockTransport(
        coordinator_steps=[
            "extract",
            "plan",
            "ask_plan",
            "estimate",
            "qa",
            "ask_estimate",
            "estimate",
            "qa",
            "ask_estimate",
        ]
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    result = await _run_with_approval(
        graph,
        _state_with_request(),
        tools,
        answers=("да", "нет", "да"),
        checkpoint_dir=str(tmp_path),
    )

    assert result["plan"] != ""
    assert result["estimate"] != ""
    assert result["qa_feedback"] == ""
    # the planner ran once; QA ran after the initial estimate and after the recalc
    planner_calls = sum(1 for c in mock.calls if "Planner" in c)
    assert planner_calls == 1
    assert sum(1 for c in mock.calls if "QA Agent" in c) >= 2
    assert _last_assistant(result)["content"].startswith("План:")
    assert "Смета:" in _last_assistant(result)["content"]


@pytest.mark.asyncio
async def test_qa_fix_loop_revises_and_finalizes(monkeypatch, tmp_path):
    """A not-ok QA verdict makes the coordinator re-run prepare_estimate +
    run_qa_check, then ask the human; the ok verdict completes."""
    mock = _MockTransport(
        coordinator_steps=[
            "extract",
            "plan",
            "ask_plan",
            "estimate",
            "qa",
            "estimate",
            "qa",
            "ask_estimate",
        ],
        qa_verdicts=[
            json.dumps({"ok": False, "message": "Смета не сходится с планом."}),
            json.dumps({"ok": True, "message": ""}),
        ],
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    result = await _run_with_approval(
        graph, _state_with_request(), tools, checkpoint_dir=str(tmp_path)
    )

    assert mock.qa_calls == 2
    assert result["qa_feedback"] == ""  # final QA passed
    planner_calls = sum(1 for c in mock.calls if "Planner" in c)
    assert planner_calls == 1  # the plan was never re-run
    assert _last_assistant(result)["content"].startswith("План:")
    assert "Смета:" in _last_assistant(result)["content"]


@pytest.mark.asyncio
async def test_coordinator_answers_directly_without_tools(monkeypatch, tmp_path):
    """A non-renovation question gets a plain reply — no tool calls."""
    mock = _MockTransport(coordinator_steps=[])
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Привет! Расскажи про ремонт."}]
    result = await _run_with_approval(graph, state, tools, checkpoint_dir=str(tmp_path))

    assert mock.coordinator_calls == 1
    assert mock.coordinator_steps == []  # no pending steps
    assert "Помогу" in _last_assistant(result)["content"]


def test_project_info_schema_allows_null_for_unknown_fields():
    """Extractor prompt tells the model to return ``null`` for missing
    fields; the schema must accept that (regression: number fields rejected
    ``null``, so the extractor burned 3 attempts then raised NodeError)."""
    from src.domain.models import PROJECT_INFO_SCHEMA

    from teff.schema import validate_json

    unknown = {
        "room_type": "bathroom",
        "area": 5.0,
        "ceiling_height": None,
        "budget": None,
        "style": None,
        "walls_area": None,
        "floor_area": None,
        "ceiling_area": None,
    }
    assert validate_json(unknown, PROJECT_INFO_SCHEMA) == []
    full = {
        "room_type": "bathroom",
        "area": 5.0,
        "ceiling_height": 2.7,
        "budget": 150000,
        "style": "modern",
        "walls_area": 20.0,
        "floor_area": 5.0,
        "ceiling_area": 5.0,
    }
    assert validate_json(full, PROJECT_INFO_SCHEMA) == []


def test_detect_room_type_maps_russian_keywords():
    from src.nodes.extractor import detect_room_type

    assert detect_room_type("Помоги спланировать ремонт ванной комнаты") == "bathroom"
    assert detect_room_type("Сделайте ремонт на кухне") == "kitchen"
    assert detect_room_type("нужен ремонт спальни") == "bedroom"
    assert detect_room_type("гостиная 20 метров") == "living_room"
    assert detect_room_type("прихожая маленькая") == "hallway"
    assert detect_room_type("какой стиль выбрать?") is None


def test_find_similar_accepts_materials_and_category_kwargs():
    """llama3.1:8b calls find_similar_material with ``materials`` and
    ``category`` kwargs; the tool must accept them instead of raising
    ``unexpected keyword argument`` (regression from obs run 99)."""
    from src.tools.rag import FindSimilarMaterial

    calls = []

    class _Catalog:
        async def find_similar(self, name, top_k=3):
            calls.append((name, top_k))
            return f"similar:{name}"

    tool = FindSimilarMaterial(_Catalog())
    result = asyncio.run(tool.arun(materials=["грунтовка"], category="кухня", top_k=3))
    assert result == "similar:грунтовка"
    assert calls == [("грунтовка", 3)]

    result = asyncio.run(tool.arun(name="краска", top_k=5))
    assert result == "similar:краска"
    assert calls[-1] == ("краска", 5)


@pytest.mark.asyncio
async def test_search_materials_accepts_string_max_price():
    """A string max_price (e.g. ``"2000"``) must be coerced to float and not
    blow up with ``'<=' not supported between instances of 'float' and 'str'``."""
    import json as _json

    from src.tools.rag import SearchMaterials

    from teff.harness.tools import execute_tool_calls

    seen = {}

    class _Catalog:
        async def search(self, query, category=None, max_price=None):
            seen.update(query=query, category=category, max_price=max_price)
            return "found"

    tool = SearchMaterials(_Catalog())
    result = await execute_tool_calls(
        [
            {
                "id": "call_1",
                "name": "search_materials",
                "args": _json.dumps({"query": "дверь", "max_price": "2000"}),
            }
        ],
        {"search_materials": tool},
    )
    assert result == ["found"]
    assert isinstance(seen["max_price"], float)
    assert seen["max_price"] == 2000.0


@pytest.mark.asyncio
async def test_search_retries_without_category_filter(monkeypatch):
    """A room-type category (``кухня``) matches nothing; the catalog must
    retry without the filter so the agent gets results instead of a dead
    ``Nothing found`` loop."""
    from src.rag.catalog import MaterialCatalog

    calls = []

    class _Store:
        def __init__(self):
            self.calls = calls

        async def search(self, query_vector, k=3, filter=None):
            calls.append(filter)
            if filter:
                return []
            return [
                (0, 0.9, {"name": "Грунтовка", "price": 200, "category": "грунтовка"})
            ]

    class _Embedder:
        async def embed(self, text):
            return [0.0] * 4

    catalog = MaterialCatalog(_Embedder(), store=_Store())
    catalog._docs = [("грунтовка", {})]
    catalog._ingested = 1

    result = await catalog.search("грунтовка", category="кухня", max_price=500)
    assert "Грунтовка" in result
    assert calls == [{"category": "кухня"}, None]


@pytest.mark.asyncio
async def test_extractor_falls_back_when_model_drops_room_type(monkeypatch):
    """llama3.1:8b often omits room_type; the extractor fallback must fill it
    from the first user message so downstream agents see the room."""
    from src.nodes.extractor import room_from_first_user

    from teff.node import ExecContext, Fallback

    node = Fallback(
        input_key="project_info",
        field="room_type",
        fn=room_from_first_user,
    )
    result = await node.execute(
        ExecContext(state={}, tools={}),
        {
            "messages": [
                {"role": "user", "content": "Помоги спланировать ремонт ванной 5 м²."},
                {"role": "assistant", "content": "План готов."},
            ],
            "project_info": {"area": 5.0},
        },
    )
    assert result["project_info"]["room_type"] == "bathroom"
    assert result["project_info"]["area"] == 5.0


@pytest.mark.asyncio
async def test_room_type_fallback_preserves_model_room(monkeypatch):
    """When the model already filled room_type the fallback is a no-op."""
    from src.nodes.extractor import room_from_first_user

    from teff.node import ExecContext, Fallback

    node = Fallback(
        input_key="project_info",
        field="room_type",
        fn=room_from_first_user,
    )
    result = await node.execute(
        ExecContext(state={}, tools={}),
        {
            "messages": [
                {"role": "user", "content": "Помоги спланировать ремонт кухни 5 м²."},
            ],
            "project_info": {"room_type": "kitchen", "area": 5.0},
        },
    )
    assert result == {}


@pytest.mark.asyncio
async def test_llm_with_messages_key_prepends_system_prompt(monkeypatch):
    """core LLM injects the system prompt into a messages_key history, so a
    plain LLM node replaces the old Extractor subclass."""
    from teff.node import LLM, ExecContext

    captured = {}

    async def mock_post(*a, **kw):
        captured["body"] = kw.get("json")

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"message": {"content": '{"room_type": "bathroom"}'}}]
                }

        return MockResponse()

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    node = LLM(
        {
            "model": "gpt-4",
            "system": "Ты извлекаешь JSON.",
            "messages_key": "messages",
            "output_key": "project_info",
            "json_schema": {
                "type": "object",
                "properties": {"room_type": {"type": "string"}},
            },
            "provider": "openai",
        }
    )
    ctx = ExecContext(state={}, tools={})
    await node.execute(
        ctx,
        {
            "messages": [
                {"role": "user", "content": "Помоги спланировать ремонт ванной."},
                {"role": "assistant", "content": "План готов."},
            ]
        },
    )

    messages = captured["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Ты извлекаешь JSON."
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_route_wiring_in_example(transport):
    """The example's flow is a single coordinator ReAct loop plus the
    context builder and the final append."""
    flow, _ = build_flow()
    graph = flow.compile()
    edges = {(e.source_id, e.target_id, e.condition) for e in graph.edges}

    assert ("context", "coordinator/agent", None) in edges
    assert ("coordinator/agent", "coordinator/tool", "_tool_call_name!=") in edges
    assert ("coordinator/tool", "coordinator/agent", None) in edges
    assert ("coordinator/agent", "append", None) in edges

    node_types = {n.type for nid, n in graph.nodes.items()}
    assert "react_agent" in node_types
    assert "tool_exec" in node_types
    assert "context_builder" in node_types
    assert "append_assistant" in node_types


@pytest.mark.asyncio
async def test_subagent_tools_write_state(monkeypatch, tmp_path):
    """The sub-agent tools read the shared state and write their output back
    into it via ``__state__``."""
    mock = _MockTransport(coordinator_steps=["extract", "plan", "ask_plan", "finish"])
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)

    flow, tools = build_flow()
    graph = flow.compile()

    state = _state_with_request()
    result = await _run_with_approval(graph, state, tools, checkpoint_dir=str(tmp_path))

    assert result["project_info"]["room_type"] == "bathroom"
    assert result["plan"] == _PLAN


def test_api_chat_and_stream(monkeypatch, tmp_path):
    """The FastAPI server serves chat + SSE and persists sessions."""
    pytest.importorskip("fastapi")
    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)
    monkeypatch.setattr(httpx.AsyncClient, "stream", mock)
    from app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(
        create_app(checkpoint_dir=str(tmp_path)), raise_server_exceptions=False
    )

    created = client.post(
        "/api/chat", json={"message": "Помоги спланировать ремонт ванной 5 м²."}
    )
    assert created.status_code == 200
    data = created.json()
    assert isinstance(data["message"], str)
    assert data["message"]  # the ask_human question is surfaced as the reply
    assert isinstance(data["run_id"], str)
    assert data["run_id"]
    chat_id = data["session_id"]

    # resuming the same session with "да" approves the plan, then the estimate
    # approval pauses again; answering "да" once more completes the pipeline
    resumed = client.post(
        "/api/chat",
        json={"message": "да", "session_id": chat_id},
    )
    assert resumed.status_code == 200
    rdata = resumed.json()
    assert rdata["message"]  # waiting on the estimate-approval interrupt

    resumed = client.post(
        "/api/chat",
        json={"message": "да", "session_id": chat_id},
    )
    assert resumed.status_code == 200
    rdata = resumed.json()
    assert rdata["message"].startswith("План:")
    assert "Смета:" in rdata["message"]

    # a fresh session streams until the first ask_human pause
    mock.coordinator_steps = _HAPPY_PATH[:]
    stream = client.post(
        "/api/chat/stream", json={"message": "Помоги спланировать ремонт ванной 5 м²."}
    )
    assert stream.status_code == 200
    assert "event: chat_id" in stream.text
    assert "event: waiting" in stream.text
    assert "event: message" not in stream.text  # nothing to say while paused
    assert "event: run_start" not in stream.text  # internal events are hidden

    # ?raw=1 keeps the underlying framework events for debugging
    mock.coordinator_steps = _HAPPY_PATH[:]
    raw = client.post(
        "/api/chat/stream?raw=1",
        json={"message": "Помоги спланировать ремонт ванной 5 м²."},
    )
    assert raw.status_code == 200
    assert "event: run_start" in raw.text
    assert "event: waiting" in raw.text
    assert "event: message" not in raw.text

    saved = client.get(f"/api/runs/{chat_id}")
    assert saved.status_code == 200
    assert "state" in saved.json()

    deleted = client.delete(f"/api/runs/{chat_id}")
    assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_catalog_reingest_task_detects_changes(tmp_path, monkeypatch):
    """The beat re-ingest rebuilds only when a seed CSV's content changes."""
    from src.queue.ingest import _fingerprint, reingest_if_changed

    catalog = __import__(
        "src.rag.catalog", fromlist=["MaterialCatalog"]
    ).MaterialCatalog
    store = __import__(
        "teff.rag.stores", fromlist=["InMemoryVectorStore"]
    ).InMemoryVectorStore(dim=4)

    async def _embed_many(texts):
        return [
            list(__import__("numpy").random.default_rng(sum(map(ord, t))).random(4))
            for t in texts
        ]

    stub = type("_Stub", (), {"embed_many": staticmethod(_embed_many)})()
    (tmp_path / "prices.csv").write_text(
        "Наименование,Цена,Ед\nКирпич,10,₽/шт\n", encoding="utf-8"
    )
    cat = catalog(embedder=stub, store=store)
    cat.add_csv(
        str(tmp_path / "prices.csv"),
        fieldmap={"name": "Наименование", "price": "Цена", "unit": "Ед"},
    )

    state_file = tmp_path / "ingest_state.json"
    monkeypatch.setattr("src.queue.ingest._state_path", lambda: state_file)
    monkeypatch.setattr("src.queue.ingest.DEFAULT_CATALOG", tmp_path / "prices.csv")
    monkeypatch.setattr("src.queue.ingest.DEFAULT_PRICE_LIST", tmp_path / "prices.csv")

    # first run embeds; fingerprint recorded
    first = await reingest_if_changed(catalog=cat)
    assert first["status"] == "ok"
    assert first["stored"] == 1  # the single row was embedded
    assert state_file.exists()

    # unchanged source -> no-op, no duplicate work
    again = await reingest_if_changed(catalog=cat)
    assert again["status"] == "unchanged"

    # changed source -> rebuild happens again
    (tmp_path / "prices.csv").write_text(
        "Наименование,Цена,Ед\nКирпич,15,₽/шт\n", encoding="utf-8"
    )
    changed = await reingest_if_changed(catalog=cat)
    assert changed["status"] == "ok"
    assert _fingerprint([tmp_path / "prices.csv"]) != ""


def test_queue_fingerprint_tracks_content():
    """Fingerprint is content-based, not path-based."""
    from src.queue.ingest import _fingerprint

    a = __import__("tempfile").NamedTemporaryFile("w", delete=False, suffix=".csv")
    a.write("a,b\n1,2\n")
    a.close()
    b = __import__("tempfile").NamedTemporaryFile("w", delete=False, suffix=".csv")
    b.write("a,b\n1,3\n")
    b.close()

    assert _fingerprint([a.name, b.name]) != _fingerprint([b.name, a.name])
    assert _fingerprint([a.name]) != _fingerprint([b.name])
    assert _fingerprint([a.name]) == _fingerprint([a.name])
