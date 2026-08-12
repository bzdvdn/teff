"""Tests for graph.stream() — streaming events during execution."""

import pytest

from teff.provider import ProviderRegistry


class TestStreamLifecycle:
    @pytest.mark.asyncio
    async def test_emits_lifecycle_events(self):
        from teff.graph import Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "uppercase", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        types = [ev.type async for ev in g.stream({"x": "hi"})]
        assert types == ["run_start", "node_start", "node_end", "run_end"]

    @pytest.mark.asyncio
    async def test_node_events_carry_ids_and_types(self):
        from teff.graph import Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "uppercase", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        events = [ev async for ev in g.stream({"x": "hi"})]
        node_start = next(e for e in events if e.type == "node_start")
        assert (node_start.node_id, node_start.node_type) == ("a", "transform")
        run_end = next(e for e in events if e.type == "run_end")
        assert run_end.data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_run_end_error_on_failure(self):
        from teff.graph import Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "bogus", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        events = [ev async for ev in g.stream({"x": "hi"})]
        run_end = events[-1]
        assert run_end.type == "run_end"
        assert run_end.data["status"] == "error"
        assert "bogus" in run_end.data["error"]
        node_error = next(e for e in events if e.type == "node_error")
        assert node_error.node_id == "a"


class TestStreamEdges:
    @pytest.mark.asyncio
    async def test_emits_conditional_edge_events(self):
        from teff.graph import Edge, Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "value", "value": "yes", "output_key": "approved"}
                ),
                "b": Transform(
                    {"action": "value", "value": "ok", "output_key": "final"}
                ),
                "c": Transform(
                    {"action": "value", "value": "no", "output_key": "final"}
                ),
            },
            edges=[
                Edge("a", "b", "approved=yes"),
                Edge("a", "c", "approved!=yes"),
            ],
            entry_point="a",
        )
        edge_events = [ev async for ev in g.stream({}) if ev.type == "edge"]
        assert len(edge_events) == 1
        assert edge_events[0].node_id == "a"
        assert edge_events[0].data == {"target_id": "b", "condition": "approved=yes"}


class TestStreamTokens:
    @pytest.mark.asyncio
    async def test_llm_tokens_stream_through(self, monkeypatch):
        from teff.graph import Graph
        from teff.node import LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
            "data: [DONE]\n",
        ]

        class MockStreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        class MockStreamCM:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *a):
                pass

        def mock_stream(*a, **kw):
            return MockStreamCM()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        g = Graph(
            nodes={
                "llm": LLM(
                    {"model": "gpt-4", "output_key": "answer", "provider": "openai"}
                ),
            },
            edges=[],
            entry_point="llm",
            providers=ProviderRegistry.from_presets("openai"),
        )
        events = [ev async for ev in g.stream({})]
        tokens = [ev.data["token"] for ev in events if ev.type == "token"]
        assert tokens == ["Hel", "lo"]
        run_end = events[-1]
        assert run_end.type == "run_end"
        assert run_end.data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_stream_uses_default_model(self, monkeypatch):
        from teff.graph import Graph
        from teff.node import LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        bodies = []

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
            "data: [DONE]\n",
        ]

        class MockStreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        class MockStreamCM:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *a):
                pass

        def mock_stream(*a, **kw):
            bodies.append(kw.get("json") or {})
            return MockStreamCM()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        g = Graph(
            nodes={
                "llm": LLM({"output_key": "answer", "provider": "openai"}),
            },
            edges=[],
            entry_point="llm",
            providers=ProviderRegistry.from_presets("openai"),
            default_model="gpt-default",
        )
        events = [ev async for ev in g.stream({})]
        assert events[-1].type == "run_end"
        assert bodies[0]["model"] == "gpt-default"

    @pytest.mark.asyncio
    async def test_stream_requires_checkpoint_id_with_checkpointer(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Graph
        from teff.node import Transform

        cp = JSONFileCheckpointer(str(tmp_path))
        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "uppercase", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        with pytest.raises(ValueError, match="checkpoint_id"):
            async for _ in g.stream({}, checkpointer=cp):
                pass

    @pytest.mark.asyncio
    async def test_run_does_not_emit_tokens(self, monkeypatch):
        from teff.graph import Graph
        from teff.node import LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hi"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "llm": LLM(
                    {"model": "gpt-4", "output_key": "answer", "provider": "openai"}
                ),
            },
            edges=[],
            entry_point="llm",
            providers=ProviderRegistry.from_presets("openai"),
        )
        result = await g.run({})
        assert result["answer"] == "hi"

    @pytest.mark.asyncio
    async def test_ollama_native_sse_tokens(self, monkeypatch):
        from teff.graph import Graph
        from teff.node import LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        raw_lines = [
            '{"model":"x","message":{"role":"assistant","content":"Од"}}\n',
            '{"model":"x","message":{"role":"assistant","content":"ин"}}\n',
            '{"model":"x","message":{"role":"assistant","content":""},"done":true}\n',
        ]

        class MockStreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in raw_lines:
                    yield line

        class MockStreamCM:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *a):
                pass

        def mock_stream(*a, **kw):
            return MockStreamCM()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        g = Graph(
            nodes={
                "llm": LLM(
                    {"model": "gpt-4", "output_key": "answer", "provider": "ollama"}
                ),
            },
            edges=[],
            entry_point="llm",
            providers=ProviderRegistry.from_presets("ollama"),
        )
        events = [ev async for ev in g.stream({})]
        tokens = [ev.data["token"] for ev in events if ev.type == "token"]
        assert tokens == ["Од", "ин"]


class TestStreamCustomNode:
    @pytest.mark.asyncio
    async def test_custom_node_can_emit_events(self):
        from teff.graph import Graph
        from teff.node import Node
        from teff.stream import StreamEvent

        class EmittingNode(Node):
            type = "emitter"

            def __init__(self, config=None, **kwargs):
                merged = dict(config or {})
                merged.update(kwargs)
                super().__init__(**merged)

            async def execute(self, ctx, state: dict) -> dict:
                await ctx.emit(
                    StreamEvent(
                        "progress",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={"percent": 50},
                    )
                )
                state["out"] = "done"
                return {"out": "done"}

        g = Graph(nodes={"n": EmittingNode({})}, edges=[], entry_point="n")
        events = [ev async for ev in g.stream({})]
        progress = next(e for e in events if e.type == "progress")
        assert progress.data == {"percent": 50}
        assert events[-1].type == "run_end"


class TestStreamInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_event_and_resume(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Edge, Graph
        from teff.node import Interrupt, Transform

        g = Graph(
            nodes={
                "ask": Interrupt({"key": "approved", "prompt": "ok?"}),
                "done": Transform(
                    {"action": "value", "value": "APPROVED", "output_key": "final"}
                ),
            },
            edges=[Edge("ask", "done", "approved=да")],
            entry_point="ask",
        )
        cp = JSONFileCheckpointer(str(tmp_path))

        first = [ev async for ev in g.stream({}, checkpointer=cp, checkpoint_id="s1")]
        types = [ev.type for ev in first]
        assert "interrupt" in types
        # an interrupted run ends with a run_end event carrying total_ms
        assert "run_end" in types
        run_end = next(e for e in first if e.type == "run_end")
        assert run_end.data["status"] == "interrupted"
        assert run_end.data["total_ms"] >= 0
        interrupt = next(e for e in first if e.type == "interrupt")
        assert interrupt.data == {
            "key": "approved",
            "prompt": "ok?",
            "question": "ok?",
        }
        assert interrupt.node_id == "ask"
        assert types[-1] == "run_end"

        resumed = [
            ev
            async for ev in g.stream(
                {}, checkpointer=cp, checkpoint_id="s1", resume={"approved": "да"}
            )
        ]
        types = [ev.type for ev in resumed]
        assert "interrupt_resume" in types
        assert types[-1] == "run_end"
        resumed_ev = next(e for e in resumed if e.type == "interrupt_resume")
        assert resumed_ev.data["keys"] == ["approved"]

    @pytest.mark.asyncio
    async def test_stream_matches_run_state(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Edge, Graph
        from teff.node import Interrupt, Transform

        g = Graph(
            nodes={
                "ask": Interrupt({"key": "approved", "prompt": "ok?"}),
                "done": Transform(
                    {"action": "value", "value": "APPROVED", "output_key": "final"}
                ),
            },
            edges=[Edge("ask", "done", "approved=да")],
            entry_point="ask",
        )
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(Exception):
            await g.run({}, checkpointer=cp, checkpoint_id="s2")
        result = await g.run(
            {}, checkpointer=cp, checkpoint_id="s2", resume={"approved": "да"}
        )
        assert result["final"] == "APPROVED"


class TestStreamCheckpoint:
    @pytest.mark.asyncio
    async def test_emits_checkpoint_events(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "uppercase", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        cp = JSONFileCheckpointer(str(tmp_path))
        events = [
            ev
            async for ev in g.stream({"x": "hi"}, checkpointer=cp, checkpoint_id="c1")
        ]
        checkpoints = [ev for ev in events if ev.type == "checkpoint"]
        assert checkpoints[0].data["action"] == "load"
        assert checkpoints[-1].data["action"] == "save"
        assert checkpoints[-1].data["next_node_id"] is None


class TestRunEmit:
    @pytest.mark.asyncio
    async def test_run_emit_receives_events_and_returns_state(self):
        from teff.graph import Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "uppercase", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        events = []

        async def sink(ev):
            events.append(ev)

        state = await g.run({"x": "hi"}, emit=sink)
        assert state["out"] == "HI"
        types = [ev.type for ev in events]
        assert types[0] == "run_start"
        assert types[-1] == "run_end"
        assert events[-1].data["status"] == "ok"
        assert "node_start" in types
        assert "node_end" in types

    @pytest.mark.asyncio
    async def test_run_emit_error_emits_run_end_and_raises(self):
        from teff.graph import Graph
        from teff.node import Transform

        g = Graph(
            nodes={
                "a": Transform(
                    {"action": "bogus", "input_key": "x", "output_key": "out"}
                ),
            },
            edges=[],
            entry_point="a",
        )
        events = []

        async def sink(ev):
            events.append(ev)

        with pytest.raises(Exception):
            await g.run({}, emit=sink)
        run_end = events[-1]
        assert run_end.type == "run_end"
        assert run_end.data["status"] == "error"
        assert "bogus" in run_end.data["error"]

    @pytest.mark.asyncio
    async def test_run_emit_streams_llm_tokens(self, monkeypatch):
        from teff.graph import Graph
        from teff.node import LLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
            "data: [DONE]\n",
        ]

        class MockStreamResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        class MockStreamCM:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *a):
                pass

        def mock_stream(*a, **kw):
            return MockStreamCM()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

        g = Graph(
            nodes={
                "llm": LLM(
                    {"model": "gpt-4", "output_key": "answer", "provider": "openai"}
                ),
            },
            edges=[],
            entry_point="llm",
            providers=ProviderRegistry.from_presets("openai"),
        )
        events = []

        async def sink(ev):
            events.append(ev)

        state = await g.run({}, emit=sink)
        assert state["answer"] == "Hello"
        tokens = [ev.data["token"] for ev in events if ev.type == "token"]
        assert tokens == ["Hel", "lo"]
