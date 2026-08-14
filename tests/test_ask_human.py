"""Tests for the ``ask_human`` human-in-the-loop ReAct tool."""

import json

import pytest

from teff.provider import ProviderRegistry


def _mock_response(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


def _ask_human_response(call_id: str, question: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": "ask_human",
                                "arguments": json.dumps({"question": question}),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _final_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestAskHumanTool:
    def test_schema_and_description(self):
        from teff.harness import tool_to_schema
        from teff.tool.builtin import AskHuman

        tool = AskHuman()
        assert tool.name == "ask_human"
        schema = tool_to_schema(tool)
        fn = schema["function"]
        assert fn["name"] == "ask_human"
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["question"]
        assert params["properties"]["question"]["type"] == "string"
        assert "pause" in fn["description"].lower()

    def test_registered_in_default_registry(self):
        from teff.tool.builtin import AskHuman
        from teff.tool.registry import default_tool_registry

        assert "ask_human" in default_tool_registry.list()
        assert isinstance(default_tool_registry.create("ask_human"), AskHuman)

    def test_direct_arun_is_intercepted_only(self):
        from teff.tool.builtin import AskHuman

        with pytest.raises(NotImplementedError, match="intercepted"):
            import asyncio

            asyncio.run(AskHuman().arun(question="hello?"))


@pytest.mark.asyncio
async def test_ask_human_pauses_without_a_reply(monkeypatch):
    """An ask_human call with no pending reply pauses the run as an
    interrupt carrying the question."""
    import httpx

    from teff.graph import Edge, Graph
    from teff.node.agent import ReActAgent, ToolExec
    from teff.node.interrupt import GraphInterrupt
    from teff.tool.builtin import AskHuman

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def mock_post(*a, **kw):
        return _mock_response(_ask_human_response("c1", "Which city?"))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    g = Graph(
        nodes={
            "agent": ReActAgent(
                {"model": "gpt-4", "input_key": "input", "use_tools": ["ask_human"]}
            ),
            "tool": ToolExec({"use_tools": ["ask_human"]}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        providers=ProviderRegistry.from_presets("openai"),
        default_provider="openai",
    )
    with pytest.raises(GraphInterrupt) as excinfo:
        await g.run(
            state={"input": "book a trip"}, tools=[AskHuman()], max_iterations=5
        )
    assert excinfo.value.key == "ask_human"
    assert excinfo.value.prompt == "Which city?"


@pytest.mark.asyncio
async def test_ask_human_resume_delivers_the_answer(monkeypatch, tmp_path):
    """After the operator's answer is resumed, the tool call completes with
    it and the agent continues (the re-invoke pattern of tool_approval)."""
    import httpx

    from teff.checkpoint import JSONFileCheckpointer
    from teff.graph import Edge, Graph
    from teff.node.agent import ReActAgent, ToolExec
    from teff.node.interrupt import GraphInterrupt
    from teff.tool.builtin import AskHuman

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # First run: the agent asks; the run pauses.  On resume the graph
    # continues at the agent (past the interrupted ToolExec), so the agent
    # re-emits the ask_human call before ToolExec consumes the resumed
    # answer and delivers it as the tool result.
    responses = [
        _ask_human_response("c1", "Which city?"),
        _ask_human_response("c2", "Which city?"),
        _final_response("Booking Moscow."),
    ]

    async def mock_post(*a, **kw):
        return _mock_response(responses.pop(0))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    g = Graph(
        nodes={
            "agent": ReActAgent(
                {"model": "gpt-4", "input_key": "input", "use_tools": ["ask_human"]}
            ),
            "tool": ToolExec({"use_tools": ["ask_human"]}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        providers=ProviderRegistry.from_presets("openai"),
        default_provider="openai",
    )
    cp = JSONFileCheckpointer(str(tmp_path / "cp.json"))
    state = {"input": "book a trip"}
    with pytest.raises(GraphInterrupt):
        await g.run(
            state=state,
            tools=[AskHuman()],
            max_iterations=5,
            checkpointer=cp,
            checkpoint_id="run-1",
        )

    result = await g.run(
        state=state,
        tools=[AskHuman()],
        max_iterations=5,
        checkpointer=cp,
        checkpoint_id="run-1",
        resume={"ask_human": "Moscow"},
    )
    assert result["output"] == "Booking Moscow."
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "Moscow"
    assert tool_msgs[0]["tool_call_id"] == "c2"
    # the consumed answer does not linger in the final state
    assert result.get("ask_human") is None


@pytest.mark.asyncio
async def test_ask_human_through_flow_harness(monkeypatch, tmp_path):
    """The high-level ``Flow.react(...)`` harness supports ask_human."""
    import httpx

    from teff.checkpoint import JSONFileCheckpointer
    from teff.flow import Flow
    from teff.node.interrupt import GraphInterrupt
    from teff.tool.builtin import AskHuman

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    responses = [
        _ask_human_response("q1", "Preferred color?"),
        _ask_human_response("q2", "Preferred color?"),
        _final_response("You chose blue."),
    ]

    async def mock_post(*a, **kw):
        return _mock_response(responses.pop(0))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    graph = (
        Flow(
            "ask",
            providers=ProviderRegistry.from_presets("openai"),
            default_provider="openai",
        )
        .react(
            model="gpt-4",
            system="Ask when unsure.",
            input_key="input",
            use_tools=["ask_human"],
        )
        .compile()
    )
    cp = JSONFileCheckpointer(str(tmp_path / "cp.json"))
    state = {"input": "pick a color"}
    with pytest.raises(GraphInterrupt):
        await graph.run(
            state=state,
            tools=[AskHuman()],
            max_iterations=8,
            checkpointer=cp,
            checkpoint_id="run-1",
        )

    result = await graph.run(
        state=state,
        tools=[AskHuman()],
        max_iterations=8,
        checkpointer=cp,
        checkpoint_id="run-1",
        resume={"ask_human": "blue"},
    )
    assert result["output"] == "You chose blue."
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "blue"


@pytest.mark.asyncio
async def test_force_tool_nudges_resume_ask_human(monkeypatch, tmp_path):
    """Tool-call enforcement: on resume the model first answers with plain
    text instead of re-emitting the pending ask_human call; the nudge pushes
    it to call ask_human so the operator's answer is delivered as the tool
    result and the run completes."""
    import httpx

    from teff.checkpoint import JSONFileCheckpointer
    from teff.graph import Edge, Graph
    from teff.node.agent import ReActAgent, ToolExec
    from teff.node.interrupt import GraphInterrupt
    from teff.tool.builtin import AskHuman

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # First run: ask (pauses).  Resume: the model wrongly replies with plain
    # text (no tool call) -> the nudge forces another LLM call which re-emits
    # ask_human (c2); ToolExec then consumes the resumed answer.
    responses = [
        _ask_human_response("c1", "Which city?"),
        _final_response("Let me think."),
        _ask_human_response("c2", "Which city?"),
        _final_response("Booking Moscow."),
    ]

    async def mock_post(*a, **kw):
        return _mock_response(responses.pop(0))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    g = Graph(
        nodes={
            "agent": ReActAgent(
                {
                    "model": "gpt-4",
                    "input_key": "input",
                    "use_tools": ["ask_human"],
                    "force_tool_rounds": 2,
                }
            ),
            "tool": ToolExec({"use_tools": ["ask_human"]}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        providers=ProviderRegistry.from_presets("openai"),
        default_provider="openai",
    )
    cp = JSONFileCheckpointer(str(tmp_path / "cp.json"))
    state = {"input": "book a trip"}
    with pytest.raises(GraphInterrupt):
        await g.run(
            state=state,
            tools=[AskHuman()],
            max_iterations=8,
            checkpointer=cp,
            checkpoint_id="run-1",
        )

    result = await g.run(
        state=state,
        tools=[AskHuman()],
        max_iterations=8,
        checkpointer=cp,
        checkpoint_id="run-1",
        resume={"ask_human": "Moscow"},
    )
    assert result["output"] == "Booking Moscow."
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "Moscow"


@pytest.mark.asyncio
async def test_force_tool_if_question_nudges_ask_human(monkeypatch, tmp_path):
    """Tool-call enforcement: a question written as plain text is nudged into
    an ask_human call so the run pauses for the user's answer instead of
    ending the turn."""
    import httpx

    from teff.checkpoint import JSONFileCheckpointer
    from teff.graph import Edge, Graph
    from teff.node.agent import ReActAgent, ToolExec
    from teff.node.interrupt import GraphInterrupt
    from teff.tool.builtin import AskHuman

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    responses = [
        _final_response("Одобряешь план? Ответь: да или нет"),
        _ask_human_response("q1", "Одобряешь план? Ответь: да или нет"),
    ]

    async def mock_post(*a, **kw):
        return _mock_response(responses.pop(0))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    g = Graph(
        nodes={
            "agent": ReActAgent(
                {
                    "model": "gpt-4",
                    "input_key": "input",
                    "use_tools": ["ask_human"],
                    "force_tool_rounds": 2,
                    "force_tool_if_question": True,
                }
            ),
            "tool": ToolExec({"use_tools": ["ask_human"]}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        providers=ProviderRegistry.from_presets("openai"),
        default_provider="openai",
    )
    cp = JSONFileCheckpointer(str(tmp_path / "cp.json"))
    state = {"input": "plan the kitchen"}
    with pytest.raises(GraphInterrupt) as exc:
        await g.run(
            state=state,
            tools=[AskHuman()],
            max_iterations=8,
            checkpointer=cp,
            checkpoint_id="run-1",
        )
    assert exc.value.key == "ask_human"
