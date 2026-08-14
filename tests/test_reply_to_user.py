"""Tests for the ``reply_to_user`` terminal ReAct tool."""

import json
import sys
from pathlib import Path

import pytest

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "applications"
    / "repair-ai-chat"
)


def _mock_response(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


def _reply_tool_response(call_id: str, message: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": "reply_to_user",
                                "arguments": json.dumps({"message": message}),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _final_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestReplyToUserTool:
    def test_schema_and_description(self):
        from teff.harness import tool_to_schema
        from teff.tool.builtin import ReplyToUser

        tool = ReplyToUser()
        assert tool.name == "reply_to_user"
        schema = tool_to_schema(tool)
        fn = schema["function"]
        assert fn["name"] == "reply_to_user"
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["message"]
        assert params["properties"]["message"]["type"] == "string"

    def test_registered_in_default_registry(self):
        from teff.tool.builtin import ReplyToUser
        from teff.tool.registry import default_tool_registry

        assert "reply_to_user" in default_tool_registry.list()
        assert isinstance(default_tool_registry.create("reply_to_user"), ReplyToUser)

    def test_direct_arun_is_intercepted_only(self):
        import asyncio

        from teff.tool.builtin import ReplyToUser

        with pytest.raises(NotImplementedError, match="intercepted"):
            asyncio.run(ReplyToUser().arun(message="hello"))


@pytest.mark.asyncio
async def test_reply_to_user_writes_output_and_ends_loop(monkeypatch):
    """A reply_to_user call writes the message to output and short-circuits
    the ReAct loop — no second LLM call happens."""
    import httpx

    from teff.graph import Edge, Graph
    from teff.node.agent import ReActAgent, ToolExec
    from teff.tool.builtin import ReplyToUser

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    post_calls = []

    async def counting_post(*a, **kw):
        post_calls.append(a)
        return _mock_response(_reply_tool_response("c1", "Здравствуйте! Чем помочь?"))

    monkeypatch.setattr(httpx.AsyncClient, "post", counting_post)

    g = Graph(
        nodes={
            "agent": ReActAgent(
                {
                    "model": "gpt-4",
                    "input_key": "input",
                    "use_tools": ["reply_to_user"],
                }
            ),
            "tool": ToolExec({"use_tools": ["reply_to_user"]}),
        },
        edges=[
            Edge("agent", "tool", "_tool_call_name!="),
            Edge("tool", "agent"),
        ],
        entry_point="agent",
        providers=__import__("teff.provider", fromlist=["ProviderRegistry"]).ProviderRegistry.from_presets(
            "openai"
        ),
        default_provider="openai",
    )
    result = await g.run(
        state={"input": "Привет"}, tools=[ReplyToUser()], max_iterations=5
    )
    # the message lands in output and the loop did not re-invoke the LLM
    assert result["output"] == "Здравствуйте! Чем помочь?"
    assert len(post_calls) == 1
    assert result["_react_done"] is False


@pytest.mark.asyncio
async def test_greeting_flow_goes_through_reply_tool(monkeypatch, tmp_path):
    """End to end: the repair coordinator answers a greeting via
    reply_to_user instead of a bare-text answer or pipeline tools."""
    import sys

    if str(_EXAMPLE_PATH) not in sys.path:
        sys.path.insert(0, str(_EXAMPLE_PATH))

    import httpx

    from src.graphs.build import build_flow
    from src.graphs.state import STATE_REDUCERS, initial_state
    from teff.checkpoint import JSONFileCheckpointer

    class _MockTransport:
        def __init__(self):
            self.coordinator_calls = 0

        def _content_for(self, body: dict) -> dict:
            system = "".join(
                m.get("content", "")
                for m in body.get("messages", [])
                if m.get("role") == "system"
            )
            if "Координатор" in system:
                self.coordinator_calls += 1
                return _reply_tool_response(
                    "c_greet", "Здравствуйте! Готов помочь с ремонтом."
                )
            return _final_response("")

        def __call__(self, *args, **kwargs):
            data = self._content_for(kwargs.get("json") or {})

            class _Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return data

            async def _post():
                return _Resp()

            return _post()

    mock = _MockTransport()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock)

    flow, tools = build_flow()
    graph = flow.compile()
    cp = JSONFileCheckpointer(str(tmp_path / "cp"))
    state = initial_state()
    state["messages"] = [{"role": "user", "content": "Привет!"}]
    result = await graph.run(
        state,
        tools=tools,
        reducers=STATE_REDUCERS,
        checkpointer=cp,
        checkpoint_id="run-1",
        max_iterations=40,
    )
    assert mock.coordinator_calls == 1
    assert result["output"] == "Здравствуйте! Готов помочь с ремонтом."
    last = [m for m in result["messages"] if m.get("role") == "assistant"][-1]
    assert last["content"] == "Здравствуйте! Готов помочь с ремонтом."
    assert result["project_info"] == {}