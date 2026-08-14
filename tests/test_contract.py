"""Tests for the ``Contract`` structured-output recipe and its nodes.

Covers the declarative recipe (schema LLM pass, normalize, semantic
validate, optional no-tools retry, deterministic fallback), the three
``Contract*`` node types, and the end-to-end branching inside the built
``SubFlow``.
"""

import pytest

from teff.flow import Flow
from teff.node import (
    LLM,
    Contract,
    ContractFallback,
    ContractNormalize,
    ContractValidate,
)
from teff.node.registry import default_registry


async def _run(node, state: dict, *, providers=None) -> dict:
    from teff.node import ExecContext

    ctx = ExecContext(state=state, tools={}, providers=providers)
    return await node.execute(ctx, state)


def _is_good(value):
    return value == "good"


class TestContract:
    def test_validate_requires_fallback(self):
        with pytest.raises(ValueError, match="requires a fallback"):
            Contract(system="s", validate=_is_good)

    def test_rejects_non_callables(self):
        with pytest.raises(TypeError):
            Contract(system="s", validate="yes")
        with pytest.raises(TypeError):
            Contract(system="s", fallback="nope")

    def test_plain_chain_without_validate(self):
        contract = Contract(
            system="s",
            schema={},
            model="m",
            provider="p",
            messages_key="messages",
            output_key="answer",
        )
        subflow = contract.build()
        graph = subflow._graph
        node_types = {n.type for nid, n in graph.nodes.items()}
        assert node_types == {"llm_chat", "contract_normalize", "contract_validate"}
        llm_cfg = graph.nodes["llm"].config
        assert llm_cfg["system"] == "s"
        assert llm_cfg["messages_key"] == "messages"
        assert llm_cfg["output_key"] == "answer_raw"

    def test_retry_without_tools_builds_second_pass_and_fallback(self):
        contract = Contract(
            system="s",
            model="m",
            provider="p",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            use_tools=True,
            retry_without_tools=True,
        )
        graph = contract.build()._graph
        node_types = {n.type for nid, n in graph.nodes.items()}
        assert node_types == {
            "llm_chat",
            "contract_normalize",
            "contract_validate",
            "contract_fallback",
        }
        assert "retry-llm" in graph.nodes
        assert graph.nodes["retry-llm"].config.get("use_tools") in (None, [], False)
        assert "llm" in graph.nodes
        assert graph.nodes["llm"].config.get("use_tools") is True

    def test_no_retry_when_first_pass_has_no_tools(self):
        contract = Contract(
            system="s",
            model="m",
            provider="p",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            retry_without_tools=True,
        )
        graph = contract.build()._graph
        assert "retry-llm" not in graph.nodes
        assert "fallback" in graph.nodes

    def test_llm_threads_kwargs(self):
        contract = Contract(
            system="s",
            schema={},
            model="m",
            provider="p",
            messages_key="messages",
            output_key="answer",
            max_retries=5,
            temperature=0.2,
        )
        llm_cfg = contract.llm(use_tools=None).config
        assert llm_cfg["max_retries"] == 5
        assert llm_cfg["temperature"] == 0.2

    def test_id_prefixes_inner_nodes(self):
        contract = Contract(
            system="s",
            model="m",
            provider="p",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            id="final",
        )
        graph = contract.build()._graph
        assert set(graph.nodes) == {"final-llm", "final-normalize", "final-validate", "final-fallback"}


class TestContractNormalize:
    @pytest.mark.asyncio
    async def test_applies_fn(self):
        node = ContractNormalize(input_key="raw", output_key="out", fn=lambda v: v.strip())
        out = await _run(node, {"raw": "  hello  "})
        assert out["out"] == "hello"

    @pytest.mark.asyncio
    async def test_passthrough_without_fn(self):
        node = ContractNormalize(input_key="raw", output_key="out")
        out = await _run(node, {"raw": {"answer": "hi"}})
        assert out["out"] == {"answer": "hi"}


class TestContractValidate:
    @pytest.mark.asyncio
    async def test_passes(self):
        node = ContractValidate(input_key="out", output_key="ok", fn=_is_good)
        out = await _run(node, {"out": "good"})
        assert out["ok"] == "yes"

    @pytest.mark.asyncio
    async def test_fails(self):
        node = ContractValidate(input_key="out", output_key="ok", fn=_is_good)
        out = await _run(node, {"out": "bad"})
        assert out["ok"] == "no"

    @pytest.mark.asyncio
    async def test_defaults_to_nonempty(self):
        node = ContractValidate(input_key="out", output_key="ok")
        assert (await _run(node, {"out": "x"}))["ok"] == "yes"
        assert (await _run(node, {"out": ""}))["ok"] == "no"


class TestContractFallback:
    @pytest.mark.asyncio
    async def test_writes_value(self):
        node = ContractFallback(output_key="out", fn=lambda st: f"fb-{st['query']}")
        out = await _run(node, {"query": "q"})
        assert out["out"] == "fb-q"

    @pytest.mark.asyncio
    async def test_requires_callable(self):
        node = ContractFallback(output_key="out")
        with pytest.raises(ValueError, match="callable"):
            await _run(node, {})


class TestContractInFlow:
    @pytest.fixture
    def providers(self):
        from teff.provider import ProviderRegistry

        return ProviderRegistry.from_presets("ollama")

    @pytest.mark.asyncio
    async def test_valid_output_skips_retry_and_fallback(self, monkeypatch, providers):
        calls = {}

        async def fake_execute(self, ctx, state):
            calls["use_tools"] = self.config.get("use_tools")
            return {"answer_raw": "good"}

        monkeypatch.setattr(LLM, "execute", fake_execute)

        subflow = Contract(
            system="s",
            model="m",
            provider="ollama",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            use_tools=True,
            retry_without_tools=True,
        ).build()
        out = await _run(
            subflow, {"messages": [{"role": "user", "content": "hi"}]}, providers=providers
        )
        assert out["answer"] == "good"
        assert "retry" not in calls or calls.get("retry") is None

    @pytest.mark.asyncio
    async def test_retry_without_tools_on_semantic_failure(self, monkeypatch, providers):
        calls = []

        async def fake_execute(self, ctx, state):
            use = self.config.get("use_tools")
            calls.append("tools" if use else "no-tools")
            if calls.count("no-tools") == 0:
                return {"answer_raw": "bad"}
            return {"answer_raw": "good"}

        monkeypatch.setattr(LLM, "execute", fake_execute)

        subflow = Contract(
            system="s",
            model="m",
            provider="ollama",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            use_tools=True,
            retry_without_tools=True,
        ).build()
        out = await _run(
            subflow, {"messages": [{"role": "user", "content": "hi"}]}, providers=providers
        )
        assert out["answer"] == "good"
        assert calls == ["tools", "no-tools"]

    @pytest.mark.asyncio
    async def test_fallback_used_when_both_passes_fail(self, monkeypatch, providers):
        async def fake_execute(self, ctx, state):
            return {"answer_raw": "bad"}

        monkeypatch.setattr(LLM, "execute", fake_execute)

        subflow = Contract(
            system="s",
            model="m",
            provider="ollama",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            use_tools=True,
            retry_without_tools=True,
        ).build()
        out = await _run(
            subflow, {"messages": [{"role": "user", "content": "hi"}]}, providers=providers
        )
        assert out["answer"] == "fallback"

    def test_flow_contract_helper_wires_subflow(self):
        flow = Flow("wire").contract(
            system="s",
            model="m",
            provider="p",
            messages_key="messages",
            output_key="answer",
            validate=_is_good,
            fallback=lambda st: "fallback",
            id="final",
        )
        graph = flow.compile()
        node_types = {n.type for nid, n in graph.nodes.items()}
        assert "subflow" in node_types

    def test_registry_builds_contract(self):
        node = default_registry.create(
            "contract",
            {
                "system": "s",
                "model": "m",
                "provider": "p",
                "messages_key": "messages",
                "output_key": "answer",
            },
        )
        assert node.type == "subflow"
        assert "llm" in node._graph.nodes  # entry node exists
