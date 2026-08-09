"""Tests for the ``Command`` node return value (update + goto routing)."""

import pytest

from teff.node import Command, Transform
from teff.node.node import Node


class _Set(Node):
    """A node that writes *value* under *key* and optionally routes."""

    type = "set"

    def __init__(self, key, value, goto=None, *, update_only=False):
        super().__init__()
        self._key = key
        self._value = value
        self._goto = goto
        self._update_only = update_only

    async def execute(self, ctx, state):
        if self._goto is not None:
            return Command(update={self._key: self._value}, goto=self._goto)
        if self._update_only:
            return Command(update={self._key: self._value})
        return {self._key: self._value}


async def _run(graph, state=None, **kwargs):
    return await graph.run(state or {}, max_iterations=20, **kwargs)


class TestCommandRouting:
    @pytest.mark.asyncio
    async def test_goto_routes_directly(self):
        from teff.flow import Flow

        flow = Flow("cmd")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto="admin"), id="decider")
        flow.step(
            Transform(action="value", value="ADMIN_OK", output_key="who"), id="admin"
        )
        r = await _run(flow.compile())
        assert r["who"] == "ADMIN_OK"

    @pytest.mark.asyncio
    async def test_goto_skips_intermediate_edge(self):
        """goto jumps straight to the target, bypassing condition edges."""
        from teff.flow import Flow

        flow = Flow("cmd")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto="admin"), id="decider")
        # an unconditional sibling the Command must skip
        flow.step(
            Transform(action="value", value="WRONG", output_key="who"), id="other"
        )
        flow.step(
            Transform(action="value", value="ADMIN_OK", output_key="who"), id="admin"
        )
        r = await _run(flow.compile())
        assert r["who"] == "ADMIN_OK"

    @pytest.mark.asyncio
    async def test_command_update_only_keeps_normal_routing(self):
        """Command(update=...) without goto routes along the normal edges."""
        from teff.flow import Flow

        flow = Flow("cmd")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto=None, update_only=True), id="decider")
        flow.step(Transform(action="value", value="OK", output_key="who"), id="next")
        r = await _run(flow.compile())
        assert r["who"] == "OK"

    @pytest.mark.asyncio
    async def test_goto_targets_must_be_reachable(self):
        from teff.errors import WorkflowError
        from teff.flow import Flow

        flow = Flow("bad")
        flow.step(_Set("role", "keep"), id="start")
        flow.step(_Set("who", "ADMIN", goto="missing"), id="decider")
        flow.step(Transform(action="value", value="OK", output_key="who"), id="next")
        with pytest.raises(WorkflowError, match="unknown node"):
            await _run(flow.compile())

    @pytest.mark.asyncio
    async def test_stop_terminates_run(self):
        from teff.flow import Flow

        flow = Flow("stop")
        flow.step(_Set("who", "FIRST"), id="start")
        flow.step(_Set("who", "STOPPED", goto=Command.STOP), id="decider")
        flow.step(
            Transform(action="value", value="NEVER", output_key="who"), id="never"
        )
        r = await _run(flow.compile())
        assert r["who"] == "STOPPED"

    @pytest.mark.asyncio
    async def test_command_merges_via_reducers(self):
        from teff.flow import Flow

        def append(old, new):
            return (old or []) + new

        flow = Flow("reducers")
        flow.step(_Set("items", ["a"], goto=None, update_only=True), id="s1")
        flow.step(_Set("items", ["b"], goto=None, update_only=True), id="s2")
        r = await flow.compile().run({}, max_iterations=20, reducers={"items": append})
        assert r["items"] == ["a", "b"]


class TestCommandWithPlainFunctions:
    @pytest.mark.asyncio
    async def test_step_accepts_async_function(self):
        from teff.flow import Flow

        async def double(ctx, state):
            return {"doubled": int(state.get("n", 0)) * 2}

        flow = Flow("fn")
        flow.step(Transform(action="value", value="4", output_key="n"))
        flow.step(double)
        r = await _run(flow.compile())
        assert r["doubled"] == 8

    @pytest.mark.asyncio
    async def test_step_accepts_sync_function(self):
        from teff.flow import Flow

        def square(ctx, state):
            return {"squared": int(state.get("n", 0)) ** 2}

        flow = Flow("fn")
        flow.step(Transform(action="value", value="4", output_key="n"))
        flow.step(square)
        r = await _run(flow.compile())
        assert r["squared"] == 16

    @pytest.mark.asyncio
    async def test_step_function_may_return_command(self):
        from teff.flow import Flow

        def gate(ctx, state):
            return Command(update={"seen": True}, goto=Command.STOP)

        flow = Flow("fn")
        flow.step(gate, id="gate")
        flow.step(Transform(action="value", value="NEVER", output_key="x"), id="never")
        r = await _run(flow.compile())
        assert r["seen"] is True
        assert "x" not in r

    @pytest.mark.asyncio
    async def test_step_function_bad_return_type(self):
        from teff.flow import Flow

        def bad(ctx, state):
            return "nope"

        flow = Flow("fn")
        flow.step(bad)
        with pytest.raises(TypeError, match="must return a dict or Command"):
            await _run(flow.compile())

    def test_function_node_type_is_function_name(self):
        from teff.flow import Flow

        async def my_worker(ctx, state):
            return {}

        g = Flow("fn").step(my_worker).compile()
        types = {n.type for nid, n in g.nodes.items()}
        assert "my_worker" in types

    def test_step_rejects_non_callable(self):
        from teff.flow import Flow

        with pytest.raises(TypeError, match="must be a Node or function"):
            Flow("fn").step({"action": "uppercase"})  # type: ignore[arg-type]


class TestCommandNodeDeclarative:
    def _yaml(self, config: str) -> str:
        return f"""\
name: cmd
steps:
  - id: route
    type: command
    config: {config}
  - id: approve
    type: transform
    config: {{action: value, value: APPROVED, output_key: verdict}}
  - id: reject
    type: transform
    config: {{action: value, value: REJECTED, output_key: verdict}}
  - id: review
    type: transform
    config: {{action: value, value: REVIEW, output_key: verdict}}
"""

    @pytest.mark.asyncio
    async def test_routes_by_condition(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            self._yaml(
                "{routes: [{when: 'score >= 0.8', goto: approve}], goto: review}"
            )
        )
        graph, _, _, _ = load_workflow(str(path))
        r = await graph.run({"score": "0.92"}, max_iterations=20)
        assert r["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_falls_back_to_goto(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            self._yaml(
                "{routes: [{when: 'score >= 0.8', goto: approve}], goto: reject}"
            )
        )
        graph, _, _, _ = load_workflow(str(path))
        r = await graph.run({"score": "0.2"}, max_iterations=20)
        assert r["verdict"] == "REJECTED"

    @pytest.mark.asyncio
    async def test_stop_terminates(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(self._yaml("{goto: STOP}"))
        graph, _, _, _ = load_workflow(str(path))
        r = await graph.run({"score": "0.5"}, max_iterations=20)
        assert "verdict" not in r

    @pytest.mark.asyncio
    async def test_update_merges_state(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(self._yaml("{goto: review, update: {routed: true}}"))
        graph, _, _, _ = load_workflow(str(path))
        r = await graph.run({}, max_iterations=20)
        assert r["routed"] is True
        assert r["verdict"] == "REVIEW"

    @pytest.mark.asyncio
    async def test_unknown_goto_raises(self, tmp_path):
        from teff.errors import WorkflowError
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(self._yaml("{goto: missing}"))
        graph, _, _, _ = load_workflow(str(path))
        with pytest.raises(WorkflowError, match="unknown node"):
            await graph.run({}, max_iterations=20)

    def test_registered_in_registry(self):
        from teff.node import default_registry

        assert "command" in default_registry.list()
        node = default_registry.create("command", {"goto": "STOP"})
        assert node.type == "command"


def _verdict_reader(ctx, state):
    return {"verdict": state.get("verdict", "needs_work")}


def _rework_reader(ctx, state):
    verdict = state.get("verdict", "needs_work")
    return {
        "verdict": verdict,
        "decision": "approve" if verdict == "pass" else "rework",
    }


class TestFlowLabelCommand:
    """``Flow.label`` + ``Flow.command`` resolve declarative goto targets."""

    def _flow(self):
        from teff.flow import Flow
        from teff.node.transform import Transform

        flow = Flow("rt")
        flow.step(_verdict_reader, id="check_verdict")
        flow.loop(
            key="verdict",
            until="pass",
            body=[
                Transform({"action": "value", "value": "pass", "output_key": "verdict"})
            ],
            done=[
                Transform({"action": "value", "value": "ok", "output_key": "status"})
            ],
        )
        flow.label("refine")
        flow.command(
            routes=[{"when": "decision=rework", "goto": "refine"}], goto="STOP"
        )
        return flow

    def test_label_resolves_to_loop_decider(self):
        flow = self._flow()
        assert flow._loop_labels == {"refine": "check_verdict"}

    def test_command_compiles_goto_to_decider_id(self):
        flow = self._flow()
        graph = flow.compile()
        node = graph.nodes["command_4"]
        assert node.config["routes"] == [
            {"when": "decision=rework", "goto": "check_verdict"}
        ]
        assert node.config["goto"] == "STOP"

    @pytest.mark.asyncio
    async def test_e2e_loopable_returns_without_hanging(self):
        flow = self._flow()
        graph = flow.compile()
        r = await graph.run(
            {"decision": "approve", "verdict": "needs_work"}, max_iterations=20
        )
        assert r["verdict"] == "pass"
        assert r["status"] == "ok"

    @pytest.mark.asyncio
    async def test_e2e_label_can_loop_again(self):
        from teff.flow import Flow
        from teff.node.transform import Transform

        flow = Flow("rt2")
        flow.step(_rework_reader, id="check_verdict")
        flow.loop(
            key="verdict",
            until="pass",
            body=[
                Transform({"action": "value", "value": "pass", "output_key": "verdict"})
            ],
            done=[
                Transform({"action": "value", "value": "ok", "output_key": "status"})
            ],
        )
        flow.label("refine")
        flow.command(
            routes=[{"when": "decision=rework", "goto": "refine"}],
            goto="STOP",
        )
        graph = flow.compile()
        # start in "rework": verdict needs_work -> body -> pass; decider updates
        # verdict to pass so the second pass through the loop terminates.
        r = await graph.run(
            {"decision": "rework", "verdict": "needs_work"}, max_iterations=20
        )
        assert r["verdict"] == "pass"
        assert r["status"] == "ok"
