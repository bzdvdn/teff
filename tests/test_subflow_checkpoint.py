"""Tests for checkpoints inside SubFlow nodes.

A :class:`~teff.flow.sub_flow.SubFlow` forwards the enclosing run's
checkpointer to its nested ``graph.run()`` under a namespaced checkpoint id
(``<parent-cid>:sub:<node-id>``).  That makes two things work:

* an ``Interrupt`` inside a sub-flow resumes *inside* the sub-flow instead
  of being skipped over by the parent;
* an exception inside a sub-flow resumes at the failing node on retry
  instead of re-running the whole sub-flow.
"""

import asyncio

import pytest

from teff.flow import Flow
from teff.node import Node, Transform
from teff.node.interrupt import GraphInterrupt


class _Flaky(Node):
    """Transient failure: fails on its first call, succeeds afterwards."""

    type = "flaky"

    def __init__(self, *, key: str):
        super().__init__(key=key)
        self._calls = 0

    async def execute(self, ctx, state):
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("boom")
        state[self.config["key"]] = "recovered"
        return {self.config["key"]: "recovered"}


class TestSubFlowInterruptResume:
    def test_interrupt_inside_subflow_resumes_in_place(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer

        inner = (
            Flow("inner")
            .step(
                Transform(
                    {"action": "uppercase", "input_key": "text", "output_key": "draft"}
                )
            )
            .interrupt("approved", prompt="Approve?")
            .step(
                Transform(
                    {"action": "uppercase", "input_key": "draft", "output_key": "final"}
                )
            )
        )
        outer = Flow("outer").add_flow(inner, id="inner").step(
            Transform({"action": "uppercase", "input_key": "final", "output_key": "done"})
        )
        g = outer.compile()
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt) as excinfo:
            asyncio.run(
                g.run(
                    state={"text": "hello"},
                    checkpointer=cp,
                    checkpoint_id="run-1",
                )
            )
        exc = excinfo.value
        assert exc.key == "approved"
        assert exc.node_id == "inner"
        assert exc.nested_checkpoint_id == "run-1:sub:inner"

        # nested checkpoint recorded the pause inside the sub-flow
        saved = asyncio.run(cp.load("run-1:sub:inner"))
        assert saved is not None
        assert saved.state.get("draft") == "HELLO"

        result = asyncio.run(
            g.run(
                state={"text": "ignored"},
                checkpointer=cp,
                checkpoint_id="run-1",
                resume={"approved": "yes"},
            )
        )
        # the sub-flow continued past the interrupt (draft came from the
        # durable nested state, final from the resumed step) and the outer
        # flow continued after the sub-flow.
        assert result["draft"] == "HELLO"
        assert result["final"] == "HELLO"
        assert result["done"] == "HELLO"
        assert result["approved"] == "yes"
        assert "__interrupt__" not in result

    def test_resume_without_answer_reraises_nested(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer

        inner = Flow("inner").interrupt("ok", "Ok?").step(
            Transform({"action": "uppercase", "input_key": "x", "output_key": "y"})
        )
        g = Flow("outer").add_flow(inner, id="inner").compile()
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt):
            asyncio.run(g.run(state={"x": "a"}, checkpointer=cp, checkpoint_id="r1"))
        with pytest.raises(GraphInterrupt) as excinfo:
            asyncio.run(g.run(state={}, checkpointer=cp, checkpoint_id="r1"))
        assert excinfo.value.key == "ok"

    def test_completed_subflow_drops_nested_checkpoint(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer

        inner = Flow("inner").step(
            Transform({"action": "uppercase", "input_key": "x", "output_key": "y"})
        )
        g = Flow("outer").add_flow(inner, id="inner").compile()
        cp = JSONFileCheckpointer(str(tmp_path))

        asyncio.run(g.run(state={"x": "a"}, checkpointer=cp, checkpoint_id="run-1"))
        # the nested run completed, so its checkpoint is cleaned up
        assert asyncio.run(cp.load("run-1:sub:inner")) is None


class TestSubFlowErrorRetry:
    def test_error_retry_resumes_at_failing_node(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer

        inner = (
            Flow("inner")
            .step(Transform({"action": "uppercase", "input_key": "x", "output_key": "a"}))
            .step(_Flaky(key="b"))
        )
        outer = Flow("outer").add_flow(inner, id="inner")
        g = outer.compile()
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(g.run(state={"x": "a"}, checkpointer=cp, checkpoint_id="run-1"))

        # first node ran and was checkpointed before the flaky node failed
        assert asyncio.run(cp.load("run-1:sub:inner")) is not None

        result = asyncio.run(
            g.run(state={"x": "ignored"}, checkpointer=cp, checkpoint_id="run-1")
        )
        # "a" survived from the durable nested checkpoint (the uppercase
        # node did NOT re-run on retry), and the flaky node recovered.
        assert result["a"] == "A"
        assert result["b"] == "recovered"
