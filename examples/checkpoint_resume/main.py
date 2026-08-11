"""Durable graph execution with an LLM call: crash, then resume in place.

An LLM drafts a summary, the next node crashes on the first run. The run
raises, but the checkpoint (which includes the LLM's answer) was already
persisted. Re-running with the same ``checkpoint_id`` continues from the
node that failed — the proof is in the counters:

- ``processed == 1`` — the first node did NOT re-run, and
- ``llm_calls == 1`` — the LLM did NOT re-run either: no tokens wasted.

Requires Ollama running locally with ``llama3.1:8b``:

    ollama pull llama3.1:8b

Usage:
    python examples/checkpoint_resume/main.py
"""

import asyncio
import os
from typing import TypedDict

from teff.checkpoint import SQLiteCheckpointer
from teff.flow import Flow
from teff.node import LLM, Node, Transform
from teff.provider import ProviderRegistry
from teff.state import State

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "checkpoints.db")

# Simulates a transient external failure (network blip, timeout): the node
# fails once, then succeeds.  Must live outside the state, because state is
# restored to the pre-node checkpoint on resume.
_crash_once = {"armed": True}


class Count(Node):
    """Increments ``processed`` on every execution."""

    type = "count"

    async def execute(self, ctx, state):
        state["processed"] = state.get("processed", 0) + 1
        return state


class CountingLLM(LLM):
    """An LLM node that counts how many times the model is actually called."""

    type = "counting_llm"

    def __init__(self, *, counter: dict, **config):
        super().__init__(**config)
        self._counter = counter

    async def execute(self, ctx, state):
        self._counter["llm_calls"] += 1
        return await super().execute(ctx, state)


class FailingNode(Node):
    """Raises on the first execution, succeeds afterwards."""

    type = "failing"

    async def execute(self, ctx, state):
        if _crash_once["armed"]:
            _crash_once["armed"] = False
            raise RuntimeError("simulated transient failure")
        state["done"] = True
        return state


class RunState(TypedDict):
    processed: int
    topic: str
    summary: str
    loud: str
    done: bool


async def main():
    counters = {"llm_calls": 0}
    flow = (
        Flow(
            "durable-llm",
            providers=ProviderRegistry.from_presets("ollama"),
            default_provider="ollama",
            default_model="llama3.1:8b",
        )
        .step(Count({}))  # node 1 — must run exactly once, ever
        .llm(
            CountingLLM(
                counter=counters,
                model="llama3.1:8b",
                prompt="Summarize {topic} in one short sentence.",
                output_key="summary",
            )
        )  # node 2 — the LLM call
        .step(FailingNode({}))  # node 3 — crashes on the first attempt
        .transform(
            Transform(
                {"action": "uppercase", "input_key": "summary", "output_key": "loud"}
            )
        )
    )
    graph = flow.compile()

    checkpointer = SQLiteCheckpointer(DB_PATH)
    try:
        state = State(RunState, {"topic": "rocket propulsion"})

        for attempt in (1, 2):
            try:
                result = await graph.run(
                    state=state,
                    checkpointer=checkpointer,
                    checkpoint_id="durable-run",
                )
                print(f"Run {attempt}: success -> {result}")
            except RuntimeError as e:
                print(f"Run {attempt}: crashed ({e}), checkpoint saved")
                continue

        # the checkpoint still points at the failed node
        cp = await checkpointer.load("durable-run")
        print("Saved checkpoint:", cp)

        # the proof: neither node 1 nor the LLM re-ran after the crash
        assert result["processed"] == 1, "node 1 must not re-execute on resume"
        assert counters["llm_calls"] == 1, "the LLM must not re-run on resume"
        assert result["done"] is True
        print("Durable: resume continued from the failed node — no tokens wasted")
    finally:
        checkpointer.close()
        # keep the DB when TEEF_KEEP_CHECKPOINT=1 so the CLI can inspect it
        if not os.environ.get("TEEF_KEEP_CHECKPOINT"):
            os.remove(DB_PATH)


if __name__ == "__main__":
    asyncio.run(main())
