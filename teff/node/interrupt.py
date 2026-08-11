"""Interrupt node — pause a workflow for external (human) input.

Human-in-the-loop: when execution reaches an :class:`Interrupt` node,
``graph.run()`` saves a checkpoint and raises :class:`GraphInterrupt`.
The operator provides a value and the run is resumed with the same
``checkpoint_id`` plus a ``resume`` value, which is written into the
state under the interrupt's *key* before the graph continues with the
node that follows the interrupt.
"""

from teff.errors import TeffError
from teff.node.node import Node
from teff.prompt import render_template


class GraphInterrupt(TeffError):
    """Raised by ``graph.run()`` when a workflow pauses for human input.

    Attributes:
        key: State key the resume value will be written to.
        prompt: Human-readable question shown to the operator.
        node_id: Id of the interrupt node that paused execution.
        checkpoint_id: Pass this back to ``graph.run()`` with the same
            checkpointer together with ``resume`` to continue.
        nested_checkpoint_id: When the interrupt fired inside a
            :class:`~teff.flow.sub_flow.SubFlow`, the checkpoint id the
            sub-flow paused under.  Resuming routes back into the sub-flow
            instead of continuing past it.
    """

    def __init__(
        self,
        key: str,
        prompt: str = "",
        node_id: str | None = None,
        checkpoint_id: str | None = None,
    ):
        super().__init__(f"workflow paused for input: {prompt or key}")
        self.key = key
        self.prompt = prompt
        self.node_id = node_id
        self.checkpoint_id = checkpoint_id
        self.nested_checkpoint_id: str | None = None


class Interrupt(Node):
    """Pause the workflow and wait for external (human) input.

    When execution reaches this node, ``graph.run()`` saves a checkpoint
    and raises :class:`GraphInterrupt`.  The operator provides a value
    and the graph is resumed with the same ``checkpoint_id`` and
    ``resume``::

        try:
            await graph.run(state, checkpointer=cp, checkpoint_id="run-1")
        except GraphInterrupt as interrupt:
            print(interrupt.prompt)
            value = input("> ")
            await graph.run(
                state, checkpointer=cp, checkpoint_id="run-1", resume=value
            )

    The resumed value is written to the state under *key* before
    continuing with the node that follows this one.

    Requires a checkpointer to be set on ``graph.run()``.

    Config:
        key: State key that receives the resume value.
        prompt: Human-readable question for the operator.
    """

    type = "interrupt"

    async def execute(self, ctx, state: dict) -> dict:
        prompt = self.config.get("prompt", "")
        if "{" in prompt:
            prompt = render_template(prompt, state)
        raise GraphInterrupt(
            self.config.get("key", ""),
            prompt,
        )
