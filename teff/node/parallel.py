"""Parallel node — runs independent branches concurrently."""

import time

from teff._async_util import gather_or_cancel
from teff.node.command import as_updates
from teff.node.context import ExecContext
from teff.node.node import Node
from teff.state import apply_reducers
from teff.trace import _ms


class Parallel(Node):
    """Execute several branch chains concurrently and merge their results.

    Each *branch* is a list of nodes run sequentially on an isolated
    copy of the state.  Branches run concurrently via ``gather_or_cancel``;
    only the updates each node *returns* are merged back (per-key
    reducers apply, so ``append`` branches accumulate instead of
    overwriting one another).

    Because branches read from independent copies, direct in-place
    mutation of the passed state is not propagated.  Nodes inside
    branches should return their updates — the constitution's contract:
    *receive state → return state*.

    Args:
        branches: Sequence of branches, each a single :class:`Node` or a
            list of nodes.  Nodes inside a branch run sequentially.

    Usage::

        node = Parallel([[upper_node, count_node], [tag_node]])
    """

    type = "parallel"

    def __init__(
        self,
        branches: list[Node | list[Node]],
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self._branches: list[list[Node]] = [
            [b] if isinstance(b, Node) else list(b) for b in branches
        ]

    async def execute(self, ctx: ExecContext, state: dict) -> dict:
        reducers = getattr(ctx, "reducers", None) or {}
        deltas = await gather_or_cancel(
            *(
                self._run_branch(branch, idx, ctx, state, reducers)
                for idx, branch in enumerate(self._branches)
            )
        )

        merged: dict = {}
        for delta in deltas:
            apply_reducers(merged, delta, reducers)
        return merged

    async def _run_branch(
        self,
        branch: list[Node],
        branch_idx: int,
        ctx: ExecContext,
        state: dict,
        reducers: dict,
    ) -> dict:
        branch_state = dict(state)
        delta: dict = {}
        for node_idx, node in enumerate(branch):
            node_id = f"{ctx.node_id or self.type}.b{branch_idx}.{node_idx}"
            node_ctx = ExecContext(
                branch_state,
                ctx.tools,
                node_id=node_id,
                node_type=node.type,
                tracer=ctx.tracer,
                reducers=reducers,
                providers=getattr(ctx, "providers", None),
                default_provider=getattr(ctx, "default_provider", None),
                default_model=getattr(ctx, "default_model", None),
                on_llm_payload=getattr(ctx, "on_llm_payload", None),
            )
            start = time.monotonic()
            if ctx.tracer is not None:
                ctx.tracer.node_start(node_id, node.type)
            try:
                result = await node.execute(node_ctx, branch_state) or {}
            except Exception as exc:
                if ctx.tracer is not None:
                    ctx.tracer.node_error(node_id, node.type, _ms(start), exc)
                raise
            if ctx.tracer is not None:
                ctx.tracer.node_end(node_id, node.type, _ms(start))
            apply_reducers(branch_state, as_updates(result), reducers)
            apply_reducers(delta, as_updates(result), reducers)
        return delta
