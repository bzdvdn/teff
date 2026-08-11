"""SubFlow — a node that executes a nested graph."""

import copy
from typing import Awaitable, Callable

from teff.graph import Edge, Graph
from teff.node.interrupt import GraphInterrupt
from teff.node.node import Node
from teff.state import reducer_appends
from teff.stream import StreamEvent


class SubFlow(Node):
    """A node that executes a sub-graph with optional key mapping.

    Args:
        graph: Compiled sub-graph to run.
        input_map: Parent key → sub-graph key (default: passthrough).
        output_map: Sub-graph key → parent key (default: passthrough).
        max_iterations: Max node executions inside the sub-graph
            (passed to ``graph.run()``).  ``None`` means unlimited.
    """

    type = "subflow"

    def __init__(
        self,
        graph: Graph,
        input_map: dict[str, str] | None = None,
        output_map: dict[str, str] | None = None,
        max_iterations: int | None = None,
        *,
        id_prefix: str = "",
    ):
        super().__init__(
            input_map=input_map or {},
            output_map=output_map or {},
            max_iterations=max_iterations,
            id_prefix=id_prefix,
        )
        self._graph = graph
        self._input_map = input_map or {}
        self._output_map = output_map or {}
        self._max_iterations = max_iterations
        self._id_prefix = id_prefix
        if id_prefix:
            self._graph = self._prefix_graph(graph, id_prefix)

    async def execute(self, ctx, state: dict) -> dict:
        reducers = getattr(ctx, "reducers", None)
        if self._input_map:
            sub_state = {}
            for parent_key, sub_key in self._input_map.items():
                sub_state[sub_key] = copy.deepcopy(state.get(parent_key))
        else:
            sub_state = copy.deepcopy(state)
        input_snapshot = copy.deepcopy(sub_state)

        checkpointer = getattr(ctx, "checkpointer", None)
        checkpoint_id = getattr(ctx, "checkpoint_id", None)
        nested_checkpoint_id = None
        if checkpointer is not None and checkpoint_id:
            nested_checkpoint_id = f"{checkpoint_id}:sub:{ctx.node_id or 'subflow'}"

        run_kwargs: dict = dict(
            tools=list(ctx.tools.values()),
            reducers=getattr(ctx, "reducers", None),
            hooks=getattr(ctx, "hooks", None),
            node_timeout=getattr(ctx, "node_timeout", None),
            max_iterations=self._max_iterations,
            emit=self._forward(ctx.emit),
            providers=getattr(ctx, "providers", None),
            default_provider=getattr(ctx, "default_provider", None),
            default_model=getattr(ctx, "default_model", None),
            on_llm_payload=getattr(ctx, "on_llm_payload", None),
            checkpointer=checkpointer,
            checkpoint_id=nested_checkpoint_id,
            owner=getattr(ctx, "owner", None),
            resume=getattr(ctx, "resume", None),
        )

        try:
            result = await self._graph.run(sub_state, **run_kwargs)
        except GraphInterrupt as exc:
            exc.node_id = ctx.node_id
            exc.nested_checkpoint_id = nested_checkpoint_id
            raise

        if nested_checkpoint_id is not None and checkpointer is not None:
            await checkpointer.delete(nested_checkpoint_id, owner=ctx.owner)

        out = {}
        if self._output_map:
            for sub_key, parent_key in self._output_map.items():
                out[parent_key] = result.get(sub_key)
        else:
            out = self._passthrough_delta(input_snapshot, result, reducers)
        return out

    @staticmethod
    def _passthrough_delta(
        input_state: dict, result: dict, reducers: "dict | None"
    ) -> dict:
        """Return only what the sub-graph changed, under the parent's reducers.

        In passthrough mode the parent merges the returned value through its
        own reducers, but the nested run already applied them inside
        ``sub_state`` — so returning the whole accumulated state would apply
        an ``append`` reducer twice.  To keep the parent's single merge
        correct we hand back a *delta*:

        * append-style keys (`reducer_appends`)  → the newly appended items
          gathered inside the sub-graph (`result[key][len(input):]`), so the
          parent appends them exactly once;
        * override keys → the new value (only when it actually changed), so
          untouched keys are not clobbered on the way out.
        """
        out: dict = {}
        for key, value in result.items():
            old = input_state.get(key)
            if reducer_appends((reducers or {}).get(key)):
                nv = value
                if isinstance(old, list) and isinstance(value, list):
                    prefix = value[: len(old)]
                    if prefix == old:
                        nv = value[len(old) :]
                if nv:
                    out[key] = nv
            elif old != value:
                out[key] = value
        return out

    @staticmethod
    def _prefix_graph(graph: Graph, prefix: str) -> Graph:
        """Rename every node in *graph* to ``prefix/<original>``."""
        nodes = {f"{prefix}/{nid}": node for nid, node in graph.nodes.items()}
        edges = [
            Edge(
                source_id=f"{prefix}/{e.source_id}",
                target_id=f"{prefix}/{e.target_id}",
                condition=e.condition,
            )
            for e in graph.edges
        ]
        return Graph(
            nodes=nodes,
            edges=edges,
            entry_point=f"{prefix}/{graph.entry_point}",
            providers=graph.providers,
            default_provider=graph.default_provider,
            default_model=graph.default_model,
        )

    @staticmethod
    def _forward(
        emit: "Callable[[StreamEvent], Awaitable[None]] | None",
    ) -> "Callable[[StreamEvent], Awaitable[None]] | None":
        """Wrap an outer emit sink, dropping the nested run's bookkeeping.

        The inner run emits its own ``run_start``/``run_end`` lifecycle
        events; those belong to the top-level stream, so they are
        stripped while node/token/llm/edge events stream through.
        ``interrupt``/``interrupt_resume`` are also stripped: ``SubFlow``
        re-raises :class:`~teff.node.interrupt.GraphInterrupt` to the
        enclosing run, which emits those events itself (with the sub-flow's
        node id), so emitting them here too would duplicate them.
        """
        if emit is None:
            return None

        _STRIPPED = ("run_start", "run_end", "interrupt", "interrupt_resume")

        async def forward(event: StreamEvent) -> None:
            if event.type in _STRIPPED:
                return
            await emit(event)

        return forward
