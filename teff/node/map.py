"""Map node — dynamically fan a state list out across concurrent branches."""

import asyncio
import time
from typing import Sequence

from teff._async_util import gather_or_cancel
from teff.node.command import as_updates
from teff.node.context import ExecContext
from teff.node.node import Node
from teff.state import apply_reducers
from teff.trace import _ms


class Map(Node):
    """Run a processor over each item of a state list, in parallel.

    Reads one or more lists from *input_keys*, splits them into chunks,
    and runs the *processor* (a single node or a chain) concurrently on
    each chunk — with the chunk placed back under its key in an isolated
    state copy.  The per-chunk result is gathered into a list stored at
    *output_key*, preserving the order of the source lists.

    With several *input_keys* the lists are zipped: chunk ``i`` contains
    ``key[0][i]``, ``key[1][i]``, etc., so the processor can read
    multiple per-item values straight from state.

    This is the *dynamic* sibling of :class:`Parallel`: branches are
    derived from data at runtime instead of being declared up front.

    Args:
        processor: Node or list of Nodes run per chunk (sequentially).
        input_keys: One or more state keys holding the lists to fan out.
            The processor reads these same keys from the branch state.
        output_key: State key that receives the list of per-chunk results.
        result_key: State key holding each chunk's result.  Defaults to
            the processor's own ``output_key`` (single-node processors);
            pass it explicitly for multi-node chains.
        chunk_size: Items per branch (default 1 = one item per branch).
        max_concurrency: Limit on simultaneously running branches
            (default ``None`` = no limit).

    Usage::

        node = Map(
            processor=LLM(model="llama3.1:8b",
                          input_key="chunk", output_key="summary"),
            input_keys=["chunks"],
            output_key="summaries",
            chunk_size=4,
            max_concurrency=2,
        )
    """

    type = "map"

    def __init__(
        self,
        processor: Node | list[Node] | dict | list[dict],
        *,
        input_keys: str | list[str] = "",
        output_key: str = "",
        result_key: str | None = None,
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
        config: dict | None = None,
        **kwargs,
    ):
        keys = [input_keys] if isinstance(input_keys, str) else list(input_keys)
        merged = {
            "input_keys": keys,
            "output_key": output_key,
            "result_key": result_key,
            "chunk_size": chunk_size,
            "max_concurrency": max_concurrency,
            "processor": processor,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)
        self._processor = self._build_processor(processor)

    @staticmethod
    def _build_processor(
        processor: Node | list[Node] | dict | list[dict],
    ) -> list[Node]:
        from teff.node.registry import default_registry

        def resolve(spec: Node | dict) -> Node:
            if isinstance(spec, Node):
                return spec
            if isinstance(spec, dict):
                spec = dict(spec)
                stype = spec.pop("type")
                cfg = spec.pop("config", None)
                if cfg is not None:
                    if not isinstance(cfg, dict):
                        msg = f"invalid processor spec: {spec!r}"
                        raise TypeError(msg)
                    cfg = {**spec, **dict(cfg)}
                else:
                    cfg = spec
                return default_registry.create(stype, cfg)
            msg = f"invalid processor spec: {spec!r}"
            raise TypeError(msg)

        if isinstance(processor, list):
            return [resolve(spec) for spec in processor]
        return [resolve(processor)]

    async def execute(self, ctx: ExecContext, state: dict) -> dict:
        input_keys = self.config.get("input_keys", [])
        output_key = self.config.get("output_key", "")
        result_key = self._resolve_result_key(output_key)
        chunk_size = self.config.get("chunk_size") or 1
        max_concurrency = self.config.get("max_concurrency")

        lists = [state.get(key, []) for key in input_keys]
        for key, items in zip(input_keys, lists):
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                msg = f"map input_key '{key}' is not a list"
                raise TypeError(msg)
        if not lists or not lists[0]:
            return {output_key: []}

        length = len(lists[0])
        for key, items in zip(input_keys, lists):
            if len(items) != length:
                msg = (
                    f"map input_keys length mismatch: '{key}' has "
                    f"{len(items)} items, expected {length}"
                )
                raise ValueError(msg)

        chunks = [
            [items[i : i + chunk_size] for i in range(0, length, chunk_size)]
            for items in lists
        ]
        groups = list(zip(*chunks))

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        reducers = getattr(ctx, "reducers", None) or {}

        async def run_group(group: tuple, idx: int) -> object:
            if semaphore is not None:
                async with semaphore:
                    return await self._run_chunk(
                        group,
                        idx,
                        ctx,
                        state,
                        reducers,
                        input_keys,
                        result_key,
                        chunk_size,
                    )
            return await self._run_chunk(
                group, idx, ctx, state, reducers, input_keys, result_key, chunk_size
            )

        results = await gather_or_cancel(
            *(run_group(group, idx) for idx, group in enumerate(groups))
        )
        return {output_key: list(results)}

    def _resolve_result_key(self, output_key: str) -> str:
        configured = self.config.get("result_key")
        if configured:
            return configured
        if len(self._processor) == 1:
            node_key = self._processor[0].config.get("output_key")
            if node_key:
                return node_key
        return output_key

    async def _run_chunk(
        self,
        group: tuple,
        chunk_idx: int,
        ctx: ExecContext,
        state: dict,
        reducers: dict,
        input_keys: list[str],
        result_key: str,
        chunk_size: int,
    ) -> object:
        branch_state = dict(state)
        for key, chunk in zip(input_keys, group):
            branch_state[key] = chunk[0] if chunk_size == 1 else chunk
        for node_idx, node in enumerate(self._processor):
            node_id = f"{ctx.node_id or self.type}.m{chunk_idx}.{node_idx}"
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
        return branch_state.get(result_key)
