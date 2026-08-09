"""Loop node — repeat a body chain until a state condition holds."""

from __future__ import annotations

import time

from teff.node.command import as_updates
from teff.node.context import ExecContext
from teff.node.node import Node
from teff.state import apply_reducers
from teff.trace import _ms


class Loop(Node):
    """Repeat a *body* chain until ``state[key]`` equals *until*.

    This is the self-contained sibling of :meth:`Flow.loop <teff.flow.Flow.loop>`:
    instead of wiring decider/done/body chains together with condition edges,
    the whole repeat lives inside one node, so a loop is expressible directly
    in YAML::

        - id: refine
          type: loop
          config:
            key: approved
            until: "yes"
            max_rounds: 3
            body:
              - {type: transform, config: {action: value, value: "no", output_key: approved}}

    Each round runs the *body* chain (a single node or a list), then evaluates
    the condition ``key=until`` against the merged state using the same
    expression language as ``edges:`` conditions (so ``until: "yes"`` matches
    ``"Yes"`` or ``"yes."``).  When the condition holds the loop stops; the body
    still runs at least once even if the condition already held on entry, which
    matches the Flow-loop contract where a decider writes *key* and then the
    body decides whether to re-run.

    ``max_rounds`` (default 10) bounds the repetition so a body that never
    reaches *until* cannot hang the workflow.

    Args:
        body: Node or list of Nodes (or their declarative dict specs) run per
            round, sequentially.
        key: State key the condition reads.
        until: Value of *key* that stops the loop.
        max_rounds: Maximum number of body rounds before giving up.
    """

    type = "loop"

    def __init__(
        self,
        body: Node | list[Node] | dict | list[dict],
        *,
        key: str = "",
        until: str = "",
        max_rounds: int = 10,
        config: dict | None = None,
        **kwargs,
    ):
        merged = {
            "key": key,
            "until": until,
            "max_rounds": max_rounds,
            "body": body,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)
        self._body = self._build_body(body)

    @staticmethod
    def _build_body(body: Node | list[Node] | dict | list[dict]) -> list[Node]:
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
                        msg = f"invalid loop body spec: {spec!r}"
                        raise TypeError(msg)
                    cfg = {**spec, **dict(cfg)}
                else:
                    cfg = spec
                return default_registry.create(stype, cfg)
            msg = f"invalid loop body spec: {spec!r}"
            raise TypeError(msg)

        if isinstance(body, list):
            return [resolve(spec) for spec in body]
        return [resolve(body)]

    async def execute(self, ctx: ExecContext | None, state: dict) -> dict:
        from teff.graph.conditions import evaluate

        key = self.config.get("key", "")
        until = self.config.get("until", "")
        max_rounds = int(self.config.get("max_rounds") or 10)
        if not key:
            raise ValueError("loop requires config.key")
        condition = f"{key}={until}"
        if ctx is None:
            ctx = ExecContext(state, {})
        reducers = getattr(ctx, "reducers", None) or {}

        for _ in range(max_rounds):
            for node_idx, node in enumerate(self._body):
                node_id = f"{ctx.node_id or self.type}.loop.{node_idx}"
                node_ctx = ExecContext(
                    state,
                    ctx.tools,
                    node_id=node_id,
                    node_type=node.type,
                    tracer=ctx.tracer,
                    reducers=reducers,
                    providers=getattr(ctx, "providers", None),
                    default_provider=getattr(ctx, "default_provider", None),
                    on_llm_payload=getattr(ctx, "on_llm_payload", None),
                )
                start = time.monotonic()
                if ctx.tracer is not None:
                    ctx.tracer.node_start(node_id, node.type)
                try:
                    result = await node.execute(node_ctx, state) or {}
                except Exception as exc:
                    if ctx.tracer is not None:
                        ctx.tracer.node_error(node_id, node.type, _ms(start), exc)
                    raise
                if ctx.tracer is not None:
                    ctx.tracer.node_end(node_id, node.type, _ms(start))
                apply_reducers(state, as_updates(result), reducers)
            if evaluate(condition, state):
                break

        return {}


__all__ = ["Loop"]
