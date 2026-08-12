"""The graph execution engine.

The heavy lifting of a :class:`~teff.graph.Graph` run lives here as a
standalone :func:`execute` function so the :class:`~teff.graph.Graph`
class stays a thin facade.  It walks the graph from the entry point,
executes each node, routes along edges whose conditions match the
current state, and (optionally) persists checkpoints so a crashed or
interrupted run can be resumed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from teff.checkpoint import DEFAULT_OWNER, Checkpoint, Checkpointer
from teff.errors import WorkflowError
from teff.graph.conditions import (
    find_error_edge,
    matched_condition,
    resolve_edge,
)
from teff.graph.edge import _ERROR_CONDITION, _INTERRUPT_KEY
from teff.logging import (
    get_logger,
    new_run_id,
    node_id_ctx,
    run_id_ctx,
)
from teff.node.command import Command
from teff.node.context import ExecContext
from teff.node.interrupt import GraphInterrupt
from teff.node.node import Node
from teff.node.registry import NodeRegistry, default_registry
from teff.provider import ProviderRegistry
from teff.state import Reducer, State, apply_reducers
from teff.stream import StreamEvent
from teff.tool.tool import Tool
from teff.trace import RunTracer, _ms

log = get_logger(__name__)

__all__ = ["execute"]


async def _call_hook(hooks: dict, name: str, *args: Any) -> None:
    fn = hooks.get(name)
    if fn is not None:
        res = fn(*args)
        if asyncio.iscoroutine(res):
            await res


def _restore_state(original: dict | State, data: dict) -> dict | State:
    """Rehydrate state from checkpoint data, preserving a State wrapper.

    If *original* is a :class:`State` instance its schema (and reducers)
    are kept and only the data is replaced. Plain dicts are returned as-is.
    """
    if isinstance(original, State):
        original.clear()
        original.update(data)
        return original
    return data


async def execute(
    graph,
    state: dict | State,
    tools: list[Tool] | None = None,
    registry: NodeRegistry | None = None,
    reducers: dict[str, Reducer] | None = None,
    hooks: dict[str, Callable] | None = None,
    node_timeout: float | None = None,
    max_iterations: int | None = None,
    checkpointer: Checkpointer | None = None,
    checkpoint_id: str | None = None,
    owner: str = DEFAULT_OWNER,
    resume: dict | None = None,
    tracer: RunTracer | None = None,
    state_schema: dict | None = None,
    emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
    providers: "dict | ProviderRegistry | None" = None,
    default_provider: str | None = None,
    default_model: str | None = None,
    on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
) -> dict | State:
    """Run *graph* with run/session correlation ids in the log context."""
    run = new_run_id()
    session = checkpoint_id or ""
    with run_id_ctx(run_id=run, session_id=session):
        log.info("run_start checkpoint=%s", session or "-")
        try:
            result = await _execute_impl(
                graph,
                state,
                tools=tools,
                registry=registry,
                reducers=reducers,
                hooks=hooks,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                owner=owner,
                resume=resume,
                tracer=tracer,
                state_schema=state_schema,
                emit=emit,
                providers=providers,
                default_provider=default_provider,
                default_model=default_model,
                on_llm_payload=on_llm_payload,
            )
        except GraphInterrupt:
            raise
        except Exception as exc:
            log.error("run_end status=error error=%r", str(exc))
            raise
        log.info("run_end status=ok")
        return result


async def _execute_impl(
    graph,
    state: dict | State,
    tools: list[Tool] | None = None,
    registry: NodeRegistry | None = None,
    reducers: dict[str, Reducer] | None = None,
    hooks: dict[str, Callable] | None = None,
    node_timeout: float | None = None,
    max_iterations: int | None = None,
    checkpointer: Checkpointer | None = None,
    checkpoint_id: str | None = None,
    owner: str = DEFAULT_OWNER,
    resume: dict | None = None,
    tracer: RunTracer | None = None,
    state_schema: dict | None = None,
    emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
    providers: "dict | ProviderRegistry | None" = None,
    default_provider: str | None = None,
    default_model: str | None = None,
    on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
) -> dict | State:
    """Shared execution core for :meth:`Graph.run` and :meth:`Graph.stream`.

    Walks *graph* from its entry point, shallow-merging each node's
    output into the state and routing along matching edges.  Returns the
    final state on success and raises on failure (``GraphInterrupt`` to
    pause, other exceptions for errors).  ``run_end`` bookkeeping is left
    to the caller.
    """
    if checkpointer is not None:
        if checkpoint_id is None:
            raise ValueError("checkpoint_id is required when checkpointer is set")
        cid: str = checkpoint_id
    else:
        cid = ""

    from teff.provider import to_provider_registry, validate_provider_refs

    effective_providers = to_provider_registry(providers)
    validate_provider_refs(effective_providers, default_provider, graph.nodes)
    providers = effective_providers

    if state_schema:
        from teff.state.state import validate_state

        state_dict = dict(state) if isinstance(state, State) else state
        errors = validate_state(state_dict, state_schema)
        if errors:
            from teff.errors import ConfigError

            raise ConfigError(
                "state does not match state_schema:\n"
                + "\n".join(f"  {e}" for e in errors)
            )

    registry = registry or default_registry
    tool_dict: dict[str, Tool] = {}
    if tools:
        for t in tools:
            tool_dict[t.name] = t

    hooks = hooks or {}

    if isinstance(state, State):
        reducers = reducers or state.reducers
    else:
        reducers = reducers or {}

    current_id: str | None = graph.entry_point
    iteration = 0

    if tracer is not None:
        tracer.run_start(checkpoint_id=cid or None)
    if emit is not None:
        await emit(StreamEvent("run_start", data={"checkpoint_id": cid or None}))

    pending: dict | None = None
    if checkpointer is not None:
        saved = await checkpointer.load(cid, owner=owner)
        log.debug(
            "checkpoint action=load checkpoint_id=%s next_node_id=%s",
            cid,
            saved.next_node_id if saved is not None else None,
        )
        if tracer is not None:
            tracer.checkpoint(
                "load",
                cid,
                saved.next_node_id if saved is not None else None,
            )
        if emit is not None:
            await emit(
                StreamEvent(
                    "checkpoint",
                    data={
                        "action": "load",
                        "checkpoint_id": cid,
                        "next_node_id": (
                            saved.next_node_id if saved is not None else None
                        ),
                    },
                )
            )
        if saved is not None:
            current_id = saved.next_node_id
            iteration = saved.iteration
            state = _restore_state(state, saved.state)
            pending = (
                state.pop(_INTERRUPT_KEY, None) if isinstance(state, dict) else None
            )

    if pending is not None:
        if not resume:
            raise GraphInterrupt(
                key=pending["key"],
                prompt=pending.get("prompt", ""),
                node_id=pending.get("node_id"),
                checkpoint_id=cid or None,
            )
        interrupt_node = pending.get("node_id")
        nested_checkpoint_id = pending.get("nested_checkpoint_id")
        if nested_checkpoint_id is None:
            for key, value in resume.items():
                state[key] = value
        if tracer is not None:
            tracer.interrupt_resume(interrupt_node, list(resume.keys()))
        if emit is not None:
            await emit(
                StreamEvent(
                    "interrupt_resume",
                    node_id=interrupt_node,
                    data={"keys": list(resume.keys())},
                )
            )
        if nested_checkpoint_id is not None:
            current_id = interrupt_node
        elif interrupt_node is not None:
            outgoing = [
                e
                for e in graph.edges
                if e.source_id == interrupt_node and e.condition != _ERROR_CONDITION
            ]
            current_id = resolve_edge(outgoing, state) if outgoing else None

    while current_id:
        if max_iterations is not None and iteration >= max_iterations:
            raise WorkflowError(f"graph exceeded max_iterations={max_iterations}")
        iteration += 1

        node: Node = graph.nodes[current_id]
        with node_id_ctx(node_id=current_id, node_type=node.type):
            log.info("node_start")
            ctx = ExecContext(
                state,
                tool_dict,
                node_id=current_id,
                node_type=node.type,
                tracer=tracer,
                reducers=reducers,
                emit=emit,
                providers=providers,
                default_provider=default_provider,
                default_model=default_model,
                hooks=hooks,
                node_timeout=node_timeout,
                checkpointer=checkpointer,
                checkpoint_id=cid,
                owner=owner,
                resume=resume if pending is not None else None,
                on_llm_payload=on_llm_payload,
            )
            start = time.monotonic()

            if checkpointer is not None:
                await checkpointer.save(
                    cid,
                    Checkpoint(
                        state=dict(state),
                        next_node_id=current_id,
                        iteration=iteration - 1,
                    ),
                    owner=owner,
                )
                if tracer is not None:
                    tracer.checkpoint("save", cid, current_id)
                log.debug("checkpoint action=save checkpoint_id=%s", cid)
                if emit is not None:
                    await emit(
                        StreamEvent(
                            "checkpoint",
                            data={
                                "action": "save",
                                "checkpoint_id": cid,
                                "next_node_id": current_id,
                            },
                        )
                    )

            await _call_hook(hooks, "on_node_start", current_id, node, state)
            if tracer is not None:
                tracer.node_start(current_id, node.type)
            if emit is not None:
                await emit(
                    StreamEvent("node_start", node_id=current_id, node_type=node.type)
                )

            try:
                if node_timeout is not None:
                    result = await asyncio.wait_for(
                        node.execute(ctx, state), timeout=node_timeout
                    )
                else:
                    result = await node.execute(ctx, state)
            except GraphInterrupt as exc:
                log.info("interrupt key=%s prompt=%r", exc.key, exc.prompt)
                if tracer is not None:
                    tracer.interrupt(current_id, exc.key, exc.prompt)
                if checkpointer is not None:
                    pending = dict(state)
                    pending[_INTERRUPT_KEY] = {
                        "key": exc.key,
                        "prompt": exc.prompt,
                        "node_id": current_id,
                        "nested_checkpoint_id": exc.nested_checkpoint_id,
                    }
                    await checkpointer.save(
                        cid,
                        Checkpoint(
                            state=pending,
                            next_node_id=None,
                            iteration=iteration,
                        ),
                        owner=owner,
                    )
                    if tracer is not None:
                        tracer.checkpoint("save", cid, None)
                if emit is not None:
                    await emit(
                        StreamEvent(
                            "interrupt",
                            node_id=current_id,
                            data={
                                "key": exc.key,
                                "prompt": exc.prompt,
                                "question": exc.prompt,
                            },
                        )
                    )
                    if checkpointer is not None:
                        await emit(
                            StreamEvent(
                                "checkpoint",
                                data={
                                    "action": "save",
                                    "checkpoint_id": cid,
                                    "next_node_id": None,
                                },
                            )
                        )
                exc.node_id = current_id
                if exc.checkpoint_id is None:
                    exc.checkpoint_id = cid or None
                raise
            except Exception as exc:
                log.error("node_error duration_ms=%s error=%r", _ms(start), str(exc))
                if tracer is not None:
                    tracer.node_error(current_id, node.type, _ms(start), exc)
                await _call_hook(hooks, "on_node_error", current_id, node, state, exc)
                if emit is not None:
                    await emit(
                        StreamEvent(
                            "node_error",
                            node_id=current_id,
                            node_type=node.type,
                            data={"error": str(exc), "duration_ms": _ms(start)},
                        )
                    )
                error_edge = find_error_edge(graph.edges, current_id)
                if error_edge is not None:
                    current_id = error_edge.target_id
                    if checkpointer is not None:
                        await checkpointer.save(
                            cid,
                            Checkpoint(
                                state=dict(state),
                                next_node_id=current_id,
                                iteration=iteration,
                            ),
                            owner=owner,
                        )
                        if tracer is not None:
                            tracer.checkpoint("save", cid, current_id)
                        if emit is not None:
                            await emit(
                                StreamEvent(
                                    "checkpoint",
                                    data={
                                        "action": "save",
                                        "checkpoint_id": cid,
                                        "next_node_id": current_id,
                                    },
                                )
                            )
                    continue
                raise

            log.info("node_end duration_ms=%s", _ms(start))
            updates: dict = result  # type: ignore[assignment]
            goto = None
            if isinstance(result, Command):
                updates = result.update
                goto = result.goto
            if updates:
                if isinstance(state, State):
                    state.merge(updates)
                else:
                    apply_reducers(state, updates, reducers or {})
            await _call_hook(hooks, "on_node_end", current_id, node, state, result)
            if tracer is not None:
                tracer.node_end(current_id, node.type, _ms(start))
            if emit is not None:
                await emit(
                    StreamEvent(
                        "node_end",
                        node_id=current_id,
                        node_type=node.type,
                        data={"duration_ms": _ms(start)},
                    )
                )

            if goto is Command.STOP:
                break

            outgoing = [
                e
                for e in graph.edges
                if e.source_id == current_id and e.condition != _ERROR_CONDITION
            ]

            if goto is not None and goto is not Command.STOP:
                if goto not in graph.nodes:
                    raise WorkflowError(
                        f"Command(goto={goto!r}) from node {current_id!r} targets an "
                        f"unknown node (known nodes: {sorted(graph.nodes)})"
                    )
                assert isinstance(goto, str)
                next_id: str | None = goto
                condition: "str | Callable[[dict], bool] | None" = None
            else:
                if not outgoing:
                    break
                next_id = resolve_edge(outgoing, state)
                if next_id is None:
                    break
                condition = matched_condition(outgoing, state, next_id)
            assert next_id is not None

            condition_label = (
                condition
                if isinstance(condition, str)
                else f"<{type(condition).__name__}>"
            )
            log.info("edge %s -> %s condition=%s", current_id, next_id, condition_label)
            if tracer is not None:
                tracer.edge(current_id, next_id, condition_label)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "edge",
                        node_id=current_id,
                        data={"target_id": next_id, "condition": condition_label},
                    )
                )
            current_id = next_id

    if checkpointer is not None:
        await checkpointer.save(
            cid,
            Checkpoint(state=dict(state), next_node_id=None, iteration=iteration),
            owner=owner,
        )
        if tracer is not None:
            tracer.checkpoint("save", cid, None)
        if emit is not None:
            await emit(
                StreamEvent(
                    "checkpoint",
                    data={
                        "action": "save",
                        "checkpoint_id": cid,
                        "next_node_id": None,
                    },
                )
            )

    return state
