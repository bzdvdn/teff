"""Handlers for flow-control idioms (parallel/map/loop/interrupt/branch/route)."""

from __future__ import annotations

import typing

from teff.errors import ConfigError
from teff.flow.compiler._common import (
    _as_node,
    _map_processor,
    _nodes_from_steps,
    _pop_id,
)

if typing.TYPE_CHECKING:
    from teff.flow.flow import Flow
    from teff.node.node import Node


def _h_parallel(flow: "Flow", payload: dict) -> None:
    """Compile ``parallel: {branches, converge}`` into a Parallel + join."""
    pid = _pop_id(payload) or "parallel"
    branches = payload.get("branches")
    if branches is None:
        raise ConfigError("parallel requires a `branches:` list")
    if not isinstance(branches, list) or not branches:
        raise ConfigError("parallel `branches:` must be a non-empty list")
    branch_nodes: list[list["Node"] | "Node"] = []
    for branch in branches:
        if isinstance(branch, list):
            branch_nodes.append([_as_node(flow, b) for b in branch])
        else:
            branch_nodes.append(_as_node(flow, branch))
    flow.parallel(*branch_nodes, id=pid)

    converge = payload.get("converge")
    if converge is not None:
        from teff.node.transform import Transform

        if not isinstance(converge, dict) or len(converge) != 1:
            raise ConfigError("parallel `converge:` must be a single-key step")
        node = _as_node(flow, converge)
        if isinstance(node, Transform):
            flow.converge(node)
        else:
            raise ConfigError(
                "parallel `converge:` only supports a Transform node in this release"
            )


def _h_map(flow: "Flow", payload: dict) -> None:
    pid = _pop_id(payload)
    processor, cfg = _map_processor(flow, payload)
    flow.map(
        processor,
        input_keys=cfg.get("input_keys") or "",
        output_key=cfg.get("output_key") or "",
        max_concurrency=cfg.get("max_concurrency"),
        id=pid,
    )


def _h_loop(flow: "Flow", payload: dict) -> None:
    key = payload.get("key")
    until = payload.get("until")
    if not key or until is None:
        raise ConfigError("loop requires both `key:` and `until:`")
    body = payload.get("body")
    done = payload.get("done")
    if body is None:
        raise ConfigError("loop requires a `body:` list")
    if done is None:
        raise ConfigError("loop requires a `done:` step or list")
    label = payload.get("label")
    max_rounds = payload.get("max_rounds")
    decider_id = flow._last_added
    body_nodes = _nodes_from_steps(flow, body)
    done_nodes = _nodes_from_steps(flow, done)
    if max_rounds is not None:
        max_rounds = int(max_rounds)
        if max_rounds < 1:
            raise ConfigError("loop max_rounds must be a positive integer")
    flow.loop(
        key=key,
        until=until,
        done=done_nodes,
        body=body_nodes,
        max_rounds=max_rounds,
    )
    if isinstance(label, str) and label and decider_id:
        flow._loop_labels[label] = decider_id


def _h_interrupt(flow: "Flow", payload: dict) -> None:
    pid = _pop_id(payload)
    key = payload.get("key")
    if not key:
        raise ConfigError("interrupt requires a `key`")
    strategy = payload.get("strategy")
    if strategy is None:
        flow.interrupt(key=key, prompt=payload.get("prompt") or "", id=pid)
        return
    from teff.node.ask import Ask

    try:
        accept = Ask.from_mapping(strategy)
    except ValueError as exc:
        raise ConfigError(f"interrupt strategy error: {exc}") from exc
    flow.interrupt(
        key=key,
        prompt=payload.get("prompt") or "",
        accept=accept,
        id=pid,
    )


def _h_branch(flow: "Flow", payload: dict) -> None:
    """Compile ``branch: {key, cases, default?, converge?}`` into conditional
    edges plus an optional merge node::

        - branch:
            key: sentiment
            cases:
              - {value: positive, steps: [transform: {...}]}
              - {value: negative, steps: [transform: {...}]}
            default:
              - transform: {action: value, value: neutral, output_key: reply}
            converge:
              transform: {action: uppercase, input_key: reply, output_key: result}
    """
    from teff.flow.case import Case

    _pop_id(payload)
    key = payload.get("key")
    if not key:
        raise ConfigError("branch requires a `key`")
    cases_data = payload.get("cases")
    if not isinstance(cases_data, list) or not cases_data:
        raise ConfigError("branch `cases:` must be a non-empty list")
    cases: list[Case] = []
    for case_spec in cases_data:
        if not isinstance(case_spec, dict):
            raise ConfigError(f"branch case must be a mapping, got {case_spec!r}")
        value = case_spec.get("value")
        if value is None:
            raise ConfigError("branch case requires a `value`")
        steps = case_spec.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ConfigError(f"branch case {value!r} requires a `steps:` list")
        case = Case(str(value))
        for node in _nodes_from_steps(flow, steps):
            case.add(node)
        cases.append(case)
    default_node = None
    default_spec = payload.get("default")
    if default_spec is not None:
        default_node = _branch_default_node(flow, default_spec)
    flow.branch(key, *cases, default=default_node)
    converge_spec = payload.get("converge")
    if converge_spec is not None:
        if not isinstance(converge_spec, dict) or len(converge_spec) != 1:
            raise ConfigError("branch `converge:` must be a single-key node step")
        flow.converge(_as_node(flow, converge_spec))


def _branch_default_node(flow: "Flow", spec: typing.Any) -> "Node":
    """Compile a branch ``default:`` — a single node or a chained list."""
    nodes = _nodes_from_steps(flow, spec)
    if not nodes:
        raise ConfigError("branch `default:` cannot be empty")
    if len(nodes) == 1:
        return nodes[0]
    from teff.flow.flow import Flow as _Flow
    from teff.flow.sub_flow import SubFlow

    inner = _Flow("branch-default")
    for node in nodes:
        inner.step(node)
    return SubFlow(inner.compile())


def _h_route(flow: "Flow", payload: dict) -> None:
    from teff.node.command_node import CommandNode

    pid = _pop_id(payload)
    config = dict(payload)
    config["routes"] = [_resolve_route(flow, r) for r in (config.get("routes") or [])]
    if isinstance(config.get("goto"), str):
        config["goto"] = flow.label_target(config["goto"])
    flow.step(CommandNode(config=config), id=pid)


def _resolve_route(flow: "Flow", route: dict) -> dict:
    """Rewrite a route's ``goto`` so a ``label`` name becomes a real node id."""
    out = dict(route)
    if isinstance(out.get("goto"), str):
        out["goto"] = flow.label_target(out["goto"])
    return out
