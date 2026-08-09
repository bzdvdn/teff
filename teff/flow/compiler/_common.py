"""Shared node-level builders used across the flow.yaml idiom handlers."""

from __future__ import annotations

import typing

from teff.errors import ConfigError
from teff.node.registry import default_registry

if typing.TYPE_CHECKING:
    from teff.flow.agent import SubFlow
    from teff.flow.flow import Flow
    from teff.node.node import Node

#: Idioms recognised as node-level steps (map processor, loop body/done,
#: parallel branch, branch/converge).
_NODE_DEFAULTS = {
    "llm",
    "transform",
    "agent",
    "agent_step",
    "interrupt",
    "map",
    "type",
}


def _pop_id(payload: dict) -> str | None:
    _id = payload.get("id")
    if _id is not None:
        del payload["id"]
    return _id


def _unused(value: typing.Any) -> None:
    pass


def _resolve_model(flow: "Flow", payload: dict) -> str:
    model = payload.get("model") or flow._default_model
    if not model:
        raise ConfigError(
            "`model:` required here — set `default_model:` at the top of "
            "flow.yaml or pass a per-step `model:`"
        )
    return model


def _resolve_provider(flow: "Flow", payload: dict) -> str:
    provider = payload.get("provider") or flow._default_provider
    if not provider:
        raise ConfigError(
            "`provider:` required here — set `default_provider:` at the top "
            "of flow.yaml or pass a per-step `provider:`"
        )
    return provider


def _as_node(flow: "Flow", step: typing.Any) -> "Node":
    """Compile a *node-level* step (loop body/done, parallel branch)."""
    if not isinstance(step, dict) or len(step) != 1:
        raise ConfigError(
            f"expected a single-key node step (e.g. `llm: {{...}}`), got {step!r}"
        )
    idiom, payload = next(iter(step.items()))
    if idiom == "llm":
        from teff.node.llm import LLM

        _pop_id(payload)
        config = dict(payload)
        config.setdefault("model", _resolve_model(flow, config))
        config.setdefault("provider", _resolve_provider(flow, config))
        return LLM(**config)
    if idiom == "transform":
        from teff.node.transform import Transform

        return Transform(**payload)
    if idiom in ("agent", "agent_step"):
        return _agent_subflow(
            flow,
            payload,
            system=payload.get("system") or "",
            output_key=payload.get("output_key") or payload.get("id") or "output",
        )
    if idiom == "interrupt":
        from teff.node.interrupt import Interrupt

        sid = _pop_id(payload)
        _unused(sid)
        key = payload.get("key")
        if not key:
            raise ConfigError("interrupt step requires a `key`")
        return Interrupt(key=key, prompt=payload.get("prompt") or "")
    if idiom == "map":
        from teff.node.map import Map

        processor, cfg = _map_processor(flow, payload)
        return Map(
            processor,
            input_keys=cfg.get("input_keys") or "",
            output_key=cfg.get("output_key") or "",
            **{k: v for k, v in cfg.items() if k != "input_keys" and k != "output_key"},
        )
    if idiom == "type":
        stype = payload.get("type")
        if not stype:
            raise ConfigError("`type:` node step requires a `type:` key")
        return default_registry.create(stype, payload.get("config") or {})
    raise ConfigError(
        f"unknown node step {idiom!r} — supported: {', '.join(sorted(_NODE_DEFAULTS))}"
    )


def _agent_subflow(
    flow: "Flow",
    payload: dict,
    *,
    system: str,
    output_key: str,
) -> "SubFlow":
    """Build an ``agent_step`` SubFlow, threading default model/provider."""
    from teff.flow.agent import agent_step

    allowed = ("tools", "sections", "messages_key", "use_tools", "stream")
    cfg = {k: v for k, v in payload.items() if k in allowed}
    return agent_step(
        system,
        output_key,
        model=_resolve_model(flow, payload),
        provider=_resolve_provider(flow, payload),
        id=str(payload.get("id") or ""),
        **cfg,
    )


def _map_processor(flow: "Flow", payload: dict) -> tuple[dict, dict]:
    """Split a ``map:`` step into its processor spec and pass-through config."""
    proc = payload.get("processor")
    if proc is None:
        raise ConfigError("map requires a `processor:`")
    if not isinstance(proc, dict) or len(proc) != 1:
        raise ConfigError(f"map processor must be a single-key node step, got {proc!r}")
    node = _as_node(flow, _unwrap_map_processor(flow, proc))
    spec = {"type": node.type, **node.config}
    cfg = {k: v for k, v in payload.items() if k not in ("processor", "id")}
    return spec, cfg


def _unwrap_map_processor(flow: "Flow", spec: dict) -> dict:
    """Validate a map processor spec is a supported node shorthand."""
    idiom = next(iter(spec))
    if idiom not in _NODE_DEFAULTS:
        raise ConfigError(f"unsupported map processor {idiom!r}")
    return spec


def _nodes_from_steps(flow_to_use: "Flow", steps: typing.Any) -> list["Node"]:
    """Compile a list (or single) node-level step into a node list."""
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list):
        raise ConfigError(f"expected a list of steps, got {type(steps).__name__}")
    out: list["Node"] = []
    for step in steps:
        out.extend(_nodes_from_step(flow_to_use, step))
    return out


def _nodes_from_step(flow_to_use: "Flow", step: typing.Any) -> list["Node"]:
    """Compile one node-level step, expanding strategy interrupts.

    An ``interrupt`` carrying a ``strategy:`` expands to the same chain
    ``Flow.interrupt`` builds (Interrupt → optional classifier → Validate)
    so the decision key is produced in nested loop/done contexts, not just
    at the top level.
    """
    if not isinstance(step, dict) or len(step) != 1:
        raise ConfigError(
            f"expected a single-key node step (e.g. `llm: {{...}}`), got {step!r}"
        )
    idiom, payload = next(iter(step.items()))
    if idiom == "interrupt" and payload.get("strategy"):
        from teff.node.ask import Ask
        from teff.node.interrupt import Interrupt

        _pop_id(payload)
        key = payload.get("key")
        if not key:
            raise ConfigError("interrupt step requires a `key`")
        try:
            accept = Ask.from_mapping(payload["strategy"])
        except ValueError as exc:
            raise ConfigError(f"interrupt strategy error: {exc}") from exc
        nodes: list["Node"] = [Interrupt(key=key, prompt=payload.get("prompt") or "")]
        input_key = key
        if accept.needs_classifier():
            if not accept.model_name or not accept.provider:
                raise ConfigError(
                    "interrupt with a 'llm' strategy requires model and provider"
                )
            nodes.append(accept.classifier())
            input_key = accept.verdict_key
        nodes.append(accept.validate_node(input_key=input_key))
        return nodes
    return [_as_node(flow_to_use, step)]
