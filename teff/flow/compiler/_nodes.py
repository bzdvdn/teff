"""Handlers for single-node idioms (``llm:``, ``transform:``, ``agent:``)."""

from __future__ import annotations

import typing

from teff.errors import ConfigError
from teff.flow.compiler._common import (
    _agent_subflow,
    _pop_id,
    _resolve_model,
    _resolve_provider,
)

if typing.TYPE_CHECKING:
    from teff.flow.flow import Flow


def _h_llm(flow: "Flow", payload: dict) -> None:
    pid = _pop_id(payload)
    config = dict(payload)
    config.setdefault("model", _resolve_model(flow, config))
    config.setdefault("provider", _resolve_provider(flow, config))
    flow.llm(**config, id=pid)


def _h_transform(flow: "Flow", payload: dict) -> None:
    pid = _pop_id(payload)
    flow.transform(**payload, id=pid)


def _h_agent(flow: "Flow", payload: dict) -> None:
    pid = _pop_id(payload)
    sub = _agent_subflow(
        flow,
        payload,
        system=payload.get("system") or "",
        output_key=payload.get("output_key") or pid or "output",
    )
    flow.step(sub, id=f"agent-{pid}" if pid else None)


def _h_context_builder(flow: "Flow", payload: dict) -> None:
    """Compile a context-composition step::

    - context_builder: {id: compose, sections: {plan: Plan},
                        messages_key: messages, output_key: input}
    """
    pid = _pop_id(payload)
    flow.context_builder(**payload, id=pid)


def _h_append_assistant(flow: "Flow", payload: dict) -> None:
    """Compile an append-reply step::

    - append_assistant: {output_key: draft, messages_key: messages}
    """
    pid = _pop_id(payload)
    flow.append_assistant(**payload, id=pid)


def _h_type(flow: "Flow", payload: dict) -> None:
    """Compile an arbitrary registered node type as one step::

    - type: {type: csv, config: {...}}
    """
    from teff.node.registry import default_registry

    if not isinstance(payload, dict):
        raise ConfigError("`type:` step requires a mapping")
    stype = payload.get("type")
    if not stype:
        raise ConfigError("`type:` step requires a `type:` key")
    pid = _pop_id(payload)
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        raise ConfigError(f"`type:` step config must be a mapping, got {config!r}")
    node = default_registry.create(stype, config)
    flow.step(node, id=pid)
