"""Handlers for the supervised-team idioms (``team:``, ``supervisor:``)."""

from __future__ import annotations

import typing

from teff.errors import ConfigError
from teff.flow.compiler._common import (
    _nodes_from_steps,
    _pop_id,
    _resolve_model,
    _resolve_provider,
)

if typing.TYPE_CHECKING:
    from teff.flow.flow import Flow


def _h_team(flow: "Flow", payload: dict) -> None:
    """Compile a supervisor team: leader decider + routed agent roles."""
    pid = _pop_id(payload) or "supervisor"
    leader = payload.get("leader") or payload.get("lead") or {}
    if not isinstance(leader, dict):
        raise ConfigError("team requires a `leader:` mapping")
    roles = payload.get("roles") or {}
    if not isinstance(roles, dict) or not roles:
        raise ConfigError("team requires a non-empty `roles:` mapping")

    model = leader.get("model") or _resolve_model(flow, leader)
    provider = leader.get("provider") or _resolve_provider(flow, leader)

    finish = payload.get("finish")
    finish_nodes = None
    if finish is not None:
        finish_nodes = _nodes_from_steps(flow, finish)
    flow.team(
        leader.get("system") or "",
        roles=roles,
        model=model,
        provider=provider,
        messages_key=leader.get("messages_key") or "messages",
        route_keys=leader.get("route_keys"),
        done_keys=list(leader.get("done_keys") or ()),
        done_mode=leader.get("done_mode") or "all",
        fallback=str(leader.get("fallback") or payload.get("fallback") or ""),
        max_rounds=int(leader.get("max_rounds") or payload.get("max_rounds") or 6),
        finish=finish_nodes,
        id=pid,
    )


def _h_supervisor(flow: "Flow", payload: dict) -> None:
    """Compile a native ``supervisor:`` decider — no team wrapper.

    Adds a single :class:`~teff.node.supervisor.Supervisor` that writes the
    next route to ``output_key`` (``next_agent`` by default).  An optional
    ``agents:`` mapping (plus ``finish:`` and ``key:``) wires the whole
    supervisor loop in one step — the ``team:`` shape without the team
    sugar::

        - supervisor:
            id: lead
            system: "route to coder or finish"
            route_keys: {coder: code}
            done_keys: [code]
            agents:
              coder: [agent_step: {id: coder, system: "You code.",
                                    output_key: code}]
            finish:
              - transform: {id: done, action: now, output_key: delivered_at}

    Without ``agents:`` the raw decider node is added alone, for pairing
    with a separate ``supervise:`` step that routes an existing decider to
    agent chains (full control over ``fill_order``, ``sections``, etc.).
    """
    from teff.node.supervisor import Supervisor

    pid = _pop_id(payload)
    config = dict(payload)
    config.setdefault("model", _resolve_model(flow, config))
    config.setdefault("provider", _resolve_provider(flow, config))
    if "fallback" in config and "fallback_agent" not in config:
        config["fallback_agent"] = config.pop("fallback")
    agents = config.pop("agents", None)
    finish = config.pop("finish", None)
    key = config.pop("key", None)
    flow.supervisor(Supervisor(**config), id=pid)
    if agents is not None:
        if not isinstance(agents, dict) or not agents:
            raise ConfigError("supervisor with `agents:` requires a non-empty mapping")
        if not key:
            key = config.get("output_key", "next_agent")
        finish_nodes = None
        if finish is not None:
            finish_nodes = _nodes_from_steps(flow, finish)
        chains = {name: _nodes_from_steps(flow, spec) for name, spec in agents.items()}
        flow.route(key, finish=finish_nodes, **chains)


def _h_supervise(flow: "Flow", payload: dict) -> None:
    """Compile a supervisor loop from the preceding decider.

    The low-level twin of the ``route`` wiring ``Flow.team`` does implicitly,
    for the advanced case where the decider and its agent groups are defined
    separately: the last added node (a ``supervisor:`` decider) writes *key*;
    each entry in *agents* maps a value of *key* to the chain run for it,
    after which control returns to the decider.  When *key* equals ``finish``
    the loop exits through *finish* (or terminates when omitted).  Prefer
    the single-step ``supervisor:`` with its own ``agents:``/``finish:`` for
    the common "one decider, one team" shape::

        - supervisor: {id: lead, system: "route to coder or finish",
                       route_keys: {coder: code}}
        - supervise:
            key: next_agent
            agents:
              coder: [agent_step: {id: coder, system: "You code.",
                                    output_key: code}]
            finish:
              - transform: {id: done, action: now, output_key: delivered_at}
    """
    key = payload.get("key")
    if not key:
        raise ConfigError("supervise requires a `key:`")
    agents = payload.get("agents") or {}
    if not isinstance(agents, dict) or not agents:
        raise ConfigError("supervise requires a non-empty `agents:` mapping")
    finish = payload.get("finish")
    finish_nodes = None
    if finish is not None:
        finish_nodes = _nodes_from_steps(flow, finish)
    chains = {name: _nodes_from_steps(flow, spec) for name, spec in agents.items()}
    flow.route(key, finish=finish_nodes, **chains)
