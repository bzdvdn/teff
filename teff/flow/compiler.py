"""Declarative ``flow.yaml`` compiler — the authoring layer.

A ``flow.yaml`` is a small, high-level document that describes *how* the
app should behave (teams, chains, gates) without spelling out every node
and edge.  This module compiles it into a :class:`teff.flow.Flow`, which
in turn produces a regular ``Graph`` and can be exported as the compiled
``graph.yaml`` artifact::

    flow.yaml ──compile()--> Flow ──compile()--> Graph
                               └── to_yaml() ──► graph.yaml

Invariant: the compiler only ever produces things a hand-written
``graph.yaml`` (or ``Flow``) could express — every idiom below lowers onto
the existing node types and the existing ``Flow`` methods.  There is no
second runtime.

Document grammar (0.2 MVP — see ``docs/design/two-layer.md``)::

    name: my-app
    description: ...
    default_provider: ollama
    default_model: llama3.1:8b
    providers: [...]                    # pass-through (graph providers)
    tools: [...]                        # pass-through (tool registry)
    state: {schema: ..., initial: ...}  # pass-through (graph state)

    steps:
      - llm: {id: replier, model: gpt-4, system: ..., output_key: answer}
      - transform: {id: shout, action: uppercase, input_key: answer,
                    output_key: shout}
      - agent_step: {id: coder, system: "You are...", output_key: code,
                     model: ..., tools: [...]}
      - team:
          id: lead
          leader:
            system: "You are the team lead... route to coder or finish."
            model: llama3.1:8b
          roles:
            coder:   {system: "...", output_key: code}
            planner: {system: "...", output_key: plan}
          fallback: planner
          max_rounds: 6
      - parallel:
          branches:
            - llm: {...}
            - agent: {...}
          converge: {transform: {...}}
      - map:
          input_keys: [items]
          output_key: results
          processor: {llm: {model: ..., system: "..."}}
      - loop:
          key: verdict
          until: pass
          body:
            - llm: {...}
          done:
            - llm: {...}
      - interrupt:
          id: approve
          key: decision
          prompt: "Send to work?"
          strategy: {any_of: [approve, ok], decision_key: decision,
                     pass_value: approve, fail_value: rework}
      - route:
          key: decision
          routes:
            - {when: "decision=approve", goto: final}
            - {when: "decision=rework", goto: refine}
          goto: STOP

Unknown steps raise :class:`teff.errors.ConfigError`.
"""

from __future__ import annotations

import os
import typing

from teff.errors import ConfigError
from teff.flow.flow import Flow
from teff.node.registry import default_registry
from teff.yaml import _interpolate_env, _providers_from_data, _safe_load

if typing.TYPE_CHECKING:
    from teff.flow.agent import SubFlow
    from teff.graph import Graph
    from teff.node.node import Node

#: Idioms recognised at the top level (``steps:`` entries).
_NODE_DEFAULTS = {"llm", "transform", "agent", "interrupt", "map", "type"}


def load_flow_yaml(path: str) -> dict:
    """Parse a ``flow.yaml`` file, interpolating ``${ENV}`` references.

    Returns:
        The document mapping.  Raises :class:`ConfigError` on a parse
        error or a non-mapping document.
    """
    if not os.path.exists(path):
        raise ConfigError(f"flow file not found: {path}")
    with open(path) as f:
        data = _safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: flow must be a mapping")
    return _interpolate_env(data)


def flow_from_yaml(data: dict) -> Flow:
    """Compile a parsed ``flow.yaml`` mapping into a :class:`Flow`.

    Args:
        data: The document mapping returned by :func:`load_flow_yaml`.

    Returns:
        A :class:`teff.flow.Flow` whose :meth:`~Flow.compile` renders a
        complete graph (and :meth:`~Flow.to_yaml` the ``graph.yaml``).
    """
    providers = _providers_from_data(data)
    flow = Flow(
        name=str(data.get("name") or ""),
        providers=providers,
        default_provider=data.get("default_provider"),
        default_model=data.get("default_model"),
    )
    for step in data.get("steps", []) or []:
        _compile_step(flow, step)
    if not flow._nodes:
        raise ConfigError("flow: `steps:` is empty — nothing to compile")
    return flow


def _compile_step(flow: Flow, step: typing.Any) -> None:
    """Compile one high-level idiom into ``flow``."""
    if not isinstance(step, dict) or len(step) != 1:
        raise ConfigError(
            f"each `steps` entry must be a single-key mapping (an idiom), got {step!r}"
        )
    idiom, payload = next(iter(step.items()))
    handler = _HANDLERS.get(idiom)
    if handler is None:
        raise ConfigError(
            f"unknown flow step {idiom!r} — supported: {', '.join(sorted(_HANDLERS))}"
        )
    handler(flow, payload)


# --------------------------------------------------------------------------
# Node-level builders (used inside map processors and parallel/converge)
# --------------------------------------------------------------------------


def _pop_id(payload: dict) -> str | None:
    _id = payload.get("id")
    if _id is not None:
        del payload["id"]
    return _id


def _resolve_model(flow: Flow, payload: dict) -> str:
    model = payload.get("model") or flow._default_model
    if not model:
        raise ConfigError(
            "`model:` required here — set `default_model:` at the top of "
            "flow.yaml or pass a per-step `model:`"
        )
    return model


def _resolve_provider(flow: Flow, payload: dict) -> str:
    provider = payload.get("provider") or flow._default_provider
    if not provider:
        raise ConfigError(
            "`provider:` required here — set `default_provider:` at the top "
            "of flow.yaml or pass a per-step `provider:`"
        )
    return provider


def _as_node(flow: Flow, step: typing.Any) -> "Node":
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
    if idiom == "agent":
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


def _unused(value: typing.Any) -> None:
    pass


def _agent_subflow(
    flow: Flow,
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


def _map_processor(flow: Flow, payload: dict) -> tuple[dict, dict]:
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


def _unwrap_map_processor(flow: Flow, spec: dict) -> dict:
    """Validate a map processor spec is a supported node shorthand."""
    idiom = next(iter(spec))
    if idiom not in _NODE_DEFAULTS:
        raise ConfigError(f"unsupported map processor {idiom!r}")
    return spec


# --------------------------------------------------------------------------
# Top-level idiom handlers
# --------------------------------------------------------------------------


def _h_llm(flow: Flow, payload: dict) -> None:
    pid = _pop_id(payload)
    config = dict(payload)
    config.setdefault("model", _resolve_model(flow, config))
    config.setdefault("provider", _resolve_provider(flow, config))
    flow.llm(**config, id=pid)


def _h_transform(flow: Flow, payload: dict) -> None:
    pid = _pop_id(payload)
    flow.transform(**payload, id=pid)


def _h_agent(flow: Flow, payload: dict) -> None:
    pid = _pop_id(payload)
    sub = _agent_subflow(
        flow,
        payload,
        system=payload.get("system") or "",
        output_key=payload.get("output_key") or pid or "output",
    )
    flow.step(sub, id=f"agent-{pid}" if pid else None)


def _h_parallel(flow: Flow, payload: dict) -> None:
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


def _h_team(flow: Flow, payload: dict) -> None:
    """Compile a supervisor team: leader decider + routed agent roles."""
    pid = _pop_id(payload) or "supervisor"
    leader = payload.get("leader") or payload.get("lead") or {}
    if not isinstance(leader, dict):
        raise ConfigError("team requires a `leader:` mapping")
    roles = payload.get("roles") or {}
    if not isinstance(roles, dict) or not roles:
        raise ConfigError("team requires a non-empty `roles:` mapping")

    role_keys = list(roles)
    system = leader.get("system") or ""
    model = leader.get("model") or _resolve_model(flow, leader)
    provider = leader.get("provider") or _resolve_provider(flow, leader)

    from teff.node.supervisor import Supervisor

    route_keys: "dict[str, str]" = {}
    declared_rk = leader.get("route_keys")
    if declared_rk is None:
        route_keys = {r: roles[r].get("output_key") or r for r in role_keys}
    else:
        route_keys = declared_rk
    supervisor = Supervisor(
        system=system,
        model=model,
        provider=provider,
        messages_key=leader.get("messages_key") or "messages",
        route_keys=route_keys,
        done_keys=set(leader.get("done_keys") or ()),
        done_mode=leader.get("done_mode") or "all",
        fallback_agent=str(leader.get("fallback") or payload.get("fallback") or ""),
        max_rounds=leader.get("max_rounds") or payload.get("max_rounds") or 6,
    )
    flow.supervisor(supervisor, id=pid)

    agents: dict[str, "SubFlow"] = {}
    for role_name, spec in roles.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"team role {role_name!r} must be a mapping")
        role_payload = dict(spec)
        role_payload.setdefault("model", model)
        role_payload.setdefault("provider", provider)
        agents[role_name] = _agent_subflow(
            flow,
            role_payload,
            system=spec.get("system") or "",
            output_key=spec.get("output_key") or role_name,
        )

    finish = payload.get("finish")
    finish_nodes = None
    if finish is not None:
        finish_nodes = _nodes_from_steps(flow, finish)
    flow.route("next_agent", finish=finish_nodes, **agents)


def _h_map(flow: Flow, payload: dict) -> None:
    pid = _pop_id(payload)
    processor, cfg = _map_processor(flow, payload)
    flow.map(
        processor,
        input_keys=cfg.get("input_keys") or "",
        output_key=cfg.get("output_key") or "",
        max_concurrency=cfg.get("max_concurrency"),
        id=pid,
    )


def _h_loop(flow: Flow, payload: dict) -> None:
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


def _h_interrupt(flow: Flow, payload: dict) -> None:
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


def _h_route(flow: Flow, payload: dict) -> None:
    from teff.node.command_node import CommandNode

    pid = _pop_id(payload)
    config = dict(payload)
    config["routes"] = [_resolve_route(flow, r) for r in (config.get("routes") or [])]
    if isinstance(config.get("goto"), str):
        config["goto"] = flow.label_target(config["goto"])
    flow.step(CommandNode(config=config), id=pid)


def _resolve_route(flow: Flow, route: dict) -> dict:
    """Rewrite a route's ``goto`` so a ``label`` name becomes a real node id."""
    out = dict(route)
    if isinstance(out.get("goto"), str):
        out["goto"] = flow.label_target(out["goto"])
    return out


def _nodes_from_steps(flow_to_use: Flow, steps: typing.Any) -> list["Node"]:
    """Compile a list (or single) node-level step into a node list."""
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list):
        raise ConfigError(f"expected a list of steps, got {type(steps).__name__}")
    out: list["Node"] = []
    for step in steps:
        out.extend(_nodes_from_step(flow_to_use, step))
    return out


def _nodes_from_step(flow_to_use: Flow, step: typing.Any) -> list["Node"]:
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


_HANDLERS = {
    "llm": _h_llm,
    "transform": _h_transform,
    "agent": _h_agent,
    "agent_step": _h_agent,
    "team": _h_team,
    "parallel": _h_parallel,
    "map": _h_map,
    "loop": _h_loop,
    "interrupt": _h_interrupt,
    "route": _h_route,
}


def _build_state(
    data: dict,
) -> tuple[list, dict, dict]:
    """Mirror ``load_workflow``'s tools / initial-state / reducers extraction."""
    from teff.state.state import (
        reducers_from_yaml_schema,
        validate_state,
    )
    from teff.tool.registry import default_tool_registry

    tools: list = []
    for td in data.get("tools", []) or []:
        ttype = td["type"]
        tconfig = td.get("config", {})
        tools.append(default_tool_registry.create(ttype, tconfig))

    state_block = data.get("state", {})
    if isinstance(state_block, dict):
        schema = state_block.get("schema", {})
        initial = state_block.get("initial", {})
    else:
        schema = {}
        initial = {}
    if not isinstance(initial, dict):
        raise ConfigError("state.initial must be a mapping")
    if schema:
        errors = validate_state(initial, schema)
        if errors:
            raise ConfigError(
                "state.initial does not match state.schema:\n"
                + "\n".join(f"  {e}" for e in errors)
            )
    reducers = reducers_from_yaml_schema(schema)
    return tools, initial, reducers


def compile_flow_file(path: str) -> "Graph":
    """Compile a ``flow.yaml`` file into a plain Graph."""
    doc = load_flow_yaml(path)
    return flow_from_yaml(doc).compile()


def load_flow(path: str):
    """Load a ``flow.yaml`` as a ``(graph, tools, initial, reducers)`` tuple.

    Mirrors :func:`teff.yaml.load_workflow` so callers (and the CLI) can
    treat ``flow.yaml`` and ``graph.yaml`` interchangeably::

        graph, tools, state, reducers = load_flow("app/flow.yaml")

    The compiled ``graph`` reflects all idioms in the authoring layer; the
    optional ``tools:`` / ``state:`` blocks are passed through unchanged.
    """
    data = load_flow_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    from teff.plugins import load_plugins_from_document

    load_plugins_from_document(data, base_dir)
    graph = flow_from_yaml(data).compile()
    tools, initial, reducers = _build_state(data)
    return graph, tools, initial, reducers


def looks_like_flow(data: dict) -> bool:
    """Return ``True`` when *data* is authored in the flow.yaml idiom surface.

    A low-level graph documents ``steps`` with explicit ``id``/``type``
    keys; the authoring layer uses the shorthand idiom keys instead
    (``llm:``, ``team:``, ``map:``, …).
    """
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        for key in step:
            if key in _HANDLERS:
                return True
    return False


def build_flow_to_yaml(path: str, output: str | None = None) -> str:
    """Compile ``flow.yaml`` at *path* to a ``graph.yaml`` document string.

    Returns the compiled YAML text; when *output* is given the text is
    also written to that path.
    """
    from ..yaml import workflow_to_yaml

    data = load_flow_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    from teff.plugins import load_plugins_from_document

    load_plugins_from_document(data, base_dir)
    flow = flow_from_yaml(data)
    text = workflow_to_yaml(flow.compile(), name=data.get("name") or "graph")
    if output:
        with open(output, "w") as f:
            f.write(text)
    return text
