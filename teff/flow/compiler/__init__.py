"""Declarative ``flow.yaml`` compiler — the authoring layer.

A ``flow.yaml`` is a small, high-level document that describes *how* the
app should behave (teams, chains, gates) without spelling out every node
and edge.  This package compiles it into a :class:`teff.flow.Flow`, which
in turn produces a regular ``Graph`` and can be exported as the compiled
``graph.yaml`` artifact::

    flow.yaml ──compile()--> Flow ──compile()--> Graph
                               └── to_yaml() ──► graph.yaml

Invariant: the compiler only ever produces things a hand-written
``graph.yaml`` (or ``Flow``) could express — every idiom below lowers onto
the existing node types and the existing ``Flow`` methods.  There is no
second runtime.

The implementation is split by concern:

* :mod:`teff.flow.compiler._common` — shared node-level builders;
* :mod:`teff.flow.compiler._nodes` — single-node idioms (``llm:`` etc.);
* :mod:`teff.flow.compiler._flow` — flow-control idioms (``parallel:``…);
* :mod:`teff.flow.compiler._team` — supervised-team idioms (``team:``…);
* :mod:`teff.flow.compiler._state` — tools/state extraction.

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
      - supervisor:                          # native decider — no team wrapper
          id: lead
          system: "You are the team lead... route to coder or finish."
          route_keys: {coder: code, planner: plan}
          done_keys: [code, plan]
          fallback: planner
          max_rounds: 6
          agents:                            # optional — wires the whole loop
            coder:   [agent_step: {system: "...", output_key: code}]
            planner: [agent_step: {system: "...", output_key: plan}]
          finish:
            - transform: {action: now, output_key: delivered_at}
      - supervise:                           # route an existing decider (advanced)
          key: next_agent
          agents:
            coder:   [agent_step: {system: "...", output_key: code}]
            planner: [agent_step: {system: "...", output_key: plan}]
          finish:
            - transform: {action: now, output_key: delivered_at}
      - parallel:
          branches:
            - llm: {...}
            - agent: {...}
          converge: {transform: {...}}
      - map:
          input_keys: [items]
          output_key: results
          processor: {llm: {model: ..., system: "..."}}
      - branch:
          key: sentiment
          cases:
            - {value: positive, steps: [transform: {...}]}
            - {value: negative, steps: [transform: {...}]}
          default: {transform: {...}}
          converge: {transform: {...}}
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
from teff.flow.compiler._common import _NODE_DEFAULTS  # noqa: F401
from teff.flow.compiler._flow import (
    _h_branch,
    _h_interrupt,
    _h_loop,
    _h_map,
    _h_parallel,
    _h_route,
)
from teff.flow.compiler._nodes import _h_agent, _h_llm, _h_transform, _h_type
from teff.flow.compiler._state import _build_state
from teff.flow.compiler._team import _h_supervise, _h_supervisor, _h_team
from teff.flow.flow import Flow
from teff.yaml import _interpolate_env, _providers_from_data, _safe_load

if typing.TYPE_CHECKING:
    from teff.graph import Graph


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


#: Top-level idiom name → handler (imported above, grouped by concern).
_HANDLERS = {
    "llm": _h_llm,
    "transform": _h_transform,
    "agent": _h_agent,
    "agent_step": _h_agent,
    "team": _h_team,
    "supervisor": _h_supervisor,
    "supervise": _h_supervise,
    "parallel": _h_parallel,
    "map": _h_map,
    "loop": _h_loop,
    "interrupt": _h_interrupt,
    "branch": _h_branch,
    "route": _h_route,
    "type": _h_type,
}


def compile_flow_file(path: str) -> "Graph":
    """Compile a ``flow.yaml`` file into a plain Graph."""
    doc = load_flow_yaml(path)
    return flow_from_yaml(doc).compile()


def load_flow(path: str, data: dict | None = None):
    """Load a ``flow.yaml`` as a ``(graph, tools, initial, reducers)`` tuple.

    Mirrors :func:`teff.yaml.load_workflow` so callers (and the CLI) can
    treat ``flow.yaml`` and ``graph.yaml`` interchangeably::

        graph, tools, state, reducers = load_flow("app/flow.yaml")

    The compiled ``graph`` reflects all idioms in the authoring layer; the
    optional ``tools:`` / ``state:`` blocks are passed through unchanged.
    Pass *data* (a document already resolved by :func:`teff.yaml.load_workflow`
    — env interpolation and ``include:`` blocks applied) to reuse it instead
    of re-reading the file.
    """
    if data is None:
        data = load_flow_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    from teff.plugins import load_plugins_from_document

    load_plugins_from_document(data, base_dir)
    graph = flow_from_yaml(data).compile()
    tools, initial, reducers = _build_state(data, base_dir)
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
        for key, value in step.items():
            if key == "type":
                # ``type:`` is only an idiom when its value is a mapping
                # (``- type: {type: csv, config: {...}}``); a low-level
                # ``type: transform`` string is a plain step key.
                if isinstance(value, dict) and value.get("type"):
                    return True
                continue
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
