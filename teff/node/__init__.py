from typing import TYPE_CHECKING

from teff.node.agent import ReActAgent, ToolExec
from teff.node.ask import Ask, Validate
from teff.node.command import Command
from teff.node.command_node import CommandNode
from teff.node.contract import (
    Contract,
    ContractFallback,
    ContractNormalize,
    ContractValidate,
)
from teff.node.extract import Extract, Fallback

if TYPE_CHECKING:
    from teff.provider import ProviderRegistry
from teff.node.context import (
    AppendAssistant,
    ContextBuilder,
    ExecContext,
    last_user_message,
)
from teff.node.gate import Gate
from teff.node.interrupt import GraphInterrupt, Interrupt
from teff.node.llm import LLM, StructuredOutputError
from teff.node.loop import Loop
from teff.node.map import Map
from teff.node.node import Node
from teff.node.parallel import Parallel
from teff.node.registry import NodeRegistry, default_registry, node
from teff.node.retry import Retry
from teff.node.supervisor import Supervisor
from teff.node.tool_call import ToolCall
from teff.node.transform import Transform

if TYPE_CHECKING:
    from teff.flow.sub_flow import SubFlow
    from teff.graph import Graph

default_registry.register("transform", lambda cfg: Transform(cfg))
default_registry.register("gate", lambda cfg: Gate(cfg))
default_registry.register("validate", lambda cfg: Validate(cfg))
default_registry.register("fallback", lambda cfg: Fallback(cfg))
default_registry.register("context_builder", lambda cfg: ContextBuilder(cfg))
default_registry.register("append_assistant", lambda cfg: AppendAssistant(cfg))
default_registry.register("llm_chat", lambda cfg: LLM(cfg))
default_registry.register("react_agent", lambda cfg: ReActAgent(cfg))
default_registry.register("tool_exec", lambda cfg: ToolExec(cfg))
default_registry.register("tool_call", lambda cfg: ToolCall(cfg))
default_registry.register("interrupt", lambda cfg: Interrupt(cfg))
default_registry.register("supervisor", lambda cfg: Supervisor(cfg))
default_registry.register("command", lambda cfg: CommandNode(cfg))


def _resolve_step(step) -> Node:
    """Build a node from a declarative ``{type, config, retry}`` mapping.

    Mirrors the step resolution used by the top-level loader and the
    ``subflow`` node so nested specifications (parallel branches, subflow
    graphs) round-trip through YAML identically.
    """
    from teff.errors import ConfigError

    if not isinstance(step, dict):
        raise ConfigError("steps must be mappings")
    stype = step.get("type")
    if not isinstance(stype, str):
        raise ConfigError("step requires a string `type`")
    node = default_registry.create(stype, step.get("config", {}))
    if step.get("retry"):
        from teff.node.retry import wrap_with_retry

        node = wrap_with_retry(node, step["retry"])
    return node


def _parallel_factory(cfg: dict) -> Parallel:
    """Build a :class:`Parallel` from a declarative ``branches`` list.

    Each branch is a single step mapping or a list of step mappings; every
    step is resolved through the node registry, so a ``parallel`` node can
    be expressed entirely in YAML::

        - id: fanout
          type: parallel
          config:
            branches:
              - {type: transform, config: {action: uppercase, input_key: text, output_key: a}}
              - - {type: transform, config: {action: count_lines, input_key: text, output_key: b}}
    """
    from teff.errors import ConfigError

    branches_cfg = cfg.get("branches", [])
    if not isinstance(branches_cfg, list):
        raise ConfigError("parallel requires config.branches to be a list")
    branches: list[list[Node] | Node] = []
    for br in branches_cfg:
        if isinstance(br, list):
            branches.append([_resolve_step(s) for s in br])
        elif isinstance(br, dict):
            branches.append(_resolve_step(br))
        else:
            raise ConfigError(
                "parallel branches must be a step mapping or a list of step mappings"
            )
    return Parallel(branches, config=cfg)


default_registry.register("parallel", _parallel_factory)


def _map_factory(cfg: dict) -> Map:
    processor = cfg.get("processor", {})
    return Map(processor, config=cfg)


def _subflow_factory(cfg: dict) -> "SubFlow":
    from teff.errors import ConfigError
    from teff.flow.sub_flow import SubFlow

    graph_cfg = cfg.get("graph")
    build = cfg.get("build")
    if isinstance(graph_cfg, dict):
        flow = SubFlow(
            _build_subgraph(graph_cfg, cfg.get("providers")),
            input_map=cfg.get("input_map"),
            output_map=cfg.get("output_map"),
            max_iterations=cfg.get("max_iterations"),
            id_prefix=cfg.get("id_prefix", ""),
        )
    elif build is not None:
        flow = _build_from_recipe(build, cfg)
    else:
        raise ConfigError(
            "subflow requires config.graph (a mapping with steps/edges) "
            "or config.build (an agent_step recipe)"
        )
    flow.config = cfg
    return flow


def _build_subgraph(
    graph_cfg: dict, providers: "dict | ProviderRegistry | None" = None
) -> "Graph":
    """Build a nested ``Graph`` from a declarative ``{steps, edges}`` dict.

    Mirrors the step/edge building in :mod:`teff.yaml` so a ``subflow``
    node can embed a full graph inline and round-trip through YAML.
    """
    from teff.errors import ConfigError
    from teff.graph import Edge, Graph

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    entry_point: str | None = None

    for step in graph_cfg.get("steps", []):
        if not isinstance(step, dict):
            raise ConfigError("subflow graph steps must be mappings")
        sid = step.get("id")
        stype = step.get("type")
        if not isinstance(sid, str) or not isinstance(stype, str):
            raise ConfigError("subflow graph step requires string id and type")
        node = default_registry.create(stype, step.get("config", {}))
        if step.get("retry"):
            from teff.node.retry import wrap_with_retry

            node = wrap_with_retry(node, step["retry"])
        nodes[sid] = node
        if entry_point is None:
            entry_point = sid

    for edge_data in graph_cfg.get("edges", []):
        if not isinstance(edge_data, dict):
            raise ConfigError("subflow graph edges must be mappings")
        edges.append(
            Edge(
                source_id=edge_data["from"],
                target_id=edge_data["to"],
                condition=edge_data.get("condition"),
            )
        )

    return Graph(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point or "",
        providers=providers,
    )


def _build_from_recipe(build: dict, cfg: dict) -> "SubFlow":
    """Build a ``SubFlow`` from a named recipe such as ``agent_step``."""
    from teff.errors import ConfigError
    from teff.flow.agent import agent_step

    rtype = build.get("type")
    if rtype != "agent_step":
        raise ConfigError(
            f"unknown subflow build recipe {rtype!r} (supported: agent_step)"
        )
    if build.get("providers"):
        raise ConfigError(
            "agent_step build recipe must not set `providers:` — providers "
            "come from the workflow's top-level `providers:` block"
        )
    try:
        return agent_step(
            build["system"],
            build["output_key"],
            model=build["model"],
            provider=build["provider"],
            sections=build.get("sections"),
            messages_key=build.get("messages_key", "messages"),
            use_tools=build.get("use_tools"),
            stream=build.get("stream", True),
            id=cfg.get("id_prefix") or None,
            **build.get("config", {}),
        )
    except KeyError as exc:
        raise ConfigError(
            f"agent_step build recipe is missing required key: {exc.args[0]}"
        ) from exc


default_registry.register("map", _map_factory)


def _loop_factory(cfg: dict) -> Loop:
    from teff.errors import ConfigError

    body = cfg.get("body")
    if body is None:
        raise ConfigError("loop requires config.body (a node spec or list)")
    return Loop(body, config=cfg)


default_registry.register("loop", _loop_factory)
default_registry.register("subflow", _subflow_factory)


def _contract_factory(cfg: dict) -> "SubFlow":
    """Build a :class:`Contract` subflow from a declarative config mapping.

    Registered as the ``contract`` node type so a structured-output recipe
    can be expressed in YAML::

        - id: final
          type: contract
          config:
            system: "Ты формируешь ответ по контракту."
            schema: {...}
            messages_key: messages
            output_key: answer_json

    ``normalize`` / ``validate`` / ``fallback`` are callables and can only
    be supplied programmatically (mirroring ``Ask.check`` / edge conditions).
    """
    from teff.node.contract import Contract

    return Contract(**cfg).build()


default_registry.register("contract", _contract_factory)
__all__ = [
    "Node",
    "NodeRegistry",
    "default_registry",
    "ExecContext",
    "ContextBuilder",
    "AppendAssistant",
    "last_user_message",
    "node",
    "Retry",
    "Transform",
    "Loop",
    "LLM",
    "StructuredOutputError",
    "ReActAgent",
    "ToolExec",
    "ToolCall",
    "Parallel",
    "Map",
    "Supervisor",
    "Interrupt",
    "GraphInterrupt",
    "Ask",
    "Validate",
    "Contract",
    "ContractFallback",
    "ContractNormalize",
    "ContractValidate",
    "Extract",
    "Fallback",
    "Command",
    "CommandNode",
]
