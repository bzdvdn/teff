"""Fluid flow builder for constructing graphs.

The heavy lifting lives in focused builder modules — :mod:`teff.flow.base`,
:mod:`teff.flow.team`, :mod:`teff.flow.control`, :mod:`teff.flow.harness`,
:mod:`teff.flow.compile` — each owned by :class:`Flow`, which keeps the
graph state and delegates the public methods to them.
"""

from typing import TYPE_CHECKING, Callable, TypeAlias

from teff.flow.base import BaseBuilder
from teff.flow.compile import CompileBuilder
from teff.flow.control import ControlBuilder
from teff.flow.harness import HarnessBuilder
from teff.flow.team import TeamBuilder
from teff.graph import Edge, Graph
from teff.memory.context import MemoryConfig
from teff.node.command import Command
from teff.node.llm import LLM
from teff.node.node import Node
from teff.node.registry import make_function_node
from teff.node.transform import Transform

if TYPE_CHECKING:
    from teff.flow.agent import AgentRole
    from teff.flow.case import Case
    from teff.node.ask import Ask
    from teff.node.context import AppendAssistant, ContextBuilder
    from teff.node.harness import ReActAgent
    from teff.node.supervisor import Supervisor
    from teff.provider import ProviderRegistry

#: A plain ``(ctx, state) -> dict | Command`` callable accepted by
#: :meth:`Flow.step` (wrapped into a function node at runtime).
FunctionNode = Callable[..., dict | Command]

#: One role spec accepted by :meth:`Flow.team` (and the ``team:`` YAML
#: idiom): an :class:`~teff.flow.AgentRole`, a plain dict recipe
#: (``{system, output_key, use_tools, ...}``), a :class:`Node` used as-is, a
#: :class:`Flow` embedded as a ``SubFlow``, or a list of any of those (a
#: route chain such as ``agent → interrupt``).
RoleSpec: TypeAlias = "AgentRole | dict | Node | Flow | list[RoleSpec]"


class Flow:
    """Fluid builder for constructing graphs with branching.

    Args:
        name: Optional flow name.
        default_provider: Optional default provider name used by LLM nodes
            that don't set ``provider`` themselves.  Must be declared in
            ``providers``.
        default_model: Optional default model name used by LLM nodes that
            don't set ``model`` themselves.  ``model=`` on a node always
            wins.
        providers: The ``{name: Provider}`` map,
            :class:`~teff.provider.ProviderRegistry`, or YAML-style list
            of preset names threaded into the compiled graph.  Every
            provider the graph references must be declared here.

    Usage::

        flow = Flow(
            "my-flow",
            providers=ProviderRegistry.from_presets("ollama"),
            default_provider="ollama",
        )
        flow.step(LLM(model="llama3.1:8b"))
        flow.branch("status", Case("ok").add(ok_node), default=err_node)
        graph = flow.compile()

    The graph-building methods are implemented by dedicated builders:

    - :mod:`teff.flow.base` — ``step``/``llm``/``transform`` &
      co. (linear chain).
    - :mod:`teff.flow.team` — ``team`` (supervised agent team).
    - :mod:`teff.flow.control` — ``parallel``/``map``/``branch``/
      ``interrupt``/``loop``/``route``/``command`` (control flow).
    - :mod:`teff.flow.harness` — ``harness``/``react`` (ReAct agent loop).
    - :mod:`teff.flow.compile` — ``compile``/``label``/``to_yaml``
      (serialization).
    """

    def __init__(
        self,
        name: str = "",
        *,
        providers: "dict | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ):
        self._name = name
        self._default_provider = default_provider
        self._default_model = default_model
        self._providers = providers
        self._nodes: list[Node] = []
        self._node_ids: list[str] = []
        self._edges: list[Edge] = []
        self._counter = 0
        self._last_added: str | None = None
        self._branch_ends: list[str] = []
        self._route_terminates = False
        self._guarded_step: str | None = None
        self._loop_labels: dict[str, str] = {}
        self._loop_decider: str | None = None

        self._base = BaseBuilder(self)
        self._team = TeamBuilder(self)
        self._control = ControlBuilder(self)
        self._harness = HarnessBuilder(self)
        self._compile = CompileBuilder(self)

    def _next_id(self, node: Node, id_hint: str | None = None) -> str:
        self._counter += 1
        nid = id_hint or f"{node.type}_{self._counter}"
        if nid in self._node_ids:
            raise ValueError(f"duplicate node id: {nid}")
        return nid

    def _existing_id(self, node: Node) -> str | None:
        """The id *node* was registered under, if this instance is already added.

        Loop bodies re-reference nodes that were added earlier in the flow
        (a planner, an interrupt, its classifier/validate).  Re-adding the
        same instance under a fresh auto-generated id would duplicate it in
        the compiled graph (``llm_chat_7``, ``subflow_9``, …) — instead the
        chain should route through the node's first registration.
        """
        for idx, existing in enumerate(self._nodes):
            if existing is node:
                return self._node_ids[idx]
        return None

    def _check_continuation(self) -> None:
        """Raise if the last route() terminated the flow (finish=None)."""
        if self._route_terminates:
            raise ValueError(
                "route() with finish=None terminates the flow when the decider "
                "returns 'finish'; pass finish=<chain> before adding more nodes"
            )

    @staticmethod
    def _as_chain(node_or_chain) -> list[Node]:
        if node_or_chain is None:
            return []
        if isinstance(node_or_chain, Node):
            return [node_or_chain]

        def to_node(item):
            if isinstance(item, Node):
                return item
            if callable(item):
                return make_function_node(item)
            raise TypeError(
                f"expected Node or callable in chain, got {type(item).__name__}"
            )

        return [to_node(item) for item in node_or_chain]

    # ------------------------------------------------------------------
    # Linear chain (see teff.flow.base.BaseBuilder)
    # ------------------------------------------------------------------

    def step(
        self,
        node: Node | FunctionNode,
        id: str | None = None,
        *,
        when: str | Callable[[dict], bool] | None = None,
    ) -> "Flow":
        """Append a node to the linear chain.  See
        :meth:`teff.flow.base.BaseBuilder.step`."""
        return self._base.step(node, id=id, when=when)

    def llm(
        self,
        node: LLM | None = None,
        id: str | None = None,
        *,
        memory: MemoryConfig | dict | None = None,
        **config,
    ) -> "Flow":
        """Add an :class:`~teff.node.llm.LLM` chat node.  See
        :meth:`teff.flow.base.BaseBuilder.llm`."""
        return self._base.llm(node, id=id, memory=memory, **config)

    def transform(
        self, node: Transform | None = None, id: str | None = None, **config
    ) -> "Flow":
        """Add a :class:`~teff.node.transform.Transform` node.  See
        :meth:`teff.flow.base.BaseBuilder.transform`."""
        return self._base.transform(node, id=id, **config)

    def context_builder(
        self,
        node: "ContextBuilder | None" = None,
        id: str | None = None,
        **config,
    ) -> "Flow":
        """Add a :class:`~teff.node.context.ContextBuilder` node.  See
        :meth:`teff.flow.base.BaseBuilder.context_builder`."""
        return self._base.context_builder(node, id=id, **config)

    def append_assistant(
        self,
        node: "AppendAssistant | None" = None,
        id: str | None = None,
        **config,
    ) -> "Flow":
        """Add a :class:`~teff.node.context.AppendAssistant` node.  See
        :meth:`teff.flow.base.BaseBuilder.append_assistant`."""
        return self._base.append_assistant(node, id=id, **config)

    def supervisor(
        self,
        node: "Supervisor | None" = None,
        id: str | None = None,
        **config,
    ) -> "Flow":
        """Add a :class:`~teff.node.supervisor.Supervisor` decider node.  See
        :meth:`teff.flow.base.BaseBuilder.supervisor`."""
        return self._base.supervisor(node, id=id, **config)

    def team(
        self,
        system: str = "",
        *,
        roles: dict[str, RoleSpec],
        model: str | None = None,
        provider: str | None = None,
        messages_key: str = "messages",
        sections: dict[str, str] | None = None,
        route_keys: dict[str, str] | None = None,
        done_keys: list[str] | None = None,
        done_mode: str = "all",
        fallback: str = "",
        max_rounds: int = 6,
        finish: "Node | list[Node] | None" = None,
        id: str | None = None,
    ) -> "Flow":
        """Compose a supervised agent team in one call.  See
        :meth:`teff.flow.team.TeamBuilder.team`."""
        return self._team.team(
            system,
            roles=roles,
            model=model,
            provider=provider,
            messages_key=messages_key,
            sections=sections,
            route_keys=route_keys,
            done_keys=done_keys,
            done_mode=done_mode,
            fallback=fallback,
            max_rounds=max_rounds,
            finish=finish,
            id=id,
        )

    # ------------------------------------------------------------------
    # Control flow (see teff.flow.control.ControlBuilder)
    # ------------------------------------------------------------------

    def add_flow(self, flow: "Flow", id: str | None = None, **kw) -> "Flow":
        """Embed a sub-flow as a single node (SubFlow).  See
        :meth:`teff.flow.control.ControlBuilder.add_flow`."""
        return self._control.add_flow(flow, id=id, **kw)

    def parallel(self, *branches, id: str | None = None) -> "Flow":
        """Run several branch chains concurrently from the last node.  See
        :meth:`teff.flow.control.ControlBuilder.parallel`."""
        return self._control.parallel(*branches, id=id)

    def map(
        self,
        processor: Node | list[Node] | dict | list[dict],
        *,
        input_keys: str | list[str] = "",
        output_key: str = "",
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
        id: str | None = None,
        **kwargs,
    ) -> "Flow":
        """Dynamically fan a state list out across parallel branches.  See
        :meth:`teff.flow.control.ControlBuilder.map`."""
        return self._control.map(
            processor,
            input_keys=input_keys,
            output_key=output_key,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
            id=id,
            **kwargs,
        )

    def branch(self, key: str, *cases: "Case", default: Node | None = None) -> "Flow":
        """Add conditional branching from the last added node.  See
        :meth:`teff.flow.control.ControlBuilder.branch`."""
        return self._control.branch(key, *cases, default=default)

    def default(self, node: Node, id: str | None = None) -> "Flow":
        """Add a fallback node for the most recent guarded ``step()``.  See
        :meth:`teff.flow.control.ControlBuilder.default`."""
        return self._control.default(node, id=id)

    def converge(self, node: Node, id: str | None = None) -> "Flow":
        """Merge all branch ends into a single node.  See
        :meth:`teff.flow.control.ControlBuilder.converge`."""
        return self._control.converge(node, id=id)

    def interrupt(
        self,
        key: str,
        prompt: str = "",
        *,
        accept: "Ask | None" = None,
        id: str | None = None,
    ) -> "Flow":
        """Pause the flow for human input at this point.  See
        :meth:`teff.flow.control.ControlBuilder.interrupt`."""
        return self._control.interrupt(key, prompt=prompt, accept=accept, id=id)

    def loop(
        self,
        key: str,
        until: str,
        done: Node | list[Node],
        body: Node | list[Node],
        *,
        max_rounds: int | None = None,
    ) -> "Flow":
        """Run a chain repeatedly until ``state[key]`` equals *until*.  See
        :meth:`teff.flow.control.ControlBuilder.loop`."""
        return self._control.loop(key, until, done, body, max_rounds=max_rounds)

    def interrupt_loop(
        self,
        key: str,
        *,
        accept: "Ask",
        body: Node | list[Node],
        done: Node | list[Node],
        prompt: str = "",
        id: str | None = None,
    ) -> "Flow":
        """Ask the human through an interrupt and re-ask until the answer
        passes.  See :meth:`teff.flow.control.ControlBuilder.interrupt_loop`."""
        return self._control.interrupt_loop(
            key,
            accept=accept,
            body=body,
            done=done,
            prompt=prompt,
            id=id,
        )

    def route(
        self,
        key: str,
        *,
        finish: Node | list[Node] | None = None,
        **agents,
    ) -> "Flow":
        """Route between agent chains under a supervisor decider.  See
        :meth:`teff.flow.control.ControlBuilder.route`."""
        return self._control.route(key, finish=finish, **agents)

    def command(
        self,
        *,
        routes: dict | None = None,
        goto: str | None = None,
        update: dict | None = None,
        id: str | None = None,
    ) -> "Flow":
        """Add a declarative ``command`` node that routes by state.  See
        :meth:`teff.flow.control.ControlBuilder.command`."""
        return self._control.command(routes=routes, goto=goto, update=update, id=id)

    # ------------------------------------------------------------------
    # Agent harness (see teff.flow.harness.HarnessBuilder)
    # ------------------------------------------------------------------

    def harness(
        self,
        model: str | None = None,
        system: str = "",
        *,
        agent: "ReActAgent | type[ReActAgent] | None" = None,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        memory: "MemoryConfig | dict | None" = None,
        max_tool_rounds: int = 10,
        tool_error_mode: str = "message",
        parse_text_tool_calls: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        use_tools: str | list[str] | None = None,
        skills: list | None = None,
        skill_dir: str = "skills",
        id: str | None = None,
        **config,
    ) -> "Flow":
        """Build a ReAct-style agent loop (LLM ↔ tools) inside this flow.
        See :meth:`teff.flow.harness.HarnessBuilder.harness`."""
        return self._harness.harness(
            model,
            system,
            agent=agent,
            input_key=input_key,
            output_key=output_key,
            messages_key=messages_key,
            memory=memory,
            max_tool_rounds=max_tool_rounds,
            tool_error_mode=tool_error_mode,
            parse_text_tool_calls=parse_text_tool_calls,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            use_tools=use_tools,
            skills=skills,
            skill_dir=skill_dir,
            id=id,
            **config,
        )

    def react(
        self,
        model: str | None = None,
        system: str = "",
        *,
        agent: "ReActAgent | type[ReActAgent] | None" = None,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        memory: "MemoryConfig | dict | None" = None,
        **config,
    ) -> "Flow":
        """Alias for :meth:`harness` (ReAct agent loop).  See
        :meth:`teff.flow.harness.HarnessBuilder.react`."""
        return self._harness.react(
            model,
            system,
            agent=agent,
            input_key=input_key,
            output_key=output_key,
            messages_key=messages_key,
            memory=memory,
            **config,
        )

    # ------------------------------------------------------------------
    # Compilation (see teff.flow.compile.CompileBuilder)
    # ------------------------------------------------------------------

    def compile(self) -> Graph:
        """Compile the flow into a ``Graph`` ready for execution.  See
        :meth:`teff.flow.compile.CompileBuilder.compile`."""
        return self._compile.compile()

    def label(self, name: str) -> "Flow":
        """Attach a route *name* to the most recently added node.  See
        :meth:`teff.flow.compile.CompileBuilder.label`."""
        return self._compile.label(name)

    def label_target(self, goto: str) -> str:
        """Resolve a declarative ``goto`` against labels to a real node id.
        See :meth:`teff.flow.compile.CompileBuilder.label_target`."""
        return self._compile.label_target(goto)

    def to_yaml(
        self,
        *,
        tools: list | None = None,
        initial: dict | None = None,
        reducers: dict | None = None,
    ) -> str:
        """Export the compiled flow as a ``workflow.yaml`` document.  See
        :meth:`teff.flow.compile.CompileBuilder.to_yaml`."""
        return self._compile.to_yaml(tools=tools, initial=initial, reducers=reducers)
