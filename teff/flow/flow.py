"""Fluid flow builder for constructing graphs."""

from typing import TYPE_CHECKING, Callable

from teff.flow.case import Case
from teff.flow.sub_flow import SubFlow
from teff.graph import Edge, Graph
from teff.memory.context import MemoryConfig
from teff.node.agent import ReActAgent
from teff.node.command import Command
from teff.node.llm import LLM
from teff.node.map import Map
from teff.node.node import Node
from teff.node.parallel import Parallel
from teff.node.registry import make_function_node
from teff.node.supervisor import Supervisor
from teff.node.transform import Transform
from teff.provider import ProviderRegistry

if TYPE_CHECKING:
    from teff.node.ask import Ask

#: A plain ``(ctx, state) -> dict | Command`` callable accepted by
#: :meth:`Flow.step` (wrapped into a function node at runtime).
FunctionNode = Callable[..., dict | Command]


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
        self._last_branch_key: str | None = None
        self._last_branch_values: list[str] = []
        self._branch_point: str | None = None
        self._branch_ends: list[str] = []
        self._route_terminates = False
        self._guarded_step: str | None = None
        self._loop_labels: dict[str, str] = {}
        self._loop_decider: str | None = None

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

    def step(
        self,
        node: Node | FunctionNode,
        id: str | None = None,
        *,
        when: str | Callable[[dict], bool] | None = None,
    ) -> "Flow":
        """Append a node to the linear chain.

        Accepts a Node instance::

            flow.step(Transform(action="uppercase"))
            flow.step(LLM(model="gpt-4"))
            flow.step(custom_node)

        *id* optionally names the node in the compiled graph instead of
        the auto-generated ``{type}_{n}``.  Returns ``self`` for chaining.

        *when* guards the edge into this node: a string condition
        (``"status=ok"``) or a callable ``(state) -> bool``.  The edge
        fires only when the condition matches; pair with ``default()`` for
        a fallback branch::

            flow.step(transform_llm)                        # decider
            flow.step(admin_panel, when=lambda s: s.get("role") == "admin")
            flow.default(denied_panel)                      # runs when the guard fails
        """
        self._check_continuation()
        if not isinstance(node, Node):
            if not callable(node):
                raise TypeError(
                    "step() expects a Node instance or a function "
                    "(ctx, state) -> dict; got a value of type "
                    f"{type(node).__name__} — must be a Node or function"
                )
            node = make_function_node(node)
        self._nodes.append(node)
        nid = self._next_id(node, id)
        self._node_ids.append(nid)
        self._loop_decider = None
        prev = self._last_added
        if prev is not None:
            self._edges.append(Edge(source_id=prev, target_id=nid, condition=when))
        elif when is not None:
            raise ValueError("step(when=...) requires a preceding node to branch from")
        if when is not None:
            self._guarded_step = prev
        else:
            self._guarded_step = None
        self._last_added = nid
        return self

    def llm(
        self,
        node: LLM | None = None,
        id: str | None = None,
        *,
        memory: MemoryConfig | dict | None = None,
        **config,
    ) -> "Flow":
        """Add an :class:`~teff.node.llm.LLM` chat node.

        Pass a pre-built ``LLM`` instance to reuse a shared node, or give
        keyword config that is forwarded to the ``LLM`` constructor::

            flow.llm(model="gpt-4", system="You are helpful", output_key="answer")
            flow.llm(LLM(model="gpt-4", parse=True, output_key="data"))

        *memory* enables long-term memory injection on this node.  Passing
        both an instance and config kwargs raises ``TypeError``.  *id*
        optionally names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = LLM(memory=memory, **config)
        else:
            if config or memory:
                raise TypeError(
                    "llm() accepts either an LLM instance or config kwargs, not both"
                )
            if not isinstance(node, LLM):
                raise TypeError("llm() expects an LLM instance")
        return self.step(node, id=id)

    def transform(
        self, node: Transform | None = None, id: str | None = None, **config
    ) -> "Flow":
        """Add a :class:`~teff.node.transform.Transform` node.

        Pass a pre-built ``Transform`` instance or keyword config that is
        forwarded to the ``Transform`` constructor::

            flow.transform(action="uppercase", input_key="text", output_key="shout")
            flow.transform(Transform(action="value", value="done", output_key="status"))

        Passing both an instance and config kwargs raises ``TypeError``.
        *id* optionally names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = Transform(**config)
        else:
            if config:
                raise TypeError(
                    "transform() accepts either a Transform instance or config kwargs, not both"
                )
            if not isinstance(node, Transform):
                raise TypeError("transform() expects a Transform instance")
        return self.step(node, id=id)

    def supervisor(
        self,
        node: "Supervisor | None" = None,
        id: str | None = None,
        **config,
    ) -> "Flow":
        """Add a :class:`~teff.node.supervisor.Supervisor` decider node.

        Pass a pre-built ``Supervisor`` instance to reuse a shared node, or
        keyword config that is forwarded to the ``Supervisor`` constructor::

            flow.supervisor(
                model="llama3.1:8b",
                provider="ollama",
                sections=AGENT_SECTIONS,
                route_keys={"planner": "plan", "reviewer": "review"},
                done_keys={"plan", "review"},
            ).route("next_agent", ...)

        Passing both an instance and config kwargs raises ``TypeError``.
        *id* optionally names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = Supervisor(**config)
        else:
            if config:
                raise TypeError(
                    "supervisor() accepts either a Supervisor instance or config kwargs, not both"
                )
            if not isinstance(node, Supervisor):
                raise TypeError("supervisor() expects a Supervisor instance")
        return self.step(node, id=id)

    def add_flow(self, flow: "Flow", id: str | None = None, **kw) -> "Flow":
        """Embed a sub-flow as a single node (SubFlow).

        The *flow* is compiled and wrapped in a SubFlow node.
        Pass *input_map* / *output_map* as keyword arguments for key remapping.
        Pass *max_iterations* to limit internal steps (see :class:`SubFlow`).
        *id* optionally names the node in the compiled graph.
        """
        self._check_continuation()
        sub = SubFlow(graph=flow.compile(), **kw)
        self._nodes.append(sub)
        nid = self._next_id(sub, id)
        self._node_ids.append(nid)
        self._guarded_step = None
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        return self

    def parallel(self, *branches, id: str | None = None) -> "Flow":
        """Run several branch chains concurrently from the last node.

        Each *branch* is a single :class:`Node`, a list of nodes (run
        sequentially inside the branch), or a :class:`Flow` (embedded as
        a :class:`SubFlow`).  Branches execute via ``asyncio.gather`` on
        isolated copies of the state; per-key reducers (``append`` etc.)
        merge their updates back without overwriting one another.

        Combine with ``converge()`` to rejoin the parallel paths::

            flow.parallel(
                [Transform(action="uppercase", input_key="a", output_key="a")],
                [Transform(action="uppercase", input_key="b", output_key="b")],
            ).converge(shout_node)

        *id* optionally names the node in the compiled graph.
        """
        self._check_continuation()
        branch_specs: list[Node | list[Node]] = [self._as_branch(b) for b in branches]
        node = Parallel(branch_specs)
        self._nodes.append(node)
        nid = self._next_id(node, id)
        self._node_ids.append(nid)
        self._guarded_step = None
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        self._branch_ends = [nid]
        return self

    def _as_branch(self, branch) -> list[Node]:
        """Normalise a branch spec into a list of nodes."""
        if isinstance(branch, Node):
            return [branch]
        if isinstance(branch, Flow):
            return [SubFlow(graph=branch.compile())]
        return list(branch)

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
        """Dynamically fan a state list out across parallel branches.

        Runs *processor* concurrently over each item of the list(s) at
        *input_keys*, gathering the per-item results into a list at
        *output_key*.  Branch count is derived from the data at runtime
        (see :class:`~teff.node.map.Map`)::

            flow.map(
                LLM(model="llama3.1:8b", input_key="chunk", output_key="summary"),
                input_keys=["chunks"],
                output_key="summaries",
            )

        *id* optionally names the node in the compiled graph.
        """
        self._check_continuation()
        node = Map(
            processor,
            input_keys=input_keys,
            output_key=output_key,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
            **kwargs,
        )
        self._nodes.append(node)
        nid = self._next_id(node, id)
        self._node_ids.append(nid)
        self._guarded_step = None
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        return self

    def branch(self, key: str, *cases: "Case", default: Node | None = None) -> "Flow":
        """Add conditional branching from the last added node.

        Args:
            key: State key to evaluate.
            *cases: One or more Case objects, each with a value.
            default: Optional fallback node (catches unmatched values).

        Each case creates an edge ``key=<case.value>`` from the last node.
        The default creates an edge ``key!=<all case values>``.

        Multiple nodes within a case are chained sequentially.
        Use ``converge()`` after branching to merge branches.
        """
        self._check_continuation()
        if not cases:
            raise ValueError("branch requires at least one Case")
        assert self._last_added is not None
        self._last_branch_key = key
        self._last_branch_values = []
        self._branch_point = self._last_added
        self._branch_ends = []
        self._guarded_step = None
        for case in cases:
            self._last_branch_values.append(case.value)
            prev_id: str | None = None
            for n, cid in zip(case._nodes, case._ids):
                self._nodes.append(n)
                nid = self._next_id(n, cid)
                self._node_ids.append(nid)
                parent = prev_id if prev_id is not None else self._last_added
                condition = f"{key}={case.value}" if prev_id is None else None
                self._edges.append(
                    Edge(source_id=parent, target_id=nid, condition=condition)
                )
                prev_id = nid
            if prev_id is not None:
                self._branch_ends.append(prev_id)
        if default:
            self._nodes.append(default)
            dnid = self._next_id(default)
            self._node_ids.append(dnid)
            negated = ",".join(self._last_branch_values)
            self._edges.append(
                Edge(
                    source_id=self._last_added,
                    target_id=dnid,
                    condition=f"{key}!={negated}",
                )
            )
            self._branch_ends.append(dnid)
        if self._branch_ends:
            self._last_added = self._branch_ends[-1]
        return self

    def default(self, node: Node, id: str | None = None) -> "Flow":
        """Add a fallback node for the most recent branch."""
        self._check_continuation()
        self._nodes.append(node)
        dnid = self._next_id(node, id)
        self._node_ids.append(dnid)
        bp = self._branch_point or self._last_added
        if self._last_branch_key and self._last_branch_values:
            key = self._last_branch_key
            negated = ",".join(self._last_branch_values)
            assert bp is not None, "branch() must precede default()"
            self._edges.append(
                Edge(
                    source_id=bp,
                    target_id=dnid,
                    condition=f"{key}!={negated}",
                )
            )
        elif self._guarded_step is not None:
            # fallback for a step(when=...) guard: the conditional edge was
            # added first, so this unconditional edge only fires when the
            # guard fails (first matching edge wins during execution)
            self._edges.append(Edge(source_id=self._guarded_step, target_id=dnid))
            self._guarded_step = None
        self._branch_ends.append(dnid)
        self._last_added = dnid
        return self

    def converge(self, node: Node, id: str | None = None) -> "Flow":
        """Merge all branch ends into a single node.

        Adds edges from every branch end (set by the last ``branch()``
        call) to *node*.  Use after ``branch()`` to rejoin paths::

            flow.branch("sentiment",
                Case("positive").add(on_pos),
                Case("negative").add(on_neg),
            ).converge(shout_node)

        *id* optionally names the node in the compiled graph.
        """
        self._check_continuation()
        self._nodes.append(node)
        nid = self._next_id(node, id)
        self._node_ids.append(nid)
        for src in self._branch_ends:
            self._edges.append(Edge(source_id=src, target_id=nid))
        self._last_added = nid
        self._branch_ends = []
        self._guarded_step = None
        return self

    def interrupt(
        self,
        key: str,
        prompt: str = "",
        *,
        accept: "Ask | None" = None,
        id: str | None = None,
    ) -> "Flow":
        """Pause the flow for human input at this point.

        Appends an :class:`~teff.node.interrupt.Interrupt` node.  When
        execution reaches it, ``graph.run()`` raises
        :class:`~teff.node.interrupt.GraphInterrupt`; resume by calling
        ``run()`` again with the same ``checkpoint_id`` and a ``resume``
        dict mapping *key* to the operator's answer::

            try:
                await graph.run(state, checkpointer=cp, checkpoint_id="run-1")
            except GraphInterrupt as interrupt:
                print(interrupt.prompt)
                answer = input("> ")
                await graph.run(
                    state, checkpointer=cp,
                    checkpoint_id="run-1", resume={key: answer},
                )

        With *accept* (an :class:`~teff.node.ask.Ask` strategy) the raw
        answer is not enough: an optional classifier
        (:class:`~teff.node.LLM`) normalizes free-form answers into a
        structured verdict, and a :class:`~teff.node.ask.Validate` node
        decodes it into ``<accept.decision_key>`` (and captures an
        arbitrary value into ``accept.value_key``), so "yes", "ok",
        "fine" all count as *accept.pass_value*.  See
        :meth:`interrupt_loop` for re-asking.

        Args:
            key: State key that receives the resume value.
            prompt: Human-readable question shown to the operator.
            accept: Optional :class:`~teff.node.ask.Ask` validation
                strategy.  When given, the interrupt is followed by an
                optional classifier and a ``Validate`` node.

        Returns:
            ``self`` for chaining.
        """
        from teff.node.interrupt import Interrupt

        if accept is None:
            return self.step(Interrupt(key=key, prompt=prompt), id=id)

        self.step(Interrupt(key=key, prompt=prompt), id=id)
        return self._wire_ask(key, accept, id)

    def _wire_ask(self, key: str, accept: "Ask", id: str | None) -> "Flow":
        """Append the classifier + validate nodes for an *accept* strategy.

        ``_last_added`` ends on the ``Validate`` node so a following
        :meth:`loop` can decide on ``accept.decision_key``.
        """
        prefix = f"{id}-" if id else ""
        input_key = key
        if accept.needs_classifier():
            if not accept.model_name or not accept.provider:
                raise ValueError(
                    "interrupt with a 'llm' Ask strategy requires model and provider"
                )
            self.step(accept.classifier(), id=f"{prefix}classifier")
            input_key = accept.verdict_key
        self.step(accept.validate_node(input_key=input_key), id=f"{prefix}validate")
        return self

    def loop(
        self,
        key: str,
        until: str,
        done: Node | list[Node],
        body: Node | list[Node],
        *,
        max_rounds: int | None = None,
    ) -> "Flow":
        """Run a chain repeatedly until ``state[key]`` equals *until*.

        Repeats the *body* chain, then checks a condition on
        ``state[key]``.  When the value equals *until*, execution
        proceeds to the *done* chain and continues after the loop;
        otherwise the *body* chain runs and loops back to the decider
        (the last node before this call)::

            flow.step(draft_llm)
            flow.interrupt("approved", "Approve?")   # decider
            flow.loop(
                key="approved", until="yes",
                done=final_llm, body=edit_llm,
            )

        Wires::

            decider --key=until--> done -> ...   (continue after loop)
            decider --key!=until--> body -> ... -> decider   (repeat)

        The decider is any node that writes *key* (an ``Interrupt``
        whose resume value lands there, an LLM, a ``Transform``, …).

        Passing *max_rounds* bounds the repetition: the loop is then
        compiled as a self-contained ``loop`` node that retries *body* at
        most *max_rounds* times before giving up on *until* (a safe guard
        against a body that can never reach *until* in the free-flow graph
        form).

        Args:
            key: State key to check.
            until: Value of *key* that stops the loop.
            done: Node or chain run when the loop terminates.
            body: Node or chain repeated while the loop continues.
            max_rounds: Maximum body re-runs before the loop gives up.

        Returns:
            ``self`` for chaining.
        """
        if max_rounds is not None:
            return self._loop_bounded(key, until, done, body, max_rounds)
        self._check_continuation()
        decider = self._last_added
        if decider is None:
            raise ValueError("loop requires a preceding node to decide from")
        self._loop_decider = decider
        done_chain = self._as_chain(done)
        body_chain = self._as_chain(body)
        if not done_chain:
            raise ValueError("loop requires at least one node in done")
        self._guarded_step = None

        def add_chain(chain: list[Node], first_condition: str) -> tuple[str, str]:
            first_id: str | None = None
            prev: str | None = None
            for n in chain:
                nid = self._existing_id(n)
                if nid is None:
                    self._nodes.append(n)
                    nid = self._next_id(n, n.config.get("id"))
                    self._node_ids.append(nid)
                if first_id is None:
                    self._edges.append(
                        Edge(
                            source_id=decider,
                            target_id=nid,
                            condition=first_condition,
                        )
                    )
                    first_id = nid
                else:
                    assert prev is not None
                    self._edges.append(Edge(source_id=prev, target_id=nid))
                prev = nid
            assert first_id is not None
            assert prev is not None
            return first_id, prev

        _, done_last = add_chain(done_chain, f"{key}={until}")
        body_first, body_last = add_chain(body_chain, f"{key}!={until}")
        # The body may end on the decider itself (interrupt_loop re-runs the
        # interrupt + classifier + validate, whose last node *is* the loop
        # decider).  Its ``key!=until`` edge already closes the cycle, so a
        # redundant loop-back edge would become a self-loop and, being
        # unconditional, short-circuit every decision.
        if body_last != decider:
            self._edges.append(Edge(source_id=body_last, target_id=decider))

        self._last_added = done_last
        return self

    def _loop_bounded(
        self,
        key: str,
        until: str,
        done: Node | list[Node],
        body: Node | list[Node],
        max_rounds: int,
    ) -> "Flow":
        """Compile a *max_rounds*-bounded loop as a self-contained loop node.

        The free-flow :meth:`loop` wires decider/done/body edges that repeat
        until the runtime gives up; bounding the retries needs a counter,
        which the :class:`~teff.node.loop.Loop` node already provides.  When
        *max_rounds* is requested the loop is emitted as one ``loop`` node
        (body re-run up to *max_rounds* times until ``key=until``), followed
        by the *done* chain as a plain linear continuation::

            <decider> --(plain)--> loop node -> done chain -> ...

        Returns ``self`` for chaining.
        """
        from teff.node.loop import Loop

        self._check_continuation()
        if not self._last_added:
            raise ValueError("loop requires a preceding node to decide from")
        done_chain = self._as_chain(done)
        body_chain = self._as_chain(body)
        if not done_chain:
            raise ValueError("loop requires at least one node in done")
        self._guarded_step = None
        loop_node = Loop(body_chain, key=key, until=until, max_rounds=max_rounds)
        self.step(loop_node)
        for node in done_chain:
            self.step(node)
        return self

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
        """Ask the human through an interrupt and re-ask until the answer passes.

        Composes an interrupt plus an :class:`~teff.node.ask.Ask` validation
        strategy into one re-askable unit:

        * An :class:`~teff.node.Interrupt` pauses the run and surfaces
          *prompt*; the operator's resume value lands in *key*.
        * If the strategy is ``model``, an :class:`~teff.node.LLM` normalizes
          the free-form answer into a structured verdict object and a
          :class:`~teff.node.ask.Validate` node decodes it into
          ``accept.decision_key`` (capturing an arbitrary value into
          ``accept.value_key`` when set) — so "yes", "fine", "ok"
          all count as *accept.pass_value*.
        * Otherwise the raw answer in *key* is matched by the strategy
          (``equals`` / ``any_of`` / ``regex`` / ``check``).

        Wires::

            decision --<decision_key>=<pass_value>--> done (continue)
            decision --<decision_key>=<clarify_value>--> interrupt (re-ask, no body)
            decision --<decision_key>!=<pass_value>--> body -> interrupt -> decision  (re-ask)

        *body* typically re-runs whatever produced *key* (e.g. a planner)
        plus the ask nodes, so a "no" answer regenerates and re-asks; *done*
        runs once the answer passes.  When the ``model`` strategy declares
        *clear_field* / *clarify_value*, an unclear answer (e.g. gibberish)
        routes back to the interrupt for a plain re-ask **without** re-running
        *body* — free-form replies never trigger an unwanted re-plan.

        Args:
            key: State key receiving the interrupt resume value.
            accept: :class:`~teff.node.ask.Ask` validation strategy.  Its
                *decision_key* / *pass_value* drive the surrounding loop.
            prompt: Question shown to the operator.
            body: Chain re-run while the loop continues (fail branch).
            done: Chain run when the loop terminates.
            id: Prefix for the interrupt/classifier/validate node ids.

        Returns:
            ``self`` for chaining.
        """
        from teff.node.interrupt import Interrupt

        prefix = f"{id}-" if id else ""

        self.step(Interrupt(key=key, prompt=prompt), id=f"{prefix}interrupt")

        ask_nodes: list[Node] = [self._nodes[-1]]
        interrupt_id = self._node_ids[-1]
        input_key = key
        if accept.needs_classifier():
            if not accept.model_name or not accept.provider:
                raise ValueError(
                    "interrupt_loop with a 'model' Ask strategy requires model and provider"
                )
            classifier = accept.classifier()
            self.step(classifier, id=f"{prefix}classifier")
            ask_nodes.append(classifier)
            input_key = accept.verdict_key
        validate = accept.validate_node(input_key=input_key)
        self.step(validate, id=f"{prefix}validate")
        ask_nodes.append(validate)

        # Third outcome — "unclear, re-ask": a verdict whose clear_field is
        # False lands in accept.clarify_value, which routes straight back to
        # the interrupt (re-ask) instead of re-running the body.  This edge is
        # added before loop() so it is evaluated first (resolve_edge picks the
        # first match), ahead of the key!=until fail edge that would otherwise
        # also match the clarify value.
        clarify = getattr(accept, "clarify_value", "")
        if clarify:
            self._edges.append(
                Edge(
                    source_id=self._node_ids[-1],
                    target_id=interrupt_id,
                    condition=f"{accept.decision_key}={clarify}",
                )
            )

        body_chain = self._as_chain(body)
        return self.loop(
            key=accept.decision_key,
            until=accept.pass_value,
            done=done,
            body=[*body_chain, *ask_nodes],
        )

    def route(
        self,
        key: str,
        *,
        finish: Node | list[Node] | None = None,
        **agents,
    ) -> "Flow":
        """Route between agent chains under a supervisor decider.

        Wires the last added node (the decider) into a supervisor-style
        loop.  The decider writes *key* (e.g. ``"next_agent"``); each
        keyword in *agents* maps a value of *key* to the chain run for
        it, and after that chain finishes control returns to the decider.
        When *key* equals ``"finish"`` the loop exits through *finish*::

            flow.step(supervisor)          # LLM writing "next_agent"
            flow.route(
                "next_agent",
                finish=final_llm,
                planner=planner_chain,
                estimator=estimator_chain,
            )

        Wires::

            supervisor --next_agent=planner--> planner-chain -> supervisor
            supervisor --next_agent=estimator--> estimator-chain -> supervisor
            supervisor --next_agent=finish--> finish-chain -> (continue)

        The *finish* chain is optional.  When omitted the flow simply
        terminates when *key* equals ``"finish"`` and no further nodes
        may be chained; pass ``finish=<chain>`` to run something on exit
        and keep building the flow afterwards.

        Args:
            key: State key written by the decider (the node last added
                before this call).
            finish: Chain (``Node`` or list of nodes) run when *key*
                equals ``"finish"``.  Optional; when omitted the flow
                terminates on ``"finish"``.
            **agents: Each keyword is a value of *key*; its value is the
                chain (``Node`` or list of nodes) run for that route,
                after which control loops back to the decider.

        Returns:
            ``self`` for chaining.
        """
        self._check_continuation()
        decider = self._last_added
        if decider is None:
            raise ValueError("route requires a preceding node to decide from")
        if not agents:
            raise ValueError("route requires at least one agent route")
        self._guarded_step = None

        def add_chain(
            chain: list[Node], first_condition: str, first_hint: str | None = None
        ) -> tuple[str, str]:
            first_id: str | None = None
            prev: str | None = None
            for i, n in enumerate(chain):
                if not isinstance(n, Node):
                    raise TypeError("route expects Node instances in chains")
                self._nodes.append(n)
                hint = n.config.get("id") or (first_hint if i == 0 else None)
                nid = self._next_id(n, hint)
                self._node_ids.append(nid)
                if first_id is None:
                    self._edges.append(
                        Edge(
                            source_id=decider,
                            target_id=nid,
                            condition=first_condition,
                        )
                    )
                    first_id = nid
                else:
                    assert prev is not None
                    self._edges.append(Edge(source_id=prev, target_id=nid))
                prev = nid
            if first_id is None:
                raise ValueError("route requires at least one node per route")
            assert prev is not None
            return first_id, prev

        def _chain_hint(chain: list[Node], value: str) -> str | None:
            """Name a prefixed SubFlow chain after its route value.

            ``agent_step(id="planner")`` yields a SubFlow whose internal
            nodes are ``planner/<node>``; naming the outer node ``planner``
            keeps the route key visible in the parent graph.
            """
            first = chain[0] if chain else None
            if isinstance(first, SubFlow) and first._id_prefix == value:
                return value
            return None

        finish_chain = self._as_chain(finish)
        if finish_chain:
            _, done_last = add_chain(
                finish_chain, f"{key}=finish", _chain_hint(finish_chain, "finish")
            )
        else:
            done_last = decider

        for value, chain in agents.items():
            chain = self._as_chain(chain)
            _, last = add_chain(chain, f"{key}={value}", _chain_hint(chain, value))
            self._edges.append(Edge(source_id=last, target_id=decider))

        self._last_added = done_last
        self._branch_point = None
        self._branch_ends = []
        self._route_terminates = finish is None
        return self

    def command(
        self,
        *,
        routes: dict | None = None,
        goto: str | None = None,
        update: dict | None = None,
        id: str | None = None,
    ) -> "Flow":
        """Add a declarative ``command`` node that routes by state.

        Equivalent to ``step(CommandNode(...))`` but resolves ``goto``
        /``routes[].goto`` names through any :meth:`label` labels, so a
        sugar route can jump back to a loop's decision point::

            flow.loop(key="verdict", until="pass", body=body, done=done)
            flow.label("refine")
            flow.command(
                routes=[{"when": "decision=rework", "goto": "refine"}],
                goto="STOP",
            )

        Returns ``self`` for chaining.
        """
        from teff.node.command_node import CommandNode

        resolved_routes = []
        for r in routes or []:
            item = dict(r)
            if "goto" in item and isinstance(item["goto"], str):
                item["goto"] = self.label_target(item["goto"])
            resolved_routes.append(item)
        resolved_goto = self.label_target(goto) if goto else goto
        config: dict = {}
        if resolved_routes:
            config["routes"] = resolved_routes
        if resolved_goto is not None:
            config["goto"] = resolved_goto
        if update is not None:
            config["update"] = update
        return self.step(CommandNode(config=config), id=id)

    def harness(
        self,
        model: str | None = None,
        system: str = "",
        *,
        agent: ReActAgent | type[ReActAgent] | None = None,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        memory: MemoryConfig | dict | None = None,
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

        Creates an agent node and a tool executor wired in a cycle.  The
        agent calls the LLM; if the LLM requests tools, they are signalled
        to the executor, which runs them **all in parallel** and loops back
        to the agent.  When the LLM answers without a tool call, the output
        is stored at *output_key* and execution continues after the loop
        (any node chained with ``step()``/``branch()`` after this call).

        Multiple tools can be requested in a single round (e.g. read from
        RAG *and* compute at once); the executor fans them out with
        ``asyncio.gather``.

        *id* names the two nodes created by this helper as ``{id}/agent``
        and ``{id}/tool``; when omitted they keep the auto-generated
        ``{type}_{n}`` ids.

        The agent node is a :class:`~teff.node.agent.ReActAgent`.  Pass a
        pre-built instance or a subclass to override its behaviour::

            flow.react(agent=MyAgent(model="gpt-4", system="..."))
            flow.react(agent=MyAgentClass, model="gpt-4", system="...")

        With an instance, *model*/*system* and the other agent knobs are
        ignored (the instance is used as-is); with a subclass they are
        forwarded to its constructor.  When *agent* is omitted the
        ``ReActAgent`` class is used and *model* is required.

        Args:
            model: LLM model name (e.g. ``gpt-4``).  Required unless
                *agent* is given.
            system: Optional system prompt.
            agent: A ``ReActAgent`` instance or subclass to use instead of
                building the default one.
            input_key: State key for user input (default ``"input"``).
            output_key: State key for final response (default ``"output"``).
            messages_key: State key for conversation (default ``"messages"``).
            memory: Long-term memory injection — a
                :class:`~teff.memory.context.MemoryConfig` or config dict.
            max_tool_rounds: Max model calls per graph visit.
            tool_error_mode: ``"message"`` (default) or ``"raise"`` — when
                ``"raise"`` a tool failure routes to the graph's error path.
            parse_text_tool_calls: Decode tool calls embedded in plain text.
            temperature / max_tokens / response_format: Sampling knobs.
            use_tools: ``None``/``[]`` (no tools, default), ``"all"`` (everything
                the pool offers), or a list of tool names to allow.  The
                bool shorthands ``True``/``False`` are supported for
                compatibility but an explicit list is preferred.
            skills: Skills to mount on the agent — names resolved against
                *skill_dir*, skill paths, or :class:`~teff.skill.Skill`
                objects.  Their instructions go into the system prompt and
                their ``allowed-tools``/``disallowed-tools`` narrow the
                agent's tool set.
            skill_dir: Directory to resolve bare skill names from
                (default ``"skills"``).
            **config: Extra kwargs passed to :class:`ReActAgent` /
                :class:`ToolExec` config.

        Remember to pass ``max_iterations`` to ``graph.run()``::

            result = await graph.run(state, tools=tools, max_iterations=20)
        """
        self._check_continuation()
        from teff.node.agent import ToolExec

        agent_cfg = {
            "model": model,
            "system": system,
            "input_key": input_key,
            "output_key": output_key,
            "messages_key": messages_key,
            "max_tool_rounds": max_tool_rounds,
            "parse_text_tool_calls": parse_text_tool_calls,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "use_tools": use_tools,
            "skills": skills,
            "skill_dir": skill_dir,
            "memory": memory,
            **config,
        }
        if agent is None:
            if model is None and self._default_model is None:
                raise TypeError(
                    "harness() requires a model (or a default_model on the "
                    "flow) when no agent instance is given"
                )
            agent_node: ReActAgent = ReActAgent(**agent_cfg)
        elif isinstance(agent, type):
            if not issubclass(agent, ReActAgent):
                raise TypeError(
                    "harness() agent must be a ReActAgent instance or subclass"
                )
            agent_node = agent(**agent_cfg)
        else:
            if not isinstance(agent, ReActAgent):
                raise TypeError(
                    "harness() agent must be a ReActAgent instance or subclass"
                )
            agent_node = agent
        tool_exec = ToolExec(
            messages_key=messages_key,
            tool_call_key=str(
                agent_node.config.get("tool_call_key") or "_tool_call_name"
            ),
            tool_error_mode=tool_error_mode,
            use_tools=use_tools,
            skills=skills,
            skill_dir=skill_dir,
            **config,
        )

        self._nodes.append(agent_node)
        agent_id = self._next_id(agent_node, f"{id}/agent" if id else None)
        self._node_ids.append(agent_id)

        self._nodes.append(tool_exec)
        tool_id = self._next_id(tool_exec, f"{id}/tool" if id else None)
        self._node_ids.append(tool_id)

        self._edges.append(
            Edge(
                agent_id,
                tool_id,
                f"{agent_node.config.get('tool_call_key', '_tool_call_name')}!=",
            )
        )
        self._edges.append(Edge(tool_id, agent_id))

        if self._last_added is not None:
            self._edges.append(Edge(self._last_added, agent_id))

        self._last_added = agent_id
        self._branch_point = None
        self._branch_ends = []
        self._guarded_step = None
        return self

    def react(
        self,
        model: str | None = None,
        system: str = "",
        *,
        agent: ReActAgent | type[ReActAgent] | None = None,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        memory: MemoryConfig | dict | None = None,
        **config,
    ) -> "Flow":
        """Alias for :meth:`harness` (ReAct agent loop)."""
        return self.harness(
            model,
            system,
            agent=agent,
            input_key=input_key,
            output_key=output_key,
            messages_key=messages_key,
            memory=memory,
            **config,
        )

    def compile(self) -> Graph:
        """Compile the flow into a ``Graph`` ready for execution.

        Raises:
            ValueError: If no nodes were added.
        """
        return self._compile()

    def label(self, name: str) -> "Flow":
        """Attach a route *name* to the most recently added node.

        ``Command``-style ``goto`` targets (from a declarative ``route``
        step) must name a real node id in the compiled graph.  A loop's
        body has no node of its own — the *decider* (the node that reads
        the loop key) is its re-entry point.  Label it so sugar can route
        back to it::

            flow.step(extract_verdict, id="extract_verdict")
            flow.loop(key="verdict", until="pass", body=body, done=done)
            flow.label("refine")            # ("refine" now means "extract_verdict")
            flow.step(route_node)           # route: goto "refine" loops back

        Returns ``self`` for chaining.
        """
        prev = self._last_added
        if prev is None:
            raise ValueError("label() requires a preceding node")
        target = self._loop_decider if self._loop_decider is not None else prev
        self._loop_labels[name] = target
        return self

    def label_target(self, goto: str) -> str:
        """Resolve a declarative ``goto`` against labels to a real node id.

        ``label()`` maps a route name (e.g. ``"refine"``) to the loop
        decider's node id; this turns a sugar ``route: {goto: refine}``
        into an executable ``goto: <decider_id>``.  Names that are not
        labeled (real node ids, ``STOP``) pass through unchanged.
        """
        return self._loop_labels.get(goto, goto)

    def _compile(self) -> Graph:
        """Compile the flow into a ``Graph`` ready for execution.

        Raises:
            ValueError: If no nodes were added.
        """
        if not self._nodes:
            raise ValueError("no nodes in flow")
        return Graph(
            nodes=dict(zip(self._node_ids, self._nodes)),
            edges=self._edges,
            entry_point=self._node_ids[0],
            providers=self._providers,
            default_provider=self._default_provider,
            default_model=self._default_model,
        )

    def to_yaml(
        self,
        *,
        tools: list | None = None,
        initial: dict | None = None,
        reducers: dict | None = None,
    ) -> str:
        """Export the compiled flow as a ``workflow.yaml`` document.

        The graph (``steps`` + ``edges``) is serialised faithfully —
        including the ReAct loop wiring produced by :meth:`harness` /
        :meth:`react`.  Tools and state are not tracked by ``Flow``, so
        pass them explicitly if you want them in the export::

            yaml_text = (
                Flow("repo")
                .react(model="llama3.1:8b", use_tools="all")
                .to_yaml(tools=[GitTool(), CsvQueryTool()])
            )
            with open("workflow.yaml", "w") as f:
                f.write(yaml_text)

        The result validates with ``teff validate`` and round-trips through
        :func:`teff.yaml.load_workflow`.
        """
        from teff.yaml import workflow_to_yaml

        return workflow_to_yaml(
            self.compile(),
            tools=tools,
            initial=initial,
            reducers=reducers,
            name=self._name or "graph",
        )
