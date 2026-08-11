"""Branching, looping and routing builders for :class:`~teff.flow.Flow`.

:class:`ControlBuilder` implements the control-flow methods — ``parallel``,
``map``, ``branch``, ``interrupt``, ``loop``, ``route``, ``command`` —
that wire edges between nodes.  :class:`Flow` owns an instance and
delegates to it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teff.flow.flow import Flow


class ControlBuilder:
    """Build control-flow edges on a :class:`~teff.flow.Flow`.

    Args:
        flow: The owning ``Flow`` whose graph state is mutated.
    """

    def __init__(self, flow: "Flow"):
        self.flow = flow

    def add_flow(self, flow, id=None, **kw):
        """Embed a sub-flow as a single node (SubFlow).

        The *flow* is compiled and wrapped in a SubFlow node.
        Pass *input_map* / *output_map* as keyword arguments for key remapping.
        Pass *max_iterations* to limit internal steps (see :class:`SubFlow`).
        *id* optionally names the node in the compiled graph.
        """
        from teff.flow.sub_flow import SubFlow
        from teff.graph import Edge

        target = self.flow
        target._check_continuation()
        sub = SubFlow(graph=flow.compile(), **kw)
        target._nodes.append(sub)
        nid = target._next_id(sub, id)
        target._node_ids.append(nid)
        target._guarded_step = None
        if target._last_added is not None:
            target._edges.append(Edge(source_id=target._last_added, target_id=nid))
        target._last_added = nid
        return target

    def parallel(self, *branches, id=None):
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
        from teff.graph import Edge
        from teff.node.parallel import Parallel

        target = self.flow
        target._check_continuation()
        branch_specs: list = [self._as_branch(b) for b in branches]
        node = Parallel(branch_specs)
        target._nodes.append(node)
        nid = target._next_id(node, id)
        target._node_ids.append(nid)
        target._guarded_step = None
        if target._last_added is not None:
            target._edges.append(Edge(source_id=target._last_added, target_id=nid))
        target._last_added = nid
        target._branch_ends = [nid]
        return target

    def _as_branch(self, branch):
        """Normalise a branch spec into a list of nodes."""
        from teff.flow.sub_flow import SubFlow
        from teff.node.node import Node

        if isinstance(branch, Node):
            return [branch]
        if isinstance(branch, self.flow.__class__):
            return [SubFlow(graph=branch.compile())]
        return list(branch)

    def map(
        self,
        processor,
        *,
        input_keys: str | list[str] = "",
        output_key: str = "",
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
        id: str | None = None,
        **kwargs,
    ):
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
        from teff.graph import Edge
        from teff.node.map import Map

        target = self.flow
        target._check_continuation()
        node = Map(
            processor,
            input_keys=input_keys,
            output_key=output_key,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
            **kwargs,
        )
        target._nodes.append(node)
        nid = target._next_id(node, id)
        target._node_ids.append(nid)
        target._guarded_step = None
        if target._last_added is not None:
            target._edges.append(Edge(source_id=target._last_added, target_id=nid))
        target._last_added = nid
        return target

    def branch(self, key: str, *cases, default=None):
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
        from teff.graph import Edge

        target = self.flow
        target._check_continuation()
        if not cases:
            raise ValueError("branch requires at least one Case")
        assert target._last_added is not None
        branch_point = target._last_added
        case_values: list[str] = []
        target._branch_ends = []
        target._guarded_step = None
        for case in cases:
            case_values.append(case.value)
            prev_id: str | None = None
            for n, cid in zip(case._nodes, case._ids):
                target._nodes.append(n)
                nid = target._next_id(n, cid)
                target._node_ids.append(nid)
                parent = prev_id if prev_id is not None else branch_point
                condition = f"{key}={case.value}" if prev_id is None else None
                target._edges.append(
                    Edge(source_id=parent, target_id=nid, condition=condition)
                )
                prev_id = nid
            if prev_id is not None:
                target._branch_ends.append(prev_id)
        if default:
            target._nodes.append(default)
            dnid = target._next_id(default)
            target._node_ids.append(dnid)
            negated = ",".join(case_values)
            target._edges.append(
                Edge(
                    source_id=branch_point,
                    target_id=dnid,
                    condition=f"{key}!={negated}",
                )
            )
            target._branch_ends.append(dnid)
        if target._branch_ends:
            target._last_added = target._branch_ends[-1]
        return target

    def default(self, node, id=None):
        """Add a fallback node for the most recent guarded ``step()``.

        The fallback fires only when the preceding ``step(..., when=...)``
        guard fails (the conditional edge wins when it matches)::

            flow.step(decider)
            flow.step(handler, when=lambda s: s["ok"])
            flow.default(else_handler)

        *id* optionally names the node in the compiled graph.
        """
        from teff.graph import Edge

        target = self.flow
        target._check_continuation()
        if target._guarded_step is None:
            raise ValueError(
                "default() requires a preceding step(when=...) guard; "
                "use branch(..., default=node) for branch fallbacks"
            )
        target._nodes.append(node)
        dnid = target._next_id(node, id)
        target._node_ids.append(dnid)
        target._edges.append(Edge(source_id=target._guarded_step, target_id=dnid))
        target._guarded_step = None
        target._branch_ends.append(dnid)
        target._last_added = dnid
        return target

    def converge(self, node, id=None):
        """Merge all branch ends into a single node.

        Adds edges from every branch end (set by the last ``branch()``
        call) to *node*.  Use after ``branch()`` to rejoin paths::

            flow.branch("sentiment",
                Case("positive").add(on_pos),
                Case("negative").add(on_neg),
            ).converge(shout_node)

        *id* optionally names the node in the compiled graph.
        """
        from teff.graph import Edge

        target = self.flow
        target._check_continuation()
        target._nodes.append(node)
        nid = target._next_id(node, id)
        target._node_ids.append(nid)
        for src in target._branch_ends:
            target._edges.append(Edge(source_id=src, target_id=nid))
        target._last_added = nid
        target._branch_ends = []
        target._guarded_step = None
        return target

    def interrupt(self, key: str, prompt: str = "", *, accept=None, id=None):
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
            return self.flow.step(Interrupt(key=key, prompt=prompt), id=id)

        self.flow.step(Interrupt(key=key, prompt=prompt), id=id)
        return self._wire_ask(key, accept, id)

    def _wire_ask(self, key: str, accept, id):
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
            self.flow.step(accept.classifier(), id=f"{prefix}classifier")
            input_key = accept.verdict_key
        self.flow.step(
            accept.validate_node(input_key=input_key), id=f"{prefix}validate"
        )
        return self.flow

    def loop(
        self,
        key: str,
        until: str,
        done,
        body,
        *,
        max_rounds: int | None = None,
    ):
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
        from teff.graph import Edge

        target = self.flow
        target._check_continuation()
        decider = target._last_added
        if decider is None:
            raise ValueError("loop requires a preceding node to decide from")
        target._loop_decider = decider
        done_chain = target._as_chain(done)
        body_chain = target._as_chain(body)
        if not done_chain:
            raise ValueError("loop requires at least one node in done")
        target._guarded_step = None

        def add_chain(chain: list, first_condition: str) -> tuple[str, str]:
            first_id: str | None = None
            prev: str | None = None
            for n in chain:
                nid = target._existing_id(n)
                if nid is None:
                    target._nodes.append(n)
                    nid = target._next_id(n, n.config.get("id"))
                    target._node_ids.append(nid)
                if first_id is None:
                    target._edges.append(
                        Edge(
                            source_id=decider,
                            target_id=nid,
                            condition=first_condition,
                        )
                    )
                    first_id = nid
                else:
                    assert prev is not None
                    target._edges.append(Edge(source_id=prev, target_id=nid))
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
            target._edges.append(Edge(source_id=body_last, target_id=decider))

        target._last_added = done_last
        return target

    def _loop_bounded(
        self,
        key: str,
        until: str,
        done,
        body,
        max_rounds: int,
    ):
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

        target = self.flow
        target._check_continuation()
        if not target._last_added:
            raise ValueError("loop requires a preceding node to decide from")
        done_chain = target._as_chain(done)
        body_chain = target._as_chain(body)
        if not done_chain:
            raise ValueError("loop requires at least one node in done")
        target._guarded_step = None
        loop_node = Loop(body_chain, key=key, until=until, max_rounds=max_rounds)
        target.step(loop_node)
        for node in done_chain:
            target.step(node)
        return target

    def interrupt_loop(
        self,
        key: str,
        *,
        accept,
        body,
        done,
        prompt: str = "",
        id: str | None = None,
    ):
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
        from teff.graph import Edge
        from teff.node.interrupt import Interrupt

        target = self.flow
        prefix = f"{id}-" if id else ""

        target.step(Interrupt(key=key, prompt=prompt), id=f"{prefix}interrupt")

        ask_nodes: list = [target._nodes[-1]]
        interrupt_id = target._node_ids[-1]
        input_key = key
        if accept.needs_classifier():
            if not accept.model_name or not accept.provider:
                raise ValueError(
                    "interrupt_loop with a 'model' Ask strategy requires model and provider"
                )
            classifier = accept.classifier()
            target.step(classifier, id=f"{prefix}classifier")
            ask_nodes.append(classifier)
            input_key = accept.verdict_key
        validate = accept.validate_node(input_key=input_key)
        target.step(validate, id=f"{prefix}validate")
        ask_nodes.append(validate)

        # Third outcome — "unclear, re-ask": a verdict whose clear_field is
        # False lands in accept.clarify_value, which routes straight back to
        # the interrupt (re-ask) instead of re-running the body.  This edge is
        # added before loop() so it is evaluated first (resolve_edge picks the
        # first match), ahead of the key!=until fail edge that would otherwise
        # also match the clarify value.
        clarify = getattr(accept, "clarify_value", "")
        if clarify:
            target._edges.append(
                Edge(
                    source_id=target._node_ids[-1],
                    target_id=interrupt_id,
                    condition=f"{accept.decision_key}={clarify}",
                )
            )

        body_chain = target._as_chain(body)
        return target.loop(
            key=accept.decision_key,
            until=accept.pass_value,
            done=done,
            body=[*body_chain, *ask_nodes],
        )

    def route(
        self,
        key: str,
        *,
        finish=None,
        **agents,
    ):
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
        from teff.flow.sub_flow import SubFlow
        from teff.graph import Edge
        from teff.node.node import Node

        target = self.flow
        target._check_continuation()
        decider = target._last_added
        if decider is None:
            raise ValueError("route requires a preceding node to decide from")
        if not agents:
            raise ValueError("route requires at least one agent route")
        target._guarded_step = None

        def add_chain(
            chain: list, first_condition: str, first_hint: str | None = None
        ) -> tuple[str, str]:
            first_id: str | None = None
            prev: str | None = None
            for i, n in enumerate(chain):
                if not isinstance(n, Node):
                    raise TypeError("route expects Node instances in chains")
                target._nodes.append(n)
                hint = n.config.get("id") or (first_hint if i == 0 else None)
                nid = target._next_id(n, hint)
                target._node_ids.append(nid)
                if first_id is None:
                    target._edges.append(
                        Edge(
                            source_id=decider,
                            target_id=nid,
                            condition=first_condition,
                        )
                    )
                    first_id = nid
                else:
                    assert prev is not None
                    target._edges.append(Edge(source_id=prev, target_id=nid))
                prev = nid
            if first_id is None:
                raise ValueError("route requires at least one node per route")
            assert prev is not None
            return first_id, prev

        def _chain_hint(chain: list, value: str) -> str | None:
            """Name a prefixed SubFlow chain after its route value.

            ``agent_step(id="planner")`` yields a SubFlow whose internal
            nodes are ``planner/<node>``; naming the outer node ``planner``
            keeps the route key visible in the parent graph.
            """
            first = chain[0] if chain else None
            if isinstance(first, SubFlow) and first._id_prefix == value:
                return value
            return None

        finish_chain = target._as_chain(finish)
        if finish_chain:
            _, done_last = add_chain(
                finish_chain, f"{key}=finish", _chain_hint(finish_chain, "finish")
            )
        else:
            done_last = decider

        for value, chain in agents.items():
            chain = target._as_chain(chain)
            _, last = add_chain(chain, f"{key}={value}", _chain_hint(chain, value))
            target._edges.append(Edge(source_id=last, target_id=decider))

        target._last_added = done_last
        target._branch_ends = []
        target._route_terminates = finish is None
        return target

    def command(
        self,
        *,
        routes=None,
        goto: str | None = None,
        update: dict | None = None,
        id: str | None = None,
    ):
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
                item["goto"] = self.flow.label_target(item["goto"])
            resolved_routes.append(item)
        resolved_goto = self.flow.label_target(goto) if goto else goto
        config: dict = {}
        if resolved_routes:
            config["routes"] = resolved_routes
        if resolved_goto is not None:
            config["goto"] = resolved_goto
        if update is not None:
            config["update"] = update
        return self.flow.step(CommandNode(config=config), id=id)
