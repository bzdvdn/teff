"""Supervised team builder for :class:`~teff.flow.Flow`.

:class:`TeamBuilder` composes a supervisor decider plus routed agents in a
single call (:meth:`~teff.flow.Flow.team`).  :class:`Flow` owns an instance
and delegates to it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teff.flow.flow import Flow
    from teff.node import Node


class TeamBuilder:
    """Compose a supervised agent team on a :class:`~teff.flow.Flow`.

    Args:
        flow: The owning ``Flow`` whose graph state is mutated.
    """

    def __init__(self, flow: "Flow"):
        self.flow = flow

    def team(
        self,
        system: str = "",
        *,
        roles: dict,
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
        """Compose a supervised agent team in one call.

        Builds a :class:`~teff.node.supervisor.Supervisor` decider plus one
        routed agent per *role* and wires the supervisor loop in a single
        step — the programmatic twin of the ``team:`` flow.yaml idiom::

            flow.team(
                "Route to planner or coder, then finish.",
                roles={
                    "planner": AgentRole("You plan.", output_key="plan"),
                    "coder": AgentRole("You code.", output_key="code"),
                },
                fallback="planner",
            )

        Each *role* value is an :class:`~teff.flow.AgentRole` (the
        recommended spelling), a plain dict recipe accepted for YAML parity
        (``{system, output_key, use_tools, ...}``), a :class:`Node` /
        :class:`~teff.flow.SubFlow` used as-is, a :class:`Flow` embedded as a
        :class:`SubFlow`, or a list of nodes run in sequence for that route.

        The leader decider inherits *model*/*provider* (or the flow's
        ``default_model``/``default_provider``).  *route_keys* defaults to
        ``{role: output_key}``; *done_keys*/*done_mode*/*fallback*/``max_rounds``
        drive the built-in safety guards (see :class:`Supervisor`).  When the
        decider replies ``finish`` the flow continues through *finish* (or
        terminates when omitted).  *id* names the supervisor node.

        Returns ``self`` for chaining.
        """
        from teff.flow.agent import AgentRole
        from teff.flow.sub_flow import SubFlow
        from teff.node.supervisor import Supervisor

        flow = self.flow
        flow._check_continuation()
        if not isinstance(roles, dict) or not roles:
            raise ValueError("team requires a non-empty `roles` mapping")
        model = model or flow._default_model
        provider = provider or flow._default_provider
        if not model or not provider:
            raise ValueError(
                "team requires `model` and `provider` (or `default_model`/"
                "`default_provider` on the flow)"
            )

        if route_keys is None:
            route_keys = {}
            for role_name, spec in roles.items():
                if isinstance(spec, AgentRole):
                    out = spec.output_key or role_name
                elif isinstance(spec, dict):
                    out = spec.get("output_key") or role_name
                else:
                    out = role_name
                route_keys[role_name] = out

        def build_role(spec):
            """Normalize one role spec into node(s) for ``route()``.

            An ``AgentRole`` / recipe dict / ``Flow`` becomes a single node
            (``SubFlow`` for agents/flow); a plain ``Node`` is used as-is; a
            list (a route chain such as agent → interrupt) is normalized
            element-wise so roles can be mixed with plain nodes.
            """
            if isinstance(spec, list):
                return [build_role(item) for item in spec]  # type: ignore[misc]
            if isinstance(spec, AgentRole):
                return spec.build(model=model, provider=provider, id=role_name)
            if isinstance(spec, dict):
                return AgentRole.from_mapping(spec, name=role_name).build(
                    model=model, provider=provider, id=role_name
                )
            if isinstance(spec, self.flow.__class__):
                return SubFlow(graph=spec.compile())
            return spec

        agents: dict = {}
        for role_name, spec in roles.items():
            if isinstance(spec, list):
                chain = build_role(spec)
                agents[role_name] = flow._as_chain(chain)
            else:
                agents[role_name] = build_role(spec)

        supervisor = Supervisor(
            system=system,
            model=model,
            provider=provider,
            messages_key=messages_key,
            sections=sections,
            route_keys=route_keys,
            done_keys=set(done_keys or ()),
            done_mode=done_mode,
            fallback_agent=fallback,
            max_rounds=max_rounds,
        )
        self.flow.supervisor(supervisor, id=id)
        self.flow.route("next_agent", finish=finish, **agents)
        return self.flow
