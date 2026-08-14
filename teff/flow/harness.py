"""ReAct agent harness builders for :class:`~teff.flow.Flow`.

:class:`HarnessBuilder` builds the ReAct agent loop (LLM ↔ tools) used by
``Flow.harness`` / ``Flow.react``.  :class:`Flow` owns an instance and
delegates to it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teff.flow.flow import Flow


class HarnessBuilder:
    """Build a ReAct agent loop on a :class:`~teff.flow.Flow`.

    Args:
        flow: The owning ``Flow`` whose graph state is mutated.
    """

    def __init__(self, flow: "Flow"):
        self.flow = flow

    def harness(
        self,
        model=None,
        system: str = "",
        *,
        agent=None,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        reply_key: str = "reply_to_user",
        done_key: str = "_react_done",
        memory=None,
        max_tool_rounds: int = 10,
        tool_error_mode: str = "message",
        parse_text_tool_calls: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        use_tools=None,
        skills: list | None = None,
        skill_dir: str = "skills",
        id: str | None = None,
        **config,
    ):
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
        from teff.graph import Edge
        from teff.node.agent import ReActAgent, ToolExec

        target = self.flow
        target._check_continuation()

        agent_cfg = {
            "model": model,
            "system": system,
            "input_key": input_key,
            "output_key": output_key,
            "messages_key": messages_key,
            "reply_key": reply_key,
            "done_key": done_key,
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
            if model is None and target._default_model is None:
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
            reply_key=reply_key,
            output_key=output_key,
            done_key=done_key,
            **config,
        )

        target._nodes.append(agent_node)
        agent_id = target._next_id(agent_node, f"{id}/agent" if id else None)
        target._node_ids.append(agent_id)

        target._nodes.append(tool_exec)
        tool_id = target._next_id(tool_exec, f"{id}/tool" if id else None)
        target._node_ids.append(tool_id)

        target._edges.append(
            Edge(
                agent_id,
                tool_id,
                f"{agent_node.config.get('tool_call_key', '_tool_call_name')}!=",
            )
        )
        target._edges.append(Edge(tool_id, agent_id))

        if target._last_added is not None:
            target._edges.append(Edge(target._last_added, agent_id))

        target._last_added = agent_id
        target._branch_ends = []
        target._guarded_step = None
        return target

    def react(
        self,
        model=None,
        system: str = "",
        *,
        agent=None,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        memory=None,
        **config,
    ):
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
