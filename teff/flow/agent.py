"""High-level agent helpers built on :class:`~teff.flow.Flow`.

:func:`agent_step` is the shared recipe behind every routed agent in the
scaffolds and examples: compose an ``input`` from shared state, run a
ReAct harness (LLM + optional tools) into an output slot, then append the
final reply to the shared conversation.  Wrapped as a
:class:`~teff.flow.SubFlow`, it plugs straight into
:meth:`teff.flow.Flow.route`.
"""

from __future__ import annotations

from teff.flow.flow import Flow
from teff.flow.sub_flow import SubFlow
from teff.node.context import AppendAssistant, ContextBuilder


def agent_step(
    system: str,
    output_key: str,
    *,
    model: str,
    provider: str,
    sections: dict[str, str] | None = None,
    messages_key: str = "messages",
    use_tools: str | list[str] | None = None,
    stream: bool = True,
    id: str | None = None,
    **config,
) -> SubFlow:
    """One routed agent: context builder → ReAct harness → append to conversation.

    Builds a small ``Flow`` wrapped as a :class:`~teff.flow.SubFlow`::

        ContextBuilder ──► ReAct harness ──► AppendAssistant

    * The context builder composes a plain-text ``input`` from the shared
      state sections (plus the latest user message) and resets the agent's
      scratch keys, so each run starts clean.
    * The harness runs the model against that ``input`` with *use_tools*,
      writing its final answer to *output_key*.
    * ``AppendAssistant`` copies that answer into the shared conversation.

    The agent's scratch conversation lives in a private ``_<output_key>_messages``
    state slot (reset by the context builder); only the final reply reaches
    *messages_key*.

    Args:
        system: System prompt for the agent.
        output_key: State key that receives the agent's final answer.
        model: LLM model name (e.g. ``llama3.1:8b``).
        provider: Provider name (e.g. ``ollama``).  Must be declared in the
            ``providers`` of the enclosing flow/workflow — this sub-flow
            inherits its provider set from the parent at run time.
        sections: Shared state key → label mapping rendered into the agent's
            context.  Defaults to ``{output_key: output_key.capitalize()}``.
        messages_key: State key holding the shared conversation.
        use_tools: ``None``/``[]`` (no tools, default), ``"all"`` (everything
            the pool offers), or a list of tool names the agent may call.
            Prefer an explicit allowlist; the ``True``/``False`` bool
            shorthands are kept only for backwards compatibility.
        stream: Emit tokens as stream events (live rendering).
        **config: Extra kwargs for the ReAct harness / ``ToolExec``.

    Returns:
        A ``SubFlow`` node usable with :meth:`teff.flow.Flow.route` /
        :meth:`teff.flow.Flow.step`.
    """
    scratch_key = f"_{output_key}_messages"
    inner = Flow(f"agent-{output_key}")
    inner.step(
        ContextBuilder(
            sections=sections or {output_key: output_key.capitalize()},
            reset_keys=(output_key, "input", scratch_key),
        )
    )
    inner.harness(
        model=model,
        system=system,
        input_key="input",
        output_key=output_key,
        messages_key=scratch_key,
        use_tools=use_tools,
        provider=provider,
        stream=stream,
        **config,
    )
    inner.step(AppendAssistant(output_key=output_key, messages_key=messages_key))
    return SubFlow(inner.compile(), id_prefix=id or "")


#: Backwards-compatible alias — ``agent_step`` is the preferred name.
agent_chain = agent_step


class AgentRole:
    """One routed agent role for :meth:`teff.flow.Flow.team`.

    Describes *how* a role performs its slot: the system prompt, the state
    key that receives its final answer, and optional model/provider/tool
    knobs.  :meth:`build` turns it into a :class:`~teff.flow.SubFlow`
    (the ``agent_step`` recipe) that plugs straight into the team's
    supervisor route loop::

        flow.team(
            "Route to the coder, then finish.",
            roles={
                "coder": AgentRole("You write code.", output_key="code"),
                "planner": AgentRole(
                    "You plan.", output_key="plan", use_tools=["web_search"]
                ),
            },
        )

    Args:
        system: System prompt for the role.
        output_key: State key that receives the role's final answer.
        model: Model override; when omitted the team/flow default is used.
        provider: Provider override; when omitted the team/flow default is
            used.
        sections: Shared state key → label mapping rendered into the agent's
            context (defaults to ``{output_key: Capitalized}``).
        messages_key: State key holding the shared conversation.
        use_tools: ``None``/``[]`` (no tools), ``"all"``, or an allowlist of
            tool names the role may call.
        stream: Emit tokens as stream events (live rendering).
        **config: Extra kwargs forwarded to the ReAct harness.
    """

    def __init__(
        self,
        system: str = "",
        *,
        output_key: str,
        model: str | None = None,
        provider: str | None = None,
        sections: dict[str, str] | None = None,
        messages_key: str = "messages",
        use_tools: str | list[str] | None = None,
        stream: bool = True,
        **config,
    ):
        self.system = system
        self.output_key = output_key
        self.model = model
        self.provider = provider
        self.sections = sections
        self.messages_key = messages_key
        self.use_tools = use_tools
        self.stream = stream
        self.config = dict(config)

    def build(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        id: str = "",
    ) -> SubFlow:
        """Render the role as a routed ``agent_step`` SubFlow.

        *model*/*provider* fall back from the arguments (the team/flow
        defaults) to the role's own overrides.  Raises ``ValueError`` when
        neither provides them.
        """
        model = model or self.model
        provider = provider or self.provider
        if not model or not provider:
            raise ValueError(
                f"AgentRole {self.output_key!r} needs model and provider "
                "(set them on the role or on the team/flow)"
            )
        return agent_step(
            self.system,
            self.output_key,
            model=model,
            provider=provider,
            sections=self.sections,
            messages_key=self.messages_key,
            use_tools=self.use_tools,
            stream=self.stream,
            id=id,
            **self.config,
        )

    @classmethod
    def from_mapping(cls, data: dict, *, name: str = "") -> "AgentRole":
        """Build a role from a ``{system, output_key, ...}`` mapping.

        Normalises the YAML ``team.roles`` entries (and plain dicts passed
        to :meth:`Flow.team`) into an :class:`AgentRole`.  ``tools:`` is
        accepted as the legacy alias for ``use_tools:``; any other keys are
        kept and forwarded to the ReAct harness via :meth:`build`.
        """
        data = dict(data)
        out_key = data.get("output_key") or name
        if not out_key:
            raise ValueError("a team role needs an `output_key` (or a role name)")
        system = data.pop("system", "") or ""
        known = {
            "model",
            "provider",
            "sections",
            "messages_key",
            "use_tools",
            "stream",
        }
        kwargs = {k: data.pop(k) for k in list(data) if k in known}
        if "tools" in data and "use_tools" not in kwargs:
            kwargs["use_tools"] = data.pop("tools")
        if data:
            kwargs["config"] = data
        return cls(system, output_key=out_key, **kwargs)
