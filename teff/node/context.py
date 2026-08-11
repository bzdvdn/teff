"""Execution context and context-building nodes for agent flows.

Two concerns live here:

* :class:`ExecContext` — the runtime handle passed to every node.
* :class:`ContextBuilder` / :class:`AppendAssistant` and
  :func:`last_user_message` — shared building blocks for routing a
  conversation turn to an agent: compose a plain-text ``input`` from state,
  then append the agent's reply back to the shared conversation.  They are
  used by :func:`teff.flow.agent_step` and by every scaffold/example.
"""

import typing
from typing import Any, Awaitable, Callable

from teff.node.node import Node
from teff.tool.tool import Tool

if typing.TYPE_CHECKING:
    from teff.provider import ProviderRegistry
    from teff.stream import StreamEvent
    from teff.trace import RunTracer


def last_user_message(messages: list) -> str:
    """Return the most recent ``user`` message from a conversation list.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        The latest user content, or ``""`` when there is none.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


class ContextBuilder(Node):
    """Compose a plain-text ``input`` for an agent from shared state.

    Renders each configured section as ``<label>:\\n<value>`` plus the latest
    user message, and clears scratch keys so a routed agent starts clean.

    Args:
        sections: State key → section label mapping.
        messages_key: State key holding the conversation.
        output_key: State key receiving the composed text.
        reset_keys: Scratch state keys to clear before the agent runs.
    """

    type = "context_builder"

    def __init__(
        self,
        config: dict | None = None,
        *,
        sections: dict[str, str] | None = None,
        messages_key: str = "messages",
        output_key: str = "input",
        reset_keys: tuple[str, ...] = (),
        **kwargs,
    ):
        merged = {
            "sections": sections or {},
            "messages_key": messages_key,
            "output_key": output_key,
            "reset_keys": list(reset_keys),
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        parts: list[str] = []
        for key, label in self.config["sections"].items():
            value = state.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value = "\n".join(str(item) for item in value)
            parts.append(f"{label}:\n{value}")
        last_user = last_user_message(state.get(self.config["messages_key"], []))
        if last_user:
            parts.append(f"User: {last_user}")
        output_key = self.config["output_key"]
        out: dict = {output_key: "\n\n".join(parts)}
        for key in self.config["reset_keys"]:
            if key != output_key:
                out[key] = []
        return out


class AppendAssistant(Node):
    """Append an agent's response to the shared conversation as assistant."""

    type = "append_assistant"

    def __init__(
        self,
        config: dict | None = None,
        *,
        output_key: str = "draft",
        messages_key: str = "messages",
        **kwargs,
    ):
        merged = {
            "output_key": output_key,
            "messages_key": messages_key,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        content = state.get(self.config["output_key"], "")
        if not content:
            return {}
        return {
            self.config["messages_key"]: [{"role": "assistant", "content": content}]
        }


class ExecContext:
    """Context available to nodes during graph execution.

    Provides access to registered tools and a placeholder for
    LLM calls (overridden by the built-in LLM node).

    Attributes:
        state: Current workflow state dict.
        tools: Dict of tool name to Tool instance.
        node_id: Graph node id of the running node.
        node_type: Node type string of the running node.
        tracer: Optional :class:`~teff.trace.RunTracer` collecting
            observability events for the current run.
        reducers: Per-key merge strategies for state updates.
        emit: Optional async sink receiving :class:`~teff.stream.StreamEvent`
            objects as the run progresses (used by ``graph.stream()``).
            ``None`` for plain ``graph.run()``.
        providers: Optional ``{name: Provider}`` map or
            :class:`~teff.provider.ProviderRegistry` for LLM nodes
            (custom providers declared in a workflow / passed to
            ``graph.run(providers=...)``).  ``None`` uses the built-in
            presets.
        default_provider: Optional default provider name for the graph.  LLM
            nodes use it when they don't set ``provider`` themselves
            (the graph-level ``Graph(default_provider=...)`` / workflow
            ``default_provider:``).
        default_model: Optional default model name for the graph.  LLM
            nodes use it when they don't set ``model`` themselves
            (the graph-level ``Graph(default_model=...)``).
        hooks: Observability hooks dict (forwarded to nested runs, e.g.
            :class:`~teff.flow.sub_flow.SubFlow`).
        node_timeout: Per-node timeout for nested runs (seconds).
        checkpointer: Optional persistence backend, forwarded to nested
            runs so interrupts inside a subflow stay resumable.
        checkpoint_id: Run key of the enclosing run, used to namespace
            nested run checkpoints.
        owner: Owner scope of the enclosing run.
        resume: Resume dict of the enclosing run, forwarded to nested runs
            so a sub-flow interrupted by human input resumes in place.
        on_llm_payload: Optional async callback receiving the raw request /
            response of every LLM call: ``(provider, model, messages,
            completion, usage, latency_ms, cached)``.  Observability layers
            (tracing, exporters) set this so node harnesses can forward the
            full payload, not just token counts.
    """

    def __init__(
        self,
        state: dict,
        tools: dict[str, Tool],
        *,
        node_id: str | None = None,
        node_type: str | None = None,
        tracer: "RunTracer | None" = None,
        reducers: dict[str, Any] | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        hooks: "dict | None" = None,
        node_timeout: float | None = None,
        checkpointer: Any = None,
        checkpoint_id: str | None = None,
        owner: str | None = None,
        resume: dict | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ):
        self.state = state
        self.tools = tools
        self.node_id = node_id
        self.node_type = node_type
        self.tracer = tracer
        self.reducers = reducers
        self.emit = emit
        self.providers = providers
        self.default_provider = default_provider
        self.default_model = default_model
        self.hooks = hooks
        self.node_timeout = node_timeout
        self.checkpointer = checkpointer
        self.checkpoint_id = checkpoint_id
        self.owner = owner
        self.resume = resume
        self.on_llm_payload = on_llm_payload

    def tool(self, name: str) -> Tool:
        """Look up a tool by name.

        Args:
            name: Tool name registered in the tool registry.

        Returns:
            Tool instance.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self.tools:
            msg = f"unknown tool: {name}"
            raise KeyError(msg)
        return self.tools[name]

    async def llm(self, model: str, messages: list) -> str:
        """Placeholder for LLM calls (not used by built-in LLM node)."""
        raise NotImplementedError("LLM provider not configured")
