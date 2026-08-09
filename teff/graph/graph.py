"""Graph data structure for representing agent workflows."""

from __future__ import annotations

import asyncio
import time
import typing
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
    cast,
    overload,
)

from teff.checkpoint import DEFAULT_OWNER, Checkpoint, Checkpointer
from teff.graph.edge import _INTERRUPT_KEY, Edge, Hook
from teff.graph.execution import execute
from teff.graph.render import to_mermaid
from teff.node.interrupt import GraphInterrupt
from teff.node.node import Node
from teff.node.registry import NodeRegistry
from teff.provider import (
    Provider,
    ProviderRegistry,
    to_provider_registry,
)
from teff.state import Reducer, State
from teff.stream import StreamEvent
from teff.tool.tool import Tool
from teff.trace import RunTracer, _ms

if typing.TYPE_CHECKING:
    from teff.tool.mcp import McpToolGroup

__all__ = ["Edge", "Graph", "Hook", "TurnResult"]


@dataclass
class TurnResult:
    """Structured outcome of one interrupt-aware conversation turn.

    Attributes:
        session_id: The session the turn ran against.
        reply: The latest assistant reply (``""`` when the turn is paused
            before any assistant text, e.g. ``waiting`` after a pure gate).
        waiting: ``True`` when the run paused on an
            :class:`~teff.node.Interrupt` and needs an operator answer.
        prompt: The interrupt's question (only when ``waiting``).
        key: The interrupt's state key (only when ``waiting``).
        state: Final state for a completed turn (``None`` when ``waiting``).
    """

    session_id: str
    reply: str = ""
    waiting: bool = False
    prompt: str | None = None
    key: str | None = None
    state: dict | None = None


class Graph:
    """A directed graph of nodes connected by edges with conditions.

    The graph executes by walking from the *entry_point* node,
    following edges whose conditions match the current state,
    and shallow-merging each node's output back into the state.

    Error handling::

        Edge("parse", "fallback", "__error__")   # catch exceptions

    Observability hooks::

        await graph.run(state, hooks={
            "on_node_start": callback,
            "on_node_end": callback,
            "on_node_error": callback,
        })

    Hook callbacks receive ``(node_id, node, state)``.
    ``on_node_end`` additionally receives the result dict and runs *after*
    the result is merged into state (so it observes the node's effect).
    ``on_node_error`` additionally receives the exception.
    Hooks may be sync or async; async hooks are awaited.
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        entry_point: str,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.providers: ProviderRegistry = to_provider_registry(providers)
        self.default_provider: str | None = default_provider
        self.default_model: str | None = default_model
        self._tool_groups: dict[str, Any] = {}

    async def _expand_tools(
        self, tools: "Sequence[Any] | None"
    ) -> "list[Tool | McpToolGroup] | None":
        """Open any MCP tool groups in *tools* once and return their members.

        Groups are keyed by server id and cached on the graph, so repeated
        ``run``/``stream`` calls (daemon ticks, conversation turns) reuse
        the same connection instead of re-spawning the server.  The cached
        connections are closed by :meth:`aclose`.
        """
        if not tools:
            return None if tools is None else list(tools)
        expanded: "list[Tool | McpToolGroup]" = []
        for tool in tools:
            group = getattr(tool, "is_mcp_group", None)
            if group:
                entry = self._tool_groups.get(group.id)
                if entry is None:
                    members = await group.open()
                    self._tool_groups[group.id] = (group, members)
                else:
                    members = entry[1]
                expanded.extend(members)
            else:
                expanded.append(tool)
        return expanded

    async def aclose(self) -> None:
        """Close connection-backed tools (e.g. MCP servers) opened by this
        graph.  Idempotent; safe to call after a partial or cancelled run.

        Conveniently, the graph is also an async context manager, so
        ``async with graph:`` closes everything on exit::

            async with graph:
                result = await graph.run(state, tools=tools)
        """
        for group, _members in self._tool_groups.values():
            await group.aclose()
        self._tool_groups.clear()

    async def __aenter__(self) -> "Graph":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    @overload
    async def run(
        self,
        state: dict | State,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = DEFAULT_OWNER,
        resume: dict | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
        *,
        message: None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
    ) -> "dict | State":
        """Plain-run overload; see the full :meth:`run` implementation."""
        ...

    @overload
    async def run(
        self,
        state: dict | State,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = DEFAULT_OWNER,
        resume: dict | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
        *,
        message: str,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
    ) -> TurnResult:
        """Conversation-turn overload; see the full :meth:`run` implementation."""
        ...

    async def run(
        self,
        state: dict | State,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = DEFAULT_OWNER,
        resume: dict | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
        *,
        message: str | None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
    ) -> "TurnResult | dict | State":
        """Execute the graph starting from the entry point.

        A single entry point for both plain workflows and durable,
        interrupt-aware conversation turns:

        * **Plain run** (no *message*): walk the graph from the entry
          point and return the final state.  When an ``Interrupt`` node is
          reached the run pauses and raises
          :class:`~teff.node.interrupt.GraphInterrupt`; resume by calling
          ``run`` again with the same *checkpoint_id* and a ``resume``
          dict mapping the interrupt's *key* to the operator's answer.
        * **Conversation turn** (with *message*): the run is driven as
          one durable turn against a session — *checkpoint_id* is the
          session id.  If the session is paused on an interrupt, *message*
          is the operator's answer and the run auto-resumes from the
          checkpoint; otherwise *message* starts (or continues) the
          conversation (seeded via *initial_state*, appended to the
          ``messages_key`` list).  A pause is **not** raised: it is folded
          into the returned :class:`TurnResult` (``waiting=True`` with the
          prompt and key), so the same loop works across any number of
          interrupts.  This is the same primitive the
          :class:`~teff.assistant.Assistant` wrapper exposes as
          ``run``/``stream``.

        Args:
            state: Initial workflow state (plain ``dict`` or :class:`State`).
            tools: Optional list of Tool instances available to nodes.
            providers: Optional ``{name: Provider}`` map or
                :class:`~teff.provider.ProviderRegistry` consulted by LLM
                nodes before the built-in presets.  Defaults to
                ``graph.providers`` (populated from a workflow's
                ``providers:`` block when loaded from YAML).
            default_provider: Optional default provider name used by LLM
                nodes that don't set ``provider`` themselves.  Defaults to
                ``graph.default_provider`` (``Graph(default_provider=...)``
                or a workflow's top-level ``default_provider:``).
            registry: Node registry (defaults to ``default_registry``).
            reducers: Per-key merge strategies
                (see :func:`teff.state.reducers_from_typeddict`).
                Ignored when *state* is a :class:`State` instance.
            hooks: Observability hooks (see class docstring).
            node_timeout: Max seconds per node.  ``asyncio.TimeoutError``
                triggers error edges (``__error__``) like any other exception.
            max_iterations: Max total node executions before raising
                ``RuntimeError``.  Guards against infinite loops in
                cyclic graphs (e.g. agentic loops).  ``None`` means unlimited.
            checkpointer: Optional persistence backend.  When set, a
                checkpoint is written before each node execution, so a
                crashed or interrupted run can be resumed by calling
                ``run`` again with the same *checkpoint_id* and the new
                initial state ignored in favor of the saved one.
            checkpoint_id: Key identifying a run (e.g. ``"thread-1"``).
                Required when *checkpointer* is set.  On a fresh ID the
                graph starts from *state* at the entry point; on an
                existing ID it resumes from the saved checkpoint.  With
                *message* this is the conversation session id.
            owner: Scopes *checkpoint_id* to a user/session/tenant.  The
                same ID under different owners never collides, and
                ``checkpointer.list(owner)`` enumerates a user's runs.
                Use one owner per end-user so every tenant's conversations
                stay isolated.  Defaults to
                :data:`teff.checkpoint.DEFAULT_OWNER`.
            resume: When a :class:`~teff.node.interrupt.Interrupt` node
                paused the run, pass a dict of ``{key: value}`` answers.
                Each key is written into the state before execution
                continues past the interrupt.  ``None`` on a normal run.
                Ignored when *message* is given.
            tracer: Optional :class:`~teff.trace.RunTracer` collecting
                an event log for this run — timeline, node latency,
                retries, checkpoint activity, and LLM token usage.
                Inspect ``tracer.events`` / ``tracer.summary()`` after
                the run completes.
            emit: Optional async sink receiving
                :class:`~teff.stream.StreamEvent` objects as the run
                progresses.  Behaves like :meth:`stream` (emitting a
                final ``run_end`` event) but returns the final state
                instead of yielding events; used by nodes such as
                :class:`~teff.flow.sub_flow.SubFlow` to forward nested
                events, or for programmatic streaming.
            state_schema: Optional YAML ``state.schema`` dict.  When set,
                *state* is validated against it before execution and a
                :class:`~teff.errors.ConfigError` is raised on mismatch.
                See :func:`teff.state.validate_state`.
            message: Operator message for one durable conversation turn
                (see above).  ``None`` runs the plain workflow.
            initial_state: Fresh-session seed for conversation turns
                (only used with *message*).
            transient_keys: Per-turn scratch keys cleared at the start of
                each conversation turn (only used with *message*).
            messages_key: Name of the messages list in the conversation
                state (only used with *message*).

        Raises:
            RuntimeError: If *max_iterations* is exceeded.
            GraphInterrupt: When an ``Interrupt`` node is reached on a
                plain run (no *message*).  The exception carries
                ``key``/``prompt`` for the operator; resume by calling
                ``run`` again with the same *checkpoint_id* and a
                ``resume`` dict.

        Returns:
            Final state (same type as passed in) on a plain run, or a
            :class:`TurnResult` for a conversation turn.
        """
        expanded_tools = cast("list[Tool] | None", await self._expand_tools(tools))
        if message is not None:
            if checkpointer is None or checkpoint_id is None:
                raise ValueError(
                    "run(message=...) requires checkpointer and checkpoint_id"
                )
            return await self._conversation_turn(
                checkpoint_id,
                message,
                checkpointer=checkpointer,
                tools=cast("list[Tool | McpToolGroup] | None", expanded_tools),
                reducers=reducers,
                initial_state=initial_state,
                transient_keys=transient_keys,
                messages_key=messages_key,
                owner=owner,
                max_iterations=max_iterations,
                tracer=tracer,
                on_llm_payload=on_llm_payload,
                state_schema=state_schema,
                emit=emit,
                providers=providers if providers is not None else self.providers,
                default_provider=default_provider
                if default_provider is not None
                else self.default_provider,
                default_model=default_model
                if default_model is not None
                else self.default_model,
            )
        started = time.monotonic()
        try:
            result = await execute(
                self,
                state,
                tools=expanded_tools,
                registry=registry,
                reducers=reducers,
                hooks=hooks,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                owner=owner,
                resume=resume,
                tracer=tracer,
                state_schema=state_schema,
                emit=emit,
                providers=providers if providers is not None else self.providers,
                default_provider=default_provider
                if default_provider is not None
                else self.default_provider,
                default_model=default_model
                if default_model is not None
                else self.default_model,
                on_llm_payload=on_llm_payload,
            )
        except GraphInterrupt:
            if tracer is not None:
                tracer.run_end("interrupted", _ms(started))
            if emit is not None:
                await emit(
                    StreamEvent(
                        "run_end",
                        data={"status": "interrupted", "total_ms": _ms(started)},
                    )
                )
            raise
        except Exception as exc:
            if tracer is not None:
                tracer.run_end("error", _ms(started), exc)
            if emit is not None:
                await emit(
                    StreamEvent("run_end", data={"status": "error", "error": str(exc)})
                )
            raise
        if tracer is not None:
            tracer.run_end("ok", _ms(started))
        if emit is not None:
            await emit(
                StreamEvent("run_end", data={"status": "ok", "total_ms": _ms(started)})
            )
        return result

    async def stream(
        self,
        state: dict | State,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        registry: NodeRegistry | None = None,
        reducers: dict[str, Reducer] | None = None,
        hooks: dict[str, Callable] | None = None,
        node_timeout: float | None = None,
        max_iterations: int | None = None,
        checkpointer: Checkpointer | None = None,
        checkpoint_id: str | None = None,
        owner: str = DEFAULT_OWNER,
        resume: dict | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
        *,
        message: str | None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
    ) -> AsyncIterator[StreamEvent]:
        """Stream events as the graph executes.

        Behaves like :meth:`run` but yields a :class:`StreamEvent` for
        each observable step instead of returning a final state::

            async for event in graph.stream(state):
                if event.type == "token":
                    print(event.data["token"], end="", flush=True)

        Event types: ``run_start``, ``node_start``, ``node_end``,
        ``node_error``, ``edge``, ``token``, ``llm``, ``structured``,
        ``interrupt``, ``interrupt_resume``, ``checkpoint``, and a final
        ``run_end``.
        Token events are only emitted when the running LLM node streams
        (any node with no tool calls streams automatically in this mode).

        A run paused at an ``Interrupt`` node ends with an ``interrupt``
        event followed by a ``run_end`` event with ``status: "interrupted"``;
        call :meth:`stream` or :meth:`run` again
        with a ``resume`` dict and the same *checkpoint_id* to continue.
        A failed run yields a ``run_end`` event with ``status: "error"``.

        With *message* the stream is one durable conversation turn (see
        :meth:`run`): a paused session auto-resumes with the operator's
        answer, and a re-work pause surfaces an ``interrupt`` event (with
        ``key``/``prompt`` in its ``data``) where the stream ends — call
        this again with the operator's answer to continue.

        Parameters mirror :meth:`run` (including ``owner``).
        """
        expanded_tools = cast("list[Tool] | None", await self._expand_tools(tools))
        if message is not None:
            if checkpointer is None or checkpoint_id is None:
                raise ValueError(
                    "stream(message=...) requires checkpointer and checkpoint_id"
                )
            async for event in self._conversation_stream(
                checkpoint_id,
                message,
                checkpointer=checkpointer,
                tools=cast("list[Tool | McpToolGroup] | None", expanded_tools),
                reducers=reducers,
                initial_state=initial_state,
                transient_keys=transient_keys,
                messages_key=messages_key,
                owner=owner,
                max_iterations=max_iterations,
                tracer=tracer,
                on_llm_payload=on_llm_payload,
                state_schema=state_schema,
                providers=providers if providers is not None else self.providers,
                default_provider=default_provider
                if default_provider is not None
                else self.default_provider,
                default_model=default_model
                if default_model is not None
                else self.default_model,
            ):
                yield event
            return
        if checkpointer is not None and checkpoint_id is None:
            raise ValueError("checkpoint_id is required when checkpointer is set")
        queue: "asyncio.Queue[StreamEvent | None]" = asyncio.Queue()
        started = time.monotonic()

        async def _emit(event: StreamEvent) -> None:
            await queue.put(event)

        async def _runner() -> None:
            try:
                try:
                    await execute(
                        self,
                        state,
                        tools=expanded_tools,
                        registry=registry,
                        reducers=reducers,
                        hooks=hooks,
                        node_timeout=node_timeout,
                        max_iterations=max_iterations,
                        checkpointer=checkpointer,
                        checkpoint_id=checkpoint_id,
                        owner=owner,
                        resume=resume,
                        tracer=tracer,
                        state_schema=state_schema,
                        emit=_emit,
                        providers=providers
                        if providers is not None
                        else self.providers,
                        default_provider=default_provider
                        if default_provider is not None
                        else self.default_provider,
                        default_model=default_model
                        if default_model is not None
                        else self.default_model,
                        on_llm_payload=on_llm_payload,
                    )
                except GraphInterrupt:
                    if tracer is not None:
                        tracer.run_end("interrupted", _ms(started))
                    await _emit(
                        StreamEvent(
                            "run_end",
                            data={"status": "interrupted", "total_ms": _ms(started)},
                        )
                    )
                    return
                except Exception as exc:
                    if tracer is not None:
                        tracer.run_end("error", _ms(started), exc)
                    await _emit(
                        StreamEvent(
                            "run_end", data={"status": "error", "error": str(exc)}
                        )
                    )
                    return
                if tracer is not None:
                    tracer.run_end("ok", _ms(started))
                await _emit(
                    StreamEvent(
                        "run_end", data={"status": "ok", "total_ms": _ms(started)}
                    )
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _load_or_seed(
        self,
        session_id: str,
        message: str,
        *,
        checkpointer: Checkpointer,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        owner: str = DEFAULT_OWNER,
    ) -> tuple[dict, dict]:
        """Return ``(state, run_kwargs)`` for one conversation turn.

        * Fresh session  -> seed the state with the user message; ``run``
          checkpoints as it executes.
        * Existing       -> append the message to the durable state and
          re-enter at the entry point so history drives the reply.  The
          *state* we return is empty because ``run`` restores the
          just-saved checkpoint.
        """
        saved = await checkpointer.load(session_id, owner=owner)
        if saved is None:
            state: dict[str, Any] = dict(initial_state() if initial_state else {})
            state[messages_key] = [{"role": "user", "content": message}]
            return state, {}

        state = dict(saved.state)
        messages = list(state.get(messages_key) or [])
        messages.append({"role": "user", "content": message})
        state[messages_key] = messages
        for key in transient_keys:
            state[key] = ""
        await checkpointer.save(
            session_id,
            Checkpoint(state=state, next_node_id=self.entry_point, iteration=0),
            owner=owner,
        )
        return {}, {}

    async def pending(
        self,
        session_id: str,
        *,
        checkpointer: Checkpointer,
        owner: str = DEFAULT_OWNER,
    ) -> dict | None:
        """Return the interrupt this session is paused on, or ``None``.

        The interrupt bookkeeping lives in durable state: when ``run`` pauses
        on an :class:`~teff.node.Interrupt` it writes a ``__interrupt__`` entry
        into the saved checkpoint.  This reads it back so the caller — without
        a try/except or an in-memory ``pending`` map — can tell whether the
        next message is a fresh turn or the operator's answer to resume.
        """
        saved = await checkpointer.load(session_id, owner=owner)
        if saved is None:
            return None
        return saved.state.get(_INTERRUPT_KEY)

    async def last_reply(
        self,
        session_id: str,
        *,
        checkpointer: Checkpointer,
        messages_key: str = "messages",
        owner: str = DEFAULT_OWNER,
    ) -> str:
        """Return the latest assistant reply for *session_id* (``""`` if none).

        Reads the durable checkpoint, so it works even for agents that do not
        stream tokens (e.g. tool-using agents): the CLI prints this at the end
        of a turn instead of relying on ``token`` events alone.
        """
        saved = await checkpointer.load(session_id, owner=owner)
        if saved is None:
            return ""
        for message in reversed(saved.state.get(messages_key) or []):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
        return ""

    async def get_state(
        self,
        checkpoint_id: str,
        *,
        checkpointer: Checkpointer,
        owner: str = DEFAULT_OWNER,
    ) -> dict | None:
        """Return the durable state for *checkpoint_id*, or ``None``.

        Reads the latest checkpoint (paused or completed).  The internal
        ``__interrupt__`` bookkeeping key is stripped — use
        :meth:`pending` to inspect a paused run's interrupt.

        Pairs with :meth:`update_state` for human-in-the-loop
        corrections: read the state, fix a value, save it back, then
        resume with ``run(resume=...)``.
        """
        saved = await checkpointer.load(checkpoint_id, owner=owner)
        if saved is None:
            return None
        state = dict(saved.state)
        state.pop(_INTERRUPT_KEY, None)
        return state

    async def update_state(
        self,
        checkpoint_id: str,
        values: dict,
        *,
        checkpointer: Checkpointer,
        owner: str = DEFAULT_OWNER,
        as_node: str | None = None,
    ) -> dict:
        """Edit the durable state of an existing run and persist it.

        Loads the latest checkpoint for *checkpoint_id*, overrides the
        given keys with *values*, and saves it back — the HITL "fix the
        data then resume" primitive.  A run paused on an interrupt stays
        paused: the next ``run(resume=...)`` continues from the interrupt
        with the edited state.

        *as_node* is accepted for LangGraph parity but is informational —
        DraftFlow resumes from the interrupt automatically, so update
        attribution is not needed.

        Raises:
            KeyError: If *checkpoint_id* has no checkpoint yet.
        """
        saved = await checkpointer.load(checkpoint_id, owner=owner)
        if saved is None:
            raise KeyError(f"no checkpoint for {checkpoint_id!r} to update")
        state = dict(saved.state)
        state.update(values)
        await checkpointer.save(
            checkpoint_id,
            Checkpoint(
                state=state,
                next_node_id=saved.next_node_id,
                iteration=saved.iteration,
            ),
            owner=owner,
        )
        return state

    async def _conversation_turn(
        self,
        session_id: str,
        message: str,
        *,
        checkpointer: Checkpointer,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        reducers: dict[str, Reducer] | None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> TurnResult:
        """Run one durable conversation turn (see :meth:`run` with *message*)."""
        try:
            pending = await self.pending(
                session_id, checkpointer=checkpointer, owner=owner
            )
            if pending is not None:
                state = await self._resume_turn(
                    session_id,
                    {pending["key"]: message},
                    checkpointer=checkpointer,
                    tools=tools,
                    reducers=reducers,
                    owner=owner,
                    max_iterations=max_iterations,
                    tracer=tracer,
                    state_schema=state_schema,
                    emit=emit,
                    providers=providers,
                    default_provider=default_provider,
                    default_model=default_model,
                    on_llm_payload=on_llm_payload,
                )
            else:
                state = await self._run_turn(
                    session_id,
                    message,
                    checkpointer=checkpointer,
                    tools=tools,
                    reducers=reducers,
                    initial_state=initial_state,
                    transient_keys=transient_keys,
                    messages_key=messages_key,
                    owner=owner,
                    max_iterations=max_iterations,
                    tracer=tracer,
                    state_schema=state_schema,
                    emit=emit,
                    providers=providers,
                    default_provider=default_provider,
                    default_model=default_model,
                    on_llm_payload=on_llm_payload,
                )
            return TurnResult(
                session_id=session_id,
                reply=await self.last_reply(
                    session_id,
                    checkpointer=checkpointer,
                    messages_key=messages_key,
                    owner=owner,
                ),
                state=state,
            )
        except GraphInterrupt as exc:
            return TurnResult(
                session_id=session_id,
                reply=await self.last_reply(
                    session_id,
                    checkpointer=checkpointer,
                    messages_key=messages_key,
                    owner=owner,
                ),
                waiting=True,
                prompt=exc.prompt,
                key=exc.key,
            )

    async def _run_turn(
        self,
        session_id: str,
        message: str,
        *,
        checkpointer: Checkpointer,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        reducers: dict[str, Reducer] | None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> dict:
        """Run one turn against the graph, returning the final state."""
        state, run_kwargs = await self._load_or_seed(
            session_id,
            message,
            checkpointer=checkpointer,
            initial_state=initial_state,
            transient_keys=transient_keys,
            messages_key=messages_key,
            owner=owner,
        )
        return await self.run(
            state,
            tools=tools,
            reducers=reducers,
            checkpointer=checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations,
            tracer=tracer,
            state_schema=state_schema,
            emit=emit,
            providers=providers,
            default_provider=default_provider,
            default_model=default_model,
            on_llm_payload=on_llm_payload,
            **run_kwargs,
        )

    async def _resume_turn(
        self,
        session_id: str,
        resume: dict,
        *,
        checkpointer: Checkpointer,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        reducers: dict[str, Reducer] | None = None,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        emit: "Callable[[StreamEvent], Awaitable[None]] | None" = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> dict:
        """Resume a turn paused by an :class:`~teff.node.interrupt.Interrupt`.

        *resume* maps the interrupt's state key to the operator's answer,
        e.g. ``{"approved": "yes"}``.  ``run`` restores the checkpoint saved
        when the interrupt fired and continues past it; a re-interrupt (e.g. a
        "rework" branch) raises again (folded by :meth:`_conversation_turn`).
        """
        return await self.run(
            {},
            tools=tools,
            reducers=reducers,
            checkpointer=checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations,
            tracer=tracer,
            state_schema=state_schema,
            emit=emit,
            providers=providers,
            default_provider=default_provider,
            default_model=default_model,
            on_llm_payload=on_llm_payload,
            resume=resume,
        )

    async def _stream_run(
        self,
        session_id: str,
        message: str,
        *,
        checkpointer: Checkpointer,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        reducers: dict[str, Reducer] | None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the events of one conversation turn."""
        state, run_kwargs = await self._load_or_seed(
            session_id,
            message,
            checkpointer=checkpointer,
            initial_state=initial_state,
            transient_keys=transient_keys,
            messages_key=messages_key,
            owner=owner,
        )
        async for event in self.stream(
            state,
            tools=tools,
            reducers=reducers,
            checkpointer=checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations,
            tracer=tracer,
            state_schema=state_schema,
            providers=providers,
            default_provider=default_provider,
            default_model=default_model,
            on_llm_payload=on_llm_payload,
            **run_kwargs,
        ):
            yield event

    async def _stream_resume(
        self,
        session_id: str,
        resume: dict,
        *,
        checkpointer: Checkpointer,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        reducers: dict[str, Reducer] | None = None,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a resume of a paused conversation turn."""
        async for event in self.stream(
            state={},
            tools=tools,
            reducers=reducers,
            checkpointer=checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations,
            tracer=tracer,
            state_schema=state_schema,
            providers=providers,
            default_provider=default_provider,
            default_model=default_model,
            on_llm_payload=on_llm_payload,
            resume=resume,
        ):
            yield event

    async def _conversation_stream(
        self,
        session_id: str,
        message: str,
        *,
        checkpointer: Checkpointer,
        tools: "Sequence[Tool | McpToolGroup] | None" = None,
        reducers: dict[str, Reducer] | None = None,
        initial_state: "Callable[[], Mapping[str, object]] | None" = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer: RunTracer | None = None,
        state_schema: dict | None = None,
        providers: "dict[str, Provider] | ProviderRegistry | None" = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        on_llm_payload: "Callable[..., Awaitable[None]] | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one durable conversation turn (see :meth:`stream`)."""
        pending = await self.pending(session_id, checkpointer=checkpointer, owner=owner)
        source = (
            self._stream_resume(
                session_id,
                {pending["key"]: message},
                checkpointer=checkpointer,
                tools=tools,
                reducers=reducers,
                owner=owner,
                max_iterations=max_iterations,
                tracer=tracer,
                state_schema=state_schema,
                providers=providers,
                default_provider=default_provider,
                default_model=default_model,
                on_llm_payload=on_llm_payload,
            )
            if pending is not None
            else self._stream_run(
                session_id,
                message,
                checkpointer=checkpointer,
                tools=tools,
                reducers=reducers,
                initial_state=initial_state,
                transient_keys=transient_keys,
                messages_key=messages_key,
                owner=owner,
                max_iterations=max_iterations,
                tracer=tracer,
                state_schema=state_schema,
                providers=providers,
                default_provider=default_provider,
                default_model=default_model,
                on_llm_payload=on_llm_payload,
            )
        )
        async for event in source:
            yield event

    def to_yaml(self) -> str:
        """Serialize this graph to a YAML string."""
        from teff.yaml import graph_to_yaml

        return graph_to_yaml(self)

    def to_mermaid(self, show_conditions: bool = True) -> str:
        """Render this graph as a Mermaid flowchart diagram.

        Produces a ``flowchart TD`` definition: every node becomes a box
        labelled ``node_id[node.type]`` and every edge an arrow.  The entry
        point is filled blue, ``__error__`` edges are dashed and red, and
        conditional edges carry their condition as an edge label (when
        *show_conditions* is true).

        Returns:
            The Mermaid diagram as a string (no code fence).
        """
        return to_mermaid(self, show_conditions=show_conditions)
