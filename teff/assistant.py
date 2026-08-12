"""Durable conversation turns against a compiled graph.

:class:`Assistant` is the app-facing service object for one durable
conversation: it holds the compiled :class:`~teff.graph.Graph`, its tools,
the :class:`~teff.checkpoint.Checkpointer`, and the conversation state shape
(reducers, fresh-session seed, transient keys, messages key), then exposes a
single interrupt-aware entry point:

* :meth:`Assistant.run`    — one turn, returns a :class:`TurnResult`
  (a pause is folded into ``waiting=True`` instead of raised).
* :meth:`Assistant.stream` — the streaming equivalent, yielding
  :class:`~teff.stream.StreamEvent` objects.

Both auto-detect a paused session from durable state and either resume it
with the message (the operator's answer) or start/continue the conversation.
The turn machinery itself lives on :class:`~teff.graph.Graph`
(``run(message=...)`` / ``stream(message=...)``); :class:`Assistant` just
binds the settings so apps call one object.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Mapping, Sequence

from teff.checkpoint import DEFAULT_OWNER, Checkpointer
from teff.graph import Graph, TurnResult
from teff.stream import StreamEvent
from teff.tool import McpToolGroup, Tool


class Assistant:
    """Runs durable conversation turns against a compiled graph."""

    def __init__(
        self,
        graph: Graph,
        tools: "Sequence[Tool | McpToolGroup]",
        checkpointer: Checkpointer,
        *,
        reducers: dict | None = None,
        initial_state: Callable[[], Mapping[str, object]] | None = None,
        transient_keys: tuple[str, ...] = (),
        messages_key: str = "messages",
        max_iterations: int = 80,
    ):
        self.graph = graph
        self.tools = tools
        self.checkpointer = checkpointer
        self.reducers = reducers
        self.initial_state = initial_state
        self.transient_keys = transient_keys
        self.messages_key = messages_key
        self.max_iterations = max_iterations

    async def run(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> TurnResult:
        """Run one turn, resuming a paused session transparently.

        This is the single entry point apps call::

            result = await assistant.run(session_id, message)
            if result.waiting:
                # surface result.prompt to the operator, await their answer
                ...                       # and call run() again with it
            else:
                print(result.reply)

        * If the session is paused on an interrupt (:meth:`pending`), *message*
          is the operator's answer and the run resumes from the checkpoint.
        * Otherwise *message* starts (or continues) the conversation.
        * A pause is **not** raised to the caller: it is folded into the
          returned :class:`TurnResult` (``waiting=True`` with the prompt and
          key), so the loop above keeps working across an arbitrary number of
          interrupts (e.g. a "rework" branch that re-asks).
        """
        return await self.graph.run(
            {},
            tools=self.tools,
            reducers=self.reducers,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
            message=message,
            initial_state=self.initial_state,
            transient_keys=self.transient_keys,
            messages_key=self.messages_key,
        )

    async def stream(
        self,
        session_id: str,
        message: str,
        *,
        owner: str = DEFAULT_OWNER,
        max_iterations: int | None = None,
        tracer=None,
        on_llm_payload=None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn, resuming a paused session transparently.

        The streaming equivalent of :meth:`run`.  Relays the underlying
        ``graph.stream(message=...)`` events verbatim; a paused session
        auto-resumes with the message, and a re-work pause surfaces an
        ``interrupt`` event (with ``key`` and the question in ``question``,
        mirrored under ``prompt`` for compatibility) where the stream ends —
        call this again with the operator's answer to continue.
        """
        async for event in self.graph.stream(
            state={},
            tools=self.tools,
            reducers=self.reducers,
            checkpointer=self.checkpointer,
            checkpoint_id=session_id,
            owner=owner,
            max_iterations=max_iterations or self.max_iterations,
            tracer=tracer,
            on_llm_payload=on_llm_payload,
            message=message,
            initial_state=self.initial_state,
            transient_keys=self.transient_keys,
            messages_key=self.messages_key,
        ):
            yield event

    async def last_reply(self, session_id: str, *, owner: str = DEFAULT_OWNER) -> str:
        """Return the latest assistant reply for *session_id* (``""`` if none).

        Reads the durable checkpoint, so it works even for agents that do
        not stream tokens (e.g. tool-using agents): the CLI prints this at
        the end of a turn instead of relying on ``token`` events alone.
        """
        return await self.graph.last_reply(
            session_id,
            checkpointer=self.checkpointer,
            messages_key=self.messages_key,
            owner=owner,
        )

    async def pending(
        self, session_id: str, *, owner: str = DEFAULT_OWNER
    ) -> dict | None:
        """Return the interrupt this session is paused on, or ``None``.

        The interrupt bookkeeping lives in durable state: when ``graph.run``
        pauses on an :class:`~teff.node.Interrupt` it writes a ``__interrupt__``
        entry into the saved checkpoint.  This reads it back so the caller —
        without a try/except or an in-memory ``pending`` map — can tell whether
        the next message is a fresh turn or the operator's answer to resume.
        """
        return await self.graph.pending(
            session_id, checkpointer=self.checkpointer, owner=owner
        )

    async def get_state(
        self, session_id: str, *, owner: str = DEFAULT_OWNER
    ) -> dict | None:
        """Return the durable conversation state for *session_id*.

        Reads the latest checkpoint (paused or completed).  The internal
        ``__interrupt__`` bookkeeping key is stripped — use
        :meth:`pending` to inspect a paused run's interrupt.
        """
        return await self.graph.get_state(
            session_id, checkpointer=self.checkpointer, owner=owner
        )

    async def update_state(
        self,
        session_id: str,
        values: dict,
        *,
        owner: str = DEFAULT_OWNER,
        as_node: str | None = None,
    ) -> dict:
        """Edit the durable state of a session and persist it.

        The HITL "fix the data then resume" primitive: override the given
        keys, save the checkpoint, then the next :meth:`run` continues
        from where the session paused with the edited state.
        """
        return await self.graph.update_state(
            session_id,
            values,
            checkpointer=self.checkpointer,
            owner=owner,
            as_node=as_node,
        )
