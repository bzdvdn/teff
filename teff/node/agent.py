"""ReAct agent: graph-visible tool-calling loop."""

import asyncio
import json
import typing

from teff.harness import (
    Harness,
    execute_tool_calls,
    parse_text_tool_call,
    resolve_approval,
    tool_to_schema,
)
from teff.harness.formats import _tool_call_parts
from teff.memory.context import MemoryConfig
from teff.node.interrupt import GraphInterrupt
from teff.node.node import Node
from teff.skill import resolve_skills, scope_tools, skills_instructions
from teff.state import reducer_appends
from teff.stream import StreamEvent


def _unanswered_tool_call(messages: list[dict]) -> str | None:
    """Return the name of the first assistant tool call whose call id has no
    matching ``tool`` result in *messages* (``None`` if all are answered).

    Used by tool-call enforcement: after a resume the interrupted agent
    re-emits ``ask_human`` in its history without a result yet, so a plain
    ``ask_human`` call whose id was never answered counts as pending.
    """
    answered: set[str] = set()
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id:
                answered.add(call_id)
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            call_id = call.get("id")
            if call_id and call_id not in answered:
                return fn.get("name") or call_id
    return None


def _mentioned_tool(content: str, names: set[str]) -> str | None:
    """Return the tool name mentioned in *content* prose (``None`` if none).

    A model sometimes writes "I will call propose_plan" instead of emitting a
    real tool call; if the reply names one of the available tools we treat
    that as the expected tool and nudge it into an actual call.
    """
    lowered = (content or "").lower()
    for name in sorted(names):
        if name.lower() in lowered:
            return name
    return None


def _looks_like_question(text: str) -> bool:
    """Heuristic: whether *text* is a question addressed to the user."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith(("?", "？")):
        return True
    lowered = stripped.lower()
    markers = (
        "ответь",
        "одобряешь",
        "согласен",
        "согласны",
        "подтверди",
        "подтвердите",
        "готовы",
        "можно ли",
        "хотите",
    )
    return any(m in lowered for m in markers)


def _nudge_for(cfg: dict, *, tool: str, default: str) -> str:
    """The configured nudge text with ``{tool}`` substituted (or *default*)."""
    configured = cfg.get("force_tool_prompt")
    if configured:
        return str(configured).replace("{tool}", tool)
    return default


class ReActAgent(Node):
    """Single-step LLM node for a graph-level ReAct loop.

    Executes one LLM call, then signals any requested tools by setting
    ``state["_tool_calls"]`` (a list of ``{id, name, args}``) and a
    non-empty ``state["_tool_call_name"]``.

    When the LLM responds without calling a tool, the output key is
    set and ``_tool_call_name`` is cleared — the parent graph stops
    because no outgoing edge matches.

    Expected graph edges::

        agent  ──(_tool_call_name!=)──→  tool_exec
        tool_exec  ──(unconditional)──→  agent

    Parameters:
        model: Model name (e.g. ``gpt-4``).
        system: System prompt.
        input_key: State key for user input (default ``"input"``).
        output_key: State key for final response (default ``"output"``).
        messages_key: State key for conversation (default ``"messages"``).
        tool_call_key: Signal key (default ``"_tool_call_name"``).
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        response_format: ``{"type": "json_object"}`` etc.
        provider: Force a provider (auto-detected from model).
        base_url: Custom base URL.
        api_key_env: Custom env var name for API key.
        chat_path: Custom API path.
        auth_header: Custom auth header name.
        auth_prefix: Custom auth header prefix.
        max_tool_rounds: Round limit used by the harness loop.
        parse_text_tool_calls: Decode text-embedded tool calls.
        tool_error_mode: ``"message"`` (default) or ``"raise"``.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_retries: Extra attempts per tool call after a failure.
        max_retries: HTTP request retries (429/5xx/timeouts).
        fallbacks: Fallback model names for provider failover.
        tool_approval: Gate on tool execution — ``"auto"`` (default),
            ``"deny"``, ``"interactive"`` (ask on stdin), or a callable
            ``(name, args) -> bool | str``.  ``"pause"`` decisions pause
            the run as a :class:`~teff.node.interrupt.GraphInterrupt`.
        memory: Optional long-term memory injection — a
            :class:`~teff.memory.context.MemoryConfig` or ``{store,
            namespace, k, header}``.  ``store`` is a
            :class:`~teff.memory.base.MemoryStore` or a config dict; on
            every turn the top-*k* recalled memories for the last user
            message are prepended to the conversation as a system message.
        stream: Stream the final assistant text (tokens forwarded to
            ``on_token`` and stream events).
        on_token: Callback ``(token: str) -> None`` for streaming.

        Tool-call enforcement: by default the loop ends as soon as the
        model returns plain text, even when a tool is still expected
        (e.g. a resumed ``ask_human`` whose answer was never consumed, or
        a question that should go through ``ask_human``).  Set
        ``force_tool_rounds`` / ``force_tool_if_unanswered`` /
        ``force_tool_if_question`` to nudge the model back onto a tool
        call instead of letting it close the turn with text:

        * ``force_tool_rounds``: Extra LLM attempts after a plain-text
          reply (0 disables enforcement).  The nudge is appended as a
          system message and the model is re-invoked; only used when a
          *specific* tool is expected.
        * ``force_tool_if_unanswered``: Nudge when the conversation still
          contains an assistant ``tool_calls`` message whose call id has
          no matching ``tool`` result (default ``True``).
        * ``force_tool_if_question``: Nudge when the reply looks like a
          question to the user and ``ask_human`` is among the agent's
          tools (default ``False``).
        * ``force_tool_prompt``: Custom nudge text; ``{tool}`` is
          replaced with the expected tool name.
    """

    type = "react_agent"

    def __init__(
        self,
        config: dict | None = None,
        *,
        model: str | None = None,
        system: str = "",
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        tool_call_key: str = "_tool_call_name",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        chat_path: str | None = None,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
        use_tools: str | list[str] | None = None,
        skills: list | None = None,
        skill_dir: str = "skills",
        max_tool_rounds: int | None = None,
        parse_text_tool_calls: bool | None = None,
        tool_error_mode: str | None = None,
        tool_timeout: float | None = None,
        tool_retries: int = 0,
        max_retries: int = 2,
        fallbacks: list[str] | None = None,
        tool_approval: typing.Any = None,
        stream: bool = False,
        on_token: typing.Callable[[str], None] | None = None,
        memory: MemoryConfig | dict | None = None,
        force_tool_rounds: int = 0,
        force_tool_if_unanswered: bool = True,
        force_tool_if_question: bool = False,
        force_tool_prompt: str | None = None,
        **kwargs,
    ):
        merged = {
            "model": model,
            "system": system,
            "input_key": input_key,
            "output_key": output_key,
            "messages_key": messages_key,
            "tool_call_key": tool_call_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "provider": provider,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "chat_path": chat_path,
            "auth_header": auth_header,
            "auth_prefix": auth_prefix,
            "use_tools": use_tools,
            "skills": skills,
            "skill_dir": skill_dir,
            "max_tool_rounds": max_tool_rounds,
            "parse_text_tool_calls": parse_text_tool_calls,
            "tool_error_mode": tool_error_mode,
            "tool_timeout": tool_timeout,
            "tool_retries": tool_retries,
            "max_retries": max_retries,
            "fallbacks": fallbacks,
            "tool_approval": tool_approval,
            "stream": stream,
            "on_token": on_token,
            "memory": memory,
            "force_tool_rounds": force_tool_rounds,
            "force_tool_if_unanswered": force_tool_if_unanswered,
            "force_tool_if_question": force_tool_if_question,
            "force_tool_prompt": force_tool_prompt,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        system = cfg.get("system", "")
        input_key = cfg.get("input_key", "input")
        output_key = cfg.get("output_key", "output")
        messages_key = cfg.get("messages_key", "messages")
        tool_call_key = cfg.get("tool_call_key", "_tool_call_name")
        done_key = cfg.get("done_key", "_react_done")

        # A terminal reply (reply_to_user) already wrote the final message to
        # *output_key*; the loop-back must not re-invoke the LLM.  Short-
        # circuit: surface the message, clear the tool signals and consume
        # the done flag so the next user turn starts fresh.
        if state.get(done_key):
            result: dict = {
                output_key: state.get(output_key, ""),
                tool_call_key: "",
                "_tool_calls": [],
                done_key: False,
            }
            return result

        skills = resolve_skills(cfg)
        skill_text = skills_instructions(skills)
        if skill_text:
            system = f"{system}\n\n{skill_text}" if system else skill_text

        messages = list(state.get(messages_key, []))
        from teff.memory.context import memory_context_from_config

        block = await memory_context_from_config(cfg, state=state, ctx=ctx)
        if block:
            messages.insert(0, {"role": "system", "content": block})
        start = len(messages)
        if not messages:
            user_input = str(state.get(input_key, ""))
            if system:
                messages.append({"role": "system", "content": system})
            if user_input:
                messages.append({"role": "user", "content": user_input})

        tool_defs = [
            tool_to_schema(t) for t in scope_tools(ctx.tools, cfg, skills).values()
        ]

        harness = Harness.from_config(
            cfg,
            default_provider=getattr(ctx, "default_provider", None),
            default_model=getattr(ctx, "default_model", None),
            providers=getattr(ctx, "providers", None),
        )
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:

            async def on_llm(provider, model, prompt, completion, duration):
                tracer.llm(provider, model, prompt, completion, duration)

            harness.on_llm = on_llm

        payload_sink = getattr(ctx, "on_llm_payload", None)
        if payload_sink is not None:
            harness.on_llm_payload = payload_sink

        emit = getattr(ctx, "emit", None)
        on_token_cfg = cfg.get("on_token")

        async def token_sink(token: str) -> None:
            if on_token_cfg is not None:
                res = on_token_cfg(token)
                if asyncio.iscoroutine(res):
                    await res
            if emit is not None:
                await emit(
                    StreamEvent(
                        "token",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={
                            "token": token,
                            "provider": harness.provider_key,
                            "model": str(cfg.get("model", "")),
                        },
                    )
                )

        want_stream = bool(cfg.get("stream", False))
        if want_stream and not tool_defs:
            harness.on_token = token_sink

        reply = await harness.call(
            messages,
            tools=tool_defs or None,
            stream=want_stream and not tool_defs,
        )

        result: dict = {}

        # The graph loop (agent → tool → agent) is what repeats, so the node
        # itself must track how many times it has been visited.  Once the
        # round budget is spent we stop signalling tools even if the model
        # keeps asking, letting the loop end on this node.
        round_key = f"_react_round_{ctx.node_id}"
        round_count = int(state.get(round_key, 0)) + 1
        result[round_key] = round_count
        max_rounds = cfg.get("max_tool_rounds")
        budget_spent = max_rounds is not None and round_count > max_rounds

        if budget_spent:
            content = reply.content or ""
            messages.append({"role": "assistant", "content": content})
            result[output_key] = content
            result[tool_call_key] = ""
            result["_tool_calls"] = []
        else:
            force_rounds = int(cfg.get("force_tool_rounds", 0) or 0)
            attempts = 0
            while True:
                tool_calls = reply.message.get("tool_calls")

                if tool_calls:
                    calls: list[dict] = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        raw = fn.get("arguments", "{}")
                        if isinstance(raw, dict):
                            raw = json.dumps(raw)
                        calls.append(
                            {
                                "id": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "args": raw,
                            }
                        )
                    result[tool_call_key] = "pending"
                    result["_tool_calls"] = calls
                    messages.append(reply.message)
                    break

                content = reply.content or ""
                parse_cfg = cfg.get("parse_text_tool_calls", True)
                if parse_cfg is None:
                    parse_cfg = True
                parsed = (
                    parse_text_tool_call(content) if tool_defs and parse_cfg else None
                )
                if parsed:
                    name, args = parsed
                    result[tool_call_key] = "pending"
                    result["_tool_calls"] = [
                        {
                            "id": f"call_{len(messages)}",
                            "name": name,
                            "args": json.dumps(args),
                        }
                    ]
                    messages.append({"role": "assistant", "content": content})
                    break

                # Plain-text close.  When a tool is still expected (an
                # unanswered tool call in the history, or a question that
                # should go through ask_human), nudge the model back onto a
                # tool call instead of ending the turn — capped by
                # force_tool_rounds so it always terminates.
                nudge = self._tool_nudge(messages, content, tool_defs, cfg)
                if nudge is None or attempts >= force_rounds:
                    result[output_key] = content
                    result[tool_call_key] = ""
                    result["_tool_calls"] = []
                    messages.append({"role": "assistant", "content": content})
                    break

                attempts += 1
                messages.append({"role": "system", "content": nudge})
                reply = await harness.call(
                    messages,
                    tools=tool_defs or None,
                    stream=False,
                )

        if reducer_appends((ctx.reducers or {}).get(messages_key)):
            result[messages_key] = messages[start:]
        else:
            result[messages_key] = messages
        return result

    def _tool_nudge(
        self,
        messages: list[dict],
        content: str,
        tool_defs: list[dict],
        cfg: dict,
    ) -> str | None:
        """Return a system nudge message when the model closed the turn with
        plain text but a tool call is still expected (``None`` to accept the
        text reply).

        Two triggers, both opt-in via config:
        * an assistant ``tool_calls`` message in the history whose call id
          has no matching ``tool`` result (``force_tool_if_unanswered``);
        * a reply that reads like a question to the user while ``ask_human``
          is one of the agent's tools (``force_tool_if_question``).
        """
        names = {schemas["function"]["name"] for schemas in tool_defs}

        if cfg.get("force_tool_if_unanswered", True):
            pending = _unanswered_tool_call(messages)
            if pending is not None:
                return _nudge_for(
                    cfg,
                    tool=pending,
                    default=(
                        "You must call a tool to continue: an earlier tool "
                        f"call ('{pending}') has not been answered yet. Call "
                        "it now instead of replying with plain text."
                    ),
                )

        # The model described a tool call in prose ("I will call propose_plan")
        # instead of actually calling it — treat the mention as the expected
        # tool and nudge it into a real call.
        if cfg.get("force_tool_if_mentioned", True):
            mentioned = _mentioned_tool(content, names)
            if mentioned is not None:
                return _nudge_for(
                    cfg,
                    tool=mentioned,
                    default=(
                        "You described calling the tool "
                        f"'{mentioned}' instead of actually calling it. Call "
                        "'{tool}' now; do not describe it in plain text."
                    ),
                )

        if (
            cfg.get("force_tool_if_question", False)
            and "ask_human" in names
            and _looks_like_question(content)
        ):
            return _nudge_for(
                cfg,
                tool="ask_human",
                default=(
                    "You addressed the user with a question, but questions "
                    "must go through the 'ask_human' tool so the run can "
                    "pause for their answer. Call ask_human with your "
                    "question instead of replying with plain text."
                ),
            )

        return None


class ToolExec(Node):
    """Executes tools signalled by :class:`ReActAgent` in parallel and feeds
    the results back into the conversation history.

    Handles multiple tool calls per round: the agent writes the whole
    ``_tool_calls`` list, which is executed concurrently and appended as
    ``tool`` messages in one go.  Falls back to the legacy single-call
    signals (``_tool_call_name`` / ``_tool_call_args`` / ``_tool_call_id``).

    Parameters:
        messages_key: State key for messages (default ``"messages"``).
        tool_call_key: Signal key (default ``"_tool_call_name"``).
        tool_error_mode: ``"message"`` (default) or ``"raise"`` — when
            ``"raise"``, a tool failure propagates to the graph error path
            (e.g. an ``__error__`` edge) instead of becoming a tool message.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_retries: Extra attempts per tool call after a failure.
        tool_approval: Gate on tool execution — ``"auto"`` (default),
            ``"deny"``, ``"interactive"`` (ask on stdin), or a callable
            ``(name, args) -> bool | str`` (sync or async).  A ``"pause"``
            decision pauses the run as a :class:`GraphInterrupt`; ``"deny"``
            short-circuits the call with a "denied" tool message.
        human_key: State key / tool name of the human-in-the-loop
            question (default ``"ask_human"``).  A call to this tool is
            intercepted *before* execution: without a pending reply it
            pauses the run as a :class:`GraphInterrupt` carrying the
            question; with a reply in the state (from ``resume``) it
            consumes it and returns it as the tool result.
        reply_key: Tool name of the terminal plain-text reply
            (default ``"reply_to_user"``).  A call to this tool is
            intercepted *before* execution: its ``message`` argument is
            written to *output_key* and the ReAct loop is short-circuited
            (the message becomes the agent's final reply).
        output_key: State key the terminal reply's message is written to
            (default ``"output"``).
        done_key: State key set to ``True`` when a terminal reply is
            intercepted, so the agent node skips the LLM call on the next
            loop-back (default ``"_react_done"``).
    """

    type = "tool_exec"

    def __init__(
        self,
        config: dict | None = None,
        *,
        messages_key: str = "messages",
        tool_call_key: str = "_tool_call_name",
        tool_error_mode: str = "message",
        tool_timeout: float | None = None,
        tool_retries: int = 0,
        tool_approval: typing.Any = None,
        use_tools: str | list[str] | None = None,
        skills: list | None = None,
        skill_dir: str = "skills",
        reply_key: str = "reply_to_user",
        output_key: str = "output",
        done_key: str = "_react_done",
        **kwargs,
    ):
        merged = {
            "messages_key": messages_key,
            "tool_call_key": tool_call_key,
            "tool_error_mode": tool_error_mode,
            "tool_timeout": tool_timeout,
            "tool_retries": tool_retries,
            "tool_approval": tool_approval,
            "use_tools": use_tools,
            "skills": skills,
            "skill_dir": skill_dir,
            "reply_key": reply_key,
            "output_key": output_key,
            "done_key": done_key,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        messages_key = self.config.get("messages_key", "messages")
        tool_call_key = self.config.get("tool_call_key", "_tool_call_name")
        tool_error_mode = self.config.get("tool_error_mode", "message")
        tool_timeout = self.config.get("tool_timeout")
        tool_retries = int(self.config.get("tool_retries", 0))
        approver = self.config.get("tool_approval")
        human_key = self.config.get("human_key", "ask_human")
        reply_key = self.config.get("reply_key", "reply_to_user")
        output_key = self.config.get("output_key", "output")
        done_key = self.config.get("done_key", "_react_done")

        calls = list(state.get("_tool_calls") or [])
        if not calls and state.get(tool_call_key):
            calls = [
                {
                    "id": state.get("_tool_call_id", ""),
                    "name": state.get(tool_call_key, ""),
                    "args": state.get("_tool_call_args", "{}"),
                }
            ]

        skills = resolve_skills(self.config)
        scoped = scope_tools(ctx.tools, self.config, skills)

        # Human-in-the-loop: an ask_human call pauses the run for an
        # operator's answer.  On resume the answer arrives in the state under
        # *human_key* (the resume dict key), is consumed here and delivered
        # back as the tool result — the same re-invoke pattern as the
        # tool_approval "pause" decision.  The call never reaches
        # execute_tool_calls (AskHuman.arun raises NotImplementedError).
        # Terminal reply: a reply_to_user call ends the loop.  The message
        # argument is written straight to *output_key* and the loop is
        # short-circuited (the agent node skips the LLM on its next visit).
        reply_done = False
        ask_replies: dict[str, str] = {}
        normal_calls: list[dict] = []
        for call in calls:
            name, raw_args, _call_id = _tool_call_parts(call)
            if name == reply_key:
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                message = str(args.get("message", ""))
                state[output_key] = message
                state[done_key] = True
                reply_done = True
            elif name == human_key:
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                question = str(args.get("question", ""))
                pending = state.get(human_key)
                if pending is None:
                    raise GraphInterrupt(
                        key=human_key,
                        prompt=question or "The agent is asking for your input.",
                    )
                state.pop(human_key, None)  # consume the operator's answer
                ask_replies[call.get("id", "")] = str(pending)
            else:
                normal_calls.append(call)

        # After a pause/interrupt, the operator's decision comes back in the
        # resume payload under the interrupt key; use it instead of re-asking.
        resumed = state.get("tool_approval")
        resumed = resumed if resumed in ("approve", "deny") else None

        # A terminal reply ends the round: the message is already in the
        # state, so no other tool in this round should run.
        if reply_done:
            normal_calls = []
        to_run = normal_calls
        denied: list[tuple[str, str, str]] = []
        if approver is not None and approver != "auto" and normal_calls:
            to_run = []
            for call in normal_calls:
                name = call.get("name", "")
                try:
                    args = (
                        json.loads(call.get("args", "{}")) if call.get("args") else {}
                    )
                except json.JSONDecodeError:
                    args = {}
                if resumed is not None:
                    decision = resumed
                else:
                    decision = await resolve_approval(approver, name, args)
                if decision == "pause":
                    raise GraphInterrupt(
                        key="tool_approval",
                        prompt=(
                            f"Approve tool call '{name}' with args {json.dumps(args)}?"
                        ),
                    )
                if decision != "approve":
                    denied.append((name, call.get("id", ""), decision))
                else:
                    to_run.append(call)

        results = await execute_tool_calls(
            to_run,
            scoped,
            tool_error_mode,
            tool_timeout,
            tool_retries,
            state=state,
            ctx=ctx,
        )

        result_by_id = {
            call.get("id", ""): str(res) if res is not None else ""
            for call, res in zip(to_run, results)
        }

        messages = list(state.get(messages_key, []))
        start = len(messages)
        for call in calls:  # keep the original call order in the conversation
            call_id = call.get("id", "")
            if call_id in result_by_id:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_by_id[call_id],
                    }
                )
            elif call_id in ask_replies:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": ask_replies[call_id],
                    }
                )
        for name, call_id, decision in denied:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"Tool call '{name}' was not approved ({decision})",
                }
            )

        out: dict = {
            tool_call_key: "",
            "_tool_calls": [],
            "_tool_call_args": "",
            "_tool_call_id": "",
        }
        if reducer_appends((ctx.reducers or {}).get(messages_key)):
            out[messages_key] = messages[start:]
        else:
            out[messages_key] = messages
        return out
