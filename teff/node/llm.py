"""LLM chat node — multi-provider, tool calling, structured output."""

import json
import typing

from teff.errors import TeffError
from teff.harness import (
    Harness,
    execute_tool_calls,
    normalize_text_tool_calls,
    tool_to_schema,
)
from teff.harness import (
    extract_usage as _extract_usage,  # noqa: F401  (re-export for tests)
)
from teff.harness import (
    parse_text_tool_call as _parse_text_tool_call,  # noqa: F401  (re-export for tests)
)
from teff.memory.context import MemoryConfig
from teff.node.node import Node
from teff.prompt import render_template
from teff.schema import (
    json_schema_from_type,
    parse_json_object,
    validate_json,
)
from teff.skill import resolve_skills, scope_tools, skills_instructions
from teff.stream import StreamEvent
from teff.tool.tool import Tool


def _opt_float_cfg(value: typing.Any) -> float | None:
    """Parse *value* as ``float`` or return ``None`` when empty."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class StructuredOutputError(TeffError, ValueError):
    """Raised when an LLM response fails structured-output parsing/validation.

    Attributes:
        schema: The JSON Schema the output was validated against (or ``None``).
        content: Raw text the LLM returned.
        errors: Parse/validation error message from the last attempt.
        attempts: Number of attempts made before giving up.
    """

    def __init__(
        self,
        *,
        schema: dict | None = None,
        content: str = "",
        errors: str = "",
        attempts: int = 0,
    ):
        self.schema = schema
        self.content = content
        self.errors = errors
        self.attempts = attempts
        message = f"LLM output failed structured validation after {attempts} attempt(s): {errors}"
        super().__init__(message)


class LLM(Node):
    """Call an LLM chat API with tool calling and structured output.

    Parameters:
        model: Model name (e.g. ``gpt-4``, ``llama3.1:8b``).
        system: System prompt.  Supports ``{key}`` placeholders rendered
            from state (see :func:`teff.prompt.render_template`).
        prompt: User prompt template.  Supports ``{key}`` placeholders
            rendered from state, e.g. ``"create a repair plan for {type} "
            "up to {summ}"``.  Overrides *input_key* when set.
        input_key: State key for user message (default: whole state).
        output_key: State key for the response (default ``"output"``).
        provider: Provider name (``"openai"``, ``"ollama"``, etc.).
            Falls back to the graph-level default (``Graph(default_provider=...)``
            / workflow ``default_provider:``) when unset.
        use_tools: Tool capability for the node: a list of names restricts
            it to exactly those tools; ``"all"`` uses every ``ctx.tools``
            entry.  ``None``/``[]`` (default) — no tools are surfaced.
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        response_format: ``{"type": "json_object"}`` etc.
        stream: If ``True``, use SSE streaming.
            Automatically disabled when tool calling is active.
        on_token: Optional callback ``(token: str) -> None`` for streaming.
        json_schema: JSON Schema dict describing the expected response.
            When set, the response is parsed as JSON, validated against
            the schema, and re-asked (with the validation error fed back)
            up to *max_retries* times.  The parsed object is stored under
            *output_key*.  Adds ``response_format: {"type": "json_object"}``
            for OpenAI-compatible providers (``format: "json"`` for Ollama)
            unless *response_format* is already set.
        output_type: Python type spec — a ``TypedDict``, dataclass, or
            ``dict[str, type]`` field map — converted to a JSON Schema.
            Alternative to *json_schema*.
        parse: If ``True`` without a schema, parse the response as a JSON
            object and store the dict under *output_key* (no validation).
        max_retries: How many times to re-ask after a validation failure.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_retries: Extra attempts per tool call after a failure.
        tool_approval: Gate on tool execution — ``"auto"`` (default),
            ``"deny"``, or a callable ``(name, args) -> bool | str``
            (sync or async).  ``"pause"`` is treated as ``"deny"`` in the
            internal multi-round loop (use a :class:`~teff.node.agent.ToolExec`
            node for pause/resume).
        http_max_retries: HTTP request retries (429/5xx/timeouts).
        fallbacks: Fallback model names for provider failover.
        base_url: Custom base URL (overrides provider default).
        chat_path: Custom API path (overrides provider default).
        auth_header: Custom auth header name.
        auth_prefix: Custom auth header prefix.
        api_key_env: Custom env var for API key.
        tools: List of raw tool definition dicts.
        messages_key: State key for message history.
            If set, the conversation history is read/written from/to
            ``state[messages_key]`` instead of being built fresh each call.
        memory: Optional long-term memory injection — a
            :class:`~teff.memory.context.MemoryConfig` or ``{store,
            namespace, k, header}`` (same as
            :class:`~teff.node.agent.ReActAgent`). Recalled memories for
            the most recent user message are prepended to the call as a
            system message.
        response_path: Dot-separated path to extract content from response.
        skills: Skills to mount on this call — a :class:`~teff.skill.Skill`,
            a path to a skill folder/``SKILL.md``, or a name resolved
            against *skill_dir*.  Their instructions are merged into the
            system prompt and their ``allowed-tools``/``disallowed-tools``
            narrow the visible tools.
        skill_dir: Directory to resolve bare skill names from
            (default ``"skills"``).
    """

    type = "llm_chat"
    _MAX_TOOL_ROUNDS = 10

    def __init__(
        self,
        config: dict | None = None,
        *,
        model: str | None = None,
        system: str = "",
        prompt: str | None = None,
        input_key: str | None = None,
        output_key: str = "output",
        provider: str | None = None,
        use_tools: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        on_token: typing.Callable[[str], None] | None = None,
        json_schema: dict | None = None,
        output_type: typing.Type[typing.Any] | None = None,
        parse: bool = False,
        max_retries: int = 2,
        tool_timeout: float | None = None,
        tool_retries: int = 0,
        tool_approval: typing.Any = None,
        http_max_retries: int = 2,
        fallbacks: list[str] | None = None,
        base_url: str | None = None,
        chat_path: str | None = None,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
        api_key_env: str | None = None,
        tools: list[dict] | None = None,
        messages_key: str | None = None,
        response_path: str = "",
        skills: list | None = None,
        skill_dir: str = "skills",
        memory: MemoryConfig | dict | None = None,
        **kwargs: typing.Any,
    ):
        merged = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "input_key": input_key,
            "output_key": output_key,
            "provider": provider,
            "use_tools": use_tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "stream": stream,
            "on_token": on_token,
            "json_schema": json_schema,
            "output_type": output_type,
            "parse": parse,
            "max_retries": max_retries,
            "tool_timeout": tool_timeout,
            "tool_retries": tool_retries,
            "tool_approval": tool_approval,
            "http_max_retries": http_max_retries,
            "fallbacks": fallbacks,
            "base_url": base_url,
            "chat_path": chat_path,
            "auth_header": auth_header,
            "auth_prefix": auth_prefix,
            "api_key_env": api_key_env,
            "messages_key": messages_key,
            "response_path": response_path,
            "skills": skills,
            "skill_dir": skill_dir,
            "memory": memory,
            **(config or {}),
            **kwargs,
        }
        # ensure tools is always a list
        merged.setdefault("tools", tools or [])
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config

        skills = resolve_skills(cfg)
        skill_text = skills_instructions(skills)

        has_messages_key = cfg.get("messages_key") and state.get(cfg["messages_key"])
        if has_messages_key:
            messages = list(state[cfg["messages_key"]])
            system = render_template(cfg.get("system", ""), state)
            if skill_text:
                system = f"{system}\n\n{skill_text}" if system else skill_text
            # a conversation node injects its system prompt on every turn,
            # but must not duplicate one already persisted in history
            if system and not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system})
        else:
            prompt = cfg.get("prompt")
            input_key = cfg.get("input_key")
            if prompt:
                user_message = render_template(prompt, state)
            elif input_key:
                user_message = str(state.get(input_key, ""))
            else:
                user_message = str(state)
            messages = []
            system = render_template(cfg.get("system", ""), state)
            if skill_text:
                system = f"{system}\n\n{skill_text}" if system else skill_text
            if system:
                messages.append({"role": "system", "content": system})
            if user_message:
                messages.append({"role": "user", "content": user_message})

        from teff.memory.context import memory_context_from_config

        memory_block = await memory_context_from_config(cfg, state=state, ctx=ctx)
        if memory_block:
            messages.insert(0, {"role": "system", "content": memory_block})

        tool_defs: list[dict] = list(cfg.get("tools", []))
        if cfg.get("use_tools", False):
            scoped_tools = scope_tools(ctx.tools, cfg, skills)
            for t in scoped_tools.values():
                tool_defs.append(tool_to_schema(t))
        else:
            scoped_tools = dict(ctx.tools)

        has_tools = bool(tool_defs)
        output_key = cfg.get("output_key", "output")

        schema = self._resolve_schema(cfg)
        structured = schema is not None
        parse_only = bool(cfg.get("parse", False)) and not structured

        harness = Harness.from_config(
            cfg,
            default_provider=getattr(ctx, "default_provider", None),
            default_model=getattr(ctx, "default_model", None),
            providers=getattr(ctx, "providers", None),
        )
        if cfg.get("http_max_retries") is not None:
            harness.max_retries = int(cfg.get("http_max_retries", 2))
        if cfg.get("fallbacks") is not None:
            harness.fallbacks = [str(f) for f in cfg["fallbacks"]]
        provider_key = harness.provider_key
        harness.on_llm = self._record_llm_cb(ctx, cfg, provider_key)
        payload_sink = getattr(ctx, "on_llm_payload", None)
        if payload_sink is not None:
            harness.on_llm_payload = payload_sink

        if structured and not cfg.get("response_format"):
            if harness.type == "ollama":
                harness._body_extra["format"] = "json"
            else:
                harness._body_extra["response_format"] = {"type": "json_object"}

        graph_stream = getattr(ctx, "emit", None) is not None
        content: str | dict = ""
        if (
            (cfg.get("stream", False) or graph_stream)
            and not has_tools
            and not structured
        ):
            harness.on_token = self._token_sink(ctx, cfg, provider_key)
            content = (await harness.call(messages, stream=True)).content
        else:
            max_retries = int(cfg.get("max_retries", 2))
            rounds = harness.max_rounds if has_tools else 1
            if structured:
                rounds = max(rounds, max_retries + 1)
            attempts = 0
            for _round in range(rounds):
                reply = await harness.call(
                    messages,
                    tools=tool_defs or None,
                    content_path=cfg.get("response_path", ""),
                )
                content = reply.content
                msg = reply.message
                tool_calls = msg.get("tool_calls")

                if has_tools and not tool_calls and harness.parse_text_tool_calls:
                    tool_calls, msg = normalize_text_tool_calls(
                        content, msg, seq=len(messages)
                    )

                if has_tools and tool_calls:
                    messages.append(msg)
                    tool_timeout = _opt_float_cfg(cfg.get("tool_timeout"))
                    tool_retries = int(cfg.get("tool_retries", 0))
                    results = await execute_tool_calls(
                        tool_calls,
                        scoped_tools,
                        harness.tool_error_mode,
                        tool_timeout,
                        tool_retries,
                        cfg.get("tool_approval"),
                    )
                    for tc, res in zip(tool_calls, results):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": res,
                            }
                        )
                    continue

                if structured:
                    assert schema is not None
                    assert isinstance(content, str)
                    parsed_value, error = self._parse_structured(content, schema)
                    if error is None:
                        content = parsed_value
                        break
                    attempts += 1
                    await self._record_structured(ctx, cfg, schema, error, attempts)
                    if attempts > max_retries:
                        raise StructuredOutputError(
                            schema=schema,
                            content=content,
                            errors=error,
                            attempts=attempts,
                        )
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response failed JSON schema "
                                f"validation: {error}\n"
                                "Respond with a single JSON object conforming "
                                f"to this schema:\n{json.dumps(schema)}"
                            ),
                        }
                    )
                    continue

                if parse_only:
                    raw = content
                    assert isinstance(raw, str)
                    try:
                        parsed = parse_json_object(raw)
                    except ValueError as exc:
                        raise StructuredOutputError(
                            content=raw,
                            errors=str(exc),
                            attempts=1,
                        ) from exc
                    content = parsed
                break

        if isinstance(content, dict):
            return {output_key: content}
        return {output_key: content or ""}

    def _record_llm_cb(self, ctx, cfg: dict, provider_key: str):
        """Build the ``on_llm`` callback recording usage + ``llm`` events."""

        async def record(
            provider: str, model: str, prompt: int, completion: int, duration: float
        ) -> None:
            tracer = getattr(ctx, "tracer", None)
            if tracer is not None:
                tracer.llm(provider, model, prompt, completion, duration)
            emit = getattr(ctx, "emit", None)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "llm",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={
                            "provider": provider,
                            "model": model,
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "duration_ms": duration,
                        },
                    )
                )

        return record

    def _resolve_schema(self, cfg: dict) -> dict | None:
        """Return the JSON Schema for structured output, if configured."""
        if cfg.get("json_schema") is not None:
            return json_schema_from_type(cfg["json_schema"])
        if cfg.get("output_type") is not None:
            return json_schema_from_type(cfg["output_type"])
        return None

    def _parse_structured(
        self, content: str, schema: dict
    ) -> tuple[typing.Any, str | None]:
        """Parse *content* as JSON and validate it against *schema*.

        Returns:
            ``(value, None)`` on success, ``(None, error_message)`` otherwise.
        """
        try:
            value = parse_json_object(content)
        except ValueError as exc:
            return None, str(exc)
        errors = validate_json(value, schema)
        if errors:
            return None, "; ".join(errors)
        return value, None

    async def _record_structured(
        self, ctx, cfg: dict, schema: dict, errors: str, attempt: int
    ) -> None:
        """Record a structured-output validation failure."""
        tracer = getattr(ctx, "tracer", None)
        if tracer is not None:
            tracer.structured(ctx.node_id, ctx.node_type, errors, attempt)
        emit = getattr(ctx, "emit", None)
        if emit is not None:
            await emit(
                StreamEvent(
                    "structured",
                    node_id=ctx.node_id,
                    node_type=ctx.node_type,
                    data={"errors": errors, "attempt": attempt},
                )
            )

    def _token_sink(
        self, ctx, cfg: dict, provider_key: str
    ) -> typing.Callable[[str], typing.Any]:
        """Build the per-token callback for streaming.

        Forwards each token to the node's ``on_token`` config and, when
        running under ``graph.stream()``, emits a ``token`` :class:`StreamEvent`.
        """
        emit = getattr(ctx, "emit", None)
        on_token = cfg.get("on_token")

        async def sink(token: str) -> None:
            if on_token is not None:
                on_token(token)
            if emit is not None:
                await emit(
                    StreamEvent(
                        "token",
                        node_id=ctx.node_id,
                        node_type=ctx.node_type,
                        data={
                            "token": token,
                            "provider": provider_key,
                            "model": str(cfg.get("model", "")),
                        },
                    )
                )

        return sink

    @staticmethod
    def _tool_to_schema(tool: Tool) -> dict:
        """Alias of :func:`teff.harness.tool_to_schema` (backward compat)."""
        return tool_to_schema(tool)
