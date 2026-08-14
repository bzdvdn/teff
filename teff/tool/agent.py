"""Sub-agent tools: a ``Tool`` that drives a short ReAct loop.

An :class:`AgentTool` wraps a sub-agent (:class:`~teff.harness.Harness`)
inside a single tool, so a coordinator node can treat a domain expert as an
ordinary tool (schema, routing, retry and approval all come free) while the
expert still runs its own LLM loop over a slice of the tool set.

The framework owns all of the plumbing — building the sub-``Harness`` from
the active :class:`~teff.node.context.ExecContext` (providers, tracer,
``on_llm_payload``) and threading the enclosing workflow ``state`` in via
the runtime ``__state__`` / ``__ctx__`` kwargs (see
:func:`teff.harness.tools`).  Subclasses only declare *content*: the system
prompt, the user message and how to handle the final reply.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from string import Formatter
from typing import Callable

from teff.schema import parse_json_object, validate_json
from teff.tool.tool import Tool


class AgentTool(Tool):
    """A tool that runs a sub-agent ReAct loop over a slice of the tool set.

    Subclasses declare *content* and leave the LLM plumbing to this class:

    - ``system`` — the sub-agent system message (a constant, or override
      :meth:`system_prompt` for state-dependent text);
    - ``tools`` — the slice of the tool set the sub-agent may call, as a
      mapping ``{name: Tool}`` or a plain iterable of :class:`Tool` instances
      (keyed by ``tool.name``);
    - ``user_message`` — a template with ``{placeholder}`` fields resolved
      against *state*: ``{state_key}`` takes the value of that key, and
      derived values (e.g. a schema-rendered ``project_info``) go through
      the ``formatters`` mapping; override :meth:`user_message` for fully
      custom messages;
    - :meth:`handle_reply` — turn the final reply into the tool result and,
      optionally, write results back into *state*.

    For the common shapes the constructor takes:
    ``writes=("plan",)`` copies the raw reply text into those state keys
    (with :meth:`handle_reply` returning the text), and :meth:`json_reply`
    parses a JSON reply against a schema.

    The runtime injects ``__state__`` / ``__ctx__`` (see
    :func:`teff.harness.tools`); anything :meth:`handle_reply` writes into
    *state* is copied back into the enclosing workflow state.
    """

    #: Class-attribute defaults, overridable per instance via the constructor.
    #: Declare these on a subclass to describe a sub-agent without an
    #: ``__init__`` override.
    system: str = ""
    max_rounds: int = 10
    writes: tuple[str, ...] = ()
    user_template: str = ""
    formatters: "dict[str, Callable[[dict], str]] | None" = None
    temperature: float | None = 0.0

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        tools: "Mapping[str, Tool] | Iterable[Tool] | None" = None,
        system: str | None = None,
        max_rounds: int | None = None,
        writes: tuple[str, ...] | None = None,
        user_template: str | None = None,
        formatters: "dict[str, Callable[[dict], str]] | None" = None,
        temperature: float | None = None,
    ):
        super().__init__()
        self._model = model
        self._provider = provider
        self._tools = self._tool_map(tools)
        self._system = system if system is not None else self.system
        self._max_rounds = max_rounds if max_rounds is not None else self.max_rounds
        self._writes = tuple(writes if writes is not None else self.writes)
        self._user_template = (
            user_template if user_template is not None else self.user_template
        )
        self._formatters = dict(
            formatters if formatters is not None else (self.formatters or {})
        )
        self._temperature = (
            temperature if temperature is not None else self.temperature
        )

    # -- subclass extension points ------------------------------------------
    def system_prompt(self, state: dict) -> str:
        """Return the sub-agent's system message (may read *state*)."""
        return self._system

    def user_message(self, state: dict) -> str:
        """Return the sub-agent's user message (may read *state*).

        Default: render the ``user_template`` constructor param, resolving
        ``{name}`` fields via ``formatters[name](state)`` when registered,
        else from ``state[name]`` (empty when absent).  Override for fully
        custom messages that don't fit the template.
        """
        if not self._user_template:
            return ""
        return self._render_user(state)

    def _render_user(self, state: dict) -> str:
        names = {
            field for _, field, _, _ in Formatter().parse(self._user_template) if field
        }
        values: dict[str, str] = {}
        for name in names:
            if name in self._formatters:
                values[name] = str(self._formatters[name](state))
            elif name in state:
                values[name] = str(state.get(name) or "")
            else:
                raise ValueError(
                    f"AgentTool '{self.name}': no formatter and no state key for "
                    f"placeholder '{{{name}}}' in user_template"
                )
        return self._user_template.format(**values)

    def handle_reply(self, state: dict, reply) -> str:
        """Post-process the final reply into the tool result.

        May read and write *state*; writes are copied back into the
        enclosing workflow state.  ``reply`` is the final
        :class:`~teff.harness.loop.Step` (``reply.content`` holds the text).

        Default: copy the reply text into each ``writes`` key and return it.
        """
        content = (reply.content or "").strip()
        for key in self._writes:
            state[key] = content
        return content

    def json_reply(self, reply, schema: dict) -> dict | None:
        """Parse the final reply as a JSON object matching *schema*.

        Returns the parsed dict, or ``None`` when the reply is not valid
        JSON (or fails validation) — failures fall back to ``None`` rather
        than raising, mirroring how structured outputs are best-effort.
        """
        try:
            parsed = parse_json_object(reply.content or "")
            if isinstance(parsed, dict) and not validate_json(parsed, schema):
                return parsed
        except ValueError:
            pass
        return None

    @staticmethod
    def _tool_map(tools: "Mapping[str, Tool] | Iterable[Tool] | None") -> dict:
        """Normalize ``tools`` to a ``{name: Tool}`` map.

        Accepts either a mapping or a plain iterable of :class:`Tool`
        instances, keyed by each tool's ``name`` attribute.
        """
        if not tools:
            return {}
        if isinstance(tools, Mapping):
            return dict(tools)
        return {t.name: t for t in tools}

    # -- framework plumbing --------------------------------------------------
    def _harness(self, ctx):
        """Build a sub-agent harness wired to the active exec context."""
        from teff.harness import Harness  # local import breaks the tool↔harness cycle

        harness = Harness.from_config(
            {
                "model": self._model,
                "provider": self._provider,
                "max_tool_rounds": self._max_rounds,
                "parse_text_tool_calls": True,
                "temperature": self._temperature,
            },
            default_provider=self._provider,
            default_model=self._model,
            providers=getattr(ctx, "providers", None) if ctx is not None else None,
        )
        if ctx is not None:
            tracer = getattr(ctx, "tracer", None)
            if tracer is not None:

                async def on_llm(provider, model, prompt, completion, duration):
                    tracer.llm(provider, model, prompt, completion, duration)

                harness.on_llm = on_llm
            if getattr(ctx, "on_llm_payload", None) is not None:
                harness.on_llm_payload = ctx.on_llm_payload
        return harness

    async def arun(self, *, __state__=None, __ctx__=None, **kwargs):  # noqa: ARG002
        """Run the sub-agent and surface the handled reply as the result."""
        state = dict(__state__ or {})
        messages = [
            {"role": "system", "content": self.system_prompt(state)},
            {"role": "user", "content": self.user_message(state)},
        ]
        reply = await self._harness(__ctx__).run(messages, tools=self._tools or None)
        result = self.handle_reply(state, reply)
        if __state__ is not None:
            __state__.update(state)
        return result


__all__ = ["AgentTool"]
