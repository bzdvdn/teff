"""Extract — declarative structured-extraction recipe.

An :class:`Extract` is :class:`~teff.node.ask.Ask`'s sibling: instead of
deciding pass/fail on an interrupt answer it extracts a structured object
from the conversation.  It bundles the LLM extraction pass (a plain
:class:`~teff.node.LLM` with ``json_schema`` / ``output_type`` and optional
``messages_key``) with deterministic *fallbacks* that fill fields the model
left empty — a common failure mode of small local models.

The recipe is executed by the nodes :meth:`Extract.nodes` builds: the
extractor ``LLM`` first, then one :class:`Fallback` node per fallback.
``Fallback`` is also usable standalone.
"""

from __future__ import annotations

from typing import Any, Callable

from teff.node.llm import LLM
from teff.node.node import Node


class _FallbackSpec:
    """A single declarative fallback: fill *field* via ``fn(state)``."""

    __slots__ = ("field", "fn")

    def __init__(self, field: str, fn: Callable):
        self.field = field
        self.fn = fn


class Extract:
    """Declarative structured-extraction recipe (``Ask``'s sibling).

    Builds ``[LLM extractor, *Fallback nodes]`` from a single spec — the
    extraction half of a ``done`` chain::

        extractor = Extract.model(
            system="You extract project data...",
            schema=PROJECT_INFO_SCHEMA,
            model="llama3.1:8b",
            provider="ollama",
            messages_key="messages",
            output_key="project_info",
            fallbacks=[
                Extract.fallback("room_type", room_from_first_user),
            ],
        )
        flow.interrupt_loop(key="approved", ..., done=extractor.nodes())

    Use :meth:`model` to configure the LLM pass (equivalent to a plain
    ``LLM`` with ``json_schema``) and :meth:`fallback` to declare a
    deterministic fill for a field the model may drop.  Everything else is
    threaded through to :class:`~teff.node.LLM`.
    """

    def __init__(
        self,
        *,
        system: str = "",
        schema: dict | None = None,
        output_type: Any | None = None,
        model: str = "",
        provider: str = "",
        messages_key: str | None = None,
        output_key: str = "output",
        parse: bool = False,
        fallbacks: list | None = None,
        id: str = "",
        **llm_kwargs,
    ):
        self.system = system
        self.schema = schema
        self.output_type = output_type
        self.model_name = model
        self.provider = provider
        self.messages_key = messages_key
        self.output_key = output_key
        self.parse = parse
        self._id = id
        self._fallbacks = list(fallbacks or [])
        self._llm_kwargs = dict(llm_kwargs)

    @classmethod
    def model(
        cls,
        *,
        system: str,
        schema: dict,
        model: str,
        provider: str,
        **kwargs,
    ) -> "Extract":
        """Build an extraction recipe around a structured ``LLM`` pass.

        ``id`` (optional) names the built nodes in the compiled graph: the
        extractor ``LLM`` becomes ``<id>`` and each fallback
        ``<id>-fallback-<n>``, so the topology shows ``extractor`` instead of
        an auto-generated ``llm_chat_7``.
        """
        return cls(
            system=system,
            schema=schema,
            model=model,
            provider=provider,
            **kwargs,
        )

    @classmethod
    def fallback(cls, field: str, fn: Callable) -> "_FallbackSpec":
        """Declare a deterministic fill for *field* via ``fn(state)``.

        *fn* receives the whole workflow state and returns the field value
        (or ``None`` to skip).  Runs after the LLM pass, only when the
        model left *field* empty.
        """
        return _FallbackSpec(field=field, fn=fn)

    def llm(self) -> LLM:
        """Build the extraction ``LLM`` node."""
        node = LLM(
            system=self.system,
            json_schema=self.schema,
            output_type=self.output_type,
            model=self.model_name,
            provider=self.provider,
            messages_key=self.messages_key,
            output_key=self.output_key,
            parse=self.parse,
            **self._llm_kwargs,
        )
        if self._id:
            node.config["id"] = self._id
        return node

    def nodes(self) -> list[Node]:
        """Build ``[LLM extractor, *Fallback nodes]`` for flow wiring."""
        nodes: list[Node] = [self.llm()]
        for i, spec in enumerate(self._fallbacks, start=1):
            fb = Fallback(
                input_key=self.output_key,
                field=spec.field,
                fn=spec.fn,
            )
            if self._id:
                fb.config["id"] = f"{self._id}-fallback-{i}"
            nodes.append(fb)
        return nodes


class Fallback(Node):
    """Deterministic fallback that fills a field the model left empty.

    Reads a dict from ``input_key``; when *field* in it is empty / ``None``,
    calls ``fn(state)`` and merges the returned value under *field*.  No-op
    when the dict already has the field or *fn* returns ``None``.

    Config:
        input_key: State key holding the extracted dict.
        field: Dict field to fill when empty.
        fn: Callable ``fn(state) -> value | None``.
    """

    type = "fallback"

    def __init__(
        self,
        config: dict | None = None,
        *,
        input_key: str = "output",
        field: str = "",
        fn: Callable | None = None,
        **kwargs,
    ):
        merged = {
            "input_key": input_key,
            "field": field,
            "fn": fn,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        input_key = cfg.get("input_key", "output")
        field = cfg.get("field")
        fn = cfg.get("fn")
        if not field or not callable(fn):
            return {}
        data = state.get(input_key)
        if not isinstance(data, dict) or data.get(field):
            return {}
        value = fn(state)
        if value is None:
            return {}
        return {input_key: {**data, field: value}}
