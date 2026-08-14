"""Contract — declarative structured-output recipe with semantic validation.

A :class:`Contract` packages the full model-response pipeline that the
``extract -> normalize -> validate -> retry -> fallback`` pattern strips
down to: the schema pass, the semantic normalization, the semantic
validation, an optional no-tools retry, and a deterministic fallback.

It is a sibling of :class:`~teff.node.extract.Extract` (extraction) and
:class:`~teff.node.ask.Ask` (interrupt answers).  Unlike ``Extract`` —
which only fills fields the model dropped — a ``Contract`` enforces a
*custom semantic contract* on the whole reply:

* **schema** — the JSON Schema the reply must conform to.  It is handled
  by the LLM node itself (parse / validate / re-ask), exactly like an
  ``Extract``'s ``json_schema``.
* **normalize** — ``fn(raw) -> value`` converting the raw model output
  into a canonical form, or ``None`` to pass the raw output through.
* **validate** — ``fn(value) -> bool`` enforcing the semantic contract
  (e.g. "answer is a canonical dict and its source list is well-formed").
* **retry_without_tools** — when the first pass used tools and validation
  fails, re-run the LLM **without** tools: small models often produce a
  better reply when forced to answer from the conversation alone.
* **fallback** — ``fn(state) -> value``, a deterministic last resort used
  when validation fails even after the retry.

The recipe is executed by a :class:`~teff.flow.sub_flow.SubFlow` built by
:meth:`Contract.build` / :meth:`Contract.nodes`::

    Contract(
        system=FINAL_SYSTEM,
        schema=RESPONSE_SCHEMA,
        messages_key="messages",
        output_key="answer_json",
        normalize=normalize_final_response,
        validate=is_valid_final_response,
        fallback=lambda st: create_contract_fallback(st["query"], st["history"]),
    )

The internal graph is a flat chain when nothing is configured and branches
only when a retry / fallback exists::

    [LLM(tools) -> normalize -> validate]
        -- no --> [LLM(no tools) -> normalize -> validate] -- no --> [fallback]
        -- yes --------------------------------> done
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from teff.node.llm import LLM

if TYPE_CHECKING:
    from teff.flow.sub_flow import SubFlow
from teff.node.node import Node


class Contract:
    """Declarative structured-output contract (``Extract``'s strict sibling).

    Args:
        system: System prompt for the structured ``LLM`` pass.
        schema: JSON Schema dict the reply is parsed / validated / re-asked
            against (via the ``LLM`` node).  Alternative to ``output_type``.
        output_type: TypedDict / dataclass / ``dict[str, type]`` compiled to
            a JSON Schema.  Alternative to ``schema``.
        model: Model name for the ``LLM`` passes.
        provider: Provider name for the ``LLM`` passes.
        messages_key: State key holding the conversation read by the LLM.
        output_key: State key receiving the final canonical value.
        raw_key: State key where the LLM writes its raw output
            (default ``<output_key>_raw``).
        normalize: ``fn(raw) -> value`` converting the raw model output into
            the canonical form.  ``None`` passes the raw output through.
        validate: ``fn(value) -> bool`` semantic contract check.  ``None``
            treats any non-empty output as valid.
        fallback: ``fn(state) -> value`` deterministic last resort.
            Required when ``validate`` is provided and can fail.
        use_tools: Whether the first ``LLM`` pass surfaces tools
            (``True`` / ``"all"`` / a tool-name list).  Defaults to ``False``.
        retry_without_tools: Re-run the LLM without tools when validation
            fails.  Meaningful only when ``use_tools`` is set.
        id: Prefix for the built node ids (``<id>-llm`` etc.).
        **llm_kwargs: Extra kwargs threaded into every ``LLM`` pass
            (``temperature``, ``max_retries``, ``response_format``, …).
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
        raw_key: str = "",
        normalize: Callable | None = None,
        validate: Callable | None = None,
        fallback: Callable | None = None,
        use_tools: bool | str | list[str] | None = False,
        retry_without_tools: bool = False,
        id: str = "",
        **llm_kwargs,
    ):
        if validate is not None and not callable(validate):
            raise TypeError("validate must be a callable")
        if normalize is not None and not callable(normalize):
            raise TypeError("normalize must be a callable")
        if fallback is not None and not callable(fallback):
            raise TypeError("fallback must be a callable")
        if validate is not None and fallback is None:
            raise ValueError(
                "Contract with a semantic validate requires a fallback so a "
                "failed contract resolves deterministically"
            )

        self._system = system
        self._schema = schema
        self._output_type = output_type
        self._model = model
        self._provider = provider
        self._messages_key = messages_key
        self._output_key = output_key
        self._raw_key = raw_key or f"{output_key}_raw"
        self._normalize = normalize
        self._validate = validate
        self._fallback = fallback
        self._use_tools = use_tools
        self._retry_without_tools = bool(retry_without_tools)
        self._id = id
        self._llm_kwargs = dict(llm_kwargs)

    # ------------------------------------------------------------------
    # LLM pass
    # ------------------------------------------------------------------

    def llm(self, *, use_tools: "bool | str | list[str] | None") -> LLM:
        """Build a structured ``LLM`` node for this contract.

        *use_tools* controls whether tools are surfaced on this pass; the
        retry pass passes ``None`` to drop them.
        """
        return LLM(
            system=self._system,
            json_schema=self._schema,
            output_type=self._output_type,
            model=self._model,
            provider=self._provider,
            messages_key=self._messages_key,
            output_key=self._raw_key,
            use_tools=use_tools,
            **self._llm_kwargs,
        )

    # ------------------------------------------------------------------
    # Recipe construction
    # ------------------------------------------------------------------

    def build(self) -> SubFlow:
        """Build the :class:`~teff.flow.sub_flow.SubFlow` executing the recipe.

        The internal graph holds the LLM pass(es), a
        :class:`ContractNormalize`, a :class:`ContractValidate`, and —
        when a retry / fallback is configured — a second no-tools LLM pass
        and a :class:`ContractFallback`.
        """
        from teff.flow.case import Case
        from teff.flow.flow import Flow
        from teff.flow.sub_flow import SubFlow

        prefix = f"{self._id}-" if self._id else ""
        ok_key = f"_{self._output_key}_ok"
        has_tools = bool(self._use_tools)
        can_retry = self._retry_without_tools and has_tools
        needs_fallback = self._validate is not None or can_retry

        normalize1 = ContractNormalize(
            input_key=self._raw_key,
            output_key=self._output_key,
            fn=self._normalize,
        )
        validate1 = ContractValidate(
            input_key=self._output_key,
            output_key=ok_key,
            fn=self._validate,
        )

        inner = Flow()
        inner.step(self.llm(use_tools=self._use_tools), id=f"{prefix}llm")
        inner.step(normalize1, id=f"{prefix}normalize")
        inner.step(validate1, id=f"{prefix}validate")

        if can_retry:
            retry_llm = self.llm(use_tools=None)
            normalize2 = ContractNormalize(
                input_key=self._raw_key,
                output_key=self._output_key,
                fn=self._normalize,
            )
            validate2 = ContractValidate(
                input_key=self._output_key,
                output_key=ok_key,
                fn=self._validate,
            )
            inner.branch(
                ok_key,
                Case("yes"),
                Case("no")
                .add(retry_llm, id=f"{prefix}retry-llm")
                .add(normalize2, id=f"{prefix}retry-normalize")
                .add(validate2, id=f"{prefix}retry-validate"),
            )
            inner.branch(
                ok_key,
                Case("yes"),
                Case("no").add(
                    ContractFallback(output_key=self._output_key, fn=self._fallback),
                    id=f"{prefix}fallback",
                ),
            )
        elif needs_fallback:
            inner.branch(
                ok_key,
                Case("yes"),
                Case("no").add(
                    ContractFallback(output_key=self._output_key, fn=self._fallback),
                    id=f"{prefix}fallback",
                ),
            )

        return SubFlow(
            graph=inner.compile(),
            output_map={self._output_key: self._output_key},
            max_iterations=12,
        )

    def nodes(self) -> list[Node]:
        """Build ``[SubFlow]`` for flow wiring (mirrors ``Extract.nodes``)."""
        return [self.build()]


class ContractNormalize(Node):
    """Convert the raw model output into the canonical contract value.

    Reads the LLM output from ``input_key``; when ``fn`` is set, writes
    ``fn(raw)`` to ``output_key`` (``None`` passes the raw output through).
    """

    type = "contract_normalize"

    def __init__(
        self,
        config: dict | None = None,
        *,
        input_key: str = "output",
        output_key: str = "output",
        fn: Callable | None = None,
        **kwargs,
    ):
        merged = {
            "input_key": input_key,
            "output_key": output_key,
            "fn": fn,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        raw = state.get(cfg["input_key"])
        fn = cfg.get("fn")
        value = fn(raw) if callable(fn) else raw
        return {cfg["output_key"]: value}


class ContractValidate(Node):
    """Enforce the semantic contract on the canonical value.

    Reads the normalized value from ``input_key`` and writes ``pass_value``
    / ``fail_value`` to ``output_key`` so the surrounding graph can branch.
    Without ``fn`` any non-empty value passes.
    """

    type = "contract_validate"

    def __init__(
        self,
        config: dict | None = None,
        *,
        input_key: str = "output",
        output_key: str = "ok",
        fn: Callable | None = None,
        pass_value: str = "yes",
        fail_value: str = "no",
        **kwargs,
    ):
        merged = {
            "input_key": input_key,
            "output_key": output_key,
            "fn": fn,
            "pass_value": pass_value,
            "fail_value": fail_value,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        value = state.get(cfg["input_key"])
        fn = cfg.get("fn")
        ok = bool(fn(value)) if callable(fn) else bool(value)
        return {cfg["output_key"]: cfg["pass_value"] if ok else cfg["fail_value"]}


class ContractFallback(Node):
    """Deterministic last resort for a failed contract.

    Calls ``fn(state)`` and writes the result to ``output_key``.
    """

    type = "contract_fallback"

    def __init__(
        self,
        config: dict | None = None,
        *,
        output_key: str = "output",
        fn: Callable | None = None,
        **kwargs,
    ):
        merged = {
            "output_key": output_key,
            "fn": fn,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        fn = cfg.get("fn")
        if not callable(fn):
            raise ValueError("contract fallback requires a callable fn")
        return {cfg["output_key"]: fn(state)}


__all__ = [
    "Contract",
    "ContractFallback",
    "ContractNormalize",
    "ContractValidate",
]