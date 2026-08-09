"""Ask — declarative validation strategy for interrupt answers.

``interrupt_loop`` / ``interrupt`` with an ``Ask`` stop being hard-wired
to a single expected word: the strategy decides whether the operator's
answer passes, and can capture an arbitrary value (a discount code, a
date, …) alongside the pass/fail decision.

Strategies:

* ``equals`` — exact match on a normalized (strip + lowercase) value.
* ``any_of`` — match any of several normalized values.
* ``regex`` — match a regular expression; the first capture group (or the
  whole match) is extracted.
* ``check`` — a callable ``fn(value) -> bool`` or
  ``fn(value) -> (bool, extracted)``.
* ``llm`` — an :class:`~teff.node.LLM` turns the free-form answer into a
  structured verdict (``{ok: bool, ...}``); ``value_field`` names the
  verdict field to capture.

A ``llm`` strategy can also declare a *third* outcome — "unclear, re-ask".
When ``clear_field`` names a verdict boolean (e.g. ``clear``) that is
``False``, :class:`Validate` writes ``clarify_value`` instead of a pass/fail
decision; :meth:`~teff.flow.Flow.interrupt_loop` then routes that value back
to the interrupt (re-ask the operator) **without** re-running the body chain.
This is how a free-form reply like ``"ghskdlsjdkls"`` gets re-asked
while ``"yes"`` / ``"sure"`` approve and ``"no"`` re-plans.

The strategy is executed by the :class:`Validate` node, which decodes the
verdict / raw answer into a ``flow.loop`` decider value (like
:class:`~teff.node.Gate`) and optionally writes the extracted value.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from teff.node.llm import LLM
from teff.node.node import Node


def _norm(value) -> str:
    return str(value or "").strip().lower()


class Ask:
    """Declarative validation strategy for an interrupt answer.

    Use the classmethod constructors to pick a strategy::

        Ask.equals("yes")
        Ask.any_of("yes", "ok", "sure")
        Ask.regex(r"^[A-Z0-9]{4,12}$", value_key="discount_code")
        Ask.check(lambda v: v.lower() in {"yes", "ok"})
        Ask.llm(system=..., user=..., schema=..., model=..., provider=...)

    The strategy is auto-detected from the constructor kwargs, so plain
    ``Ask(equals="yes", value_key="code")`` also works.
    """

    def __init__(
        self,
        *,
        equals: Optional[str] = None,
        any_of: Optional[list] = None,
        regex: Optional[str] = None,
        check: Optional[Callable] = None,
        system: str = "",
        user: str = "",
        schema: Optional[dict] = None,
        model: str = "",
        provider: str = "",
        verdict_key: str = "verdict",
        ok_field: str = "ok",
        value_key: str = "",
        value_field: str = "",
        decision_key: str = "decision",
        pass_value: str = "да",
        fail_value: str = "нет",
        clear_field: str = "",
        clarify_value: str = "",
        rounds_key: str = "rounds",
        max_rounds: int = 100,
    ):
        # Internal names avoid colliding with the classmethod constructors.
        self._expected = equals
        self._allowed = list(any_of) if any_of else None
        self._pattern = regex
        self._predicate = check
        self.system = system
        self.user = user
        self.schema = schema
        self.model_name = model
        self.provider = provider
        self.verdict_key = verdict_key
        self.ok_field = ok_field
        self.value_key = value_key
        self.value_field = value_field
        self.decision_key = decision_key
        self.pass_value = pass_value
        self.fail_value = fail_value
        self.clear_field = clear_field
        self.clarify_value = clarify_value
        self.rounds_key = rounds_key
        self.max_rounds = max_rounds

    @property
    def strategy(self) -> str:
        if self._predicate is not None:
            return "check"
        if self._pattern:
            return "regex"
        if self._allowed:
            return "any_of"
        if self._expected is not None:
            return "equals"
        if self.system or self.schema:
            return "llm"
        return ""

    def needs_classifier(self) -> bool:
        return self.strategy == "llm"

    @classmethod
    def equals(cls, value, **kwargs) -> "Ask":
        return cls(equals=value, **kwargs)

    @classmethod
    def any_of(cls, *values, **kwargs) -> "Ask":
        return cls(any_of=list(values), **kwargs)

    @classmethod
    def regex(cls, pattern: str, **kwargs) -> "Ask":
        return cls(regex=pattern, **kwargs)

    @classmethod
    def check(cls, fn: Callable, **kwargs) -> "Ask":
        return cls(check=fn, **kwargs)

    @classmethod
    def llm(
        cls,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str,
        provider: str,
        **kwargs,
    ) -> "Ask":
        return cls(
            system=system,
            user=user,
            schema=schema,
            model=model,
            provider=provider,
            **kwargs,
        )

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Ask":
        """Build an :class:`Ask` from a declarative strategy mapping.

        Mirrors the YAML shorthand on an ``interrupt`` step::

            strategy:
              equals: yes
            # or: any_of: [yes, ok]  |  regex: "^[A-Z0-9]{4}$"
            # or: llm: {system, user, schema, model, provider}

        The mapping's other keys (``value_key``, ``decision_key``,
        ``pass_value``, ``fail_value``, ``verdict_key``, ``ok_field``,
        ``clear_field``, ``clarify_value``, ``rounds_key``, ``max_rounds``)
        are passed through to the chosen strategy constructor.

        Raises:
            ValueError: When no known strategy key is present.
        """
        if "equals" in mapping:
            spec = {k: v for k, v in mapping.items() if k != "equals"}
            return cls(equals=mapping["equals"], **spec)
        if "any_of" in mapping:
            spec = {k: v for k, v in mapping.items() if k != "any_of"}
            return cls(any_of=list(mapping["any_of"]), **spec)
        if "regex" in mapping:
            spec = {k: v for k, v in mapping.items() if k != "regex"}
            return cls(regex=mapping["regex"], **spec)
        if isinstance(mapping.get("llm"), dict):
            llm_cfg = mapping["llm"]
            spec = {k: v for k, v in mapping.items() if k != "llm"}
            return cls(
                system=llm_cfg.get("system", ""),
                user=llm_cfg.get("user", ""),
                schema=llm_cfg.get("schema"),
                model=llm_cfg.get("model", ""),
                provider=llm_cfg.get("provider", ""),
                **spec,
            )
        raise ValueError(
            "strategy requires one of equals / any_of / regex / llm, "
            f"got {sorted(mapping)}"
        )

    def classifier(self) -> LLM:
        """Build the verdict classifier ``LLM`` for the ``"llm"`` strategy."""
        return LLM(
            system=self.system,
            prompt=self.user,
            output_key=self.verdict_key,
            json_schema=self.schema or {},
            model=self.model_name,
            provider=self.provider,
        )

    def validate_node(self, input_key: str) -> "Validate":
        """Build the :class:`Validate` node wired to *input_key*."""
        return Validate(
            input_key=input_key,
            verdict_key=self.verdict_key,
            ok_field=self.ok_field,
            output_key=self.decision_key,
            pass_value=self.pass_value,
            fail_value=self.fail_value,
            clear_field=self.clear_field,
            clarify_value=self.clarify_value,
            value_key=self.value_key,
            value_field=self.value_field,
            rounds_key=self.rounds_key,
            max_rounds=self.max_rounds,
            strategy=self.strategy,
            equals=self._expected,
            any_of=self._allowed,
            regex=self._pattern,
            check=self._predicate,
        )


class Validate(Node):
    """Decode an interrupt answer into a ``flow.loop`` decider value.

    Works on two kinds of input:

    * a **raw answer** (a string from the interrupt resume) matched by the
      ``equals`` / ``any_of`` / ``regex`` / ``check`` strategies;
    * a **verdict dict** (from an ``LLM`` classifier) read via *ok_field*,
      with *value_field* captured into *value_key*.

    Each evaluation increments ``rounds_key``; once it reaches
    ``max_rounds`` the node is forced to ``pass_value`` so the enclosing
    loop terminates deterministically instead of spinning forever.

    Config:
        input_key: State key holding the raw answer or verdict object.
        strategy: Matching strategy for raw answers.
        equals/any_of/regex/check: Strategy parameters (raw answers).
        verdict_key: State key holding the classifier's verdict object.
        ok_field: Pass-flag field of the verdict object.
        output_key: State key receiving ``pass_value`` / ``fail_value``.
        pass_value/fail_value: Decision values written on pass / fail.
        clear_field: Optional verdict boolean naming "is this answer
            decipherable".  When it is ``False`` the node writes
            ``clarify_value`` instead of pass/fail (re-ask, no body).
        clarify_value: Decision value written when *clear_field* is
            ``False`` (falls back to *fail_value* when empty).
        value_key: State key receiving the extracted value (cleared on a
            fail).  Empty to skip.
        value_field: Verdict field captured into *value_key*.
        rounds_key: State key with the evaluation counter (incremented).
        max_rounds: After this many evaluations the node is forced to pass.
        missing_is_ok: Treat a missing / non-dict input as a pass.
    """

    type = "validate"

    def __init__(
        self,
        config: dict | None = None,
        *,
        input_key: str = "answer",
        strategy: str = "",
        equals: Optional[str] = None,
        any_of: Optional[list] = None,
        regex: Optional[str] = None,
        check: Optional[Callable] = None,
        verdict_key: str = "verdict",
        ok_field: str = "ok",
        output_key: str = "decision",
        pass_value: str = "да",
        fail_value: str = "нет",
        clear_field: str = "",
        clarify_value: str = "",
        value_key: str = "",
        value_field: str = "",
        rounds_key: str = "rounds",
        max_rounds: int = 100,
        missing_is_ok: bool = False,
        **kwargs,
    ):
        merged = {
            "input_key": input_key,
            "strategy": strategy,
            "equals": equals,
            "any_of": any_of,
            "regex": regex,
            "check": check,
            "verdict_key": verdict_key,
            "ok_field": ok_field,
            "output_key": output_key,
            "pass_value": pass_value,
            "fail_value": fail_value,
            "clear_field": clear_field,
            "clarify_value": clarify_value,
            "value_key": value_key,
            "value_field": value_field,
            "rounds_key": rounds_key,
            "max_rounds": max_rounds,
            "missing_is_ok": missing_is_ok,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    def _match(self, raw):
        """Return ``(ok, extracted)`` for a raw answer."""
        cfg = self.config
        strategy = cfg["strategy"]
        if strategy == "equals":
            ok = _norm(raw) == _norm(cfg["equals"])
            return ok, (raw if ok else None)
        if strategy == "any_of":
            ok = _norm(raw) in {_norm(v) for v in cfg["any_of"]}
            return ok, (raw if ok else None)
        if strategy == "regex":
            m = re.search(cfg["regex"], str(raw or ""))
            ok = m is not None
            value = None
            if m:
                value = m.group(1) if m.groups() else m.group(0)
            return ok, value
        if strategy == "check":
            res = cfg["check"](raw)
            if isinstance(res, tuple):
                ok, value = res
                return bool(ok), value
            ok = bool(res)
            return ok, (raw if ok else None)
        if isinstance(raw, dict):
            ok = bool(raw.get(cfg["ok_field"], cfg["missing_is_ok"]))
            value = raw.get(cfg["value_field"]) if cfg["value_field"] else None
            return ok, value
        return bool(cfg["missing_is_ok"]), None

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        rounds = int(state.get(cfg["rounds_key"], 0) or 0) + 1

        data = state.get(cfg["input_key"])
        if isinstance(data, dict):
            ok = bool(data.get(cfg["ok_field"], cfg["missing_is_ok"]))
            value = data.get(cfg["value_field"]) if cfg["value_field"] else None
            clear = cfg["clear_field"] == "" or bool(
                data.get(cfg["clear_field"], False)
            )
        else:
            ok, value = self._match(data)
            clear = True

        forced = rounds >= int(cfg["max_rounds"])
        if not forced and not clear:
            # the verdict is unclear — route to the "re-ask" branch (no body)
            decision = cfg["clarify_value"] or cfg["fail_value"]
            passed = False
        else:
            passed = bool(ok or forced)
            decision = cfg["pass_value"] if passed else cfg["fail_value"]

        out: dict = {
            cfg["rounds_key"]: rounds,
            cfg["output_key"]: decision,
        }
        if cfg["value_key"]:
            out[cfg["value_key"]] = value if passed else ""
        return out
