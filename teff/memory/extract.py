"""LLM-based fact extraction for long-term memory.

The :class:`MemoryExtractor` turns a conversation into durable facts by
asking a model to summarise what should be remembered *beyond* the current
session.  It is a thin layer over a :class:`~teff.harness.loop.Harness`:

- ``extract`` calls the model once and parses a JSON array of facts from
  the reply (tolerating code fences and surrounding prose).
- ``save`` extracts facts and writes them into a
  :class:`~teff.memory.base.MemoryStore`, keyed by a stable hash of the
  fact text so re-extracting the same fact upserts it instead of
  duplicating it.

The extractor never stores anything itself; pass it a :class:`MemoryStore`
(which owns the vector store / embedder) to persist results.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from teff.harness.loop import Harness

#: Default system prompt used when no custom one is given.
DEFAULT_SYSTEM_PROMPT = """\
You extract durable, long-term facts about the user from a conversation.

A durable fact is something that stays true beyond this session: identity, \
profession, preferences, relationships, recurring needs, work details, \
habits, and explicit decisions ("I prefer X over Y").

Only extract facts about the user; ignore what the assistant says.  Do NOT \
extract greetings, thanks, timestamps, or anything that only matters for \
the current request.

Examples:
User: "Hi, I'm a DevOps engineer and I love coffee"
-> [{"text": "The user is a DevOps engineer"}, {"text": "The user likes coffee"}]

User: "thanks, bye"
-> []

Return ONLY a JSON array of objects, one per fact:
[{"text": "the fact as a self-contained sentence"}]

Each "text" must be a single factual sentence that can be understood \
without the surrounding conversation.  Respond in the language of the \
conversation.  If there are no durable facts, return an empty array: []"""


class MemoryExtractor:
    """Extract durable facts from a conversation using an LLM.

    Args:
        harness: A :class:`~teff.harness.loop.Harness` used for the single
            extraction call.  When omitted, one is built from *model* and
            *provider*.
        model: Model name for a self-built harness (ignored when *harness*
            is given).
        provider: Provider key for a self-built harness.
        system_prompt: Overrides the default extraction prompt.
        temperature: Sampling temperature for the extraction call.
    """

    def __init__(
        self,
        harness: Harness | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ):
        if harness is None:
            if not model:
                raise ValueError(
                    "MemoryExtractor needs a harness or a model to build one"
                )
            harness = Harness(model=model, provider=provider, temperature=temperature)
        self._harness = harness
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    async def extract(self, conversation: list[dict]) -> list[str]:
        """Return the durable facts found in *conversation*.

        Args:
            conversation: OpenAI-style messages (``{"role", "content"}``);
                the text of every message is included in the prompt.
        """
        transcript = _transcript(conversation)
        reply = await self._harness.call(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": transcript},
            ]
        )
        return _parse_facts(reply.content)

    async def save(
        self,
        memory: Any,
        conversation: list[dict],
        namespace: tuple[str, ...],
        *,
        ttl: float | None = None,
    ) -> list[tuple[str, str]]:
        """Extract facts and write them into *memory*.

        Each fact is stored under a stable key derived from its text (a
        short SHA-1), so re-extracting the same fact updates it in place.

        Args:
            memory: A :class:`~teff.memory.base.MemoryStore`.
            conversation: The messages to extract facts from.
            namespace: Namespace to store the facts under.
            ttl: Per-item TTL in seconds, or ``None`` for no expiry.

        Returns:
            The ``(key, fact)`` pairs that were written.
        """
        written: list[tuple[str, str]] = []
        for fact in await self.extract(conversation):
            key = _fact_key(fact)
            await memory.put(
                namespace, key, {"text": fact, "source": "extractor"}, ttl=ttl
            )
            written.append((key, fact))
        return written


def _transcript(conversation: list[dict]) -> str:
    parts: list[str] = []
    for msg in conversation:
        role = str(msg.get("role", "user"))
        content = msg.get("content")
        if isinstance(content, list):
            text = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content)
        parts.append(f"{role}: {text.strip()}")
    return "\n".join(parts)


def _fact_key(fact: str) -> str:
    return "fact_" + hashlib.sha1(fact.encode("utf-8")).hexdigest()[:16]


def _parse_facts(content: str) -> list[str]:
    """Parse a JSON array of facts from the model reply."""
    raw = _find_json_array(content)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    facts: list[str] = []
    for entry in data:
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
        else:
            continue
        if text:
            facts.append(text)
    return facts


def _find_json_array(text: str) -> str | None:
    """Return the balanced JSON array starting at the first ``[``, or None."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
