"""Generic webhook channel: run a workflow on any inbound JSON payload.

The payload shape is declared in the workflow's ``channels.webhook`` block
as a JSON Schema, and a template maps the payload onto the one-turn
``message`` a conversation consumes::

    channels:
      webhook:
        - path: /hooks/order
          schema:
            type: object
            properties:
              id: {type: integer}
              total: {type: number}
              customer_id: {type: string}
            required: [id, total]
          input:
            message: "new order {id} for {total}"
          session_key: id
          owner: "payload.customer_id"

* ``schema``      — JSON Schema the payload is validated against
  (:func:`teff.schema.validate_json`, stdlib subset).
* ``input.message`` — ``render_template`` source; placeholders are the
  payload fields.
* ``session_key`` — payload field used as the durable ``session_id``
  (fallback: ``message``'s sha1, so every payload is its own session).
* ``owner`` — who owns the checkpoint (session isolation).  One of:
  ``payload.<field>`` (take from the body), ``header.<Name>`` (take from a
  request header), ``fixed:<value>`` (a constant), or omitted (``default``).

The channel deliberately reuses the same :class:`~teff.assistant.Assistant`
as every other channel: checkpoints, reducers and interrupts behave
identically, so a webhook-triggered run can pause on an interrupt just
like a chat message would.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from teff.assistant import Assistant
from teff.channels.reply import turn_response
from teff.prompt import render_template
from teff.schema import validate_json


class WebhookChannel:
    """One inbound webhook route bound to a shared ``Assistant``."""

    def __init__(self, assistant: Assistant, spec: dict[str, Any]):
        self.assistant = assistant
        self.spec = spec
        self.path = spec.get("path", "/webhook")
        self.schema = spec.get("schema") or {}
        self.session_key = spec.get("session_key")
        self.message_template = spec.get("input", {}).get("message", "{message}")
        self.max_iterations = spec.get("max_iterations", 80)
        self.owner_spec = spec.get("owner") or "default"

    def session_id_for(self, payload: dict[str, Any]) -> str:
        """Derive the durable session id from the payload.

        Uses ``session_key`` when configured; otherwise a content hash, so
        the same payload always resumes the same conversation.
        """
        if self.session_key is not None:
            value = payload.get(self.session_key)
            if value is not None:
                return str(value)
        raw = str(payload).encode("utf-8", "replace")
        return "wh-" + hashlib.sha1(raw).hexdigest()[:24]

    def message_for(self, payload: dict[str, Any]) -> str:
        """Render the one-turn ``message`` from the payload fields."""
        return render_template(self.message_template, payload)

    def owner_for(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, Any] | None = None,
    ) -> str:
        """Resolve the checkpoint owner from the configured ``owner`` spec.

        ``payload.<field>`` reads the body, ``header.<Name>`` reads a
        request header (case-insensitive), ``fixed:<value>`` is a constant,
        and anything else falls back to the spec verbatim (``default``).
        """
        spec = self.owner_spec
        if isinstance(spec, str) and spec.startswith("payload."):
            return str(payload.get(spec[len("payload.") :], "default"))
        if isinstance(spec, str) and spec.startswith("header."):
            name = spec[len("header.") :].lower()
            if headers:
                for key, value in headers.items():
                    if str(key).lower() == name and value is not None:
                        return str(value)
            return "default"
        if isinstance(spec, str) and spec.startswith("fixed:"):
            return spec[len("fixed:") :]
        return str(spec)

    def validate(self, payload: Any) -> list[str]:
        """Return schema errors for *payload* (empty when valid)."""
        if not self.schema:
            return []
        if not isinstance(payload, dict):
            return ["payload must be an object"]
        return validate_json(payload, self.schema)

    async def handle(
        self,
        payload: dict[str, Any],
        *,
        owner: str | None = None,
        headers: Mapping[str, Any] | None = None,
    ) -> dict:
        """Validate *payload*, run one turn, return the channel response.

        *owner* overrides the configured ``owner:`` spec (the CLI passes the
        resolved value when it wants to override).  The return value matches
        the HTTP channel's shape: ``ok`` plus a turn of ``session_id`` /
        ``waiting`` / ``message`` (the reply, or the interrupt prompt when
        ``waiting``).
        """
        errors = self.validate(payload)
        if errors:
            return {"ok": False, "errors": errors}
        session_id = self.session_id_for(payload)
        result = await self.assistant.run(
            session_id,
            self.message_for(payload),
            owner=owner or self.owner_for(payload, headers),
            max_iterations=self.max_iterations,
        )
        return {"ok": True, **turn_response(result, session_id)}
