"""Room-type detection for the repair workflow.

The extractor itself is a plain ``LLM`` node (``messages_key`` + system
prompt + ``json_schema``) — since core ``LLM`` prepends the system prompt to
the message history when ``messages_key`` is set, no custom subclass is
needed.  This module keeps only the *deterministic* part, which is domain
logic: local models frequently drop the room even when the user named it
explicitly, so :func:`room_from_first_user` (an ``Extract.fallback``) fills
``room_type`` from the first user message when the extraction leaves it
empty.
"""

from __future__ import annotations

#: Russian room keywords → canonical ``room_type`` value.
ROOM_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ванн", "санузел", "с/у", "сануз"), "bathroom"),
    (("кухн",), "kitchen"),
    (("спальн",), "bedroom"),
    (("гостин", "зал", "living"), "living_room"),
    (("детск",), "kids_room"),
    (("коридор", "прихож"), "hallway"),
)


def detect_room_type(first_user_message: str) -> str | None:
    """Map the first user message to a ``room_type`` via keywords."""
    text = first_user_message.lower()
    for keywords, room in ROOM_KEYWORDS:
        if any(k in text for k in keywords):
            return room
    return None


def room_from_first_user(state: dict) -> str | None:
    """Fallback fn: detect the room from the user messages.

    Runs only when the extractor ``LLM`` left ``project_info.room_type``
    empty.  Returns the first detected room, or ``None`` to skip.
    """
    for message in state.get("messages", []):
        if message.get("role") == "user" and message.get("content"):
            room = detect_room_type(str(message["content"]))
            if room:
                return room
    return None
