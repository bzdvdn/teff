"""Probe ``POST /api/chat/stream`` on a running repair-ai-chat server.

Runs one turn against the chat-style SSE stream and asserts the event
sequence matches the contract::

    chat_id -> (status | content)* -> (waiting | message)

Exits ``0`` when the stream looks correct, ``1`` otherwise, printing every
event as it arrives.  Raw framework events (``run_start``, ``node_start``,
…) are rejected in the default mode and allowed with ``--raw``.

Usage::

    uv run python examples/applications/repair-ai-chat/check_stream.py
    uv run python examples/applications/repair-ai-chat/check_stream.py \\
        --message да --session <session_id>          # answer an ask_human pause
    uv run python examples/applications/repair-ai-chat/check_stream.py --raw
    uv run python examples/applications/repair-ai-chat/check_stream.py \\
        --api-key "$TEFF_API_KEY" --url http://localhost:8000

Requires the server from ``main.py`` to be running (and an LLM backend).
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

TERMINAL = ("waiting", "message")
CHAT_EVENTS = {"chat_id", "status", "content"}

_DATA_KEYS = {
    "chat_id": {"session_id"},
    "status": {"session_id", "message"},
    "content": {"session_id", "content"},
    "waiting": {"session_id", "question"},
    "message": {"session_id", "reply", "waiting"},
}

_ERRORS: list[str] = []


def fail(message: str) -> None:
    _ERRORS.append(message)
    print(f"  ✗ {message}", file=sys.stderr)


def check_event(name: str, data: dict, raw: bool) -> None:
    required = _DATA_KEYS.get(name)
    if required is None:
        if not raw:
            fail(f"unexpected event type: {name}")
        return
    missing = required - set(data)
    if missing:
        fail(f"{name}: missing keys {sorted(missing)}")


def run(url: str, message: str, session: str, api_key: str, raw: bool) -> int:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload: dict = {"message": message}
    if session:
        payload["session_id"] = session
    target = f"{url}/api/chat/stream" + ("?raw=1" if raw else "")

    try:
        with httpx.stream(
            "POST",
            target,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(300.0, read=300.0),
        ) as resp:
            if resp.status_code != 200:
                fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return 1
            return _read(resp.iter_lines(), raw=raw)
    except httpx.HTTPError as exc:
        fail(f"transport error: {exc}")
        return 1


def _read(lines, raw: bool) -> int:
    current: str | None = None
    events: list[tuple[str, dict]] = []
    first = True
    for line in lines:
        line = (line or "").strip()
        if not line:
            continue
        if line.startswith("event:"):
            current = line.removeprefix("event:").strip()
            continue
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            fail(f"bad JSON in data ({current}): {exc}")
            return 1
        if current is None:
            fail("data frame without a preceding event: line")
            return 1
        if first:
            if current != "chat_id":
                fail(f"stream must open with `chat_id`, got `{current}`")
                return 1
            first = False
        print(f"  {current:<10} {json.dumps(data, ensure_ascii=False)[:140]}")
        check_event(current, data, raw)
        events.append((current, data))
    return _summarise(events, raw=raw)


def _summarise(events: list[tuple[str, dict]], raw: bool) -> int:
    if not events:
        fail("empty stream: no events received")
        return 1
    types = [name for name, _ in events]
    terminal = types[-1]
    if terminal not in TERMINAL:
        fail(f"stream must end with `message` or `waiting`, ended with `{terminal}`")
    if not raw:
        foreign = set(types) - CHAT_EVENTS - set(TERMINAL)
        if foreign:
            fail(f"framework events leaked into the chat stream: {sorted(foreign)}")
    if _ERRORS:
        print(f"\nFAIL: {len(_ERRORS)} problem(s)")
        return 1
    print(f"\nok: {len(events)} event(s): {' → '.join(types)}")
    for name, data in events:
        if name == "message":
            print(f"reply: {data['reply']}")
        elif name == "waiting":
            print(
                f"question: {data['question']}  "
                f'(resume: --message "да" --session {data["session_id"]})'
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the repair-ai-chat SSE stream contract."
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="server base URL"
    )
    parser.add_argument(
        "--message",
        default="Помоги спланировать ремонт ванной комнаты, 5 м².",
        help="message to send (or the answer when resuming a paused turn)",
    )
    parser.add_argument("--session", default=None, help="resume an existing session")
    parser.add_argument("--api-key", default=None, help="X-API-Key header value")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="allow underlying framework events (debugging)",
    )
    args = parser.parse_args()
    ok = run(args.url, args.message, args.session, args.api_key, args.raw)
    raise SystemExit(0 if ok == 0 else 1)


if __name__ == "__main__":
    main()
