"""Tests for the channel adapters: HTTP/SSE, generic webhooks, Telegram.

All transports bind the same :class:`~teff.assistant.Assistant` built from a
``workflow.yaml`` and run against the offline ``mock_llm`` fixture (patches
the ``Harness`` transport — no network, no API keys).  Covers:

* ``build_assistant``: a durable, interrupt-aware service from YAML.
* ``WebhookChannel``: schema validation, session derivation, owner scoping.
* ``HTTPChannel``: ``/api/chat`` single-shot + durable runs + interrupts.
* ``TelegramChannel``: update handling, session mapping, interrupt resume.
"""

import json
import textwrap
from pathlib import Path

import pytest

from teff.channels import (
    HTTPChannel,
    TelegramChannel,
    WebhookChannel,
    build_assistant,
    reply_text,
    turn_response,
)


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    p = tmp_path / "chat.yaml"
    p.write_text(
        textwrap.dedent(
            f"""
            name: chat_demo
            state:
              schema:
                messages: {{reducer: append, type: list}}
              initial:
                messages: []
            providers:
              - name: ollama
                type: ollama
                base_url: http://localhost:11434
                chat_path: /api/chat
            checkpoint:
              type: file
              path: {tmp_path / "cp"}
            steps:
              - llm:
                  id: reply
                  system: "Reply with one word."
                  model: llama3.1:8b
                  provider: ollama
                  output_key: answer
                  messages_key: messages
            """
        ),
        encoding="utf-8",
    )
    return p


class TestBuildAssistant:
    def test_compiles_workflow_with_checkpoint(self, workflow):
        assistant = build_assistant(str(workflow))
        assert assistant.graph is not None
        assert assistant.checkpointer is not None

    async def test_run_durable_turn(self, workflow, mock_llm):
        mock_llm.content = "hello"
        assistant = build_assistant(str(workflow))
        result = await assistant.run("s1", "hi there")
        assert reply_text(result) == "hello"
        assert not result.waiting
        # the session is durable: a second turn continues the same history
        result2 = await assistant.run("s1", "again")
        assert reply_text(result2) == "hello"

    async def test_turn_response_shape(self, workflow, mock_llm):
        mock_llm.content = "hi"
        assistant = build_assistant(str(workflow))
        result = await assistant.run("s1", "hello")
        payload = turn_response(result, "s1")
        assert payload == {"session_id": "s1", "waiting": False, "message": "hi"}


class TestWebhookChannel:
    def _hook(self, assistant):
        return WebhookChannel(
            assistant,
            {
                "path": "/hooks/x",
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "input": {"message": "summarize: {text}"},
                "session_key": "text",
            },
        )

    async def test_handle_runs_turn(self, workflow, mock_llm):
        mock_llm.content = "summary"
        hook = self._hook(build_assistant(str(workflow)))
        out = await hook.handle({"text": "the quick fox"})
        assert out["ok"] is True
        assert out["message"] == "summary"
        assert not out["waiting"]
        assert out["session_id"] == "the quick fox"

    async def test_handle_validation_errors(self, workflow, mock_llm):
        hook = self._hook(build_assistant(str(workflow)))
        out = await hook.handle({"other": 1})
        assert out["ok"] is False
        assert out["errors"]

    async def test_handle_uses_header_owner(self, workflow, mock_llm):
        mock_llm.content = "owned"
        hook = WebhookChannel(
            build_assistant(str(workflow)),
            {
                "input": {"message": "{text}"},
                "owner": "header.X-User-Id",
            },
        )
        out = await hook.handle({"text": "x"}, headers={"X-User-Id": "alice"})
        assert out["ok"] is True
        assert out["message"] == "owned"
        # same payload from a different owner resolves to the same session id,
        # but checkpoints are isolated per owner by the Assistant
        assert out["session_id"] == hook.session_id_for({"text": "x"})

    async def test_session_id_fallback_content_hash(self, workflow, mock_llm):
        hook = WebhookChannel(
            build_assistant(str(workflow)),
            {"input": {"message": "{text}"}},
        )
        sid = hook.session_id_for({"text": "same"})
        assert sid == hook.session_id_for({"text": "same"})
        assert sid != hook.session_id_for({"text": "other"})
        assert sid.startswith("wh-")

    async def test_owner_resolution_specs(self, workflow, mock_llm):
        def hook(owner_spec):
            return WebhookChannel(
                build_assistant(str(workflow)),
                {"input": {"message": "{text}"}, "owner": owner_spec},
            )

        assert hook("payload.customer").owner_for({"customer": "alice"}) == "alice"
        assert (
            hook("header.X-User-Id").owner_for({"x": 1}, {"X-User-Id": "bob"}) == "bob"
        )
        # header lookup is case-insensitive
        assert hook("header.X-User-Id").owner_for({}, {"x-user-id": "carol"}) == "carol"
        assert hook("fixed:ops").owner_for({}) == "ops"
        assert hook("default").owner_for({}) == "default"
        # missing payload/header fields fall back to "default"
        assert hook("payload.customer").owner_for({}) == "default"
        assert hook("header.X-User-Id").owner_for({}, {}) == "default"


class TestHTTPChannel:
    def test_create_app_and_chat(self, workflow, mock_llm):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        mock_llm.content = "ok"
        channel = HTTPChannel(build_assistant(str(workflow)))
        client = TestClient(channel.app)

        r = client.post("/api/chat", json={"message": "hello"})
        assert r.status_code == 200
        body = r.json()
        assert body["waiting"] is False
        assert body["message"] == "ok"
        assert body["session_id"]

    def test_stream_ends_with_message(self, workflow, mock_llm):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        mock_llm.content = "streamed"
        channel = HTTPChannel(build_assistant(str(workflow)))
        client = TestClient(channel.app)

        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as resp:
            events = []
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
        assert events
        assert events[-1]["message"] == "streamed"

    async def test_interrupt_surfaces_waiting(self, workflow, mock_llm, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        # a workflow with an interrupt node
        p = tmp_path / "gate.yaml"
        p.write_text(
            textwrap.dedent(
                f"""
                name: gate_demo
                state:
                  schema:
                    messages: {{reducer: append, type: list}}
                  initial:
                    messages: []
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                checkpoint:
                  type: file
                  path: {tmp_path / "cpg"}
                steps:
                  - interrupt:
                      id: ask
                      prompt: "Approve?"
                      key: approved
                """
            ),
            encoding="utf-8",
        )
        mock_llm.content = "unused"
        channel = HTTPChannel(build_assistant(str(p)))
        client = TestClient(channel.app)

        r = client.post("/api/chat", json={"message": "do it"})
        assert r.status_code == 200
        body = r.json()
        assert body["waiting"] is True
        assert body["key"] == "approved"
        assert body["message"] == "Approve?"
        # resume with the operator's answer
        r2 = client.post(
            "/api/chat", json={"message": "yes", "session_id": body["session_id"]}
        )
        assert r2.status_code == 200

    def test_mount_router_into_existing_app(self, workflow, mock_llm):
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from teff.channels import create_http_router

        mock_llm.content = "mounted"
        app = FastAPI()
        app.include_router(create_http_router(build_assistant(str(workflow))))
        client = TestClient(app)

        r = client.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 200
        assert r.json()["message"] == "mounted"

    def test_auth_dependencies_enforced(self, workflow, mock_llm):
        pytest.importorskip("fastapi")
        from fastapi import Depends, Header, HTTPException
        from fastapi.testclient import TestClient

        def require_key(
            x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        ):
            if x_api_key != "secret":
                raise HTTPException(status_code=401, detail="missing api key")

        mock_llm.content = "authed"
        channel = HTTPChannel(
            build_assistant(str(workflow)),
            dependencies=[Depends(require_key)],
        )
        client = TestClient(channel.app)

        assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
        r = client.post(
            "/api/chat", json={"message": "hi"}, headers={"X-API-Key": "secret"}
        )
        assert r.status_code == 200
        assert r.json()["message"] == "authed"
        # health stays open
        assert client.get("/api/health").status_code == 200

    def test_turn_kwargs_hook_sees_owner_and_session(self, workflow, mock_llm):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        seen = []
        mock_llm.content = "traced"

        def hook(owner: str, session_id: str) -> dict:
            seen.append((owner, session_id))
            return {"max_iterations": 5}

        channel = HTTPChannel(build_assistant(str(workflow)), turn_kwargs=hook)
        client = TestClient(channel.app)

        r = client.post(
            "/api/chat", json={"message": "hi"}, headers={"X-User-Id": "alice"}
        )
        assert r.status_code == 200
        assert seen, "turn_kwargs hook was never called"
        assert seen[0][0] == "alice"
        assert seen[0][1] == r.json()["session_id"]


class TestTelegramChannel:
    def _bot(self, assistant):
        return TelegramChannel(assistant, token="test:token")

    async def test_handle_update_runs_turn(self, workflow, mock_llm, monkeypatch):
        sent = []
        mock_llm.content = "hi!"

        async def fake_api(self, method, **params):
            sent.append((method, params))
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update(
            {"update_id": 1, "message": {"chat": {"id": 42}, "text": "hello"}}
        )
        assert sent, "expected a sendMessage call"
        assert sent[0][0] == "sendMessage"
        assert sent[0][1]["chat_id"] == 42
        assert sent[0][1]["text"] == "hi!"

    async def test_handle_update_session_mapping(self, workflow, mock_llm, monkeypatch):
        sent = []
        mock_llm.content = "ok"

        async def fake_api(self, method, **params):
            sent.append((method, params))
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update(
            {"update_id": 1, "message": {"chat": {"id": 99}, "text": "a"}}
        )
        assert bot.session_id_for(99) == "tg-99"
        assert sent[0][1]["text"] == "ok"

    async def test_handle_update_owner_is_user_id(self, workflow, monkeypatch):
        from teff.assistant import Assistant

        async def fake_api_ok(self, method, **params):
            return {}

        seen = {}

        async def fake_run(self, session_id, text, **kwargs):
            seen["owner"] = kwargs.get("owner")
            from teff.graph import TurnResult

            return TurnResult(session_id=session_id, reply="hi")

        monkeypatch.setattr(Assistant, "run", fake_run)
        monkeypatch.setattr(TelegramChannel, "_api", fake_api_ok)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 7},
                    "from": {"id": 12345},
                    "text": "hello",
                },
            }
        )
        assert seen["owner"] == "12345"

    async def test_ignores_non_text_updates(self, workflow, monkeypatch):
        sent = []

        async def fake_api(self, method, **params):
            sent.append((method, params))
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = self._bot(build_assistant(str(workflow)))
        await bot.handle_update({"update_id": 1, "message": {"chat": {"id": 1}}})
        assert sent == []

    async def test_mention_aware_skips_unaddressed_group(self, workflow, monkeypatch):
        sent = []

        async def fake_api(self, method, **params):
            sent.append((method, params))
            if method == "getMe":
                return {"username": "my_bot"}
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = TelegramChannel(
            build_assistant(str(workflow)), token="test:token", reply_when="mentioned"
        )
        await bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 10, "type": "group"},
                    "from": {"id": 5},
                    "text": "what does anyone think",
                },
            }
        )
        assert all(m != "sendMessage" for m, _ in sent)

    async def test_mention_aware_answers_mention(self, workflow, mock_llm, monkeypatch):
        mock_llm.content = "hi!"
        sent = []

        async def fake_api(self, method, **params):
            sent.append((method, params))
            if method == "getMe":
                return {"username": "my_bot"}
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = TelegramChannel(
            build_assistant(str(workflow)), token="test:token", reply_when="mentioned"
        )
        await bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 10, "type": "group"},
                    "from": {"id": 5},
                    "text": "hey @my_bot",
                    "entities": [{"type": "mention", "offset": 4, "length": 7}],
                },
            }
        )
        assert any(m == "sendMessage" for m, _ in sent)

    async def test_mention_aware_answers_reply_to_bot(
        self, workflow, mock_llm, monkeypatch
    ):
        mock_llm.content = "thanks!"
        sent = []

        async def fake_api(self, method, **params):
            sent.append((method, params))
            if method == "getMe":
                return {"username": "my_bot"}
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = TelegramChannel(
            build_assistant(str(workflow)), token="test:token", reply_when="mentioned"
        )
        await bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 10, "type": "supergroup"},
                    "from": {"id": 5},
                    "text": "thanks",
                    "reply_to_message": {"from": {"is_bot": True}},
                },
            }
        )
        assert sent[0][0] == "sendMessage"

    async def test_mention_aware_answers_private_chat(
        self, workflow, mock_llm, monkeypatch
    ):
        mock_llm.content = "hi!"
        sent = []

        async def fake_api(self, method, **params):
            sent.append((method, params))
            if method == "getMe":
                return {"username": "my_bot"}
            return {}

        monkeypatch.setattr(TelegramChannel, "_api", fake_api)
        bot = TelegramChannel(
            build_assistant(str(workflow)), token="test:token", reply_when="mentioned"
        )
        await bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 11, "type": "private"},
                    "from": {"id": 6},
                    "text": "hi",
                },
            }
        )
        assert sent[0][0] == "sendMessage"

    async def test_mention_aware_defaults_to_all(self, workflow):
        bot = self._bot(build_assistant(str(workflow)))
        assert bot.reply_when == "all"
        with pytest.raises(ValueError):
            TelegramChannel(assistant=bot.assistant, token="t", reply_when="nope")
