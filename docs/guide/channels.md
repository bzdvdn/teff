# Channels: run one workflow over HTTP, Telegram, webhooks

`teff[channels]` binds a **single** durable `Assistant` (built from a
`workflow.yaml`) to many transports, so interrupt handling, checkpoints and
message history behave identically on every surface.  You get:

- **HTTP/SSE** — `/api/chat`, a streaming `/api/chat/stream`, run state
  GET/DELETE, and a health check.
- **Telegram** — a bot via long-polling or a webhook, with one session per
  chat and owner scoping by user id.
- **Generic webhooks** — JSON payloads validated against a JSON Schema and
  routed into a turn with a configurable `owner:`.

Every turn answers with the same shape across all channels:

```json
{"session_id": "...", "waiting": false, "message": "<the reply>"}
```

When a workflow pauses on an `interrupt`, `waiting` is `true` and `message`
carries the prompt; the client resumes simply by sending the operator's
answer as the next `message`.

## Install

```bash
pip install "teff[channels]"
```

## Zero-code: `channels:` in YAML

```yaml
channels:
  http:
    enabled: true
    host: 0.0.0.0
    port: 8000
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    mode: polling
```

Then run the whole thing from one command:

```bash
teff serve -f workflow.yaml --host 0.0.0.0 --port 8000
teff bot -f workflow.yaml --token-env TELEGRAM_BOT_TOKEN --mode polling
teff chat -f workflow.yaml
```

`teff chat` is the terminal channel: the same durable `Assistant` as a plain
REPL.  Each line is one turn, a paused workflow (interrupt) asks in-chat and
resumes on your answer.

`teff new channels` scaffolds this YAML-first template; a code-first variant
is available too.

## Code-first: the HTTP/SSE channel

```python
from teff.channels import build_assistant, create_http_app, HTTPChannel

assistant = build_assistant("workflow.yaml")

# standalone app
app = create_http_app(assistant)
# or a mounted router inside your own FastAPI app
channel = HTTPChannel(assistant)
app.include_router(channel.router)
```

Sessions are scoped to the caller via the `X-User-Id` header (default
owner `"default"` when absent), and the session id is always generated
server-side so callers cannot hijack another user's session.

### Hooks: auth and per-turn kwargs

`create_http_app` / `create_http_router` / `HTTPChannel` accept:

- `dependencies` — FastAPI `Depends` objects applied to every endpoint
  except `GET /api/health` (health stays open for probes).
- `turn_kwargs` — a `(owner, session_id) -> kwargs` factory whose result is
  merged into every `Assistant.run` / `Assistant.stream` call.  Use it to
  attach an observability `tracer`/`on_llm_payload` per turn.

```python
from fastapi import Depends, Header, HTTPException


def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if x_api_key != "secret":
        raise HTTPException(status_code=401, detail="missing api key")


app = create_http_app(assistant, dependencies=[Depends(require_key)])


def traced(owner: str, session_id: str) -> dict:
    return {"tracer": my_tracer(owner, session_id)}


app = create_http_app(assistant, turn_kwargs=traced)
```

## Telegram

```python
from teff.channels import TelegramChannel, build_assistant

bot = TelegramChannel(build_assistant("workflow.yaml"), token="...")

# long-poll
await bot.poll()
# or webhook mode
await bot.set_webhook("https://example.com/hook")
```

Each Telegram chat maps to a session; the owner is the message sender's
user id (`from.id`, falling back to the chat id), so two people in the same
group never share a conversation.

Group chats: by default the bot replies to every group message. Set
`reply_when: mentioned` to answer only messages addressed to it — a reply
on the bot's own message or an `@bot_username` mention — so each user sees
only responses to their own message:

```yaml
channels:
  telegram:
    token_env: TELEGRAM_BOT_TOKEN
    mode: polling
    reply_when: mentioned   # all (default) | mentioned
```

The bot username is resolved once via `getMe` and cached; if it cannot be
resolved, any `@mention` in a group message counts as addressed.

## Generic webhooks

```python
from teff.channels import WebhookChannel, build_assistant

hook = WebhookChannel(
    build_assistant("workflow.yaml"),
    {
        "input": {"message": "summarize: {text}"},
        "owner": "header.X-User-Id",
        "session_key": "text",
    },
)

await hook.handle({"text": "the quick fox"}, headers={"X-User-Id": "alice"})
```

`owner:` accepts `payload.<field>`, `header.<Name>` (case-insensitive),
`fixed:<value>` or a literal; anything missing falls back to `"default"`.
The optional JSON Schema in `schema:` validates the payload before the turn.

## Growing a knowledge base: `rag_ingest`

`rag` only searches. The `rag_ingest` tool is the write side: it takes
document text (or a `path` to a csv/txt/pdf/excel file), chunks it, embeds
it and persists the vectors — so a Telegram message, webhook or terminal
turn can grow the store, and later `rag` turns answer from it. Declared in
`tools:` exactly like `rag` (same `embedder:`/`store:` config):

```yaml
tools:
  - type: rag_ingest
    config:
      embedder: { provider: ollama, model: nomic-embed-text }
      store: { type: sqlite, path: data/vectors.db, dim: 768 }
```

Run it from a workflow step with `tool_call`, feeding AI-normalized text
from an `llm_chat` step before it:

```yaml
steps:
  - id: normalize
    type: llm_chat
    config:
      system: "Turn this product row into one clean paragraph."
      prompt: "Row:\n{row}"
      output_key: doc_text
  - id: ingest
    type: tool_call
    config:
      tool: rag_ingest
      args: { text: "{doc_text}" }
      output_key: ingest_result
      on_error: message
```

See [`examples/channels/rag_ingest`](../examples.md) for the full pipeline
(ingest a CSV row over a webhook, then query the grown base from any
channel) and [`examples/channels/supervisor`](../examples.md) for the
multi-agent supervisor wrapped in the `channels:` block.
