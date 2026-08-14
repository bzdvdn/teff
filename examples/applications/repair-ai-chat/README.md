# repair-ai-chat — supervisor repair assistant (production scaffold app)

A runnable instance of the **production scaffold template** (`teff/scaffold`):
a `src/` package with typed state, domain services, per-agent
tools, a RAG materials catalog — wired as a **supervisor Flow** built on
[`Flow.route()`](../../teff/flow/flow.py).

## Flow

```
supervisor ─ next_agent=direct ──► ContextBuilder → ReAct(direct) ─┐
    ▲                                                              │
    │  each agent runs as a SubFlow; control returns to the        │
    └────────────────────────────── supervisor ────────────────────┘

supervisor ─ next_agent=finish ──► Extractor (structured project_info)
```

The core `Supervisor` node writes `next_agent`; `route()` sends each value
to the matching agent chain and loops back to the supervisor.  Routing is
deterministic after the first decision — no subclass needed:

* **first round** the model picks the entry agent (`direct` / `planner` /
  `estimator` / `materials` / `qa`) from the user message;
* a `direct` answer fills `direct_reply` and, via `done_keys`, finishes the
  turn;
* any other pick enters the **`fill_order` pipeline**: `planner` → `estimator`
  → `materials` → `qa` run in order, one per supervisor round, and the turn
  finishes only after the QA review ran;
* a mid-chain agent picked directly (a targeted question) runs once and
  finishes without dragging the whole pipeline in.

On `finish` the route exits through the `Extractor`, which pulls structured
project info from the whole conversation.

## Highlights

- **`Flow.route("next_agent", finish=..., direct=..., planner=..., ...)`** —
  supervisor-style routing with a finish chain
- agent chains as **`SubFlow`** (context builder → ReAct harness with
  tool scoping → assistant append), each with a private scratch conversation
- **`graph.run(emit=...)`** — stream `StreamEvent`s while still returning
  the final state (nested `run_start`/`run_end` are stripped)
- **trace dashboard** — every chat turn is captured by a `GraphObserver`
  into `data/traces.db` and browsable at **`/obs/ui`** (one click per run:
  full graph, per-node LLM prompt/response, tags, notes); mounted via
  `teff.observability.attach_dashboard`, prefix from `TEFF_TRACES_PREFIX`
- production layout from `teff/scaffold`: `config/`, `src/`, `domain/`,
  `nodes/`, `tools/`, `rag/`, `graphs/`, `data/`

## Layout

```
repair-ai-chat/
├── main.py               # server entry point (uvicorn; host/port from settings)
├── app.py                # FastAPI app factory (uvicorn app:create_app)
├── cli.py                # interactive chat; or one repair-planning turn
├── src/                  # the production package
│   ├── config/           # env-driven settings (.env / TEFF_* vars)
│   ├── api/              # endpoint groups: router.py + chat/ + run/ + auth/
│   ├── core/             # dependency wiring (services, catalog)
│   ├── domain/           # entities + pure domain services (room/material/budget)
│   ├── graphs/           # typed state, prompts, JSON schemas, flow builder
│   ├── nodes/            # Extractor + context builders (Supervisor from teff.node)
│   ├── tools/            # Tool subclasses bound to services + catalog
│   ├── rag/              # materials catalog over an in-memory vector store
│   ├── service/          # Assistant: turn orchestration (HTTP + CLI)
│   └── storage/          # JSON-file checkpointer + session helpers
├── data/documents/       # materials.csv + price.csv — embedded lazily / via `load`
└── src/                  # (wiring + API tests live in tests/test_applications_repair_ai_chat.py)
```

## Run

Requires Ollama running locally:

```
ollama pull qwen2.5:7b
uv run python examples/applications/repair-ai-chat/main.py
```

Open `http://localhost:8000` for the build-free web chat UI, and
`http://localhost:8000/obs/ui` for the live trace dashboard.

### Live demo with Docker Compose

The default `docker-compose.yml` connects the app to Ollama **on your host
machine** (via `host.docker.internal`):

```
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
docker compose up --build
```

- `http://localhost:8000` — the chat web UI
- `http://localhost:8000/obs/ui` — the trace dashboard

If your Ollama only listens on `127.0.0.1`, start it with
`OLLAMA_HOST=0.0.0.0 ollama serve` first so the container can reach it.
Long-running sessions persist to the `appdata` volume.

> **No host Ollama? Use the all-in-one bundle.** `compose.ollama.yml` adds a
> containerised Ollama (app + model together — needs several GB of RAM, so
> it is for local tasting, not a small VPS):
>
> ```
> docker compose -f docker-compose.yml -f compose.ollama.yml up --build
> ```

### Point it at any OpenAI-compatible endpoint

`TEFF_PROVIDER_BASE_URL` works for any provider, not just Ollama.  This is
the recipe for a **hostable, RAM-free demo** — run the app image and set the
URL to a hosted API that already serves an 8B model (Groq, OpenRouter,
Together, …):

```
docker run --rm -p 8000:8000 \
  -e TEFF_PROVIDER=openai_compatible \
  -e TEFF_PROVIDER_BASE_URL=https://api.groq.com/openai/v1 \
  -e TEFF_API_KEY=... \
  -v teff-data:/data \
  repair-ai-chat
```

or, locally against your own Ollama:

```
TEFF_PROVIDER=ollama TEFF_PROVIDER_BASE_URL=http://localhost:11434 \
uv run python examples/applications/repair-ai-chat/main.py
```

The end-to-end wiring test (`tests/test_applications_repair_ai_chat.py`)
runs the same graph against a mocked LLM transport, so it needs no network:
`uv run pytest tests/test_applications_repair_ai_chat.py`

While the server is up, `check_stream.py` probes a turn of the chat-style
SSE stream and validates its event contract:

```
uv run python examples/applications/repair-ai-chat/check_stream.py
uv run python examples/applications/repair-ai-chat/check_stream.py \
    --message да --session <id>            # answer an ask_human pause
uv run python examples/applications/repair-ai-chat/check_stream.py --raw
```

The stream is `chat_id` → `(status | content)*` → a single terminal
`message`.  When the run pauses to ask the operator something, the terminal
`message` carries the question in `reply` and `waiting: true`; otherwise it
carries the final answer and `waiting: false`.  Posting the answer to the
same `session_id` resumes the paused run in place.
