# Changelog

## 0.2.0 — flow.yaml authoring layer

This release adds a second, higher-level YAML surface — `flow.yaml` — that
mirrors the Python `Flow` builder, plus a Python `team()` API for
supervisor-based multi-agent teams. The classic low-level `workflow.yaml`
(`id`/`type` steps + `edges`) is unchanged and still the compiled artifact.

Highlights:

- **`flow.yaml` (sugar / authoring layer)** — single-key idiom steps that
  expand into the low-level graph:
  - `llm:` / `transform:` / `agent_step:` / `agent:` linear steps.
  - `context_builder:` / `append_assistant:` — the turn-routing pair as
    first-class idioms (mirroring the new `flow.context_builder()` /
    `flow.append_assistant()` methods), replacing the `type:`
    `{type: context_builder, ...}` boilerplate in examples and the README.
  - `team:` — a `Supervisor` decider plus one routed agent per role in a
    single block; `supervisor:` / `supervise:` idioms for explicit
    supervisor loops.
  - `parallel:` / `map:` / `loop:` / `branch:` / `route:` / `interrupt:`
    control-flow idioms, with `interrupt:` accepting the `strategy:`
    shorthand (`equals` / `any_of` / `regex` / `llm`).
  - Top-level `default_provider` / `default_model` defaults inherited by
    every step, plus `providers:`, `tools:`, `state:`, `checkpoint:`,
    `hooks:`, `observability:`, `include:` and `${ENV}` interpolation.
  - `teff run -f flow.yaml`, `teff validate -f flow.yaml`,
    `teff graph -f flow.yaml` auto-detect the format; `teff build -f
    flow.yaml` compiles it into the low-level `graph.yaml` artifact
    (`build_flow_to_yaml`), including nested `Map` / `Parallel` / `Loop` /
    `SubFlow` configs.
- **`flow.team()`** — programmatic twin of the `team:` idiom: builds the
  `Supervisor` decider, one routed agent per role and the supervisor loop in
  one call. Roles are `AgentRole` instances (the recommended spelling), plain
  dict recipes, `Node` / `SubFlow` objects, or nested `Flow`s.
- **`AgentRole`** (`teff.flow.AgentRole`) — a system prompt + `output_key`
  (+ optional `use_tools`) rendered as a routed `agent_step` SubFlow; also
  constructed from a YAML role mapping for parity.
- **Two-layer validation** — `validate_flow` / `validate_flow_file` for the
  authoring surface (sibling of `validate_workflow`); the CLI picks the
  right validator from the file shape.
- **`load_workflow_document`** — reads a workflow document with env refs and
  `include:` blocks resolved, without building nodes or tools (used by the
  example parity tests).
- **`include:` improvements** — idiom steps that carry their id inside a
  single-key payload (`transform: {id: split, ...}`) are now id-prefixed too,
  so a sugar document can be included several times without collisions.
- **Example parity** — every sugar `workflow.yaml` example ships a low-level
  `graph.yaml` twin; `tests/test_examples_parity.py` checks that each pair
  loads and validates on its own surface, the twin is never sugar, and that
  both layers agree on tools and initial state. `tests/test_flow_yaml.py`
  (1000+ lines) covers the idioms, defaults, validation and
  flow.yaml → graph round-trips.
- Fix: `build_flow_to_yaml` used a wrong relative import
  (`from ..yaml import workflow_to_yaml`), which raised `ModuleNotFoundError`
  at runtime — now resolves to `teff.yaml`.
- Docs: `docs/guide/flow-yaml.md` (full `flow.yaml` key reference),
  `docs/guide/yaml-workflows.md` expanded to a complete `graph.yaml`
  reference, `docs/guide/best-practices.md` updated with a format-selection
  section. Example messages translated from Russian to English.

## 0.1.1 — docs & metadata refresh

Documentation, recipes and metadata updates only; no runtime changes.

- New guides: `why-teff.md`, recipes for FastAPI agents, KB assistants and
  more; `hello_llm` / `poem_chat` examples.
- Metadata and `uv.lock` refresh for publishing.

## 0.1.0 — first stable release

Two alphas of development polish went into this line. From here the public
API, YAML surface and CLI are considered stable: breaking changes now
require a minor version bump and a CHANGELOG entry.

Highlights:

- Channels: `teff[channels]` ships HTTP/SSE, Telegram and generic webhook
  adapters bound to one durable `Assistant` built from a `workflow.yaml`: `teff[channels]` ships HTTP/SSE, Telegram and generic webhook
  adapters bound to one durable `Assistant` built from a `workflow.yaml`:
  - `teff serve -f workflow.yaml` runs the HTTP/SSE service; `teff bot`
    runs a Telegram bot (polling or webhook); a `channels:` YAML block
    declares them declaratively.
  - `create_http_app(assistant)` serves `/api/chat` (+ SSE stream, runs
    GET/DELETE) out of the box; `create_http_router(assistant)` returns a
    mountable `APIRouter` so the same endpoints can be embedded in an
    existing FastAPI app. Both accept `dependencies` (auth gates, skipped
    on `/api/health`) and a `turn_kwargs(owner, session_id) -> kwargs`
    hook for per-turn tracing/overrides.
  - One unified turn response `{session_id, waiting, message}` across HTTP,
    webhook and Telegram; owner scoping per channel (`X-User-Id` header,
    Telegram user id, webhook `owner:` spec).
  - `teff.new` scaffolds a `channels` template (YAML-first) and a
    code-first variant.
  - `teff chat` runs the same durable `Assistant` as a terminal REPL —
    interrupts ask in-chat and resume on your answer, so a workflow that
    serves HTTP/Telegram/webhook also works from the shell.
  - `rag_ingest` tool: the write side of the vector store. Declared in
    `tools:` like `rag` (same `embedder:`/`store:` config), it takes
    `text` or a `path` (csv/txt/pdf/excel), chunks, embeds and persists
    documents at runtime — so a Telegram/webhook/terminal turn can grow a
    knowledge base, then answer via `rag`. AI-parsing is an explicit
    `llm_chat` step before the tool.
  - New examples: `examples/channels/supervisor/` (the multi-agent
    supervisor wrapped in the `channels:` block) and
    `examples/channels/rag_ingest/` (ingest CSV rows from any channel).
- `teff.testing.MockLLM` now answers in both OpenAI and Ollama wire shapes,
  so the `mock_llm` fixture works for every provider type.
- `Ask.model(...)` is renamed to `Ask.llm(...)` (the LLM-classifier strategy);
  the internal strategy name is now `"llm"`. The old `model` name clashed with
  the model-name keyword (`Ask.model(model=...)`).
- YAML workflows can now be assembled purely from YAML:
  - `transform` gains pipeline-building actions — `contains`, `compare`
    (numeric `eq/ne/gt/ge/lt/le`), `split`, `join`, `replace`, `coalesce`,
    `pick`, `to_int`, `to_float`, `now`. `contains`/`compare` emit
    `"true"`/`"false"` for direct use in `edges:` conditions.
  - `command` node type: declarative `goto`/`STOP` routing from state
    (`routes:` with `when` conditions + `update:`).
  - `loop` node type: repeat a `body` chain until `state[key]` equals
    `until`, bounded by `max_rounds`.
  - `interrupt` steps accept a `strategy:` shorthand (`equals` / `any_of` /
    `regex` / `llm`) that expands to the classifier + `validate` chain —
    the YAML counterpart of `Flow.interrupt(..., accept=...)`.
  - `include:` block composes steps/edges/tools/state from other workflow
    files (recursive, with optional `prefix:` to avoid id collisions).

## 0.1.0-alpha

First alpha release.

- Workflow as data: `Flow` / `Graph` builders with `route`, `agent_step`,
  `Map`, `Parallel`, `Retry`, `Interrupt` (human-in-the-loop).
- Typed `State` with reducers; YAML workflow definitions.
- Provider layer: Ollama, OpenAI, Anthropic and more, with concurrency control.
- ReAct agent harness, `Tool`/`ToolRegistry`, structured output validation.
- Long-term memory: `MemoryStore`, `MemoryExtractor`, per-owner context.
- RAG: chunker, embedders, vector stores (Chroma, Qdrant, PG, FAISS, ...).
- Observability: run tracing, token pricing, tool-call tracking.
- CLI: `teff new <name>` scaffolding, YAML validation.
