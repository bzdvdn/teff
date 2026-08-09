# Learning path

Teff is a small framework with a large surface. The fastest way to "get it"
is to follow a path instead of reading the docs alphabetically. Each stage
below points at the examples and guides that matter, in the order that makes
concepts build on each other.

Every example runs offline on a local Ollama (`ollama pull llama3.1:8b`) or
needs no model at all — no API keys required. After each stage, `teff run`,
`teff validate` and the tests in the example should "just work".

## Stage 0 — Fit and install (10 minutes)

Decide if Teff is for you and get it running.

- [Install](install.md) — `uv`/`pip`, extras, first `teff new`.
- [Quick start](quickstart.md) — your first graph in three minutes.
- [Concepts](concepts.md) — the mental model: graph → runtime → state.

*You should know:* what a node, an edge, state and a run are; how a workflow
is declared; how to launch it from YAML and from Python.

## Stage 1 — Declare workflows (30 minutes)

The core idea: **workflow as data**. Topology lives in YAML (or the Flow DSL),
not in prompts or node bodies.

- [basic_pipeline](https://github.com/bzdvdn/teff/tree/master/examples/basic_pipeline/) — minimal YAML pipeline.
- [branching](https://github.com/bzdvdn/teff/tree/master/examples/branching/) — conditional edges + Flow API.
- [parallel](https://github.com/bzdvdn/teff/tree/master/examples/parallel/) — concurrent branches + typed `State`.
- [map_repair_plans](https://github.com/bzdvdn/teff/tree/master/examples/map_repair_plans/) — dynamic fan-out
  (`Map`) + `{key}` prompt templates.
- Guides: [YAML workflows](../guide/yaml-workflows.md),
  [Flow YAML (sugar)](../guide/flow-yaml.md),
  [Flow builder](../guide/flow-builder.md), [State](../guide/state.md),
  [Structured output](../guide/structured-output.md).

*You should know:* how to write `workflow.yaml` (and when the concise
`flow.yaml` idiom surface is a better fit), when to prefer YAML vs the
Flow DSL, how reducers merge branch results, and how `Map` differs from
`Parallel`.

## Stage 2 — Durable runs and humans in the loop (30 minutes)

Teff's differentiator: a run is **checkpointable and resumable**, and it can
**pause for approval**.

- [checkpoint_resume](https://github.com/bzdvdn/teff/tree/master/examples/checkpoint_resume/) — crash/resume in a
  few lines.
- [checkpoint_stores](https://github.com/bzdvdn/teff/tree/master/examples/checkpoint_stores/) — file / SQLite / PG.
- [human_in_loop](https://github.com/bzdvdn/teff/tree/master/examples/human_in_loop/) — `Interrupt` + `loop()` +
  resume (Python and YAML).
- [ask_strategies](https://github.com/bzdvdn/teff/tree/master/examples/ask_strategies/) — validate interrupt
  answers (`Ask`: regex, equals, LLM classifier) — offline.
- [agent_approval](https://github.com/bzdvdn/teff/tree/master/examples/agent_approval/) — every tool call pauses
  for human sign-off.
- Guide: [Durable (checkpoints)](../guide/durable.md).

*You should know:* how to add a `checkpoint:` block, how resume works, the
difference between an interrupt and a retry, and when a workflow *must* be
durable.

## Stage 3 — Agents and tools (45 minutes)

Composition: agent loops, tool calling, skills, and MCP.

- [react_agent](https://github.com/bzdvdn/teff/tree/master/examples/react_agent/) — ReAct loop + calculator tool +
  streaming.
- [harness_agent](https://github.com/bzdvdn/teff/tree/master/examples/harness_agent/) — parallel tool calls in one
  round + `__error__` fallback.
- [skills](https://github.com/bzdvdn/teff/tree/master/examples/skills/) and
  [pdf_agent](https://github.com/bzdvdn/teff/tree/master/examples/pdf_agent/) — `SKILL.md` folders and vendored
  tool-bundling skills.
- [mcp](https://github.com/bzdvdn/teff/tree/master/examples/mcp/) — call tools from an MCP server (stdio).
- Guides: [Agents](../guide/agents.md), [Skills](../guide/skills.md),
  [Plugins](../guide/plugins.md), [Tools](../reference/tools.md).

*You should know:* `Agent` vs `ReActAgent`, how to register custom tools and
node types, how skills scope tools, and how to bridge MCP servers.

## Stage 4 — Retrieval and memory (30 minutes)

Grounded answers and long-term state.

- [rag_search](https://github.com/bzdvdn/teff/tree/master/examples/rag_search/) — RAG over a local CSV,
  in-memory store.
- [rag_stores](https://github.com/bzdvdn/teff/tree/master/examples/rag_stores/) — the same RAG agent on every
  vector store.
- [memory_assistant](https://github.com/bzdvdn/teff/tree/master/examples/memory_assistant/) and
  [memory_chat](https://github.com/bzdvdn/teff/tree/master/examples/memory_chat/) — long-term memory.
- Guides: [RAG](../guide/rag.md), [Long-term memory](../guide/memory.md).

*You should know:* how to pick a vector store, how the RAG tool is wired,
and when memory vs RAG is the right answer.

## Stage 5 — Production concerns (45 minutes)

Validation, resilience, observability, testing, evaluation.

- [streaming](https://github.com/bzdvdn/teff/tree/master/examples/streaming/) and
  [structured_output](https://github.com/bzdvdn/teff/tree/master/examples/structured_output/) — live events and
  schema-validated JSON.
- [self_refine](https://github.com/bzdvdn/teff/tree/master/examples/self_refine/),
  [plan_and_execute](https://github.com/bzdvdn/teff/tree/master/examples/plan_and_execute/),
  [deep_research](https://github.com/bzdvdn/teff/tree/master/examples/deep_research/) — multi-step agent patterns
  (mocked, no API key).
- [time_travel](https://github.com/bzdvdn/teff/tree/master/examples/time_travel/) — rewind / edit / replay a run.
- [release_features](https://github.com/bzdvdn/teff/tree/master/examples/release_features/) — validation, typed
  errors, `teff eval`, cost reports, response cache.
- Guides: [Testing](../guide/testing.md), [Evaluation](../guide/evaluation.md),
  [Observability](../guide/observability.md), [Logging](../guide/logging.md),
  [Best practices](../guide/best-practices.md).

*You should know:* how to test a graph without the LLM, how to gate a release
on `teff eval`, how to read the trace dashboard, and how to budget tokens.

## Stage 6 — Full applications (deploy as a reference)

Real, closed, production-shaped apps to copy.

- [repair-ai-chat](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/) — FastAPI app:
  `route()` supervisor, five agents, tools, RAG, streaming.
- [service_desk](https://github.com/bzdvdn/teff/tree/master/examples/applications/service_desk/) — FastAPI +
  RAG + run management.
- [fraud_gate](https://github.com/bzdvdn/teff/tree/master/examples/applications/fraud_gate/) — review/publish flow
  with HITL and durable checkpoints.
- [repo-health](https://github.com/bzdvdn/teff/tree/master/examples/applications/repo-health/),
  [github-reviewer](https://github.com/bzdvdn/teff/tree/master/examples/applications/github-reviewer/),
  [gitlab-reviewer](https://github.com/bzdvdn/teff/tree/master/examples/applications/gitlab-reviewer/) — CI-adjacent
  automations.
- Scaffolding: `teff new` (fastapi/cli/daemon variants).

*You should know:* how a real app is structured (config → container → graph →
API → storage), and where each piece of the framework appears.

Instead of re-reading these, see the [Recipes](../recipes/index.md) —
each closes a real business case (fraud review, release approval, support
triage, ops daemon) in one consistent pattern.

---

## "I have a problem X" — where to look

Not sure which example matches your need? Start here.

| I want to…                                        | Read / run                                                |
| ------------------------------------------------- | --------------------------------------------------------- |
| Write my first workflow                            | [quickstart](quickstart.md), [basic_pipeline](https://github.com/bzdvdn/teff/tree/master/examples/basic_pipeline/) |
| Branch on a condition                              | [branching](https://github.com/bzdvdn/teff/tree/master/examples/branching/)                    |
| Run steps in parallel                              | [parallel](https://github.com/bzdvdn/teff/tree/master/examples/parallel/)                      |
| Fan out over a list of inputs                      | [map_repair_plans](https://github.com/bzdvdn/teff/tree/master/examples/map_repair_plans/)      |
| Survive crashes / resume a run                     | [checkpoint_resume](https://github.com/bzdvdn/teff/tree/master/examples/checkpoint_resume/), [durable](../guide/durable.md) |
| Add human approval                                | [human_in_loop](https://github.com/bzdvdn/teff/tree/master/examples/human_in_loop/), [agent_approval](https://github.com/bzdvdn/teff/tree/master/examples/agent_approval/) |
| Build an agent that uses tools                     | [react_agent](https://github.com/bzdvdn/teff/tree/master/examples/react_agent/), [agents](../guide/agents.md) |
| Run several tool calls at once                     | [harness_agent](https://github.com/bzdvdn/teff/tree/master/examples/harness_agent/)            |
| Give an agent a skill / a bundle of tools          | [skills](https://github.com/bzdvdn/teff/tree/master/examples/skills/), [pdf_agent](https://github.com/bzdvdn/teff/tree/master/examples/pdf_agent/) |
| Call tools from an MCP server                      | [mcp](https://github.com/bzdvdn/teff/tree/master/examples/mcp/)                               |
| Answer from my own documents                       | [rag_search](https://github.com/bzdvdn/teff/tree/master/examples/rag_search/), [rag](../guide/rag.md) |
| Keep long-term user context                        | [memory_assistant](https://github.com/bzdvdn/teff/tree/master/examples/memory_assistant/), [memory](../guide/memory.md) |
| Stream tokens to the UI                            | [streaming](https://github.com/bzdvdn/teff/tree/master/examples/streaming/)                   |
| Force valid JSON output                            | [structured_output](https://github.com/bzdvdn/teff/tree/master/examples/structured_output/)   |
| Route between multiple agents                      | [simple_router](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/), [supervisors](../guide/supervisors.md) |
| Test without the LLM                               | [testing](../guide/testing.md), [simple_router](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/) |
| Evaluate / gate a release                          | [release_features](https://github.com/bzdvdn/teff/tree/master/examples/release_features/), [evaluation](../guide/evaluation.md) |
| Debug a run after the fact                         | [observability](../guide/observability.md), [time_travel](https://github.com/bzdvdn/teff/tree/master/examples/time_travel/) |
| Ship a FastAPI app                                 | [repair-ai-chat](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/), `teff new fastapi` |

Before you add a new example, re-check this list — the pattern you need is
probably already documented.
