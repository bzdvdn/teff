# Best practices

Practical guidance for building reliable, testable, observable Teff
workflows. These rules converge from the framework's own examples and
tests; follow them for maintainable agents.

## Choose the right document format first

Teff ships two declarative formats, auto-detected by the CLI and the loaders
(`looks_like_flow`). Pick the smaller one that expresses your problem, and
escalate only when you need explicitness:

| You want… | Use | Reference |
| --------- | --- | --------- |
| Concise idioms — teams, loops, gates — with less boilerplate | **`flow.yaml`** | [flow.yaml reference](flow-yaml.md) |
| Every node and arrow explicit, or a compiled/deployable artifact | **`graph.yaml`** | [graph.yaml reference](yaml-workflows.md) |
| Code-first construction that still exports to YAML | **`Flow` builder** | [Flow builder](flow-builder.md) |
| Custom node classes, dynamic wiring, full runtime control | **`Graph` API** | [Graph API](graph-api.md) |

Rule of thumb: author in `flow.yaml`, inspect and review the compiled
`graph.yaml` (`teff graph -m`), and export (`Flow.to_yaml()`) when ops needs
the artifact. Both formats validate with `teff validate`, so switching is
cheap — the compiled output always targets the same node/edge vocabulary.

## Express as much as possible in YAML

The declarative YAML formats cover graph topology,
LLM/ReAct/supervisor nodes, conditional edges, reducers, built-in tools,
RAG & memory stores, `retry:`, `observability:`, `checkpoint:`, `hooks:`
and `${ENV}` secrets. Start there — a YAML workflow is inspectable
(`teff graph -m`), validatable (`teff validate`), and portable.

Reach for Python (the `Flow` builder or bare `Graph`) only when you need:

- custom node types or tools (or register them via `plugins:` to keep YAML);
- callable edge conditions / custom reducers / `Ask.check` predicates;
- custom checkpointer, observer, hook or `redact` functions;
- conversational multi-turn serving (`Assistant`, `graph.run(message=...)`).

Rule of thumb: if it's graph shape or configuration, put it in YAML; if
it's a bespoke piece of logic, isolate it in a plugin file and reference it
by name.

## Reference the field tables when you change a file

The two format references list every key, its type, its default and which
code path consumes it — use them to find *where to change what*:

- [`flow.yaml`](flow-yaml.md) — top-level keys, every idiom (`llm:`,
  `transform:`, `context_builder:`, `append_assistant:`, `agent:`, `team:`,
  `supervisor:`, `parallel:`, `map:`, `loop:`, `interrupt:`, `branch:`,
  `route:`, `type:`), node-level steps, `providers:`, `tools:`, `state:`,
  `checkpoint:`, `hooks:`, `observability:`.
- [`graph.yaml`](yaml-workflows.md) — top-level keys, `steps:` node types,
  `edges:` conditions, `retry:`, subflows, `include:`, `command`, `loop`,
  interrupts with `strategy:`.

## Keep nodes small and functions pure

A node should *receive state → return state*. Prefer returning the delta
over mutating shared state in place: nodes inside `parallel` branches run
on isolated copies, so in-place writes don't propagate. Side effects
(HTTP, files, databases) belong in `Tool`s or leaf nodes, never inside
prompts.

## Reducers define the merge contract up front

Declare per-key reducers in `state.schema` (`override`, `append`, `keep`)
before wiring branches or a supervisor. `append` prevents parallel
branches and conversation turns from clobbering one another (e.g. a shared
`messages` list). A key with no reducer overwrites by default — make that
explicit so multi-branch graphs behave predictably.

## Make runs durable for anything long or interactive

Enable a `checkpoint:` block (or pass a checkpointer programmatically)
for agentic loops, human-in-the-loop gates, or daemons. A crashed or
interrupted run then resumes on the next `--checkpoint-id` instead of
restarting. Give each tenant their own `owner` so sessions never collide.

## Guard against runaway loops

Always set `max_iterations` on agentic/supervisor graphs and prefer
explicit `done_keys`/`fill_order` guards and edge conditions over relying
on the prompt to terminate. Add `retry: {retry_on:[429], backoff:...}` to
LLM and HTTP tool nodes, and route failures through `__error__` edges to
a `fallback` node rather than failing the whole run.

## Keep prompts free of secrets

Pass keys via `${ENV_VAR}` in YAML or environment-specific tool config,
never baked into `system:`/`prompt:` text. Strip anything sensitive from
state with a redacting observer before persistence and telemetry export.

## Observe in two layers

- **Local trace**: `teff run -t` prints a JSON run trace (nodes, latencies,
  token usage) — enough for debugging.
- **Observability block**: `observability:` with a `db:` records full runs
  to a dashboard you inspect with `teff obs-server`; add `export:`
  (`langfuse`/`langsmith`/`webhook`) for remote. Wire named `hooks:` for
  custom metrics instead of scattering logging through node bodies.

## Test the graph, not the prompts

Use `with_llm_mock`/the harness to pin node outputs, then assert final
state + side effects. Test branch merges, reducers, error edges and
timeouts deterministically; keep the LLM out of the critical assertions
and test prompt fragility separately with the evaluation harness
(`teff eval`).

## Structure a real app

- `workflow.yaml` (or `flow.yaml`) — the graph + config;
- `plugins/` — registered custom nodes/tools/hooks;
- `prompts/` — prompt text kept out of code;
- `data/` — checkpoints, traces, stores (gitignored);
- `tests/` — state/harness tests.

Keep prompts in their own modules so they can be versioned, reviewed and
swapped independently of logic. Name the file `flow.yaml` when you author in
the idiom surface and `graph.yaml` for the compiled/explicit form — both are
auto-detected, so renaming is safe.