# Choosing your abstraction

Teff gives you several ways to build the same graph. The rules below are the
decision map — pick the *smallest* abstraction that expresses your problem,
and escalate only when you need more control. "Smaller" here means: more
declarative, more checked, less boilerplate.

## 1. Build the graph: YAML vs Flow DSL vs `graph.py`

All three produce the same `Graph`. They differ in where the structure lives
and how much control you have.

| You want to…                                      | Use                |
| ------------------------------------------------- | ------------------ |
| Declare topology as data, own it via git/diff     | **YAML** (`workflow.yaml`) |
| Build ergonomically from Python, still export to YAML | **Flow DSL** (`Flow`) |
| Use custom node classes, dynamic wiring, full runtime control | **`graph.py`** (`Graph` + `Node`) |

Rules of thumb:

- **Start in YAML** if the workflow is mostly built-in nodes (`llm`,
  `transform`, `map`, `parallel`, `interrupt`) and you want ops ownership of
  the topology. `teff validate` checks it; `observability:`/`checkpoint:` are
  one line.
- **Use the Flow DSL** when you are already in Python: `flow.llm(...)` /
  `flow.transform(...)` / `flow.branch(...)` / `flow.map(...)` give you
  type-checked construction, and `flow.to_yaml()` still lets you export the
  graph to YAML when ops needs it.
- **Drop to `graph.py`** when you need a custom `Node` subclass, imperative
  wiring (edges computed at build time), or hooks only the low-level API
  exposes. YAML and Flow are views over the same runtime — you can mix them
  (`from_yaml()` → run, `flow.compile()` → same engine).

## 2. Pick the agent shape

The most common confusion. An "agent" in Teff is just a node that calls an
LLM in a loop.

| Shape               | What it does                                         | Use when… |
| ------------------- | ---------------------------------------------------- | --------- |
| **`LLM`** node      | One model call, no loop                              | A single step in a larger graph |
| **`Agent`**         | Conversational loop (multi-turn chat state)          | Chat assistant; you manage turns via state |
| **`ReActAgent`**    | Tool-calling loop (think → act → observe → repeat)   | Agent that must call tools to answer |
| **`ToolExec`**      | Executes a single tool call from state               | Build your own tool loop at low level |
| **`route()`** (Flow) | Supervisor: route `state[key]` to one of N sub-agents | Multi-agent / heterogeneous workers |
| **`harness()` / `react()`** (Flow) | Opinionated agent loops with tool orchestration | Get a capable agent in 2 lines |
| **`Supervisor`**    | Node that delegates one turn to a worker             | Hand-rolled supervisor with custom policy |

Decision flow:

1. One model call? → `LLM`.
2. Loops, and needs tools? → `ReActAgent` (or `Flow.react()`).
3. Multiple specialist agents? → `Flow.route()` (see
   [supervisors](../guide/supervisors.md)).
4. Loop but no tools, chat-like? → `Agent`.
5. Need full control? → build a `@node`-based loop with `ToolExec`.

## 3. Concurrency and fan-out

| Construct          | Semantics                                                  | Use when… |
| ------------------ | ---------------------------------------------------------- | --------- |
| **`Parallel`**     | Fixed set of branches run concurrently; results merged by reducers | You know the branches at build time |
| **`Map`**          | Dynamic fan-out over `state[input_keys]`, then reduce      | The number of items is only known at runtime |
| **`SubFlow` / `add_flow()`** | Reuse a sub-graph as one node                      | Sharing a reusable sub-workflow |

`Map` and `Parallel` are not interchangeable: `Map` iterates a runtime list
(`plan` → one pass per plan item), `Parallel` fans out fixed branches (e.g.
summarize `title` and `body` at once). If the fan-out is data-driven, use
`Map`; if it is structural, use `Parallel`.

## 4. Durable state: checkpoints

| Checkpoint type    | Persistence          | Use when… |
| ------------------ | -------------------- | --------- |
| none               | in-memory            | Short, stateless demo graphs; tests |
| `file`             | JSON files on disk   | Dev, simple deploys, no DB wanted |
| `sqlite`           | single `.db` file    | Default production choice; multi-tenant via `owner` |
| `pg`               | PostgreSQL           | Multi-instance, central state, shared DB |
| `sqlite_history` / `pg_history` | full run history | Time travel (`time_travel` example), audit |

Rules:

- Anything interactive or long-running **must** be durable — add
  `checkpoint:` (or a `Checkpointer` in Python).
- `owner` scoping gives you multi-tenant checkpoints on one store
  (`JSONFileCheckpointer`/`SQLiteCheckpointer`/`PGCheckpointer` take an owner).
- Start with `sqlite`, move to `pg` when you outgrow one process.
  See [durable](../guide/durable.md).

## 5. Humans in the loop

| Approach            | What happens                                    | Use when… |
| ------------------- | ----------------------------------------------- | --------- |
| `Flow.interrupt(key)` | Run pauses, resumes with a new value          | Approve / edit a result (HITL) |
| **`Agent` approval**  | Each tool call requires sign-off              | Safety-critical tool usage |
| **`Ask`** validation  | Validate the answer (regex/equals/LLM) before resuming | Bad input must be rejected/re-asked |

Combine them: interrupt for *who decides*, `Ask` for *is the answer valid*.
See [human_in_loop](https://github.com/bzdvdn/teff/tree/main/examples/human_in_loop/) and
[ask_strategies](https://github.com/bzdvdn/teff/tree/main/examples/ask_strategies/).

## 6. Resilience

| Mechanism               | Scope          | Use when… |
| ----------------------- | -------------- | --------- |
| Workflow `retry:` / `Retry` node | step-level retries with backoff | Transient failures (rate limits, timeouts) |
| `__error__` edge / `Fallback`     | catch and redirect on failure       | Handle a failing node explicitly |
| `Extract`               | repair malformed LLM output          | Structured output that can fail |

Layer them: `Retry` for *transient*, `__error__` for *expected-but-handleable*,
model failover at the provider level for *whole-model* issues.

## 7. Retrieval

| Vector store     | When to use |
| ---------------- | ----------- |
| `memory` / `sqlite` | Local dev, small corpora, no external service |
| `chroma`/`faiss`/`lance`/… | Feature parity on your infra |
| `qdrant`/`milvus`/`weaviate`/`pinecone`/`pgvector` | Scale, teams, managed services |

Pick the store, not the app: the RAG agent is identical across stores —
see [rag_stores](https://github.com/bzdvdn/teff/tree/main/examples/rag_stores/).

---

## Quick-reference matrix

| I need to…                                  | Start with            | Escalate to            |
| ------------------------------------------- | --------------------- | ---------------------- |
| Declare a pipeline                          | YAML                  | Flow DSL               |
| Custom node logic                           | `@node` / `Node`      | `graph.py` wiring      |
| One tool-using agent                        | `Flow.react()`        | `ReActAgent` / custom loop |
| Several agents                              | `Flow.route()`        | `Supervisor` node      |
| Fixed parallel work                         | `Parallel`            | —                      |
| Fan out over runtime data                   | `Map`                 | —                      |
| Long/interactive run                        | `checkpoint: sqlite`  | `pg` + history         |
| Pause for human input                       | `Flow.interrupt()`    | `Ask` validation       |
| Survive flaky LLMs                          | workflow `retry:`     | provider failover      |
| Ground answers in documents                 | `RAGTool` + sqlite    | managed vector store   |
