# Why Teff

> **Workflow as data. Agents as graphs.**

Before we tell you what Teff *is*, let's talk about the problem it exists to
solve — and why the obvious ways of building agents quietly stop working the
moment your workflow outgrows a single function.

## The problem: agents stop being trustworthy the moment they become real

The first LLM call is easy. The first *agent* is where most projects start
bleeding:

- **LLM output is non-deterministic — and so is your application.** A
  conditional that lives inside a prompt loop (`"if the user seems angry…"`)
  cannot be tested, traced, or repeated. You get a different answer every run
  and no way to explain *why*.
- **Business flow is buried in control-flow code.** The moment you have two or
  three agents, branching, retries and error handling become `if`/`else` and
  `try`/`except` scattered across `async` functions. The logic of the
  application is now *implicit* — you can't read it, you can only step through
  it in a debugger.
- **A crash means starting over.** A long pipeline dies at step 40 of 42 and
  the entire run state is gone, because it lived in process memory. You re-run
  everything, re-pay for every LLM token, and hope it works this time.
- **You are flying blind.** No per-node timeline, no prompt/response log, no
  way to replay "the run that returned the wrong answer two hours ago".
- **Humans have no seat at the table.** The one tool call that needs
  sign-off either fires silently, or the whole agent collapses because pausing
  to ask was never designed in.

None of this is about "bad code". They are the natural failure modes of a
single pattern: *execution logic embedded in imperative code*.

## The alternative: the graph owns the behaviour

Teff is a Python library that moves the intelligence out of the code and into
the **structure**. A workflow is a graph: nodes that transform state, edges
that decide what runs next, and a runtime that owns execution.

- The **topology is data** — YAML/JSON you can version, diff, validate and
  inspect without running anything.
- Every branch, retry, checkpoint and interrupt is **visible in the graph**,
  never hidden inside a loop.
- The **runtime owns execution**: scheduling, retries, checkpointing,
  parallelism, resuming, pauses, streaming, observability. You write business
  logic — never execution infrastructure.

That is where the tagline comes from: *Workflow as data. Agents as graphs.* An
agent is just a graph the runtime happens to execute — inspectable,
recoverable and testable, at any scale.

## The pain it answers

| The pain (from real projects)                              | How Teff answers it                                                    |
| ---------------------------------------------------------- | ---------------------------------------------------------------------- |
| "One run picks a branch, the next doesn't — I can't reproduce it" | Branching lives on **edges**, a deterministic diagram, not a hidden `if` |
| "The workflow logic is buried in nested `async` loops"   | The flow is data: you read the YAML, not the stack            |
| "He crashed at step 40 — everything restarted from scratch" | **Resume from the node that failed** — finished work is never re-run |
| "A risky tool call needs a human before it fires"        | `Interrupt` — pause mid-run, review, resume from that point       |
| "I neither know it failed, nor what the LLM actually saw" | A checkpointed trace of every node, tool call and LLM request  |
| "I want Ollama locally today, OpenAI in prod tomorrow"    | Raw-HTTP providers (Ollama, OpenAI-compatible, Anthropic) — no SDKs, no lock |
| "The "framework" really means a platform I must host"     | A library you import, not a server you operate                       |

## The killer feature: resume from the node that failed

Most "durable" flavours restart the whole pipeline when anything breaks. Teff
is different: a checkpoint is written **before every node**. If a run crashes,
calling `run()` with the same `checkpoint_id` loads the last safe checkpoint
and continues **from the exact node that failed** — every completed node stays
completed.

```python
from teff import Graph
from teff.checkpoint import SQLiteCheckpointer
from teff.node import Transform

nodes = {"shout": Transform(action="uppercase", input_key="text", output_key="loud")}
graph = Graph(nodes, edges=[], entry_point="shout")
cp = SQLiteCheckpointer("checkpoints.db")

# first run dies somewhere in the middle of the graph
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")

# same id: resumes from the exact node that crashed, and finishes
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")
```

Only the failed node is re-run (it never finished); everything before it is
preserved. That means **fewer LLM calls**, **faster recovery**, and process
failure becoming an annoyance rather than a catastrophe.

The same checkpoints power three more workflows: human-in-the-loop interrupts
(pause → operator → resume), run inspection (`teff inspect`), and time travel
(replay a finished run from any checkpoint, branch, and continue).

## From a script to a full product — on the same primitives

The interesting thing is that this isn't only for the small graphs. The same
five primitives — nodes, edges, checkpoints, interrupts, reducers — compose
into production systems.

- **Multi-agent supervisor.** A `route()` supervisor picks an agent per round
  and loops until the work is done — planning, coding, QA and an approval
  gate, all durable. It is described as a plain flow and a team of sub-flows
  ([`examples/supervisor_complex`](https://github.com/bzdvdn/teff/tree/master/examples/supervisor_complex/) runs it
  offline with no API key).
- **A five-agent customer repair assistant.** RAG over a materials catalog,
  tools, streaming, a trace dashboard and a FastAPI app — a real, runnable
  composition: [`examples/applications/repair-ai-chat`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/).
- **Support triage with human escalation.** A ticket is grounded in your
  knowledge base, a supervisor routes it to a specialist, then an escalation
  gate pauses for a human when confidence is low — routing is index-something:
  [`docs/recipes/support-triage.md`](recipes/support-triage.md).

The building blocks that made the small example above (a `Transform` node and
an SQLite checkpointer) are the same ones that make a five-agent assistant
work. The complexity does not leak into the core.

Even the same supervisor can be pure YAML — a team of agents included from
their own files, quality-gated by a Map + loop, with an operator interrupt:

```yaml
name: supervisor_complex
include: [team/lead.yaml, team/planner.yaml, team/coder.yaml]
steps:
  - id: review_quality       # LLM review step
    type: llm_chat
    config: { model: llama3.1:8b, output_key: critique, parse: true }
  - id: route               # conditional routing on the verdict
    type: command
    config:
      routes:
        - { when: "verdict=pass", goto: approve }
        - { when: "verdict=needs_work", goto: refine }
  - id: refine              # product-rewrite loop with parallel fixes
    type: loop
    config:
      key: verdict
      until: pass
      max_rounds: 3
      body: [{ type: map, input_keys: [issues], output_key: fixes, processor: ... }]
  - id: approve             # human-in-the-loop gate
    type: interrupt
    config: { key: approved, strategy: { any_of: [approve, ok, yes] } }
```

Full version: [`examples/supervisor_complex/workflow.yaml`](https://github.com/bzdvdn/teff/blob/master/examples/supervisor_complex/workflow.yaml).

## The principles behind it

Teff was designed around a handful of firm rules:

- **The graph owns behaviour.** Business logic lives in the structure, never in
  prompts or hidden code.
- **Nodes transform state.** A node cannot tell "run X"; it only describes how
  the state should change.
- **The runtime owns execution.** Scheduling, retries, durability and
  observability are the framework's job, not yours.
- **State is a plain dict.** One serializable snapshot per run, no hidden
  behaviour.

The full frame is in [the Constitution](https://github.com/bzdvdn/teff/blob/master/CONSTITUTION.md).

## Not a comparison chart — a comparison of patterns

Vendors change, names change, and a table "Teff vs X" goes stale quickly. The
comparison that stays true is the one against the pattern you would otherwise
reach for:

|                        | Imperative loops | Big platform SDKs | **Teff** |
| ---------------------- | ---------------- | ----------------- | -------- |
| Flow is visible        | No (in code)     | Usually yes       | **First-class — the graph IS the app** |
| Durable / resumable    | No               | Yes, often on their runtime | **Yes, no server: file/SQLite/PG** |
| Crash recovery         | Start over       | Resume at job level | **Resume from the failed node** |
| Human-in-the-loop      | Hand-rolled      | Available | `Interrupt` → resume, a step like any other |
| Dependencies           | Your code only   | Heavy SDK + runtime | **4 core runtime deps, no SDKs, raw HTTP** |
| Embeddable in your app | Yes | Only on their platform | **Yes — you import us, a library** |
| Observability          | print/log        | Their dashboard, you write adapters | **Built in — traces, token usage, cost** |
| Vendor lock-in         | None             | Strong            | **None** |

That's the trade-off we are honest about: you don't get a "library of literally
everything". You get **structure, durability and instrumentability** — without
operating any server.

## What we won't promise

Honest boundaries build trust:

- Teff is **not** a batch/ETL scheduler. A large job orchestrator is better
  served by dedicated tools.
- Teff is **not** an "everything but the kitchen sink" SDK with an integration
  for every vendor. We trade that ecosystem for minimal-to-go, inspectable,
  no-lock-in design.
- Teff is **not** a managed platform. Many of the biggest dependencies of an
  app are easy to own precisely because it stays a library.

## When it fits

- You are shipping agent flows **to production**, where a silent wrong answer
  is unacceptable — where you need to observe, replay and pause.
- You want the flow to be **reviewable data**, not a pile of `async`
  functions nobody can read.
- You are **local-first** (Ollama in dev, any OpenAI-compatible endpoint in
  prod) and want no SDK or platform lock-in.
- You run workflows from **Docker, cron, a queue or a FastAPI app** — as a
  caller, not a host.

## Get started

- [Install](getting-started/install.md)
- [Quick start](getting-started/quickstart.md)
- [Durable runs (checkpoints)](guide/durable.md)
- [The Constitution — the principles](https://github.com/bzdvdn/teff/blob/master/CONSTITUTION.md)