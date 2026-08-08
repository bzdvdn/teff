# Tutorial — build your first workflow

This walks you through writing, running, and reading a real workflow. We use a
**deterministic** example — no LLM, no network, no API key — so you can run it
the moment `teff` is installed. Everything here lives, runnable, in
[`examples/hello_workflow/`](https://github.com/bzdvdn/teff/tree/master/examples/hello_workflow/).

---

## 1. The job we want to automate

Given a note (some `text`), decide whether it is a single-line or multi-line
note and tag it accordingly, then mark the job `done`.

That's deliberately boring. The point is the **shape** — a count, a branch, and
a converge — which every real workflow repeats.

## 2. Express it as YAML (the "workflow as data" view)

```yaml
name: hello-offline
state:
  initial:
    text: "just one line"

steps:
  - id: count
    type: transform
    config: {action: count_lines, input_key: text, output_key: lines}

  - id: single
    type: transform
    config: {action: value, value: single-line note, output_key: note}

  - id: multi
    type: transform
    config: {action: value, value: multi-line note, output_key: note}

  - id: status
    type: transform
    config: {action: value, value: done, output_key: status}

edges:
  - {from: count, to: single, condition: "lines=1"}
  - {from: count, to: multi, condition: "lines!=1"}
  - {from: single, to: status}
  - {from: multi, to: status}
```

The nodes are the `steps`; the arrows are the `edges`. An edge can carry a
`condition` — here `lines=1` routes to `single`, anything else to `multi`.

Run it with the built-in CLI:

```bash
teff run --file examples/hello_workflow/workflow.yaml
```

```
{"text": "just one line", "lines": "1", "note": "single-line note", "status": "done"}
```

That's the whole result: every node's `output_key` has landed on the state.
**YAML needs only the data — teff infers everything else from the node types.**

## 3. The same graph with the Flow builder

For workflows with real logic, the **Flow** DSL reads top-to-bottom
like a script. `Transform` is invoked as `.transform(...)`, and the branch is
`.branch("lines", Case("1").add(...), default=...)`; both ends re-join at
`.converge(...)`:

```python
import asyncio
from teff.flow import Flow
from teff.flow.case import Case
from teff.node import Transform

flow = (
    Flow("hello-offline")
    .transform(action="count_lines", input_key="text", output_key="lines", id="count")
    .branch(
        "lines",
        Case("1").add(
            Transform(action="value", value="single-line note", output_key="note"),
            id="single",
        ),
        default=Transform(action="value", value="multi-line note", output_key="note"),
    )
    .converge(Transform(action="value", value="done", output_key="status"), id="status")
)

graph = flow.compile()
result = asyncio.run(graph.run(state={"text": "two\nlines"}))
print(result["note"])  # multi-line note
```

Same topology — but now the code tells the story left-to-right.

## 4. The same graph, hand-wired

The low-level API is explicit about *every* node and *every* edge. Useful when
you need full control (custom edges, one-off behaviour):

```python
from teff.graph import Edge, Graph
from teff.node import Transform

graph = Graph(
    nodes={
        "count": Transform(action="count_lines", input_key="text", output_key="lines"),
        "single": Transform(
            action="value", value="single-line note", output_key="note"
        ),
        "multi": Transform(action="value", value="multi-line note", output_key="note"),
        "status": Transform(action="value", value="done", output_key="status"),
    },
    edges=[
        Edge("count", "single", "lines=1"),
        Edge("count", "multi", "lines!=1"),
        Edge("single", "status"),
        Edge("multi", "status"),
    ],
    entry_point="count",
)
result = asyncio.run(graph.run(state={"text": "only"}))
print(result["note"])  # single-line note
```

## 5. See the topology

Neither guess the shape — ask `teff`:

```bash
teff graph examples/hello_workflow/workflow.yaml
```

prints the same graph as Mermaid, and `teff run` shows the state through the whole run.

## 6. Where this leads

Now swap the `Transform` count/`value` nodes for real work:

- `llm_chat` / agent for a reasoning step,
- a `tool` (search, fetch) for an action,
- `checkpoint` to make it **durable** (pause on an `Interrupt`, resume later),
- `parallel` / `map` for fan-out.

Easiest next step: open
[`choosing-your-abstraction`](../guide/choosing-your-abstraction.md) to pick
when each view is right, then build the runnable
[`support_triage`](https://github.com/bzdvdn/teff/tree/master/examples/recipes/support_triage/) example (supervisor
+ human-in-the-loop + checkpoints), which uses this same branch-and-converge
shape under the hood.