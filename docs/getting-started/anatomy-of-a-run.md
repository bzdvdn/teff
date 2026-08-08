# Anatomy of a run

What actually happens between `graph.run(...)` and the result you get back?
Everything below is real output from
[`examples/hello_workflow/`](https://github.com/bzdvdn/teff/tree/main/examples/hello_workflow/) — a run with no
LLM, so you can replay it locally in seconds.

## 1. The CLI trace

```bash
teff run --file examples/hello_workflow/workflow.yaml --trace
```

prints this trace to stderr:

```
run_start
node_start   count
node_end     count
edge         count → single
node_start   single
node_end     single
edge         single → status
node_start   status
node_end     status
run_end      status: "ok"
```

A run is exactly this sequence, repeatedly: **start → run nodes → follow edges
→ end**. The `summary` block on top of the trace is the first thing to read:

```json
{"status": "ok", "total_ms": 0.128, "node_count": 3,
 "llm_calls": 0, "cost_usd": 0.0, "nodes": {"count": {"runs": 1, ...}}}
```

It tells you: did it finish, how long, how many nodes ran, how many model
calls happened, and what they cost.

## 2. The phases of a run

1. **Planning** — `graph.run(state, reducers, ...)` resolves the entry node
   and prepares an isolated copy of the state (reducers handle per-key merge).
2. **Execution** — the scheduler picks nodes reachable from the entry point,
   runs them (optionally in `parallel` / `map` fan-out), and follows the edges
   their outputs trigger (including `condition`-guarded edges).
3. **Merging** — each node returns the keys it produced; they are merged back
   with the state's reducers (e.g. `append` for message lists).
4. **Completion** — `run_end` fires with a status; the final state is returned
   (or thrown, if a node failed and no `__error__` edge caught it).

With a checkpointer (see below) a checkpoint is saved **before every node**, so
phase 2 can always be re-entered.

## 3. What model calls add

When a node calls an LLM, the trace gains `llm` events and the summary shows
`llm_calls` and `cost_usd`. A model call is just another node that happens to
have a cost — the lifecycle above does not change. That's the whole mental
model: **agents are graphs; an LLM call is a node in the graph.**

## 4. Durability: checkpoints and interrupts

With a checkpointer, the lifecycle gains two events:

- `checkpoint` — written before each node, so a crash is never a lost run;
- `interrupt` — a run that reaches an `Interrupt` node **pauses**:
  `GraphInterrupt` is raised, the run ends with `status: "interrupted"`, and
  the checkpoint records exactly where it stopped.

```python
try:
    result = await graph.run(state=state, checkpointer=cp, checkpoint_id="sess-1")
except GraphInterrupt as exc:
    print(exc.prompt)  # "Эскалация: …"
    answer = input("> ")
    result = await graph.run(
        state=state,
        checkpointer=cp,
        checkpoint_id="sess-1",
        resume={"escalate": answer},
    )
```

The second `run` continues from the pending interrupt instead of restarting —
even in a different process, as long as it uses the same checkpointer and
`checkpoint_id`. See [durable](../guide/durable.md) for the full contract.

## 5. Failures

A node that raises becomes `node_error` in the trace; the run ends with
`status: "error"` **unless** an `__error__` edge catches it and reroutes. Retry
wrappers and fallback nodes change the *outcome*, not the trace shape — every
attempt is still a `node_start`/`node_end` pair. See
[errors](../guide/errors.md) for the playbook.

## 6. Streaming: the same lifecycle, event by event

`graph.stream(state)` is `run()` cut into events — `run_start`, `node_start`,
`node_end`, `node_error`, `edge`, `token`, `llm`, `structured`, `interrupt`,
`checkpoint`, `run_end`:

```python
async for event in graph.stream(state):
    if event.type == "token":
        print(event.data["token"], end="", flush=True)
```

The durable conversation-turn form — `run(message=...)` / `Assistant` — wraps
this same lifecycle: a paused session auto-resumes with the operator's answer,
and a fresh interrupt surfaces as an `interrupt` event instead of a raise.

## 7. Where to look next

- Watch your own runs: `teff run --trace`, or the
  [observability dashboard](../guide/observability.md).
- Make runs durable: [durable](../guide/durable.md).
- See the whole lifecycle in action with human-in-the-loop:
  [`examples/human_in_loop/`](https://github.com/bzdvdn/teff/tree/main/examples/human_in_loop/) and the
  runnable [`support_triage`](https://github.com/bzdvdn/teff/tree/main/examples/recipes/support_triage/).