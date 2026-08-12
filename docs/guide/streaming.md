# Streaming execution

`graph.stream()` runs the same execution core as `graph.run()` but yields a
`StreamEvent` for every observable step, so callers can render tokens and
progress before the run finishes:

```python
from teff.flow import Flow
from teff.node import LLM
from teff.provider import ProviderRegistry

flow = Flow(
    "chat",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.step(LLM(prompt="Скажи привет", output_key="answer"))
graph = flow.compile()

async for event in graph.stream(state):
    if event.type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.type == "run_end":
        print("\nstatus:", event.data["status"])
```

Event types: `run_start`, `node_start`, `node_end`, `node_error`, `edge`,
`token`, `llm`, `structured`, `interrupt`, `interrupt_resume`, `checkpoint`,
`run_end`.

LLM tokens are emitted as they arrive (any node without tool calls streams
automatically in this mode); routing decisions, checkpoints, and interrupt
pauses stream the same way. `stream()` accepts the same parameters as
`run()` — tools, checkpointer, resume, tracer, `max_iterations`.

For chat applications, `stream(message=...)` drives one durable conversation
turn: a paused session auto-resumes with the message, and a re-work pause
surfaces an `interrupt` event (with `key` and the question you'll show the
operator in `question` — the same value is mirrored under `prompt` for
compatibility) where the stream ends — call `stream(message=...)` again
with the operator's answer.
See [Durable execution](durable.md#conversation-turns-runmessage).

## Observability (telemetry)

Pass a `RunTracer` to `graph.run()` to collect a JSON-serialisable event log:
node start/end with latency, edge routing, checkpoints, retries, and LLM token
usage. Fold it into a summary afterwards.

```python
from teff import Graph, RunTracer

tracer = RunTracer()
await graph.run(state, tracer=tracer)

print(tracer.to_json())  # {"summary": {...}, "events": [...]}
print(tracer.summary())  # RunSummary(status, total_ms, nodes, tokens, ...)
```

The CLI exposes the same report: `teff -f workflow.yaml --trace`. Cost and
token accounting live in [Providers](../reference/providers.md#cost-token-reports).