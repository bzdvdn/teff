# Durable execution (checkpoints)

`Graph.run()` accepts a `checkpointer` and a `checkpoint_id`. A checkpoint is
written **before** every node, so a crash or error resumes from the last safe
point instead of starting over.

```python
from teff import Graph
from teff.checkpoint import SQLiteCheckpointer
from teff.node import Transform

nodes = {"shout": Transform(action="uppercase", input_key="text", output_key="loud")}
graph = Graph(nodes, edges=[], entry_point="shout")
cp = SQLiteCheckpointer("checkpoints.db")

# first run crashes at some node
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")

# same id resumes from the saved checkpoint and completes
await graph.run(state, checkpointer=cp, checkpoint_id="demo-run")
```

## Backends

| Backend | Extra | Notes |
| ------- | ----- | ----- |
| `JSONFileCheckpointer` | core | file-based, `owner/` subdirectories |
| `SQLiteCheckpointer` | core | stdlib SQLite, composite `(owner, checkpoint_id)` key |
| `PGCheckpointer` | `teff[pg-checkpoint]` | needs PostgreSQL |

On resume the saved state wins over the passed-in state; a `State` instance
keeps its schema and reducers.

## Multi-tenant checkpoints (owner scoping)

Pass `owner=` to scope checkpoints to a user/session/tenant. The same
`checkpoint_id` under different owners never collides, so one store can serve
many users:

```python
await graph.run(state, checkpointer=cp, checkpoint_id="chat-1", owner="alice")
await graph.run(state, checkpointer=cp, checkpoint_id="chat-1", owner="bob")  # separate

chats = await cp.list("alice")  # ["chat-1", ...] — enumerate a user's runs
```

**Use `owner` for anything that should be isolated per end-user.** File
checkpoints land in an `owner/` subdirectory; SQLite/PG store a composite
`(owner, checkpoint_id)` primary key (existing single-owner databases migrate
automatically).

When `owner` is omitted, runs fall under the default owner `"default"`
(`teff.checkpoint.DEFAULT_OWNER`). The CLI exposes the same knob:
`--checkpoint-owner` on `teff run` and `teff inspect` (defaults to `default`).

## Human-in-the-loop (interrupts)

Pause a workflow for operator input with an `Interrupt` node. When execution
reaches it, `graph.run()` raises `GraphInterrupt`; resume with the same
`checkpoint_id` plus a `resume` dict:

```python
from teff.checkpoint import JSONFileCheckpointer
from teff.node.interrupt import GraphInterrupt
from teff.flow import Flow
from teff.node import LLM
from teff.provider import ProviderRegistry

flow = Flow(
    "approval",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.step(LLM(prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.step(LLM(prompt="{draft}\nВердикт: {approved}", output_key="final"))

graph = flow.compile()
cp = JSONFileCheckpointer("checkpoints")

try:
    await graph.run(state=state, checkpointer=cp, checkpoint_id="run-1")
except GraphInterrupt as interrupt:
    print(interrupt.prompt)
    answer = input("> ")
    result = await graph.run(
        state=state,
        checkpointer=cp,
        checkpoint_id="run-1",
        resume={"approved": answer},
    )
```

The answer lands in `state["approved"]` and execution continues. Interrupts
require a `checkpointer`.

### Conversation turns: `run(message=...)`

For chat applications the interrupt bookkeeping is better handled for you.
Call `graph.run()` with a `message` and the checkpoint id as the *session id*:
the run auto-detects a paused session from durable state and either resumes
it with the message (the operator's answer) or starts/continues the
conversation. A pause is **not** raised — it is folded into a `TurnResult`:

```python
from teff.graph import TurnResult

session_id = "chat-1"

result: TurnResult = await graph.run(
    state={},
    message="Спланируй ремонт ванной 5 м².",
    checkpointer=cp,
    checkpoint_id=session_id,
    initial_state=lambda: {"task": "remont"},
)
if result.waiting:
    # the run paused on an Interrupt: surface result.prompt to the operator
    answer = input(result.prompt + " ")
    result = await graph.run(
        state={}, message=answer, checkpointer=cp, checkpoint_id=session_id
    )
else:
    print(result.reply)  # latest assistant reply from the durable state
```

The loop above survives any number of interrupts (e.g. a "rework" branch that
re-asks). `initial_state` seeds a fresh session, `transient_keys` are cleared
at the start of every turn, and `messages_key` names the message list.
`graph.stream(message=...)` is the streaming equivalent — a pause surfaces an
`interrupt` event (with `key` and the question in `question`, mirrored under
`prompt` for compatibility) where the stream ends.

### Validating the answer

A bare `interrupt` compares the resume value verbatim. To validate the answer
(and capture a value, e.g. a promo code) instead, pair the interrupt with an
[`Ask`](../reference/nodes.md#ask) strategy and re-ask until it passes with
`interrupt_loop`:

```python
from teff.node import Ask

flow.interrupt_loop(
    key="code",
    prompt="Введите промокод (формат XX-1234):",
    accept=Ask.regex(
        r"^[A-Z]{2}-[0-9]{4}$", decision_key="code_ok", value_key="discount_code"
    ),
    body=LLM(model="llama3.1:8b", prompt="Введите корректный код.", output_key="hint"),
    done=LLM(
        model="llama3.1:8b",
        prompt="Примени скидку {discount_code}.",
        output_key="final",
    ),
)
```

`Ask.llm(...)` adds an LLM classifier so free-form answers ("конечно", "ок")
count as approval. See the
[`ask_strategies` example](../examples.md) and the
[`validate` reference](../reference/nodes.md#validate).

### Revision loop

Wire a conditional cycle back to the `Interrupt` with `Flow.loop()`:

```python
flow.step(LLM(model="llama3.1:8b", prompt="Составь план: {task}", output_key="draft"))
flow.interrupt(key="approved", prompt="Одобрить? (да / правки)")
flow.loop(
    key="approved",
    until="да",
    done=LLM(model="llama3.1:8b", prompt="{draft}", output_key="final"),
    body=LLM(
        model="llama3.1:8b",
        prompt="Переработай {draft} с учётом: {approved}",
        output_key="draft",
    ),
)
```

`loop()` wires `decider --key=until--> done` (stop) and
`decider --key!=until--> body -> decider` (repeat). `max_iterations` caps the
rounds. The decider can be any node that writes `key`, not just an `Interrupt`.