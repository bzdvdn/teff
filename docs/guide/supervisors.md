# Multi-agent supervisors (`route` + `agent_step`)

The most powerful (and most complex) pattern in teff is the **supervisor
loop**: one *decider* node picks which specialist agent handles the next
turn; that agent runs and control returns to the decider, which picks again —
until it says `finish` and the loop exits. This is what
[`service_desk`](../examples.md) and the `teff new` scaffolds build.

Two building blocks make it short:

- [`Flow.route()`](#flowroute) — the supervisor wiring (conditional edges +
  loops back to the decider).
- [`agent_step()`](#agent_step) — a framework helper that wraps one specialist
  agent (context builder → ReAct harness → append reply to conversation) as a
  `SubFlow`, ready to plug into `route()`.

## `Flow.route()`

Wires the **last added node** (the decider) into a supervisor loop. The
decider writes a routing key (e.g. `next_agent`); each keyword in `agents`
maps a value of that key to the chain run for it. After the chain finishes,
control returns to the decider. When the key equals `"finish"`, the loop
exits through the optional `finish` chain.

```python
from teff.flow import Flow
from teff.node import LLM
from teff.provider import ProviderRegistry

flow = Flow(
    "support",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.step(
    LLM(
        output_key="next_agent",
        system="Reply 'planner', 'estimator' or 'finish'.",
    )
)  # decider
flow.route(
    "next_agent",  # key the decider writes
    finish=final_llm,  # run when key == "finish"
    planner=planner_chain,  # chain for key == "planner"
    estimator=estimator_chain,  # chain for key == "estimator"
)
```

The wiring this produces:

```
decider --next_agent=planner--> planner-chain -> decider
decider --next_agent=estimator--> estimator-chain -> decider
decider --next_agent=finish--> finish-chain -> (continue)
```

Chains can be a single node, a list of nodes (run sequentially), or a
`SubFlow` (e.g. from `agent_step()`). `finish` is optional — when omitted the
flow terminates on `"finish"` and nothing can be chained after `route()`.

The decider is **any node that writes the routing key** — an `LLM` node, a
`react_agent`, an `Interrupt`, or a custom node. `route()` never cares what it
is, only that the previous `flow.step(...)` wrote `key`.

## A ready-made decider: `Supervisor`

For the common case — "ask the model which agent, then route" — teff ships a
safe decider built in: `teff.node.Supervisor` (wired via `flow.supervisor()`).
It asks the model for a one-word reply, writes it to the routing key, and
applies deterministic guards so the loop can't hang and free-form text can't
silently end the graph:

```python
from teff.flow import Flow, agent_step
from teff.provider import ProviderRegistry

flow = Flow(
    "support",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
)
flow.supervisor(
    model="llama3.1:8b",
    sections={"plan": "План", "review": "Ревью"},
    route_keys={"planner": "plan", "reviewer": "review"},
    done_keys={"plan", "review"},
    fallback_agent="planner",
)  # writes "next_agent"
flow.route(
    "next_agent",
    finish=final_llm,
    planner=agent_step(PLANNER_PROMPT, "plan", model="llama3.1:8b"),
    reviewer=agent_step(REVIEWER_PROMPT, "review", model="llama3.1:8b"),
)
```

Guards provided, on top of the plain "reply a single word":

| Key | Default | What it guards |
| --- | ------- | -------------- |
| `route_keys` | `{}` | Map agent → output slot; an agent whose slot already has content is never re-run (no overwrite of finished work). |
| `done_keys` / `done_mode` | — / `"all"` | When these output slots are filled, return `finish` without another model call. `done_mode="any"` requires just one. |
| `fallback_agent` | `""` | Route to this agent when the model says `finish` (or fails to parse) before anything has been produced, so the user still gets an answer. |
| `finish` | `"finish"` | The terminator token the model answers with and that lands in `output_key` for the `finish` branch — set it to whatever your system prompt spells out (e.g. `"<end>"`). The parser normalizes `<>` and enclosing punctuation, so `<finish>` still matches the default. |
| `fill_order` | `[]` | A deterministic pipeline `[(agent, slot), ...]`. The model picks the *entry* agent once; then the chain runs in order (each missing slot → its agent) and finishes when every slot is full. A mid-chain agent picked directly runs once and finishes. No subclass needed. |
| `rounds_key` / `max_rounds` | `"supervisor_rounds"` / `6` | Bound the loop: `finish` is forced once the counter reaches `max_rounds`, so a model that never says `finish` can't hang. |

Other knobs: `output_key` (default `"next_agent"`), `messages_key` (default
`"messages"`; set `""` to always consult the model), `sections` (rendered into
the prompt alongside `Round: n/max` and the latest user message), and `agents`
(override the reply vocabulary; by default it is `route_keys ∪ fill_order ∪
{"finish"} ∪ {fallback_agent}`).

`fill_order` covers the common "run the agents in order, then finish" pipeline
without subclassing:

```python
flow.supervisor(
    model="llama3.1:8b",
    sections={"plan": "План", "estimate": "Смета"},
    fill_order=[
        ("planner", "plan"),
        ("estimator", "estimate"),
    ],
    done_keys={"direct_reply"},  # a direct answer finishes the turn
    done_mode="any",
    fallback_agent="direct",
)
flow.route("next_agent", finish=..., direct=..., planner=..., estimator=...)
```

See [`examples/applications/service_desk`](https://github.com/bzdvdn/teff/tree/master/examples/applications/service_desk/)
for a concrete chat router built on these guards (one-word dispatch, the
`done_keys` / `fallback_agent` guards, a bounded loop and a human `Interrupt`
deploy gate).  For the alternative design — a single ReAct coordinator that
drives specialists as *tools* instead of a decider node — see
[`examples/applications/repair-ai-chat`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/).

Two override hooks cover deterministic policies:

- ``_needs_model(state)`` — skip the model call when the state alone decides
  (default: skip when there's no user message or `done_keys` are filled).
- ``decide(state, proposal)`` — resolve the final route from the parsed
  *proposal* plus the guards. Override it for a state-machine policy (e.g.
  `examples/release_coordinator`, which forces a fill order and human-gate
  rules instead of letting the model roam).

A deterministic decider therefore subclasses `Supervisor`, overrides
`_needs_model` / `decide`, and can omit the model call entirely for states it
resolves from state alone.

## `flow.supervisor()`

## `agent_step()`

One routed agent as a reusable `SubFlow`:

```
ContextBuilder ──► ReAct harness ──► AppendAssistant
```

- **ContextBuilder** composes a plain-text `input` from shared state sections
  (plus the latest user message) and resets the agent's scratch keys, so each
  run starts clean.
- The **harness** runs the model against that `input` with `use_tools`,
  writing its final answer to `output_key`.
- **AppendAssistant** copies that answer into the shared conversation
  (`messages_key`).

```python
from teff.flow import agent_step

planner = agent_step(
    "You are the planning agent. Produce a step list.",
    "plan",  # output_key in shared state
    model="llama3.1:8b",
    provider="ollama",
    sections={"draft": "Draft", "review": "Review"},  # context sections
    use_tools=["current_date"],  # allowlist, or "all" / None
    stream=True,
)
```

Arguments:

| Arg | Description |
| --- | ----------- |
| `system` | System prompt for the agent. |
| `output_key` | State key that receives the agent's final answer. |
| `model` / `provider` | LLM model and provider for the harness. |
| `sections` | Shared state key → label mapping rendered into the agent's context (default `{output_key: output_key.capitalize()}`). |
| `messages_key` | State key holding the shared conversation (default `"messages"`). |
| `use_tools` | `None`/`[]` (no tools, default), `"all"` (everything the pool offers), or an explicit allowlist of tool names. Prefer an allowlist. |
| `stream` | Emit tokens as stream events (live rendering, default `True`). |
| `**config` | Extra kwargs passed to the ReAct harness / `ToolExec`. |

`agent_chain` is a backwards-compatible alias. The agent's scratch conversation
lives in a private `_<output_key>_messages` state slot (reset by the context
builder); only the final reply reaches the shared `messages`.

## The full pattern (repair-ai style)

```python
from teff.flow import Flow, agent_step
from teff.provider import ProviderRegistry

flow = Flow(
    "support",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
)

flow.step(supervisor)  # decider writes "next_agent"
flow.route(
    "next_agent",
    planner=agent_step(PLANNER_PROMPT, "plan", model="llama3.1:8b", sections=SECTIONS),
    writer=agent_step(
        WRITER_PROMPT,
        "draft",
        model="llama3.1:8b",
        sections=SECTIONS,
        use_tools=["current_date", "search_catalog"],
    ),
    reviewer=agent_step(
        REVIEWER_PROMPT, "review", model="llama3.1:8b", sections=SECTIONS
    ),
)
graph = flow.compile()
```

Shared state keys (`plan`, `draft`, `review`) are rendered into every agent's
context via `sections`, so a later agent sees what earlier ones produced —
that is how writer + reviewer collaborate through the same conversation.

### Practical tips

- **Keep the loop bounded.** The decider's system prompt should explicitly
  list the route values and say `finish` ends the conversation, so the loop
  cannot spin forever.
- **Explicit tool allowlists.** Pass `use_tools=["name", ...]` per agent
  instead of `"all"` — it keeps `secret_tool` out of a specialist's reach.
- **One conversation, many agents.** Use a single `messages_key` (default
  `"messages"`) so every `agent_step` appends to the same thread.
- **Stream everything.** `agent_step(stream=True)` + `graph.stream()` renders
  tokens live while the supervisor routes between agents.

## Run it

- [`examples/simple_router/`](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/)
  — a minimal two-agent supervisor, offline tests.
- [`examples/applications/service_desk/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/service_desk/)
  — the default `supervisor()` chat router: specialists, guards, bounded loop
  and a human approval gate.
- [`examples/applications/repair-ai-chat/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/)
  — the alternative: one ReAct coordinator driving specialists as tools (RAG + streaming).
- `teff new <name>` — scaffolds the same supervisor with `HOW TO EXTEND`
  comments (see [CLI](../cli.md)).

See also [`teff.flow.route`](../api/teff.flow.flow.md) and
[`teff.flow.agent_step`](../api/teff.flow.agent.md) in the API reference.