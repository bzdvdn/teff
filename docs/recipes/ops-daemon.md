# Recipe: Ops daemon

> A scheduled change-triage daemon: every N minutes the agent runs one **tick**
> against your infrastructure — takes a distributed lock, inspects a git repo,
> cross-references changed files against a priority table, de-duplicates alerts
> in redis, waits for a maintenance window, and sends a single summary.
> Entirely `workflow.yaml` — no per-app Python.

Runnable: [`examples/applications/repo-health/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repo-health/)

## 1. The business problem

Ops wants a background job that watches a codebase and pings owners when
something they own changed — but it must be **idempotent and safe under
concurrency**: two instances must never run the same tick, alerts must not
repeat every run, and delivery should be gated on a maintenance window. The
whole thing should be configurable in a `workflow.yaml`, not buried in app
code, so an operator can change the behaviour by editing a YAML file (and
often just the agent's prompt).

## 2. Graph design

```
teff daemon -f workflow.yaml --interval 300
```

```
reset (context_builder) ──► agent (react_agent) ──► tool_exec ──► (loop)
                                  ▲                        │
                                  └───────── _tool_call_name!= ┴─
```

- **`reset`** — rebuilds the agent's `input` with the priority-CSV path and
  clears the previous tick's conversation, so each tick starts clean even
  with `--checkpoint`.
- **`agent`** — a ReAct agent (`react_agent`) with all tools scoped in; it
  *drives the tick itself*:
  1. `lock acquire daemon:tick` — skip if another instance holds it.
  2. `csv_query read` the priority table.
  3. `git status` + `git log` — what changed recently.
  4. `csv_query filter` — owner/priority for each changed file.
  5. `redis exists`/`redis set` — alert each file only once a day.
  6. `wait_for redis_key deploy:ready` — wait for the maintenance window
     (a timeout is reported, not fatal).
  7. `send_telegram` — one summary.
  8. `lock release daemon:tick`.
- **`tool_exec`** — executes whatever tool the agent signalled and loops back
  until the agent stops calling tools.

Because the **agent drives the loop**, adding a step to a tick is just editing
the prompt — no code. Prefer this when a tick is a *sequence of coordinated
actions*; prefer explicit steps/branches when the order must be guaranteed
and testable independently of the model.

## 3. State schema

```yaml
state:
  initial:
    priority_csv: "data/priority.csv"
steps:
  - id: reset
    type: context_builder
    config:
      reset_keys: [messages, _tool_call_name, _tool_calls]  # clean tick
  - id: agent
    type: react_agent
    config:
      provider: ollama
      model: llama3.1:8b
      system: > ...
```

The `reset` node is what keeps repeated ticks isolated — the daemon re-runs
the same graph every interval, so without resetting `messages` the second tick
would inherit the first's "memory".

## 4. Durability & HITL

Two complementary mechanisms:

- **No checkpoint needed for the common case.** `redis` keys
  (`alerted:<file>`) make de-dup survive restarts, so state lives in your
  infrastructure, not in a checkpoint.
- **Add a checkpointer only if you want mid-tick persistence:**

```bash
teff daemon -f workflow.yaml --interval 300 \
  --checkpoint '{"type":"file","path":"data/daemon"}'
```

The `lock` tool is the concurrency guard: `acquire` returns "held by someone
else" and the agent stops the tick, so a queueing clone can never double-run.

## 5. Testing strategy

No real infra:

- **Tools** (`git`, `lock`, `wait_for`, `csv_query`, `redis`) are unit-tested
  against mocked subprocess / redis / httpx.
- **The workflow** is validated offline:

```bash
teff validate workflow.yaml
```

- The agent's tick *order* is guarded by tests that swap in canned tool
  responses and assert the sequence of tool calls the model makes.

## 6. Cost & observability

- **Watch:** tick duration, token spend per tick (normalize to `--interval`),
  "held by another instance" skips, telegram delivery.
- **Hook:** the daemon's per-tick timeline in the trace dashboard; the agent's
  steps are all observable via `observability:`.
- **Bound:** the wait_for timeout is reported rather than fatal, so one
  blocked infra call can't wedge the schedule.

## 7. Deploy

A cron or your scheduler, or the built-in loop:

```bash
# one tick, then exit
teff daemon -f workflow.yaml --once
# run forever, every 5 minutes
teff daemon -f workflow.yaml --interval 300
```

Point `git.path` / `csv_query.path` at the real repo and priority table;
`REDIS_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` via `${VAR}`
interpolation. Containerize the CLI image (see README → Docker) and run one
replica — `lock` keeps the rest warm.

## 8. How to adapt

- **Different infra:** swap/adjust the tools the agent can call.
- **Different cadence:** change `--interval` or move to cron.
- **Reporter changes:** `send_telegram` → any notify/messaging tool; or a
  `transform` finalizer instead of an agent step.
- **More logic in YAML:** any fixed ordering can move out of the prompt into
  explicit steps — the `flow.py` twin shows the same graph built
  programmatically when you prefer.