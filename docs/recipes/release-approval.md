# Recipe: Release approval

> A release coordinator: a supervisor routes three specialist agents
> (planner / estimator / tester), fans feature checks out in parallel, and —
> crucially — **pauses for human approval inside the supervisor loop** so that
> rejecting a release re-plans instead of ending.

Runnable: [`examples/release_coordinator/`](https://github.com/bzdvdn/teff/tree/main/examples/release_coordinator/)

## 1. The business problem

Shipping a release means gathering a plan, an estimate, and a test verdict,
then getting a human sign-off. The hard part is the **approval gate**: it must
live *inside* the loop, not after it. If the operator says "нет" (reject), the
graph must route control **back to the planner** and re-plan, not stop.
The loop must also be **bounded** — a stray free-form supervisor answer can
neither end the graph silently nor hang it.

## 2. Graph design

```
                 ┌──────────────────────── planner ──────┐
                 ▼                                        │
      supervisor ──(feature checks: Map over features)    │
                 │                                        │
                 └──(approval gate: Interrupt)──  да ──► final summary
                                    │
                                    └── нет ──► back to supervisor → planner (re-plan)
```

Built with the Flow DSL ([`main.py`](https://github.com/bzdvdn/teff/tree/main/examples/release_coordinator/main.py)):

```python
flow.supervisor(
    system=SUPERVISOR_PROMPT,
    model=model,
    provider=provider,
    route_keys={"planner": "plan", "estimator": "estimate", "tester": "test"},
    done_keys={"approved"},
    done_mode="any",
    # ...supervisor policy: fill sections, then ask for approval
)
flow.route(
    "next_agent",
    planner=agent_step(PLANNER_PROMPT, "plan", model=model, provider=provider),
    estimator=agent_step(ESTIMATOR_PROMPT, "estimate", model=model, provider=provider),
    tester=agent_step(TESTER_PROMPT, "test", model=model, provider=provider),
)
```

Key structural decisions:

- **`agent_step(...)` sub-flows** with stable `id=` so the compiled graph
  reads like the domain (`planner`), not `subflow_3`.
- **A `Map`** fans the feature list out to parallel checks after planning.
- **The `Interrupt` lives inside the supervisor cycle**: on `да` the run
  proceeds to the final summary; on `нет` control returns to the supervisor,
  which routes to `planner` to re-plan.

The approval gate is part of the `route()` loop, *not* a separate `loop()` —
this is what makes reject → re-plan a first-class behaviour rather than a
special case.

## 3. State schema

```python
class ReleaseState(TypedDict):
    goal: str  # what we are releasing
    plan: str  # planner output
    estimate: str  # estimator output
    tests: Annotated[list, add]  # Map results, appended per feature
    approved: str  # human verdict: "да" | "нет"
    supervisor_rounds: int  # bound the loop — can't hang
```

`supervisor_rounds` is the anti-hang safety valve: the policy enforces the
fill order (plan → estimate → tests → approval) and bounds the loop, so a
free-form supervisor answer can neither end silently nor loop forever.

## 4. Durability & HITL

- The `Interrupt` on `approved` pauses for the operator; the run resumes in
  the same session and reads `approved`.
- Add `--checkpoint` (or a `checkpoint:` block) if the release process can
  take long enough that the run must survive a restart between steps.

The lesson generalizes: **put the gate where the decision matters**, and let
rejection be a routing decision, not a dead end.

## 5. Testing strategy

The supervisor policy is the risk: a free-form answer must be handled. Test:

- route to each specialist agent reaches the right sub-flow;
- feature fan-out appends one `tests` entry per feature;
- approval `да` → final summary; approval `нет` → back to planner;
- the loop terminates even if the supervisor returns garbage
  (`supervisor_rounds`).

Follow the offline approach in
[`examples/simple_router/`](https://github.com/bzdvdn/teff/tree/main/examples/simple_router/) (canned transport +
bounded-loop assertion) to verify wiring without an LLM.

## 6. Cost & observability

- **Watch:** number of supervisor rounds per release (a "нет" chain is
  normal — a long chain is a signal), fan-out size, tokens per release.
- **Hook:** the trace dashboard shows the approve/reject path and every
  re-plan as a timeline — essential when a human says "нет".
- **Budget:** cap `supervisor_rounds` and per-agent token budgets so an
  expensive re-plan loop cannot run away.

## 7. Deploy

Run as a script (`python main.py`) or wrap in a small API/service that starts
a release and returns a `waiting` decision endpoint, like the fraud recipe's
`/decide`. The durable session lets the operator approve from a web UI later.

## 8. How to adapt

- **Different specialists:** change `route_keys` and the `agent_step` prompts.
- **Different gate:** any interrupt key works — e.g. `approved`, `go`,
  `signed_off`.
- **Reject semantics:** today reject → planner; change `fallback_agent` or the
  policy to route rejects to a different worker.
- **Domain swaps:** this is the template for any human-gated multi-worker
  pipeline — content publishing, architecture review, model launch approval.
