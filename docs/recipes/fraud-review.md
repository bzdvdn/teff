# Recipe: Fraud review

> Payment screening: a payment comes in, an LLM reads its free-text note and
> returns a risk score, and the next step is **decided at runtime** based on
> that score — auto-approve, route to a human analyst, or deny outright.

Runnable: [`examples/applications/fraud_gate/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/fraud_gate/)

## 1. The business problem

A payments team needs a first-pass screen on every transaction before it
moves on. The decision (approve / review / deny) depends on a **risk score the
model just produced**, so the graph cannot be wired statically — the branch is
a function of runtime data. Mid-risk payments must pause for a human analyst
and survive a restart (the analyst answers minutes later, possibly on another
node). Everything must be auditable: every routing decision is appended to a
log.

## 2. Graph design

```
POST /api/review ──▶ ingest → analyser (LLM) → router ──(approve)──▶ finalizer (LLM)
                                                      │
                                                      ├─(review)──▶ review_gate ──▶ finalizer
                                                      │              (human analyst,
                                                      │               durable resume)
                                                      └─(deny)────▶ END  (Command.STOP)
```

The **router node** is the interesting part: instead of a static `branch(key)`
on a pre-computed key, it returns a
[`Command`](https://github.com/bzdvdn/teff/tree/master/teff/node/command.py) whose `goto` decides the next node
at runtime:

| Route | What the router returns | Why it matters |
| ----- | ----------------------- | -------------- |
| **approve** | `Command(update={...}, goto="finalize")` | jumps to the finalizer even though there is **no** `router → finalize` edge |
| **review** | `Command(update={...})` (no `goto`) | normal `router → review_gate` edge runs; the analyst's answer resumes later |
| **deny** | `Command(update={...}, goto=Command.STOP)` | run ends immediately |

Every `Command` also appends to the `events` reducer, so routing and
state-writing happen in one return value — no hidden writes.

Use `Command` (dynamic) rather than a static `branch(key, ...)` when the
decision value is produced **inside the same turn** and can be `STOP`-worthy.
A static `branch` is the right tool when the routing key already lives in
state and never short-circuits.

## 3. State schema

```python
class ReviewState(TypedDict):
    tx: dict  # raw transaction (id, amount, merchant, ...)
    analyser_out: dict  # structured {score, reason, flags}
    decision: str  # "approve" | "review" | "deny"
    events: Annotated[list, add]  # reducer: append-only audit log
```

The `events` reducer is what makes the flow auditable and the offline tests
assertable — every `Command` writes one event, so a test can assert "deny →
`finalizer` never ran" by checking the log.

## 4. Durability & HITL

The `review_gate` is an `Interrupt` with a durable checkpointer. Mid-risk
payments **pause with `waiting: true`** and a human prompt; the run resumes
in a later request with the same checkpoint owner:

```bash
# pauses with waiting:true and a prompt
curl -s -X POST http://127.0.0.1:8001/api/review -H 'Content-Type: application/json' \
  -d '{"tx": {"id": "tx-2", "amount": 60_000, "note": "оплата счёта"}}'
# the analyst resumes the paused review in the same session
curl -s -X POST http://127.0.0.1:8001/api/review/<session_id>/decide \
  -H 'Content-Type: application/json' -d '{"answer": "pass"}'
```

Without the checkpointer the analyst's "pass" would vanish on a redeploy.
This is the canonical "must be durable" signal: *a human can answer later
than the machine asked*.

## 5. Testing strategy

No LLM, no network. A **canned transport** returns a fixed risk score for the
analyser, then the tests walk every routing outcome:

- `approve` → finalizer runs.
- `review` → run pauses; a later `decide` resumes and finishes.
- `deny` → `Command.STOP`, finalizer never runs.
- FastAPI layer: `/api/health`, `/api/review`, `/api/review/{id}/decide`.

```bash
uv run python -m pytest examples/applications/fraud_gate/tests -q
```

The service returns pure dataclasses (`ReviewOutcome`/`DecideOutcome`) which
FastAPI serializes from the handler's return annotation — the business layer
is testable without any HTTP.

## 6. Cost & observability

- **Watch:** risk-score distribution (what % lands in `review`?), human
  decision lag, `deny` rate, per-payment token spend.
- **Hook:** `observability:`/trace dashboard to replay a single payment's
  timeline — routing events make each step visible.
- **Bound:** the analyser's output schema (`StructuredOutput`) so a bad score
  cannot silently take the wrong branch.

## 7. Deploy

FastAPI app with `uv sync --extra fastapi`; the durable sessions live in a
file/SQLite checkpointer (swap to `pg` when the app runs on several nodes).
`uvicorn` behind your reverse proxy, `X-API-Key` auth on `/api/review`.

## 8. How to adapt

- **Different risk signal:** swap the analyser's prompt/output schema.
- **More than three outcomes:** add branches — each `Command.goto` can target
  any node.
- **No human step:** drop the interrupt; `review` becomes another LLM pass.
- **Different domain:** the pattern (score → runtime route → audit log →
  durable human gate) is the same for loan applications, content moderation,
  or onboarding checks.
