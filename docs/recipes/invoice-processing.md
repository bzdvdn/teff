# Recipe: Invoice processing

> Invoices arrive as messy PDFs and emails. The system extracts the structured
> fields (amount, vat, counterparty, due date), **validates** them against the
> source text, and routes each invoice to auto-pay or to a human approver —
> with a full audit trail and durable resume for approvals.

Runnable pieces:
[`examples/structured_output/`](https://github.com/bzdvdn/teff/tree/main/examples/structured_output/) (schema-validated
JSON) and [`examples/agent_approval/`](https://github.com/bzdvdn/teff/tree/main/examples/agent_approval/) (every
tool/step can pause for human sign-off).

## 1. The business problem

An accounts team processes hundreds of invoices. Each one needs the same
fields extracted *correctly* — a hallucinated amount is worse than no amount.
Small invoices under a threshold should pay automatically; large or unusual
ones need a human's sign-off. The whole pipeline is an audit target: who
extracted what, who approved it, when.

## 2. Graph design

```
invoice ──▶ extract (LLM → output_type) ──▶ fallback ──▶ validate (Ask/Validate)
             │    (per-field JSON +      │  (deterministic   │  (re-check against
             │     source spans)        │   fill)           │   the text)
             │                          │                    └──(fail)──▶ re-extract (loop, bounded)
             └──────────────────────────┴──────────────▶ gate ──(amount ≤ limit)──▶ pay
                                                                   └─(large/odd)──▶ approve ──▶ pay
                                                                       (human Interrupt,
                                                                        durable resume)
```

- **`extract`** produces a *structured* result via `output_type` and reports a
  failure (instead of malformed state) when it can't fit the schema — the
  failure is caught by an `__error__` edge, not by the business logic.
- **`fallback`** deterministically fills fields the model left empty (e.g. a
  default VAT rate) — see [`Extract`/`Fallback`](../guide/errors.md).
- **`validate`** re-checks the extracted values against the source text and
  can send the invoice back for a **bounded** `re-extract` round rather than
  paying on bad data.

## 3. State schema

```python
class InvoiceState(TypedDict):
    invoice_id: str
    source_text: str
    parsed: dict  # {amount, vat, counterparty, due_date, confidence} (output_type)
    decision: str  # "pay" | "approve"
    events: Annotated[list, "append"]  # reducer: who did what, when
```

Every step appends one `events` entry (extract → validated → paid / approved),
which is exactly what the audit report and the offline tests assert against.

## 4. Durability & HITL

`approve` is an `Interrupt` behind a durable checkpointer. A large invoice
pauses with the extracted fields in the prompt; the approver answers later —
from another process, as long as they use the same checkpointer and
`checkpoint_id`. This is the canonical "a human answers later than the machine
asked" signal (see [fraud review](fraud-review.md) for the same shape).

```python
result = await graph.run(
    state=state,
    checkpointer=cp,
    checkpoint_id=f"invoice-{invoice_id}",
    resume={"approve": "approved"},
)
```

## 5. Testing strategy

Offline, no LLM: a canned transport returns a **fixed parsed invoice**, then
the tests walk every route:

- amount ≤ threshold → `pay`, events = `[extracted, validated, paid]`.
- amount > threshold → run pauses on `approve`; a later `resume` pays it.
- extraction fails schema → `__error__` edge to `re-extract`, never a payment.
- `validate` rejects the first parse → bounded retry, then escalate if still bad.

## 6. Cost & observability

- **Watch:** extraction `confidence`, validation failure rate (how often the
  model "remembers" the wrong amount), approval lag, per-invoice tokens.
- **Bound:** `output_type` schema + `Validate`; a **bounded** re-extract loop so
  a bad invoice can't spin forever.
- **Audit:** the `events` reducer is the report — every run's trace replays
  each decision.

## 7. Deploy

A `teff new cli` app or a FastAPI endpoint per queue. The durable sessions live
in a file/SQLite checkpointer locally; swap to `pg` (`pg-checkpoint`) when
several workers approve invoices concurrently. `teff eval` scores extraction
accuracy against a labeled dataset before you touch real money.

## 8. How to adapt

- **Different document:** receipts, PO orders, medical claims — same
  extract→validate→approve skeleton, different `output_type` schema.
- **Different threshold:** the gate is a static `branch` on `amount` — one
  state field, no code change.
- **Tiered approval:** > threshold → manager, > 10× threshold → CFO — chain
  two `Interrupt`s with separate keys.
- **Receipts with photos:** swap `source_text` for an image field; the extract
  prompt changes, the graph doesn't.
