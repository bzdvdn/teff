# Recipe: Knowledge-base assistant

> A self-service assistant that answers support questions from an internal
> knowledge base. RAG finds the source passages; a confidence gate decides
> whether the answer is safe to send — and a low-confidence question escalates
> to a human agent instead of guessing.

Runnable pieces:
[`examples/rag_search/`](https://github.com/bzdvdn/teff/tree/master/examples/rag_search/) (RAG over a local CSV) and
[`examples/applications/repair-ai-chat/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/)
(a production RAG + streaming version).

## 1. The business problem

Support teams answer the same ~80% of questions repeatedly. A KB assistant
answers those instantly from curated documents, but **never invents answers**:
when the retrieval is empty or the model's confidence is low, the question must
reach a human agent with the best evidence attached. Every interaction is
logged so the KB gaps become visible (and the docs get fixed).

## 2. Graph design

```
ask ──▶ retrieve (RAG) ──▶ answer (LLM, cited) ──▶ gate ──(confident)──▶ reply
  │                           │  (uses only cited passages)       │
  └───────────────────────────┴──────────────(low)──▶ escalate ──▶ ticket
                                                        (human Interrupt,
                                                         durable resume)
```

- **`retrieve`** is a `RAGTool`-backed node — it *returns the source
  passages*, which is the whole point: the answer is grounded, and the human
  gets the evidence if it escalates.
- **`answer`** is an `LLM` node that receives **only the cited passages** (not
  the whole KB) and must produce a structured `{answer, confidence, source_ids}`
  via `output_type` — so the gate operates on a well-typed value, not free
  text.
- **`gate`** is a static `branch()` on `confidence` — this routing key *lives
  in state* and never short-circuits, so a plain `branch` (not `Command`) is
  the right tool.

This is the RAG+support case from the [index](index.md) pattern: entry
(`ask`) → driver (`answer`) → fan-out by `gate` → human gate (`escalate`) →
finalizer (`reply`/`ticket`).

## 3. State schema

```python
class AnswerState(TypedDict):
    question: str
    passages: Annotated[list, "replace"]  # retrieved by RAGTool
    structured: dict  # {answer, confidence, source_ids} from output_type
    routed_to: str  # "reply" | "escalate"
    transcript: Annotated[list, "append"]  # reducer: full audit log
```

The `passages` reducer uses `replace` (each retrieval overwrites the previous
turn's), while `transcript` **appends** — one entry per turn, exactly like the
`events` log in [fraud review](fraud-review.md).

## 4. Durability & HITL

Low-confidence questions hit the `escalate` `Interrupt`. The agent's drafted
answer, the cited passages, and the original question are all in state at the
pause point, so the human resolves a **complete context**:

```python
except GraphInterrupt as exc:
    # exc.prompt contains the drafted answer + sources; the human replies
    result = await graph.run(state=state, checkpointer=cp,
                             checkpoint_id=f"ticket-{ticket_id}",
                             resume={"escalate": answer})
```

Same checkpoint+`checkpoint_id` from a different process resumes the ticket —
an agent might answer in a queue UI minutes later.

## 5. Testing strategy

Offline, no LLM: mock the retrieval store and the model transport, then walk
both gate outcomes:

- `confidence >= 0.7` → `reply` sends the cited answer, transcript has one
  entry.
- `confidence < 0.7` or empty passages → run pauses on `escalate`; a later
  `resume` completes and the transcript records the human's answer.

Assert the **gate invariant**: the `structured` value is the only input to the
branch — a malformed `output_type` cannot silently take the wrong path.

## 6. Cost & observability

- **Watch:** retrieval-hit rate (how often `passages` is empty), escalation
  rate, average confidence, tokens per answer.
- **Hook:** the trace dashboard replays a question's full timeline — passages
  → answer → gate → escalate — so a bad answer is debuggable.
- **Bound:** `output_type` on `answer`; a RAG max-passage cap keeps prompt
  cost flat.

## 7. Deploy

`rag_search` runs standalone; the production shape is the FastAPI app in
`repair-ai-chat` — vector store on Postgres (`pgvector`) or Qdrant, streaming
answers, sessions per user with a durable checkpointer. The human-escalation
queue is any ticketing API: the `escalate` node writes the ticket, the resume
reads the agent's reply.

## 8. How to adapt

- **Different corpus:** swap the store — same RAG agent on every backend
  (`examples/rag_stores/`).
- **Retry instead of escalate:** low-confidence on a *rephrase*; escalate only
  after `max_rounds` — a `loop()` + confidence threshold.
- **Auto-escalate on topic:** route on a classifier output first (a
  `supervisor` or a `branch` on `topic`) and only then answer.
- **Different domain:** the pattern (grounded retrieval → typed answer →
  confidence gate → human fallback) is the same for legal Q&A, product docs,
  or ops runbooks.
