# Recipe: Support triage

> A customer-support SLA hub: a ticket comes in, the system grounds the query
> in your knowledge base (RAG), a supervisor routes it to the right specialist
> agent, answers with confidence automatically, and **escalates to a human**
> when it cannot commit to an answer — everything durable so a follow-up reply
> lands in the right conversation.

Runnable:
[`examples/recipes/support_triage/`](https://github.com/bzdvdn/teff/tree/master/examples/recipes/support_triage/) (the
compact supervisor + HITL version, offline-tested), plus the larger pieces it
composes:
[`examples/simple_router/`](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/) (the `route()`
supervisor skeleton) and
[`examples/applications/repair-ai-chat/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/)
(the RAG + streaming + five-agent app).

## 1. The business problem

A support team gets tickets across billing, technical issues, and refunds. The
system should answer straightforward questions from the company's own docs
(RAG), route anything ambiguous to a specialist, and when confidence is low
**pause for a human** rather than risk a wrong autopilot answer. Replies must
be durable: the customer may send a follow-up and expect the thread to stay
intact.

## 2. Graph design

```
ticket ─► context_builder ─► supervisor ──(billing / technical / refund)──▶ agent_step
   ▲                              │                                          │
   └───── append reply ───────────┘                                          │
                                      │ (confidence too low / user asks for human)
                                      ▼
                                  escalation_gate (Interrupt) ── resume later
```

Flow sketch (following the `simple_router` shape):

```python
agent = Flow("triage", providers=ProviderRegistry.from_presets("ollama"))
agent.supervisor(
    system=SUPERVISOR_PROMPT,        # route by ticket intent
    model=model, provider=provider,
    route_keys={"billing": "billing", "technical": "tech", "refund": "refund"},
    done_keys={"answered"},          # finished when a reply is appended
    done_mode="any",
    fallback_agent="billing",
)
agent.route(
    "next_agent",
    billing=agent_step(BILLING_PROMPT, "billing", model=model, provider=provider,
                       tools=[rag_tool, tools.scoped...]),
    technical=agent_step(TECH_PROMPT, "tech", model=model, provider=provider),
    refund=agent_step(REFUND_PROMPT, "refund", model=model, provider=provider),
)
agent.interrupt("escalate", prompt="Route this ticket to a human:")
```

Two decisions shape this graph:

- **RAG is a tool, not a step.** The knowledge base is exposed as a
  [`RAGTool`](https://github.com/bzdvdn/teff/tree/master/teff/rag/tool.py) so each specialist can ground answers
  in the docs — the retriever works *inside* the agent loop, not as a
  one-shot prefix. See [RAG](../guide/rag.md).
- **Escalation is an `Interrupt`, driven by confidence.** When the supervisor
  cannot commit (or the user asks for a human), the run pauses at
  `escalation_gate` and resumes with the human's answer — same model
  [fraud-review](fraud-review.md) uses.

## 3. State schema

```python
class TicketState(TypedDict):
    ticket: dict  # subject, body, customer id
    messages: Annotated[list, add]  # shared thread across agents
    reply: str  # final answer / human note
    escalated: bool  # human takeover
    next_agent: str  # selected by the supervisor
```

All specialists share one `messages` thread (via `add` reducer) so routing
never loses context even when different agents answer different turns.

## 4. Durability & HITL

- **Durable session** per `ticket_id` (owner scoping) — a follow-up message
  resumes the same run instead of starting cold.
- **Escalation gate** pauses for the human and resumes; the human's answer
  becomes the next state and flows back into `messages`.

The support case needs durability for a different reason than fraud: not
"crash recovery" but **long-lived conversations across turns**.

## 5. Testing strategy

Offline, mocked LLM + canned RAG corpus:

- ticket → supervisor routes to the right specialist;
- an answer with a known knowledge base hit terminates and appends a reply;
- a low-confidence / "human" request pauses at `escalation_gate`, then
  resumes with the human note;
- the loop cannot hang (bounded like `simple_router`).

Start from the offline suite in
[`examples/simple_router/tests/`](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/tests/) and the
FastAPI tests in
[`examples/applications/repair-ai-chat/tests/`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/tests/).

## 6. Cost & observability

- **Watch:** auto-answered vs escalated ratio (the SLA metric), RAG hit rate,
  per-ticket tokens, human lag on escalations.
- **Hook:** the trace dashboard shows the route, the retrieval, and where
  each ticket lands — replay any escalated thread.
- **Bound:** `RAGTopK` and per-agent budgets so long threads stay cheap.

## 7. Deploy

FastAPI app (host the checked `repair-ai-chat` variant), vector store from
`teff[stores-*]` for the knowledge base, SQLite (→ PG) checkpointer for the
ticket sessions, streaming over SSE for the chat UI.

## 8. How to adapt

- **More queues** (sales / devops / legal): add `route_keys` + `agent_step`s.
- **Confidence rule:** drive escalation off a `Command` or a supervisor
  `route_keys` value instead of the fallback.
- **No human pool:** replace the interrupt with a "we'll get back" finalizer.
- **This is the pattern:** rooted in RAG + `route()` + durable HITL. The
  [fraud-review](fraud-review.md) and [release-approval](release-approval.md)
  cases reuse the same gate; [ops-daemon](ops-daemon.md) shows the pure-YAML
  variant.