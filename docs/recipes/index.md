# Recipes: closed real-world cases

A recipe is a **real business case from start to finish** — not a feature
demo. Each one follows the same template so you can compare cases and lift
the structure for your own:

1. **The business problem** — what the system is for, who uses it.
2. **Graph design** — the topology (and *why* it is shaped this way).
3. **State schema** — what flows between nodes.
4. **Durability & HITL** — why this case needs checkpoints / interrupts.
5. **Testing strategy** — how it is verified offline, no LLM.
6. **Cost & observability** — what to watch, what it costs.
7. **Deploy** — how it runs in production.
8. **How to adapt** — the knobs you turn for a different business.

Each recipe points at a **runnable example** in `examples/` so you are never
reading abstract code.

| Recipe | Business case | Runnable example | Core pattern |
| ------ | ------------- | ---------------- | ------------ |
| [FastAPI agent in 10 minutes](fastapi-agent.md) | Durable AI agent over HTTP | [`fastapi_server`](https://github.com/bzdvdn/teff/tree/master/examples/fastapi_server/) + [`teff/scaffold/fastapi`](https://github.com/bzdvdn/teff/tree/master/teff/scaffold/fastapi/) | `Flow.react()` + checkpointer + SSE streaming — the zero-to-30s on-ramp |
| [Fraud review](fraud-review.md) | Payment screening | [`fraud_gate`](https://github.com/bzdvdn/teff/tree/master/examples/applications/fraud_gate/) | Dynamic `Command` routing + HITL + durable resume |
| [Release approval](release-approval.md) | Ship-gate with human sign-off | [`release_coordinator`](https://github.com/bzdvdn/teff/tree/master/examples/release_coordinator/) | Supervisor + `route()` + `Map` + approval gate |
| [Support triage](support-triage.md) | Customer-support SLA hub | [`simple_router`](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/) + [`repair-ai-chat`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/) | `route()` supervisor + RAG + HITL escalation |
| [Ops daemon](ops-daemon.md) | Scheduled change-triage | [`repo-health`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repo-health/) | YAML-only agent-driven tick via CLI daemon |
| [KB assistant](kb-assistant.md) | Grounded self-service Q&A | [`rag_search`](https://github.com/bzdvdn/teff/tree/master/examples/rag_search/) + [`repair-ai-chat`](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/) | RAG retrieval + confidence gate + human fallback |
| [Invoice processing](invoice-processing.md) | AP automation with approval | [`structured_output`](https://github.com/bzdvdn/teff/tree/master/examples/structured_output/) + [`agent_approval`](https://github.com/bzdvdn/teff/tree/master/examples/agent_approval/) | `Extract`/`Fallback` + `Validate` + durable approval |

## Reading the pattern

The six cases differ in *business*, but the *graph anatomy* repeats:

- **One entry point** — a node that builds context (ingest / reset / context
  builder).
- **A driver** — either an agent loop (`react_agent`, `ReActAgent`) or a
  supervisor (`route()` / `Supervisor`) that decides what happens next.
- **A fan-out** — `Parallel` (fixed branches) or `Map` (runtime list).
- **A human gate** — `Interrupt` / `Ask` where a wrong answer is costly.
- **A finalizer** — a node that commits, sends, or writes the outcome.

If your case is a mix, combine the recipes — e.g. a support hub that also
re-triages nightly (triage + ops daemon) or a release gate with a fraud check
(approval + fraud).
