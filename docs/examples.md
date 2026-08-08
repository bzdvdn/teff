# Examples

Runnable examples live under `examples/` — one per feature. Most require
local [Ollama](https://ollama.com) (no API keys); check each directory's
README for exact commands.

| Example | What it shows |
| ------- | ------------- |
| [basic_pipeline](https://github.com/bzdvdn/teff/tree/master/examples/basic_pipeline/) | Minimal YAML pipeline, no API keys |
| [branching](https://github.com/bzdvdn/teff/tree/master/examples/branching/) | Conditional edges + Flow API |
| [parallel](https://github.com/bzdvdn/teff/tree/master/examples/parallel/) | Concurrent branches + typed `State` reducers |
| [map_repair_plans](https://github.com/bzdvdn/teff/tree/master/examples/map_repair_plans/) | Dynamic fan-out (`Map`) + `{key}` prompt templates + typed `State` |
| [human_in_loop](https://github.com/bzdvdn/teff/tree/master/examples/human_in_loop/) | Approve/Edit LLM output via `Interrupt` + `loop()` + resume (Python and YAML) |
| [ask_strategies](https://github.com/bzdvdn/teff/tree/master/examples/ask_strategies/) | Validate interrupt answers with `Ask` — regex (capture a promo code), `equals`, and an LLM `model` classifier (offline, no API key) |
| [react_agent](https://github.com/bzdvdn/teff/tree/master/examples/react_agent/) | ReAct agent loop with a calculator tool and live token streaming |
| [memory_assistant](https://github.com/bzdvdn/teff/tree/master/examples/memory_assistant/) | Long-term memory: LLM fact extraction, `MemoryStore` + provider-aware embedder, context injection |
| [memory_chat](https://github.com/bzdvdn/teff/tree/master/examples/memory_chat/) | Multi-user streaming chat — owner picked at the console, per-owner memory (`${owner}`), live tokens, auto fact extraction |
| [harness_agent](https://github.com/bzdvdn/teff/tree/master/examples/harness_agent/) | `flow.harness()` — parallel tool calls in one round + `__error__` fallback |
| [hello_workflow](https://github.com/bzdvdn/teff/tree/master/examples/hello_workflow/) | The same deterministic workflow at three levels — YAML, Flow DSL, low-level graph; no API key, CLI-runnable |
| [hello_llm](https://github.com/bzdvdn/teff/tree/master/examples/hello_llm/) | Minimal LLM workflow (`llm_chat` + `transform`) — CLI-runnable with durable SQLite checkpoints |
| [poem_chat](https://github.com/bzdvdn/teff/tree/master/examples/poem_chat/) | Two-agent poem chat with human approval — `context_builder` → `llm_chat` poet → critic → `interrupt`; the approval answer is classified by a small LLM (no hard-coded keywords) and a failed approval loops the poem back for a rewrite (`teff chat`) |
| [agent_approval](https://github.com/bzdvdn/teff/tree/master/examples/agent_approval/) | Tool approval (HITL) — every tool call pauses for human sign-off and resumes |
| [agent_resilience](https://github.com/bzdvdn/teff/tree/master/examples/agent_resilience/) | Retries, model failover, context trimming and a token budget (mocked, no API key) |
| [skills](https://github.com/bzdvdn/teff/tree/master/examples/skills/) | Skills folder (`SKILL.md`) — instructions + tool scoping on a harness agent |
| [pdf_agent](https://github.com/bzdvdn/teff/tree/master/examples/pdf_agent/) | Skill with its own tools — vendored `pdf` skill whose bundled scripts run via the shell tool |
| [mcp](https://github.com/bzdvdn/teff/tree/master/examples/mcp/) | ReAct agent calling tools from an MCP server (stdio) |
| [plugins](https://github.com/bzdvdn/teff/tree/master/examples/plugins/) | Custom nodes/tools via decorators and via subclasses; offline + agent variants |
| [streaming](https://github.com/bzdvdn/teff/tree/master/examples/streaming/) | Live LLM tokens + graph events via `graph.stream()` |
| [observability](https://github.com/bzdvdn/teff/tree/master/examples/observability/) | langfuse-style trace viewer — `GraphObserver` captures topology, node spans and full LLM prompt/response into SQLite; FastAPI dashboard UI + push exporters (webhook/langfuse/langsmith) and `teff obs-server` |
| [structured_output](https://github.com/bzdvdn/teff/tree/master/examples/structured_output/) | Schema-validated LLM JSON via `output_type` / `json_schema` |
| [rag_search](https://github.com/bzdvdn/teff/tree/master/examples/rag_search/) | RAG over a local CSV, in-memory store |
| [rag_stores](https://github.com/bzdvdn/teff/tree/master/examples/rag_stores/) | Same RAG agent on every vector store |
| [checkpoint_resume](https://github.com/bzdvdn/teff/tree/master/examples/checkpoint_resume/) | Crash/resume in a few lines |
| [checkpoint_stores](https://github.com/bzdvdn/teff/tree/master/examples/checkpoint_stores/) | Durable workflow on file/sqlite/pg |
| [release_features](https://github.com/bzdvdn/teff/tree/master/examples/release_features/) | Release API tour — validation, typed errors, `teff eval`, cost reports, response cache (mocked, no API key) |
| [simple_router](https://github.com/bzdvdn/teff/tree/master/examples/simple_router/) | Minimal `Flow.route()` supervisor — two agents, a bounded loop (can't hang), offline tests |
| [command_routing](https://github.com/bzdvdn/teff/tree/master/examples/command_routing/) | Dynamic per-node routing with `Command` — `update`+`goto`, `goto=Command.STOP`, edge bypass (offline, no API key) |
| [yaml_compose](https://github.com/bzdvdn/teff/tree/master/examples/yaml_compose/) | Pure-YAML composition — `include:`, `loop`, `command` routing, new `transform` actions (offline, no API key) |
| [fraud_gate](https://github.com/bzdvdn/teff/tree/master/examples/applications/fraud_gate/) | Production FastAPI payment gate — LLM scorer + `Command` routing (approve / mid-risk human review / deny-and-stop) |
| [service_desk](https://github.com/bzdvdn/teff/tree/master/examples/applications/service_desk/) | Default `supervisor()` chat router — one-word dispatch, `done_keys`/`fallback_agent` guards, bounded loop and a human `Interrupt` deploy gate (Russian support desk) |
| [repair-ai-chat](https://github.com/bzdvdn/teff/tree/master/examples/applications/repair-ai-chat/) | Full FastAPI app — one ReAct coordinator driving specialist *tools* (extract, plan, materials, estimate, QA), RAG, streaming (Russian repair workflow) |
| [channels](https://github.com/bzdvdn/teff/tree/master/examples/channels/) | One durable `Assistant` over every transport — HTTP/SSE (`teff serve`), Telegram (`teff bot`) and terminal (`teff chat`); zero-code `channels:` YAML block |
| [channels/supervisor](https://github.com/bzdvdn/teff/tree/master/examples/channels/supervisor/) | Multi-agent supervisor (planner/coder/QA) wrapped in the `channels:` block — `llm_chat` JSON verdicts, `Command` routing, loop-until-pass refine, human `approve` gate (Russian prompts) |
| [channels/rag_ingest](https://github.com/bzdvdn/teff/tree/master/examples/channels/rag_ingest/) | Grow the vector store from any channel — `llm_chat` normalizes a raw row, then the `rag_ingest` tool chunks/embeds it to SQLite; query the grown base with `rag` |

All LLM examples use `llama3.1:8b` (`ollama pull llama3.1:8b`);
[pdf_agent](https://github.com/bzdvdn/teff/tree/master/examples/pdf_agent/) uses `qwen2.5:7b`
(`ollama pull qwen2.5:7b`).