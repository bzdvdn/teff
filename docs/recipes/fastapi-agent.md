# Recipe: from zero to a FastAPI agent in 10 minutes

> The shortest path from a bare checkout to a working, durable AI agent over
> HTTP. You end with a FastAPI app that runs a ReAct agent, keeps every
> conversation durable across requests and restarts, and streams tokens — no
> workflow.yaml required.

Runnable end-state:
[`examples/fastapi_server/`](https://github.com/bzdvdn/teff/tree/master/examples/fastapi_server/)
and the productionised scaffold in
[`teff/scaffold/fastapi/`](https://github.com/bzdvdn/teff/tree/master/teff/scaffold/fastapi/).

## 0. Install

```bash
pip install 'teff[fastapi]'
# optional but needed to run the example verbatim:
ollama pull llama3.1:8b
```

## 1. The agent as a Flow

Define the agent in code with the `Flow` builder. One `react()` line is a
ReAct loop: system prompt, LLM, tools when it needs them, and the message
history that makes it multi-turn.

```python
from teff.flow import Flow
from teff.provider import ProviderRegistry

flow = Flow(
    "chat",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.react(
    system="You are a concise assistant.",
    input_key="query",
    output_key="answer",
    messages_key="messages",
    stream=True,
)
graph = flow.compile()
```

## 2. Durable sessions with a checkpointer

Without a checkpointer every request starts from scratch. With one, each
conversation lives in a file-scoped checkpoint keyed by `chat_id` — a crash,
a restart, a different process, all resume from exactly where the run
paused.

```python
from teff.checkpoint import JSONFileCheckpointer
from teff.node.interrupt import GraphInterrupt

checkpointer = JSONFileCheckpointer("data/checkpoints")

try:
    result = await graph.run(
        state={"query": message},
        checkpointer=checkpointer,
        checkpoint_id=chat_id,
        owner=owner,
    )
except GraphInterrupt as exc:
    return {"status": "interrupt", "key": exc.key, "prompt": exc.prompt}
```

## 3. The FastAPI app

Three thin endpoints are enough to be useful: `POST /api/chat` (single-shot
or continuation), `POST /api/chat/stream` (SSE tokens as they arrive), and
`GET /api/runs/{chat_id}` (the durable state — debugging for free).

```python
from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI()
checkpointer = JSONFileCheckpointer("data/checkpoints")


class ChatRequest(BaseModel):
    message: str = ""
    chat_id: str | None = None


@app.post("/api/chat")
async def chat(req: ChatRequest, x_user_id: str | None = Header(None)):
    owner = x_user_id or "anonymous"
    chat_id = req.chat_id or uuid4().hex
    result = await graph.run(
        state={"message": req.message},
        checkpointer=checkpointer,
        checkpoint_id=chat_id,
        owner=owner,
    )
    return {"chat_id": chat_id, "answer": result.get("answer", "")}
```

## 4. Run it

```bash
uvicorn app:app --port 8000
```

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" -H "X-User-Id: alice" \
  -d '{"message": "Hi! Write me a haiku about async."}'
```

Send the same `chat_id` again — the conversation continues from its saved
checkpoint, not from scratch.

## 5. Where to go next

- **Human-in-the-loop** — add an `Interrupt` for the one step that needs
  sign-off before it fires.
- **Tools** — a `react()` with tools, or wire an MCP server.
- **Supervisors** — several agents and a `Flow.route()` that decides who
  works next.
- **Real deployment** — the scaffold from `teff new` gives you API-key auth,
  an env-driven supervisor, tests and a trace dashboard in one package
  (`teff/scaffold/fastapi`).