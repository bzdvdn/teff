# hello_llm — the durable "hello world"

A minimal **LLM workflow** you can run straight from the CLI — no Python
needed. It calls Ollama (`llama3.1:8b`), uppercases the answer with a plain
`transform`, and can be made **durable** with a checkpointer.

## Run it

```bash
teff run --file workflow.yaml
# {"topic": "Rocket engineering", "greeting": "Hello, ...!", "loud": "HELLO, ...!"}
```

Render the graph:

```bash
teff graph workflow.yaml
```

## Durable resume

Add a SQLite checkpointer and re-running with the same `--checkpoint-id`
resumes from the last checkpoint instead of starting over:

```bash
teff run --file workflow.yaml \
  --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id hello

teff run --file workflow.yaml \
  --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id hello  # resumes

teff inspect --checkpoint '{"type":"sqlite","path":"cp.db"}' --checkpoint-id hello
```

Requires local [Ollama](https://ollama.com) with `llama3.1:8b`:

```bash
ollama pull llama3.1:8b
```

## The workflow

- `greet` — an `llm_chat` node reads `{topic}` into its prompt and writes
  `greeting`.
- `shout` — a `transform` node uppercases `greeting` into `loud`.

A deterministic, LLM-free sibling lives in
[`examples/hello_workflow/`](../hello_workflow/) — runnable offline with
`teff run --file workflow.yaml`, no API keys.