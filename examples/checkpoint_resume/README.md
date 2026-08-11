# checkpoint_resume — durable execution with an LLM call

Proves crash-recovery end to end: an LLM drafts a summary, the next node
crashes on the first run, and re-running with the same `checkpoint_id`
continues **from the node that failed** — not from the start.

## The proof

The run prints two counters after recovery:

- `processed == 1` — the first node executed exactly once, never re-ran;
- `llm_calls == 1` — the LLM was called exactly once, so no tokens were
  wasted re-generating the same summary.

```
Run 1: crashed (simulated transient failure), checkpoint saved
Run 2: success -> {'processed': 1, 'summary': 'Rocket propulsion is ...', 'done': True, ...}
Durable: resume continued from the failed node — no tokens wasted
```

## How it works

Teff writes a checkpoint **before every node**. When `FailingNode` raises,
the run dies but the checkpoint — which already contains the LLM's
`summary` — is on disk. A second `graph.run` with the same
`checkpoint_id="durable-run"` reads it, skips the completed nodes, and
re-executes only the failed one.

## Run it

Requires local [Ollama](https://ollama.com) with `llama3.1:8b`:

```bash
ollama pull llama3.1:8b
python examples/checkpoint_resume/main.py
```

## Files

- `main.py` — the durable LLM workflow: `Count` → `CountingLLM` (summarize)
  → `FailingNode` (crashes once) → `Transform` (uppercase).
