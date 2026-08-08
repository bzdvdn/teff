# CLI cheat-sheet

The `teff` CLI is how you drive a workflow without writing Python. All examples
use the runnable [`examples/hello_workflow/`](https://github.com/bzdvdn/teff/tree/master/examples/hello_workflow/).

## The five commands

| Command              | What it does                                          |
| -------------------- | ----------------------------------------------------- |
| `teff run FILE`      | Execute a workflow YAML and print the final state.    |
| `teff graph FILE`    | Print the graph topology / a Mermaid diagram.         |
| `teff eval`          | Score a workflow against a dataset (exact or LLM judge). |
| `teff daemon`        | Run a workflow on a ticker (`--interval`, durable state). |
| `teff new`           | Scaffold an app from a template (`cli`/`daemon`/`fastapi`). |

Pass the file as a positional argument (not `--file`) except for `teff run`.
Note `graph [file]` and `eval [file] --data ds.{json,jsonl,csv}`.

## `teff run` — the flags that matter

```bash
teff run --file examples/hello_workflow/workflow.yaml --pretty
```

| Flag                  | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `--pretty` / `-p`     | Pretty-print the JSON result.                       |
| `--trace` / `-t`      | Print a run trace to stderr (see **Anatomy of a run**). |
| `--interactive`      | Prompt the operator on stdin when the workflow pauses for input (e.g. at an `Interrupt`). |
| `--resume <json>`    | Resume a paused run, e.g. `'{"approved":"да"}'`.    |
| `--checkpoint <json>`| Checkpointer config, e.g. `'{"type":"file","path":"cp"}'`. |
| `--checkpoint-id`  | Key identifying the run/session.                     |
| `--checkpoint-owner`  | Owner/session scope (default `default`).           |
| `--node-timeout`  | Max seconds per node (triggers `__error__`).      |
| `--max-iterations` | Loop guard: max node executions.                     |
| `--output`           | Write the result to a file.                          |

A durable + human-in-the-loop run from a single shell line:

```bash
teff run app.yaml \
  --checkpoint '{"type":"file","path":"cp"}' --checkpoint-id ticket-1 \
  --interactive --trace
```

1. Run pauses at the `Interrupt`, prints its prompt, reads your answer.
2. `teff run` resumes internally with your answer as the `resume` value.
3. `--trace` shows the `interrupt` → `interrupt_resume` events.

## Which node for which job

A mental map (full detail in [reference/nodes](../reference/nodes.md)):

| Need                            | Node / type                            |
| ------------------------------- | -------------------------------------- |
| Deterministic string/data op    | `transform` (`count_lines`, `value`, `json_get`, …) |
| A single model call             | `llm_chat`                            |
| An agentic loop with tools      | `react_agent` / `tool_exec`           |
| Pick among subagents            | `supervisor`                          |
| Call an external tool           | `tool_call` (or built-in tools like `web_search`, `sql_query`, `slack_send`, `s3_*`) |
| Fan-out branches, merge later   | `parallel`                            |
| Apply a step to every item      | `map`                                 |
| Pause for a human               | `interrupt` + checkpointer            |
| Block until a condition         | `gate`                               |
| Retry a transient call          | `retry` (wrap any node)               |
| Get structured JSON, fill gaps  | `extract` + `fallback`                |
| Check / re-ask an LLM answer    | `validate` / `Ask`                    |
| Conversation plumbing           | `context_builder`, `append_assistant` |

## Seeing the graph

Before running anything new, render it:

```bash
teff graph examples/hello_workflow/workflow.yaml
```

and validate with `teff run` on a tiny input before wiring it to real
services — the CLI runs the exact same runtime as your Python code.