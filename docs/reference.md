# Reference

## Design principles

See [`CONSTITUTION.md`](https://github.com/bzdvdn/teff/blob/master/CONSTITUTION.md)
for the framework's principles and non-negotiable rules — the short version:

- **Workflow as data.** YAML/JSON are canonical; code is optional.
- **Async by default.** Nodes, tools, and LLM calls are `async`.
- **Durable.** Checkpoint/resume across file, SQLite, and PostgreSQL.
- **Embeddable.** A library, not a platform — you import us, we never import
  you.
- **Zero runtime magic.** No hidden registries beyond the documented plugin
  mechanism.

## The `${ENV}` interpolation

Every value in a workflow document is interpolated against the process
environment. A variable that is not set stays as a literal placeholder —
nothing crashes. This is the only supported way to pass secrets; never
hardcode API keys in a workflow file.

## Typed errors

Loading or running a broken workflow throws errors from a public hierarchy
rooted at `teff.TeffError`; subclasses multiple-inherit from builtins for
back-compat:

```
TeffError
├── ConfigError             (also KeyError)      — invalid config / unknown types
├── WorkflowError           (also RuntimeError)  — workflow-level failures
│   ├── NodeError                                 — a node raised (node_id/type)
│   └── LLMError                                  — a model call failed after retries
├── InterruptError                                — HITL resume misuse
├── GraphInterrupt                                — workflow paused for human input
└── StructuredOutputError   (also ValueError)    — schema validation failed
```

## Workflow validation

```python
from teff.yaml_schema import validate_workflow_file, format_errors

errors = validate_workflow_file("workflow.yaml")
if errors:
    print(format_errors(errors, source="workflow.yaml"))
```

```bash
teff validate workflow.yaml    # exits non-zero on errors
```

To validate an already-parsed dict (no file) use
`teff.yaml_schema.validate_workflow(data, node_types=..., tool_types=...)` —
same error list. See the [API reference](api/teff.yaml_schema.md).

## Development

```bash
uv sync                        # install deps
uv run pytest tests/ -q        # tests
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy .                  # types
```

## Running these docs

```bash
uv run pip install -e ".[docs]"
uv run mkdocs serve            # http://localhost:8000
uv run mkdocs build            # static site in site/
```