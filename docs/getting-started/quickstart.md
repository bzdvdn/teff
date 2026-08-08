# Quick start

> New here? **[From zero to a FastAPI agent in 10 minutes](../recipes/fastapi-agent.md)**
> is the fastest on-ramp. For a 5-minute feel of the core ideas without any
> LLM, read the [tutorial](tutorial.md) instead.

## YAML workflow

Workflows are **data**. The canonical form is YAML — no code required to
describe routing.

```yaml
name: text-pipeline
state:
  initial:
    title: "  hello world  "

steps:
  - id: trim
    type: transform
    config: {action: trim, input_key: title, output_key: trimmed}
  - id: uppercase
    type: transform
    config: {action: uppercase, input_key: trimmed, output_key: loud}
  - id: count
    type: transform
    config: {action: count_lines, input_key: loud, output_key: line_count}

edges:
  - from: trim
    to: uppercase
  - from: uppercase
    to: count
```

Run it:

```python
import asyncio
from teff.yaml import load_workflow


async def main():
    graph, tools, state, reducers = load_workflow("workflow.yaml")
    result = await graph.run(state, tools=tools, reducers=reducers)
    print(result)


asyncio.run(main())
```

Or from the command line:

```bash
teff -f workflow.yaml
teff validate workflow.yaml   # check before running
```

## Flow API (Python)

Prefer composing graphs in code? The `Flow` builder is the ergonomic
equivalent of the YAML above:

```python
import asyncio
from teff.flow import Flow, Case
from teff.node import LLM, Transform
from teff.provider import ProviderRegistry


async def main():
    flow = Flow(
        "sentiment-router",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
        default_model="llama3.1:8b",
    )
    flow.step(
        LLM(
            system='Classify the sentiment. Reply "positive" or "negative".',
            input_key="text",
            output_key="sentiment",
        )
    )
    flow.branch(
        "sentiment",
        Case("positive").add(
            Transform(action="value", value="Glad you liked it!", output_key="reply")
        ),
        Case("negative").add(
            Transform(action="value", value="Sorry to hear that.", output_key="reply")
        ),
    ).converge(Transform(action="uppercase", input_key="reply", output_key="result"))

    result = await flow.compile().run(state={"text": "I love this product!"})
    print(result)


asyncio.run(main())
```

Both describe the same graph — YAML for deployment, `Flow` for code, and
`Flow.to_yaml()` to serialize a code-built graph back into a deployable
workflow. See [YAML workflows](../guide/yaml-workflows.md).