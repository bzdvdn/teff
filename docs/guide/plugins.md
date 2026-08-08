# Plugins

A plugin is any Python module that registers types with teff. **Loading the
file is the whole mechanism** — no separate plugin API.

## Discovery

`load_workflow` and `teff validate` discover plugin files two ways:

1. **`plugins:` key** — a list of file paths (relative to the workflow) in the
   workflow document:
   ```yaml
   plugins:
     - nodes.py
     - tools.py
   ```
2. **`plugins_folder`** — an auto-loaded folder next to the workflow
   (default `plugins`). Every `.py` file there is imported. Override with:
   ```yaml
   plugins_folder: my_plugins
   ```

Registration is idempotent per absolute path, so double-loading is harmless.

## Decorator style

```python
# plugins/nodes.py
from teff.node.registry import node


@node("slugify_node", SlugConfig)
async def slugify_node(ctx, config, state): ...
```

```python
# plugins/tools.py
from teff.tool.registry import tool


@tool("slugify", "Convert a string to a lowercase URL slug")
def slugify(text: str = "") -> str: ...
```

## Class style

No decorators needed — subclass `Node` / `Tool` and register explicitly:

```python
from teff.node.node import Node
from teff.node.registry import default_registry
from teff.tool.tool import Tool
from teff.tool.registry import default_tool_registry


class UpperTool(Tool):
    name = "upper"
    description = "Uppercase a string"

    def run(self, text: str = "") -> str:
        return text.upper()


class UppercaseNode(Node):
    type = "uppercase_node"

    async def execute(self, ctx, state):
        input_key = self.config.get("input_key", "text")
        output_key = self.config.get("output_key", "out")
        state[output_key] = ctx.tools["upper"].run(text=state.get(input_key, ""))
        return state


default_tool_registry.register(UpperTool)
default_registry.register("uppercase_node", UppercaseNode)
```

A `Node` subclass reads its workflow `config:` from `self.config` and calls
other tools through `ctx.tools[name]`.

## Reusing types

Registered types are ordinary: they appear in the same shared registries
(`default_registry`, `default_tool_registry`) as built-ins, so a plugin node
can be used in YAML (`type: slugify_node`), in the `Flow` API, or from another
plugin.

## Programmatic loading

```python
from teff.plugins import load_plugins

load_plugins(["nodes.py", "tools.py"])  # files
load_plugins(["plugins"])  # folder
```

See [`examples/plugins/`](https://github.com/bzdvdn/teff/tree/master/examples/plugins/) for a complete offline
pipeline (both styles) plus a ReAct agent that uses the custom tools.