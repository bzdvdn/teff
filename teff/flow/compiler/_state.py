"""Extract tools / initial-state / reducers from a flow document."""

from __future__ import annotations

import os

from teff.errors import ConfigError


def _build_state(
    data: dict,
    base_dir: str | None = None,
) -> tuple[list, dict, dict]:
    """Mirror ``load_workflow``'s tools / initial-state / reducers extraction."""
    import teff.rag  # noqa: F401 — registers the "rag" tool
    import teff.tool.builtin  # noqa: F401 — registers built-in tools
    from teff.state.state import (
        reducers_from_yaml_schema,
        validate_state,
    )
    from teff.tool.registry import default_tool_registry
    from teff.yaml import _mcp_group_from_config, _resolve_rag_config

    tools: list = []
    for td in data.get("tools", []) or []:
        ttype = td["type"]
        tconfig = td.get("config", {})
        if ttype in ("rag", "rag_ingest"):
            tconfig = _resolve_rag_config(tconfig, base_dir or os.getcwd())
        if ttype == "mcp":
            tools.append(_mcp_group_from_config(tconfig))
            continue
        tools.append(default_tool_registry.create(ttype, tconfig))

    state_block = data.get("state", {})
    if isinstance(state_block, dict):
        schema = state_block.get("schema", {})
        initial = state_block.get("initial", {})
    else:
        schema = {}
        initial = {}
    if not isinstance(initial, dict):
        raise ConfigError("state.initial must be a mapping")
    if schema:
        errors = validate_state(initial, schema)
        if errors:
            raise ConfigError(
                "state.initial does not match state.schema:\n"
                + "\n".join(f"  {e}" for e in errors)
            )
    reducers = reducers_from_yaml_schema(schema)
    return tools, initial, reducers
