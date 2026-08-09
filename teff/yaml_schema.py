"""Validation of workflow YAML documents against a JSON Schema.

A workflow file is validated *before* execution so that typos and
structural mistakes are reported with a clear path and message instead
of a stack trace mid-run.

Use :func:`validate_workflow_file` / :func:`validate_workflow` directly,
or rely on :func:`teff.yaml.load_workflow`, which validates by default
and raises :class:`~teff.errors.ConfigError` with all findings.
"""

from __future__ import annotations

import os
from typing import Any

import jsonschema

from teff.errors import ConfigError

WORKFLOW_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Teff workflow",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "include": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "required": ["path"],
                                "properties": {
                                    "path": {"type": "string"},
                                    "prefix": {"type": "string"},
                                },
                                "additionalProperties": True,
                            },
                        ]
                    },
                },
            ]
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "config": {"type": "object"},
                    "retry": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "max_retries": {"type": "integer", "minimum": 1},
                            "delay": {"type": "number", "minimum": 0},
                            "backoff": {"type": "number", "minimum": 0},
                            "timeout": {"type": "number", "minimum": 0},
                            "retry_on": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                            },
                        },
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"type": "string"},
                    "config": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "schema": {"type": "object"},
                "initial": {},
            },
            "additionalProperties": True,
        },
        "plugins": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
        "plugins_folder": {"type": "string"},
        "checkpoint": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "path": {"type": "string"},
                "dsn": {"type": "string"},
                "table": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "hooks": {
            "type": "object",
            "properties": {
                "on_node_start": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "on_node_end": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "on_node_error": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
            },
            "additionalProperties": True,
        },
        "default_provider": {"type": "string"},
        "default_model": {"type": "string"},
        "observability": {
            "type": "object",
            "properties": {
                "db": {"type": "string"},
                "export": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"enum": ["webhook", "langfuse", "langsmith"]},
                            "url": {"type": "string"},
                            "url_env": {"type": "string"},
                            "host": {"type": "string"},
                            "api_url": {"type": "string"},
                            "api_key_env": {"type": "string"},
                            "public_key_env": {"type": "string"},
                            "secret_key_env": {"type": "string"},
                            "project": {"type": "string"},
                            "timeout": {"type": "number", "minimum": 0},
                            "retries": {"type": "integer", "minimum": 0},
                            "backoff": {"type": "number", "minimum": 0},
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "additionalProperties": True,
        },
        "providers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "enum": [
                            "openai_compatible",
                            "anthropic_compatible",
                            "ollama",
                        ]
                    },
                    "base_url": {"type": "string"},
                    "chat_path": {"type": "string"},
                    "api_key_env": {"type": "string"},
                    "auth_header": {"type": "string"},
                    "auth_prefix": {"type": "string"},
                    "timeout": {"type": "number", "minimum": 0},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}

_VALIDATOR = jsonschema.Draft202012Validator(WORKFLOW_JSON_SCHEMA)

#: Idiom keys recognised in the authoring (workflow) layer — a single-key
#: ``steps:`` mapping wrapping a step definition (``team:``, ``map:`` …).
FLOW_IDIOMS = (
    "llm",
    "transform",
    "agent",
    "agent_step",
    "team",
    "parallel",
    "map",
    "loop",
    "interrupt",
    "branch",
    "route",
    "type",
)


def _node_types() -> list[str]:
    from teff.node.registry import default_registry

    return default_registry.list()


def _tool_types() -> list[str]:
    import teff.rag  # noqa: F401 — registers the "rag" tool
    import teff.tool.builtin  # noqa: F401 — registers built-in tools
    from teff.tool.registry import default_tool_registry

    return default_tool_registry.list()


def validate_workflow(
    data: dict,
    *,
    node_types: list[str] | None = None,
    tool_types: list[str] | None = None,
) -> list[dict]:
    """Validate a parsed workflow dict.

    Checks the structural JSON Schema plus node/tool type membership and
    edge references.

    Args:
        data: The parsed workflow document.
        node_types: Allowed node type names (defaults to the registry).
        tool_types: Allowed tool type names (defaults to the registry).

    Returns:
        A list of ``{"path", "message"}`` errors (empty when valid).
    """
    errors: list[dict] = []
    for err in _VALIDATOR.iter_errors(data):
        path = _err_path(err.absolute_path)
        errors.append({"path": path, "message": err.message})

    node_types = node_types or _node_types()
    tool_types = tool_types or _tool_types()

    steps = data.get("steps") or []
    step_ids: set[str] = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        if isinstance(sid, str):
            step_ids.add(sid)
        stype = step.get("type")
        if isinstance(stype, str) and stype not in node_types:
            errors.append(
                {
                    "path": f"steps[{i}].type",
                    "message": (
                        f"unknown node type {stype!r} (registered: "
                        f"{', '.join(sorted(node_types))})"
                    ),
                }
            )

    for i, tool in enumerate(data.get("tools") or []):
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if isinstance(ttype, str) and ttype not in tool_types:
            errors.append(
                {
                    "path": f"tools[{i}].type",
                    "message": f"unknown tool type {ttype!r}",
                }
            )

    for i, edge in enumerate(data.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        for key in ("from", "to"):
            target = edge.get(key)
            if isinstance(target, str) and step_ids and target not in step_ids:
                errors.append(
                    {
                        "path": f"edges[{i}].{key}",
                        "message": f"edge references unknown step {target!r}",
                    }
                )

    return errors


def validate_flow(
    data: dict,
    *,
    node_types: list[str] | None = None,
) -> list[dict]:
    """Validate a parsed *authoring-layer* document (``workflow.yaml``).

    This is the sibling of :func:`validate_workflow` for the high-level
    formatting surface that mirrors the Python :class:`~teff.flow.Flow`
    API (single-key idiom steps: ``team:``, ``map:``, ``loop:``, …):

        steps:
          - team: {leader: {system: "…"}, roles: {coder: {…}}}
          - loop: {key: verdict, until: pass, body: […], done: […]}

    Checks the structural JSON Schema common block plus the idiom surface:
    every ``steps:`` entry must be a single-key mapping with a known idiom
    name, and the structural invariants each idiom requires.

    Args:
        data: The parsed workflow document.
        node_types: Allowed node type names (defaults to the registry).

    Returns:
        A list of ``{"path", "message"}`` errors (empty when valid).
    """
    errors: list[dict] = []
    for err in _VALIDATOR.iter_errors(data):
        path = _err_path(err.absolute_path)
        # The classic validator demands id/type on every step — irrelevant
        # for the idiom surface, so skip step-level schema findings here.
        if path.startswith("steps"):
            continue
        errors.append({"path": path, "message": err.message})

    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append({"path": "steps", "message": "`steps` must be a non-empty list"})
        return errors

    for i, step in enumerate(steps):
        path = f"steps[{i}]"
        if not isinstance(step, dict) or len(step) != 1:
            errors.append(
                {
                    "path": path,
                    "message": (
                        "each step must be a single-key idiom "
                        f"(one of: {', '.join(FLOW_IDIOMS)})"
                    ),
                }
            )
            continue
        idiom, spec = next(iter(step.items()))
        if idiom not in FLOW_IDIOMS:
            errors.append(
                {
                    "path": f"{path}.{idiom}",
                    "message": (
                        f"unknown flow idiom {idiom!r} (registered: "
                        f"{', '.join(FLOW_IDIOMS)})"
                    ),
                }
            )
            continue
        if not isinstance(spec, dict):
            errors.append(
                {
                    "path": f"{path}.{idiom}",
                    "message": f"expected a mapping after {idiom!r}, "
                    f"got {type(spec).__name__}",
                }
            )
            continue
        if idiom in ("team",):
            if (
                "roles" not in spec
                or not isinstance(spec["roles"], dict)
                or not spec["roles"]
            ):
                errors.append(
                    {
                        "path": f"{path}.{idiom}.roles",
                        "message": "team requires a non-empty `roles:` mapping",
                    }
                )
            if "leader" not in spec or not isinstance(spec["leader"], dict):
                errors.append(
                    {
                        "path": f"{path}.{idiom}.leader",
                        "message": "team requires a `leader:` mapping",
                    }
                )
        elif idiom == "map":
            if "processor" not in spec:
                errors.append(
                    {
                        "path": f"{path}.{idiom}.processor",
                        "message": "map requires a `processor:`",
                    }
                )
        elif idiom == "loop":
            if "key" not in spec:
                errors.append(
                    {"path": f"{path}.{idiom}.key", "message": "loop requires a `key:`"}
                )
            if "body" not in spec:
                errors.append(
                    {
                        "path": f"{path}.{idiom}.body",
                        "message": "loop requires a `body:`",
                    }
                )
            if "until" not in spec:
                errors.append(
                    {
                        "path": f"{path}.{idiom}.until",
                        "message": "loop requires an `until:`",
                    }
                )
        elif idiom == "interrupt":
            if "key" not in spec:
                errors.append(
                    {
                        "path": f"{path}.{idiom}.key",
                        "message": "interrupt requires a `key:`",
                    }
                )
        elif idiom == "parallel":
            if (
                "branches" not in spec
                or not isinstance(spec["branches"], list)
                or not spec["branches"]
            ):
                errors.append(
                    {
                        "path": f"{path}.{idiom}.branches",
                        "message": "parallel requires a non-empty `branches:` list",
                    }
                )

    return errors


def validate_flow_file(path: str) -> list[dict]:
    """Validate a ``workflow.yaml`` file on disk.

    Resolves env refs, ``include:`` blocks, and loads plugins the same way
    :func:`validate_workflow_file` does, so custom node/tool types
    referenced inside idioms are registered before validation.  Returns
    ``{"path", "message"}`` errors (empty when valid); a missing file
    raises :class:`ConfigError`.
    """
    if not os.path.exists(path):
        raise ConfigError(f"workflow file not found: {path}")
    from teff.yaml import load_workflow_document

    data = load_workflow_document(path)
    from teff.plugins import load_plugins_from_document

    load_plugins_from_document(data, os.path.dirname(os.path.abspath(path)))
    return validate_flow(data)


def _err_path(parts: Any) -> str:
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f"{'.' if out else ''}{part}"
    return out or "$"


def format_errors(errors: list[dict], *, source: str = "workflow") -> str:
    """Render validation *errors* as human-readable lines."""
    lines = []
    for err in errors:
        lines.append(f"{source}: {err['path']}: {err['message']}")
    return "\n".join(lines)


def validate_workflow_file(path: str) -> list[dict]:
    """Validate a workflow YAML file on disk.

    Auto-detects the document layer: a ``flow.yaml``-style document (the
    sugar idiom surface) is validated against the flow schema via
    :func:`validate_flow_file`; a low-level graph document is validated
    against the graph schema via :func:`validate_workflow`.

    Loads any plugins referenced by the ``plugins`` key (or the default
    ``plugins/`` folder) and resolves ``include:`` blocks the same way
    :func:`teff.yaml.load_workflow` does, so custom node/tool types and
    sub-included steps are all registered before validation.

    Returns a list of ``{"path", "message"}`` errors (empty when valid).
    A missing or unparseable file raises :class:`ConfigError`.
    """
    if not os.path.exists(path):
        raise ConfigError(f"workflow file not found: {path}")
    from teff.yaml import load_workflow_document

    data = load_workflow_document(path)
    from teff.plugins import load_plugins_from_document

    load_plugins_from_document(data, os.path.dirname(os.path.abspath(path)))
    from teff.flow.compiler import looks_like_flow

    if looks_like_flow(data):
        return validate_flow(data)
    return validate_workflow(data)


def raise_for_validation(errors: list[dict], *, source: str = "workflow") -> None:
    """Raise :class:`ConfigError` listing *errors* if any exist."""
    if errors:
        raise ConfigError(format_errors(errors, source=source))
