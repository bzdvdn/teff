"""Build the durable ``Assistant`` service from a workflow YAML file.

``workflow.yaml`` is the single source of truth: its ``steps``/``edges``
compile into the graph, its ``state``/``checkpoint:`` blocks drive the
durable conversation store, and its optional ``channels:`` block declares
which adapters to run.  Everything the channel layer needs is produced
here, exactly once, so every adapter talks to the same graph and the same
checkpoints.

The YAML format mirrors :func:`teff.yaml.load_workflow` with three
additional top-level blocks consumed by the channel layer::

    name: my-workflow
    checkpoint: {type: file, path: data/checkpoints}
    channels:
      server:
        host: 127.0.0.1
        port: 8000
      telegram:
        token_env: TELEGRAM_BOT_TOKEN
        mode: polling
      webhook:
        - path: /hooks/order
          schema: {type: object, properties: {...}}
          input:
            message: "new order {id}"
          session_key: id
    steps: [...]
    edges: [...]

``channels`` is optional; ``server`` defaults to 127.0.0.1:8000 when a
``serve``/``bot`` command is invoked without an explicit transport.
"""

from __future__ import annotations

import os
from typing import Any

from teff.assistant import Assistant
from teff.checkpoint import Checkpointer
from teff.yaml import checkpointer_from_workflow, load_workflow


def build_assistant(
    path: str,
    *,
    checkpointer: Checkpointer | None = None,
    max_iterations: int = 80,
) -> Assistant:
    """Compile *path* into a durable, interrupt-aware :class:`Assistant`.

    The workflow's ``checkpoint:`` block is honored by default (a
    ``JSONFileCheckpointer`` whose ``path`` resolves relative to the YAML
    file); pass *checkpointer* to override.  The ``state.initial`` mapping
    becomes the fresh-session seed, and ``state.schema`` reducers apply on
    every turn.

    Args:
        path: Path to the workflow YAML file.
        checkpointer: Override the checkpointer declared in the file.
        max_iterations: Cap on graph iterations per turn.

    Returns:
        A compiled :class:`Assistant` ready for ``run()``/``stream()``.
    """
    graph, tools, initial_state, reducers = load_workflow(path)
    if checkpointer is None:
        checkpointer = checkpointer_from_workflow(path)
    from typing import Callable

    make_state: Callable[[], dict] | None = None
    if initial_state:
        seed = dict(initial_state)

        def make_state():
            return dict(seed)

    return Assistant(
        graph,
        tools,
        checkpointer,
        reducers=reducers,
        initial_state=make_state,
        messages_key="messages",
        max_iterations=max_iterations,
    )


def load_channels(path: str) -> dict:
    """Return the parsed ``channels:`` block of *path* (``{}`` when absent).

    The block is read from the raw YAML document, after environment
    interpolation and include resolution, so ``${VAR}`` references and
    ``team/`` includes behave like everywhere else in the workflow.
    """
    from teff.yaml import _interpolate_env, _load_workflow_document, _resolve_includes

    base_dir = os.path.dirname(os.path.abspath(path))
    data = _interpolate_env(_load_workflow_document(path))
    data = _resolve_includes(data, base_dir)
    channels = data.get("channels", {})
    if channels is None:
        return {}
    if not isinstance(channels, dict):
        raise TypeError("channels: must be a mapping")
    return channels


def build_webhook(
    assistant: Assistant,
    spec: dict[str, Any],
):
    """Build one generic webhook channel from a ``channels.webhook`` entry."""
    from teff.channels.webhook import WebhookChannel

    return WebhookChannel(assistant, spec)
