"""CLI for running teff workflows from YAML files.

The app doubles as the default ``run`` command: ``teff --file wf.yaml``
and ``teff run --file wf.yaml`` are equivalent.  Additional subcommands
cover validation, inspection, evaluation, and versioning.
"""

import asyncio
import json
import os

import typer

from teff._version import __version__
from teff.checkpoint import DEFAULT_OWNER
from teff.errors import ConfigError
from teff.scaffold import TEMPLATES

#: Hosts where unauthenticated trace serving is still permitted.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

app = typer.Typer(
    name="teff",
    help="Workflow as data. Agents as graphs.",
    invoke_without_command=True,
)


def _checkpointer_from_config(config: dict):
    """Build a checkpointer from a ``{type, ...}`` dict.

    Delegates to the declarative builder shared with a workflow's
    ``checkpoint:`` block, surfacing errors as :class:`typer.BadParameter`.
    """
    from teff.checkpoint.from_config import checkpointer_from_config

    try:
        return checkpointer_from_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_workflow(
    file: str,
    *,
    output: str | None = None,
    pretty: bool = False,
    trace: bool = False,
    checkpoint: str | None = None,
    checkpoint_id: str | None = None,
    checkpoint_owner: str = DEFAULT_OWNER,
    resume: dict | None = None,
    node_timeout: float | None = None,
    max_iterations: int | None = None,
    interactive: bool = False,
) -> None:
    from teff.flow.compiler import load_flow, looks_like_flow
    from teff.yaml import load_workflow

    try:
        cfg = _load_yaml(file)
        if looks_like_flow(cfg):
            graph, tools, initial_state, reducers = load_flow(file)
        else:
            graph, tools, initial_state, reducers = load_workflow(file)
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    checkpointer = None
    base_dir = os.path.dirname(os.path.abspath(file))
    cp_config = _resolve_workflow_checkpoint(cfg, checkpoint, base_dir)
    if cp_config:
        checkpointer = _checkpointer_from_config(cp_config)

    observer_factory = _observer_factory(file, cfg, graph, base_dir)
    hooks = _resolve_workflow_hooks(cfg)

    try:
        result = asyncio.run(
            _run_loop(
                graph,
                initial_state,
                tools=tools,
                reducers=reducers,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                checkpoint_owner=checkpoint_owner,
                resume=resume,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                interactive=interactive,
                trace=trace,
                observer_factory=observer_factory,
                hooks=hooks,
            )
        )
    except Exception as e:
        typer.echo(f"error: workflow failed: {e}", err=True)
        raise typer.Exit(1)

    text = json.dumps(result, indent=2 if pretty else None, default=str) + "\n"
    if output:
        with open(output, "w") as f:
            f.write(text)
    else:
        typer.echo(text)


async def _run_loop(
    graph,
    state,
    *,
    tools,
    reducers,
    checkpointer,
    checkpoint_id,
    checkpoint_owner,
    resume,
    node_timeout,
    max_iterations,
    interactive,
    trace,
    observer_factory=None,
    hooks=None,
) -> dict | None:
    """Run a graph, handling interrupts interactively or via resume."""
    from teff.node.interrupt import GraphInterrupt
    from teff.trace import RunTracer

    observer = observer_factory() if observer_factory else None
    tracer = observer.tracer if observer else (RunTracer() if trace else None)
    async with graph:
        try:
            while True:
                try:
                    result = await graph.run(
                        state,
                        tools=tools,
                        reducers=reducers,
                        checkpointer=checkpointer,
                        checkpoint_id=checkpoint_id,
                        owner=checkpoint_owner,
                        resume=resume,
                        node_timeout=node_timeout,
                        max_iterations=max_iterations,
                        tracer=tracer,
                        hooks=hooks,
                        on_llm_payload=observer.on_llm_payload if observer else None,
                    )
                    if tracer is not None and observer is None:
                        typer.echo(tracer.to_json(), err=True)
                    return result
                except GraphInterrupt as interrupt:
                    if not interactive:
                        if tracer is not None:
                            typer.echo(tracer.to_json(), err=True)
                        raise
                    typer.echo(
                        f"\n-- paused: {interrupt.prompt or interrupt.key} "
                        f"(checkpoint {interrupt.checkpoint_id!r}) --",
                        err=True,
                    )
                    answer = input("> ").strip()
                    resume = {interrupt.key: answer}
        finally:
            if observer is not None:
                observer.export()


def _resolve_workflow_checkpoint(
    cfg: dict, flag: str | None, base_dir: str
) -> dict | None:
    """Pick the checkpointer config: an explicit ``--checkpoint`` wins,
    otherwise the workflow's ``checkpoint:`` block (resolved against base_dir)."""
    from teff.checkpoint.from_config import resolve_checkpoint_config

    if flag:
        return resolve_checkpoint_config(json.loads(flag), base_dir)
    block = cfg.get("checkpoint")
    if isinstance(block, dict):
        return resolve_checkpoint_config(block, base_dir)
    return None


def _resolve_workflow_hooks(cfg: dict) -> dict | None:
    """Resolve the workflow's ``hooks:`` block into graph hook callables."""
    block = cfg.get("hooks")
    if not isinstance(block, dict) or not block:
        return None
    from teff.hooks import resolve_hooks

    return resolve_hooks(block)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"{path}: expected a YAML mapping, got {type(data).__name__}"
        )
    return data


def _observer_factory(file: str, cfg: dict, graph, base_dir: str):
    """Build a per-run GraphObserver factory from the workflow's YAML.

    Returns ``None`` when the workflow has no ``observability:`` block, so
    existing workflows are untouched.
    """
    from teff.observability import build_observer_factory

    return build_observer_factory(
        cfg.get("observability"),
        base_dir=base_dir,
        graph=graph,
        name=cfg.get("name", "workflow"),
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    file: str = typer.Option(None, "--file", "-f", help="Path to workflow YAML file"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write result to file"
    ),
    pretty: bool = typer.Option(
        False, "--pretty", "-p", help="Pretty-print JSON output"
    ),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print a JSON run trace to stderr"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help='JSON checkpointer config, e.g. \'{"type":"file","path":"cp"}\'',
    ),
    checkpoint_id: str | None = typer.Option(
        None, "--checkpoint-id", help="Checkpoint key identifying the run"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
    resume: str | None = typer.Option(
        None, "--resume", help='Resume values as JSON, e.g. \'{"approved":"yes"}\''
    ),
    node_timeout: float | None = typer.Option(
        None, "--node-timeout", help="Max seconds per node"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Max node executions (loop guard)"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Prompt the operator on stdin when a workflow pauses for input",
    ),
) -> None:
    """Run a workflow from a YAML file (default command)."""
    if ctx.invoked_subcommand is not None:
        return
    if not file:
        typer.echo(ctx.get_usage(), err=True)
        raise typer.Exit(1)
    _run_workflow(
        file,
        output=output,
        pretty=pretty,
        trace=trace,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        checkpoint_owner=checkpoint_owner,
        resume=json.loads(resume) if resume else None,
        node_timeout=node_timeout,
        max_iterations=max_iterations,
        interactive=interactive,
    )


@app.command()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Path to workflow YAML file"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write result to file"
    ),
    pretty: bool = typer.Option(
        False, "--pretty", "-p", help="Pretty-print JSON output"
    ),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print a JSON run trace to stderr"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help='JSON checkpointer config, e.g. \'{"type":"file","path":"cp"}\'',
    ),
    checkpoint_id: str | None = typer.Option(
        None, "--checkpoint-id", help="Checkpoint key identifying the run"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
    resume: str | None = typer.Option(
        None, "--resume", help='Resume values as JSON, e.g. \'{"approved":"yes"}\''
    ),
    node_timeout: float | None = typer.Option(
        None, "--node-timeout", help="Max seconds per node"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Max node executions (loop guard)"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Prompt on stdin when a workflow pauses for input"
    ),
) -> None:
    """Run a workflow from a YAML file."""
    _run_workflow(
        file,
        output=output,
        pretty=pretty,
        trace=trace,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        checkpoint_owner=checkpoint_owner,
        resume=json.loads(resume) if resume else None,
        node_timeout=node_timeout,
        max_iterations=max_iterations,
        interactive=interactive,
    )


@app.command()
def build(
    file: str = typer.Argument(..., help="Path to flow.yaml (authoring surface)"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Path for the compiled graph.yaml"
    ),
) -> None:
    """Compile a flow.yaml into the low-level graph.yaml artifact.

    Previews the two-layer compile: ``flow.yaml → Graph → graph.yaml``.
    Unless ``--output`` is given the compiled YAML is written to
    ``graph.yaml`` next to the source flow file.
    """
    from teff.flow.compiler import build_flow_to_yaml, looks_like_flow

    try:
        cfg = _load_yaml(file)
        if not looks_like_flow(cfg):
            typer.echo(
                f"error: {file} does not look like an authoring flow.yaml "
                "(no idiom steps like `team:`, `llm:`, `map:`); use it "
                "directly with `teff run -f {file}` as a graph",
                err=True,
            )
            raise typer.Exit(2)
        if output is None:
            base = file if file.endswith(".yaml") else f"{file}.yaml"
            output = base[:-5] + "_graph.yaml" if base.endswith(".yaml") else None
        text = build_flow_to_yaml(file, output=output)
        target = output or "<stdout>"
        typer.echo(f"ok: compiled {file} → {target}")
        if output is None:
            typer.echo(text)
    except Exception as e:
        typer.echo(f"error: failed to compile flow: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def daemon(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    interval: float = typer.Option(
        60.0, "--interval", "-i", help="Seconds between ticks"
    ),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print a JSON run trace to stderr"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help='JSON checkpointer config, e.g. \'{"type":"file","path":"cp"}\'',
    ),
    checkpoint_id: str = typer.Option(
        "daemon", "--checkpoint-id", help="Checkpoint key for durable daemon state"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
    node_timeout: float | None = typer.Option(
        None, "--node-timeout", help="Max seconds per node"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Max node executions (loop guard)"
    ),
) -> None:
    """Run a workflow as a daemon: poll on an interval, keeping state between ticks.

    The workflow itself defines what a *tick* does — e.g. list open GitLab
    merge requests, review new ones, post verdicts and notify Telegram.
    Durable state (already-reviewed MRs, counters, …) is carried across ticks
    via the optional ``--checkpoint``.
    """
    from teff.flow.compiler import load_flow, looks_like_flow
    from teff.yaml import load_workflow

    try:
        cfg = _load_yaml(file)
        if looks_like_flow(cfg):
            graph, tools, initial_state, reducers = load_flow(file)
        else:
            graph, tools, initial_state, reducers = load_workflow(file)
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    checkpointer = None
    base_dir = os.path.dirname(os.path.abspath(file))
    cp_config = _resolve_workflow_checkpoint(cfg, checkpoint, base_dir)
    if cp_config:
        checkpointer = _checkpointer_from_config(cp_config)

    observer_factory = _observer_factory(file, cfg, graph, base_dir)
    hooks = _resolve_workflow_hooks(cfg)

    try:
        asyncio.run(
            _daemon_loop(
                graph,
                initial_state,
                tools=tools,
                reducers=reducers,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                checkpoint_owner=checkpoint_owner,
                interval=interval,
                once=once,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                trace=trace,
                observer_factory=observer_factory,
                hooks=hooks,
            )
        )
    except Exception as e:
        typer.echo(f"error: daemon failed: {e}", err=True)
        raise typer.Exit(1)


async def _daemon_loop(
    graph,
    initial_state,
    *,
    tools,
    reducers,
    checkpointer,
    checkpoint_id,
    checkpoint_owner,
    interval,
    once,
    node_timeout,
    max_iterations,
    trace,
    observer_factory=None,
    hooks=None,
) -> None:
    """Run *graph* once per tick, persisting state between ticks.

    Each tick re-runs the workflow from its entry point (so sources like
    GitLab are re-polled), starting from the durable state of the previous
    tick.  ``GraphInterrupt`` pauses are logged and skipped rather than
    blocking the daemon.
    """
    from teff.checkpoint import Checkpoint
    from teff.node.interrupt import GraphInterrupt
    from teff.trace import RunTracer

    state = dict(initial_state)
    if checkpointer is not None:
        saved = await checkpointer.load(checkpoint_id, owner=checkpoint_owner)
        if saved is not None:
            state = dict(saved.state)

    tick = 0
    async with graph:
        while True:
            tick += 1
            observer = observer_factory() if observer_factory else None
            tracer = observer.tracer if observer else (RunTracer() if trace else None)
            try:
                try:
                    state = await graph.run(
                        state,
                        tools=tools,
                        reducers=reducers,
                        node_timeout=node_timeout,
                        max_iterations=max_iterations,
                        tracer=tracer,
                        hooks=hooks,
                        on_llm_payload=observer.on_llm_payload if observer else None,
                    )
                    if tracer is not None and observer is None:
                        typer.echo(tracer.to_json(), err=True)
                    typer.echo(json.dumps(state, ensure_ascii=False, default=str))
                    if checkpointer is not None:
                        await checkpointer.save(
                            checkpoint_id,
                            Checkpoint(
                                state=dict(state), next_node_id=None, iteration=0
                            ),
                            owner=checkpoint_owner,
                        )
                except GraphInterrupt as interrupt:
                    if tracer is not None and observer is None:
                        typer.echo(tracer.to_json(), err=True)
                    typer.echo(
                        f"tick {tick}: paused at {interrupt.node_id!r} "
                        f"({interrupt.prompt or interrupt.key}) — skipped in daemon mode",
                        err=True,
                    )
            finally:
                if observer is not None:
                    observer.export()
            if once:
                return
            await asyncio.sleep(interval)


@app.command()
def graph(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    mermaid: bool = typer.Option(
        False, "--mermaid", help="Render the workflow graph as a Mermaid diagram"
    ),
) -> None:
    """Inspect a workflow graph: YAML topology or a Mermaid diagram."""
    from teff.flow.compiler import load_flow, looks_like_flow
    from teff.yaml import load_workflow

    try:
        cfg = _load_yaml(file)
        if looks_like_flow(cfg):
            graph_, _tools, _state, _reducers = load_flow(file)
        else:
            graph_, _tools, _state, _reducers = load_workflow(file)
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    if mermaid:
        typer.echo(graph_.to_mermaid())
        return
    typer.echo(graph_.to_yaml())


@app.command()
def validate(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
) -> None:
    """Validate a workflow YAML file without running it.

    Detects whether *file* is authored in the idiom surface (``team:``,
    ``map:``, ``loop:`` …) or the classic graph format and runs the
    matching validator, resolving ``include:`` blocks and declared plugins
    first.
    """
    import yaml as _yaml

    from teff.flow.compiler import looks_like_flow
    from teff.yaml_schema import (
        format_errors,
        validate_flow_file,
        validate_workflow_file,
    )

    try:
        with open(file) as f:
            cfg = _yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ConfigError(f"{file}: workflow must be a mapping")
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    kind = "workflow" if looks_like_flow(cfg) else "graph"
    try:
        if kind == "workflow":
            errors = validate_flow_file(file)
        else:
            errors = validate_workflow_file(file)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if errors:
        typer.echo(format_errors(errors, source=file), err=True)
        typer.echo(f"invalid: {len(errors)} error(s)", err=True)
        raise typer.Exit(1)
    typer.echo(f"ok: {file} is a valid workflow ({kind})")


@app.command("eval")
def eval_(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    data: str = typer.Option(
        ..., "--data", "-d", help="Dataset file (.json/.jsonl/.csv)"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write the JSON report to a file"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Model used to score outputs (LLM judge)"
    ),
    judge_provider: str | None = typer.Option(
        None, "--judge-provider", help="Provider key for the judge model"
    ),
    exact: bool = typer.Option(
        False, "--exact", help="Score by exact (normalised) string match"
    ),
    max_examples: int | None = typer.Option(
        None, "--max-examples", help="Limit the number of examples"
    ),
    output_key: str | None = typer.Option(
        None, "--output-key", help="State key holding the answer"
    ),
) -> None:
    """Evaluate a workflow against a dataset and report pass/fail."""
    import json as _json

    from teff.eval import format_report, load_dataset, run_eval
    from teff.flow.compiler import load_flow, looks_like_flow
    from teff.yaml import load_workflow

    try:
        cfg = _load_yaml(file)
        if looks_like_flow(cfg):
            workflow = load_flow(file)
        else:
            workflow = load_workflow(file)
        dataset = load_dataset(data)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    try:
        report = asyncio.run(
            run_eval(
                workflow,
                dataset,
                judge_model=judge_model,
                judge_provider=judge_provider,
                exact=exact,
                max_examples=max_examples,
                output_key=output_key,
            )
        )
    except Exception as e:
        typer.echo(f"error: eval failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(format_report(report), err=True)
    text = _json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    if output:
        with open(output, "w") as f:
            f.write(text)
    else:
        typer.echo(text)


@app.command()
def inspect(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="JSON checkpointer config"
    ),
    checkpoint_id: str = typer.Option(
        ..., "--checkpoint-id", help="Run key to inspect"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
) -> None:
    """Print the saved state for a checkpointed run."""
    try:
        cp = _checkpointer_from_config(json.loads(checkpoint))
        saved = asyncio.run(cp.load(checkpoint_id, owner=checkpoint_owner))
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if saved is None:
        typer.echo(f"no checkpoint for {checkpoint_id!r}", err=True)
        raise typer.Exit(1)
    from teff.checkpoint import checkpoint_to_dict

    typer.echo(json.dumps(checkpoint_to_dict(saved), indent=2, default=str))


@app.command()
def prune(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="JSON checkpointer config"
    ),
    checkpoint_owner: str | None = typer.Option(
        None,
        "--checkpoint-owner",
        help="Only prune this owner (default: all owners)",
    ),
    max_age: float | None = typer.Option(
        None,
        "--max-age",
        help="Delete checkpoints older than this many seconds",
    ),
    keep_last: int | None = typer.Option(
        None, "--keep-last", help="Keep only the N most recent per owner"
    ),
) -> None:
    """Delete stale checkpoints (TTL / keep-last GC)."""
    try:
        cp = _checkpointer_from_config(json.loads(checkpoint))
        removed = asyncio.run(
            cp.cleanup(
                owner=checkpoint_owner,
                max_age=max_age,
                keep_last=keep_last,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"removed {removed} checkpoint(s)")


@app.command()
def obs_server(
    db: str = typer.Option("traces.db", "--db", help="SQLite file holding the traces"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Address to bind (use 0.0.0.0 to expose)"
    ),
    port: int = typer.Option(8001, "--port", help="Port to listen on"),
    prefix: str = typer.Option(
        "/obs", "--prefix", help="URL prefix for the dashboard and ingest"
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="TEFF_OBS_API_KEY",
        help="Shared key required in the X-API-Key header (mandatory on 0.0.0.0)",
    ),
) -> None:
    """Serve the trace dashboard + ingest endpoint (standalone obs server).

    Workflows with no API push their traces here via ``observability:``
    (``type: webhook``), and this process serves the dashboard UI::

        teff obs-server --db traces.db --host 127.0.0.1 --port 8001
        # open http://localhost:8001/obs/ui

    Traces contain full prompts/responses.  Binding to a non-loopback host
    (``0.0.0.0``) without ``--api-key`` is refused: the server refuses to
    start rather than expose them unauthenticated.
    """
    if api_key is None and host not in _LOOPBACK_HOSTS:
        raise typer.BadParameter(
            "--api-key is required when binding outside 127.0.0.1 "
            "(traces contain full prompts/responses)",
            param_hint="--host",
        )
    try:
        from teff.observability.server import serve
    except ImportError as e:
        typer.echo(
            f"error: 'teff[observability]' is required for obs-server: {e}",
            err=True,
        )
        raise typer.Exit(1)
    serve(db, host=host, port=port, prefix=prefix, api_key=api_key)


@app.command()
def new(
    name: str = typer.Argument(..., help="Project name, e.g. 'support-ai'"),
    dest: str | None = typer.Option(
        None, "--dest", help="Destination directory (default: ./<slug>)"
    ),
    template: str = typer.Option(
        "fastapi",
        "--template",
        "-t",
        help=f"App template: {', '.join(TEMPLATES)}",
    ),
    with_variants: str = typer.Option(
        "",
        "--with",
        help="Comma-separated feature variants: postgres,rag,celery",
    ),
) -> None:
    """Scaffold a new teff app from a template (fastapi|cli|daemon)."""
    from teff.scaffold import new_project

    variants = tuple(v for v in (p.strip() for p in with_variants.split(",")) if v)
    try:
        path = new_project(name, dest=dest, template=template, variants=variants)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"created {path}")
    typer.echo(
        f"next: uv sync && uv run pytest tests/ && uv run {TEMPLATES[template].entry}"
    )
    if variants:
        typer.echo(f"variants: {', '.join(variants)}")


@app.command()
def serve(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
) -> None:
    """Serve a workflow over HTTP/SSE plus any configured webhook channels.

    The ``channels:`` block of the workflow YAML is the source of truth:
    ``server`` (host/port) is overridable via ``--host``/``--port``, and
    every ``channels.webhook`` entry is mounted as a POST endpoint.  The
    same compiled :class:`~teff.assistant.Assistant` serves all routes, so
    checkpoints and interrupts behave identically everywhere.
    """
    try:
        import uvicorn
        from fastapi import Request
    except ImportError:
        typer.echo(
            "error: serving over HTTP requires the fastapi extra: "
            "uv sync --extra fastapi",
            err=True,
        )
        raise typer.Exit(1)

    from teff.channels import build_assistant, build_webhook, create_http_app
    from teff.channels.factory import load_channels

    try:
        assistant = build_assistant(file)
    except Exception as e:
        typer.echo(f"error: failed to build workflow: {e}", err=True)
        raise typer.Exit(1)

    app = create_http_app(assistant)
    channels = load_channels(file)
    webhooks = channels.get("webhook") or []
    for spec in webhooks:
        hook = build_webhook(assistant, spec)

        @app.post(hook.path)
        async def _webhook_endpoint(request: Request) -> dict:
            payload = await request.json()
            return await hook.handle(payload, headers=dict(request.headers))

    typer.echo(
        f"serving {os.path.basename(file)} on http://{host}:{port}"
        f" ({len(webhooks)} webhook route(s))"
    )
    uvicorn.run(app, host=host, port=port)


@app.command()
def bot(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    token_env: str = typer.Option(
        "TELEGRAM_BOT_TOKEN", "--token-env", help="Env var holding the bot token"
    ),
    mode: str = typer.Option("polling", "--mode", help="Transport: polling or webhook"),
    once: bool = typer.Option(False, "--once", help="Process pending updates and exit"),
) -> None:
    """Run a workflow as a Telegram bot (long-polling or webhook).

    Reads the workflow's ``channels.telegram`` block for ``mode``/``url``
    (CLI flags win), then binds the same compiled ``Assistant`` to every
    chat: each chat is a durable session, so interrupts ask questions
    in-chat and resume on the operator's answer.
    """
    from teff.channels import TelegramChannel, build_assistant
    from teff.channels.factory import load_channels

    try:
        assistant = build_assistant(file)
    except Exception as e:
        typer.echo(f"error: failed to build workflow: {e}", err=True)
        raise typer.Exit(1)

    import os as _os

    token = _os.environ.get(token_env, "")
    if not token:
        typer.echo(f"error: {token_env} is not set", err=True)
        raise typer.Exit(1)

    cfg = load_channels(file).get("telegram") or {}
    effective_mode = mode if mode != "polling" else cfg.get("mode", "polling")
    bot = TelegramChannel(assistant, token, reply_when=cfg.get("reply_when", "all"))

    if effective_mode == "webhook":
        url = cfg.get("url")
        if not url:
            typer.echo("error: webhook mode requires channels.telegram.url", err=True)
            raise typer.Exit(1)
        try:
            import uvicorn
            from fastapi import Request

            from teff.channels import create_http_app
        except ImportError:
            typer.echo(
                "error: webhook mode requires the fastapi extra: "
                "uv sync --extra fastapi",
                err=True,
            )
            raise typer.Exit(1)
        app = create_http_app(assistant)

        @app.post("/api/telegram")
        async def _telegram_webhook(request: Request) -> dict:
            update = await request.json()
            await bot.handle_update(update)
            return {"ok": True}

        async def _main() -> None:
            await bot.set_webhook(url)
            await uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=8000)
            ).serve()

        asyncio.run(_main())
        return

    typer.echo(f"telegram bot polling (token env {token_env})")
    asyncio.run(bot.run(once=once))


@app.command()
def chat(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    session: str | None = typer.Option(
        None, "--session", "-s", help="Durable session id (default: chat-<user>)"
    ),
    owner: str = typer.Option(
        DEFAULT_OWNER, "--owner", help="Owner scoping this session's checkpoints"
    ),
    prompt: str = typer.Option(
        "> ", "--prompt", help="Input prompt shown before each turn"
    ),
) -> None:
    """Chat with a workflow interactively from the terminal.

    Builds the durable :class:`~teff.assistant.Assistant` from *file* and
    runs a REPL: each line is one turn, the reply is printed, and a paused
    workflow (interrupt) asks in-chat and resumes on your answer — so the
    same ``workflow.yaml`` that serves HTTP/Telegram/webhook also runs as a
    plain terminal conversation.  Ctrl-D or Ctrl-C exits.
    """
    from teff.channels import build_assistant
    from teff.channels.reply import turn_response

    try:
        assistant = build_assistant(file)
    except Exception as e:
        typer.echo(f"error: failed to build workflow: {e}", err=True)
        raise typer.Exit(1)

    session_id = session or f"chat-{owner}"
    typer.echo(f"teff chat: session={session_id} owner={owner} (Ctrl-D to exit)")

    async def _loop() -> None:
        while True:
            try:
                message = input(prompt)
            except EOFError:
                typer.echo("\nbye")
                return
            if not message.strip():
                continue
            result = await assistant.run(session_id, message, owner=owner)
            payload = turn_response(result, session_id)
            typer.echo(payload["message"] if payload["message"] else "(no reply)")

    try:
        asyncio.run(_loop())
    except (KeyboardInterrupt, EOFError):
        typer.echo("\nbye")


@app.command()
def version() -> None:
    """Print the teff version."""
    typer.echo(f"teff {__version__}")


if __name__ == "__main__":
    app()
