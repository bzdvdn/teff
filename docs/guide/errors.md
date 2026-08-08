# Errors & resilience

Teff treats failure as a first-class part of the graph. This page maps the
error types, shows how the runtime routes failures, and gives the
retry/recovery levers — from a simple per-step retry to a catch-all
`__error__` edge.

## 1. The error hierarchy

Everything the framework raises derives from `teff.TeffError`, so a single
`except te.TeffError` catches any library error:

| Type              | Raised when                                            |
| ----------------- | ------------------------------------------------------ |
| `TeffError`       | base class for all framework errors                    |
| `ConfigError`     | bad workflow YAML, unknown node/tool type (works as `KeyError`) |
| `WorkflowError`   | runtime failures: loop guards, `max_iterations`, invariants (works as `RuntimeError`) |
| `NodeError`       | a node failed; carries `node_id` and `node_type`       |
| `LLMError`        | a model call failed after retries and fallbacks (transport is `__cause__`) |
| `InterruptError`  | an interrupt/resume contract violation                 |
| `GraphInterrupt`  | the run paused for human input (a **normal** pause, not an error) |

`NodeError` and `LLMError` inherit from both `WorkflowError` and `RuntimeError`
on purpose, so legacy `except RuntimeError` blocks keep working.

## 2. How execution surfaces failure

During a run:

- A node that raises propagates up as a `NodeError` (wrapping the original as
  `__cause__`) unless it is wrapped in a `Retry` or is caught by an
  `__error__` edge.
- A paused human step raises `GraphInterrupt` — this is expected control flow,
  catch it and resume with a `resume=` answer (see
  [durable](durable.md) / [human_in_loop](https://github.com/bzdvdn/teff/tree/main/examples/human_in_loop/)).
- Loop guards raise `WorkflowError` (`graph exceeded max_iterations=…`) so a
  run can never spin forever.

## 3. `Retry` — the first lever

Wrap any step with retry logic, either the `Retry` node or the YAML `retry:`
block next to `config:`.

```yaml
steps:
  - id: search
    type: web_search
    config: {query_key: q, output_key: results}
    retry:
      max_retries: 4
      delay: 0.5
      backoff: 2.0        # delays: 0.5s, 1s, 2s, 4s
      timeout: 30
      retry_on: ["httpx.HTTPStatusError", 429]   # only these
```

- `retry_on` matches exception type names or HTTP status codes; empty means
  *everything* is retried. `[]`→ all.
- Use `retry: {enabled: false}` to keep the schema valid but disable.
- The wrapper preserves the inner node's behaviour — an `__error__` edge still
  fires after the final failed attempt.

Use `Retry` for **transient** failures (rate limits, timeouts). It is the
in-process complement to provider-level fallback (see §6).

## 4. `__error__` — handle, don't crash

An `__error__` edge routes control somewhere else when a node's error bypasses
(or exhausts) its retry. It is the "catch" of the graph:

- the failing node's error goes to the edge target instead of the whole run;
- the target can be a `transform` that records the failure, a `Fallback`, or
  the desired re-entry point.

This is the right tool for **expected-but-handleable** failures: one optional
external call failing should degrade the result, not abort the workflow.

## 5. Repair with `Extract` / `Fallback`

For structured output that can fail, don't just retry — repair.

- `Extract` asks the model for JSON in a controlled shape and reports a
  failure (instead of malformed state) if it can't.
- `Fallback` fills a field the model left empty with a deterministic
  `fn(state)` — a safe default, e.g. `None` → a sentinel or an empty list.

Chain them: `Extract` (parse) → `Fallback` (fill missing) → use. See
[structured-output](structured-output.md).

## 6. Whole-model failover

When a *provider* is down, per-step retry still fails. Move up a level:
configure `providers` with a fallback sequence so the runtime switches model
provider when one exhausts. This is orthogonal to node `Retry`; use both —
`Retry` for the transient spam, provider failover for a dead upstream.

## 7. Failures inside a fan-out

`Map` / `Parallel` run children concurrently. Decide *how much* a single
child failure should cost:

- Keep each child's behaviour local: wrap the child step in a `Retry` (a
  flaky per-item call) so a retryable blip doesn't nudge the batch.
- Give each child an `__error__` path that writes a partial result / marks the
  item `failed` in its own slot, so the run's final state reflects "these
  items succeeded, these didn't" instead of aborting.
- Let a fatal, non-retryable item failure propagate to the whole run when the
  batch must be all-or-nothing — then catch it at the `Map`'s `__error__`
  edge or a `converge` node.

Choose per batch: an audit/summary job tolerates partial items (record
failures), a money-moving job rejects the batch on any failure.

## 8. Putting it together

```yaml
steps:
  - id: lookup        # transient upstream that must not kill the run
    type: web_search
    config: {query_key: q, output_key: results}
    retry:
      max_retries: 4
      retry_on: [429, "httpx.HTTPStatusError"]
    on_error: handle_missing   # __error__ edge -> degrade
```

Decision flow:

1. **Transient?** → `Retry` (backoff, `retry_on`).
2. **Expected but degradeable?** → `__error__` edge to a handler node.
3. **Empty/malformed LLM output?** → `Extract` + `Fallback`.
4. **Provider truly down?** → provider failover.
5. **Hard, irreversible, irrecoverable?** → let it raise `NodeError` / abort the
   run (and observe it on the trace dashboard).