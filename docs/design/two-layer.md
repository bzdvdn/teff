# Two-layer workflow: `flow.yaml` → `graph.yaml`

> Status: **design proposal for 0.2.0.** Nothing here is implemented yet.

Today there is one file format (`workflow.yaml`) that is really two things
at once: an authoring surface and a compiled graph. This design splits it
into two named layers that mirror the Python API 1:1 — `Flow` (how you
build) and `Graph` (how it executes) — and demotes the low-level form to a
generated artifact.

## 1. The model

```
Python:   Flow()  --compile()-->  Graph()
YAML:     flow.yaml --compile()--> graph.yaml
                 <-- export ----
```

| | Authoring (sugar) | Runtime (artifact) |
| --- | --- | --- |
| Python | `teff.flow.Flow` | `teff.graph.Graph` |
| YAML | `flow.yaml` | `graph.yaml` |
| Human job | describe the app | debug / inspect the run |
| VCS | commit it | usually generated, not stored |
| Extends | your logic | — |

`graph.yaml` is what `workflow.yaml` is today, renamed honestly: `steps:` +
`edges:` + node `config:`. `flow.yaml` is a new declarative sugar surface
that compiles down into the same nodes and edges.

## 2. Naming: why `workflow.yaml` goes away

`workflow` is a third term that collides with `Flow` and `Graph`. The
Python docs already teach two concepts (`Flow builder` vs `the compiled
graph`); the file names should say the same two concepts. Keeping
`flow.yaml` and `graph.yaml` means **one mental model across code and
configuration** — authors never translate between `workflow` and anything
else.

Backward compatibility rule: `teff -f workflow.yaml` and `load_workflow()`
keep working as aliases for **graph.yaml during 0.2.x**, with a deprecation
notice pointing at the new names.

## 3. Mapping: which sugar compiles into which graph piece

`flow.yaml` exposes high-level idioms only. Everything below compiles into
ordinary `graph.yaml` nodes, edges, `Supervisor`, `route`, `Parallel`,
`Map`, `Interrupt` — reusing what already exists (no second runtime).

| `flow.yaml` idiom | compiles to (`graph.yaml`) |
| --- | --- |
| `flow:` / `steps:` | `steps:` + `edges:` (chain) |
| `team:` + `lead:` / roles | `supervisor` node + `route` edges + role `context_builder → harness → append` chains |
| `strategy: chain/parallel/hierarchical` | linear edges / `parallel` node + `converge` / supervisor loop |
| `human:` / `review_human:` | `interrupt` node (+ `Ask`/classifier when the accept is fuzzy) |
| `map:` | `Map` node |
| `case:` / `branch` | `command` routes or conditional `edges` |
| `loop:` | `loop` node |

Invariant: **`flow.yaml` only generates pieces a hand-written `graph.yaml`
can also express.** If an idiom has no well-defined, checkable expansion,
it does not belong in `flow.yaml`.

## 4. Authoring rule (the "B" scope boundary)

`flow.yaml` is deliberately a **subset**, not a second low-level format:

- Covers: teams, agents, chain / parallel / hierarchical flow, HITL gates,
  Map fans-out, the common route-and-converge patterns.
- Does **not** attempt: arbitrary `config:` verbatim, custom `edge`
  conditions beyond the idiomatic ones, custom node options. For those —
  escape hatch: `use: graph` / `include:` a hand `graph.yaml` or Python `Flow`.

Rule of thumb shipped in docs: *start in `flow.yaml`, drop to
`graph.yaml`/Python where the idiomatic surface stops.*

## 5. Round-trip as the safety net

`Flow.to_yaml()` already serialises the compiled graph
(`workflow_to_yaml`). The 0.2 work extends it so one can ask for either
level:

- `Flow.to_yaml(layer="graph")` → `graph.yaml` (current behaviour).
- `Flow.to_yaml(layer="flow")` → the sugar form of the same topology
  (best-effort for graphs that already have a sugar expression).

Combined with `compile()`, this guarantees two equal editors: whatever is
authored in `flow.yaml` round-trips into a valid `graph.yaml`, and both
types are equivalent to the Python `Flow`/`Graph` boolean reality:

```
flow.yaml ⇄ Flow ⇄ graph  (one model — two surface forms)
```

## 6. Layout

```
app/
├── flow.yaml          # authoring surface (commit this)
├── .teff/
│   └── build/
│       └── graph.yaml # compiled artifact (like a build output)
└── ... 
```

`graph.yaml` may also live next to `flow.yaml` when a team explicitly
wants to inspect/patch the compiled result — same file format either way.

## 7. CLI next-steps

- `teff build` — compile `flow.yaml` → `graph.yaml` (printed path).
- `teff graph` — keep: render/dump the compiled graph (mermaid, `--yaml`).
- `teff validate` — checks authoring layer; compilation errors surface
  node-level with precise paths.
- `teff run -f flow.yaml` — implicit `flow.yaml → graph.yaml → graph` in
  one command (same UX as today).

## 8. Open questions to resolve in 0.2 spike

- Exact grammar: `steps:` inline sugar vs `team:` block syntax (P1).
- Whether `graph.yaml` output defaults to `.teff/build/` or commitment
  in-repo (P2, ops preference).
- Migration path & deprecation window for `workflow.yaml` aliases (P3).
- Editor/tooling: schema for `flow.yaml` (JSON Schema) to get
  autocomplete in VSCode/NeoVim (helps marketing: "editors support my
  syntax").

## 9. Marketing note (why this helps promote 0.2)

- The two-layer story is simple enough to demo in a 90-second clip:
  `flow.yaml` (5 lines) → `teff new` → `graph.yaml` (auto) → `run/resume/graph`.
- Draw a clear comparison against CrewAI-style teams: same team idiom, but
  *you still see your graph* — compilable, diffable, exportable. That is
  the "LangGraph + CrewAI, both served" pitch.
- The named pair `flow/graph` gives documentation a crisp two-column frame
  that a screen-comparison hero section can lean on.