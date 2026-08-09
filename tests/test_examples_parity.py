"""Every sugar ``workflow.yaml`` ships a low-level ``graph.yaml`` twin.

The authoring layer (single-key idiom steps) and the classic low-level
surface (``id``/``type`` steps + ``edges``) must agree on the pieces that
cross the two layers: the tool set and the initial state.  This test walks
every example pair and checks both files load and validate on their own
surface, that ``graph.yaml`` really is a low-level graph (never sugar), and
that the two layers describe the same tools and initial state.
"""

import glob
from pathlib import Path

import pytest

from teff.flow.compiler import looks_like_flow
from teff.yaml import load_workflow, load_workflow_document
from teff.yaml_schema import validate_workflow_file

ROOT = Path(__file__).resolve().parent.parent

WORKFLOWS = sorted(
    glob.glob(str(ROOT / "examples" / "*" / "workflow.yaml"))
    + glob.glob(str(ROOT / "examples" / "*" / "*" / "workflow.yaml"))
)
PAIRS = [wf for wf in WORKFLOWS if Path(wf).with_name("graph.yaml").is_file()]

# rag stores behind optional third-party drivers: skip when the driver is
# not installed (mirrors tests/test_stores_external.py).
_STORE_MODULE = {
    "chroma": "chromadb",
    "faiss": "faiss",
    "lance": "lancedb",
    "milvus": "pymilvus",
    "qdrant": "qdrant_client",
    "weaviate": "weaviate",
}

# Stores whose constructor boots an embedded server on fixed ports; loading
# the sugar + graph twin in one process would start it twice and collide.
# Their twins are still checked at the document level (validate, tools,
# initial) — full ``load_workflow`` is exercised once by the graph file.
_SERVER_STORES = {"weaviate"}


@pytest.mark.parametrize("workflow", PAIRS, ids=[Path(p).parent.name for p in PAIRS])
def test_workflow_graph_pair_parity(workflow):
    for store, mod in _STORE_MODULE.items():
        if f"/rag_stores/{store}/" in workflow:
            pytest.importorskip(mod)

    graph = workflow.replace("workflow.yaml", "graph.yaml")

    assert looks_like_flow(_raw(workflow)), f"{workflow} is not sugar"
    assert not looks_like_flow(_raw(graph)), f"{graph} is not a low-level graph"
    assert validate_workflow_file(workflow) == []
    assert validate_workflow_file(graph) == []

    # tools / initial agree at the document level (includes resolved, no
    # store booted), so the check is safe even for server-backed stores.
    wd = load_workflow_document(workflow)
    gd = load_workflow_document(graph)
    assert _tool_names(wd) == _tool_names(gd)
    assert _initial_state(wd) == _initial_state(gd)

    if any(f"/rag_stores/{s}/" in workflow for s in _SERVER_STORES):
        return

    wg, wtools, winit, _ = load_workflow(workflow)
    gg, gtools, ginit, _ = load_workflow(graph)

    assert wg.entry_point, f"{workflow} produced no entry point"
    assert gg.entry_point, f"{graph} produced no entry point"
    assert sorted(t.name for t in wtools) == sorted(t.name for t in gtools)
    assert winit == ginit


def _tool_names(data: dict) -> list[str]:
    return sorted(td.get("type") for td in (data.get("tools") or []))


def _initial_state(data: dict) -> dict:
    state = data.get("state") or {}
    return dict(state.get("initial") or {})


def _raw(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
