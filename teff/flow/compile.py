"""Compilation and serialization builders for :class:`~teff.flow.Flow`.

:class:`CompileBuilder` implements ``compile``, ``label``/``label_target``
and ``to_yaml``.  :class:`Flow` owns an instance and delegates to it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teff.flow.flow import Flow


class CompileBuilder:
    """Compile a :class:`~teff.flow.Flow` into an executable graph.

    Args:
        flow: The owning ``Flow`` whose graph state is read.
    """

    def __init__(self, flow: "Flow"):
        self.flow = flow

    def compile(self):
        """Compile the flow into a ``Graph`` ready for execution.

        Raises:
            ValueError: If no nodes were added.
        """
        return self._compile()

    def label(self, name: str):
        """Attach a route *name* to the most recently added node.

        ``Command``-style ``goto`` targets (from a declarative ``route``
        step) must name a real node id in the compiled graph.  A loop's
        body has no node of its own — the *decider* (the node that reads
        the loop key) is its re-entry point.  Label it so sugar can route
        back to it::

            flow.step(extract_verdict, id="extract_verdict")
            flow.loop(key="verdict", until="pass", body=body, done=done)
            flow.label("refine")            # ("refine" now means "extract_verdict")
            flow.step(route_node)           # route: goto "refine" loops back

        Returns ``self`` for chaining.
        """
        target = self.flow
        prev = target._last_added
        if prev is None:
            raise ValueError("label() requires a preceding node")
        loop_target = target._loop_decider if target._loop_decider is not None else prev
        target._loop_labels[name] = loop_target
        return target

    def label_target(self, goto: str) -> str:
        """Resolve a declarative ``goto`` against labels to a real node id.

        ``label()`` maps a route name (e.g. ``"refine"``) to the loop
        decider's node id; this turns a sugar ``route: {goto: refine}``
        into an executable ``goto: <decider_id>``.  Names that are not
        labeled (real node ids, ``STOP``) pass through unchanged.
        """
        return self.flow._loop_labels.get(goto, goto)

    def _compile(self):
        """Compile the flow into a ``Graph`` ready for execution.

        Raises:
            ValueError: If no nodes were added.
        """
        from teff.graph import Graph

        target = self.flow
        if not target._nodes:
            raise ValueError("no nodes in flow")
        return Graph(
            nodes=dict(zip(target._node_ids, target._nodes)),
            edges=target._edges,
            entry_point=target._node_ids[0],
            providers=target._providers,
            default_provider=target._default_provider,
            default_model=target._default_model,
        )

    def to_yaml(
        self,
        *,
        tools: list | None = None,
        initial: dict | None = None,
        reducers: dict | None = None,
    ) -> str:
        """Export the compiled flow as a ``workflow.yaml`` document.

        The graph (``steps`` + ``edges``) is serialised faithfully —
        including the ReAct loop wiring produced by :meth:`harness` /
        :meth:`react`.  Tools and state are not tracked by ``Flow``, so
        pass them explicitly if you want them in the export::

            yaml_text = (
                Flow("repo")
                .react(model="llama3.1:8b", use_tools="all")
                .to_yaml(tools=[GitTool(), CsvQueryTool()])
            )
            with open("workflow.yaml", "w") as f:
                f.write(yaml_text)

        The result validates with ``teff validate`` and round-trips through
        :func:`teff.yaml.load_workflow`.
        """
        from teff.yaml import workflow_to_yaml

        return workflow_to_yaml(
            self.compile(),
            tools=tools,
            initial=initial,
            reducers=reducers,
            name=self.flow._name or "graph",
        )
