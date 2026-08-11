"""Linear chain builders for :class:`~teff.flow.Flow`.

The :class:`BaseBuilder` implements the ``step``/``llm``/``transform``
family: append a single node to the running chain.  :class:`Flow` owns an
instance and delegates to it.
"""


from teff.node.context import AppendAssistant, ContextBuilder
from teff.node.llm import LLM
from teff.node.node import Node
from teff.node.registry import make_function_node
from teff.node.supervisor import Supervisor
from teff.node.transform import Transform


class BaseBuilder:
    """Append linear (single-node) steps to a :class:`~teff.flow.Flow`.

    Args:
        flow: The owning ``Flow`` whose graph state is mutated.
    """

    def __init__(self, flow):
        self.flow = flow

    def step(self, node, id=None, *, when=None):
        """Append a node to the linear chain.

        Accepts a Node instance::

            flow.step(Transform(action="uppercase"))
            flow.step(LLM(model="gpt-4"))
            flow.step(custom_node)

        *id* optionally names the node in the compiled graph instead of
        the auto-generated ``{type}_{n}``.  Returns ``self`` for chaining.

        *when* guards the edge into this node: a string condition
        (``"status=ok"``) or a callable ``(state) -> bool``.  The edge
        fires only when the condition matches; pair with ``default()`` for
        a fallback branch::

            flow.step(transform_llm)                        # decider
            flow.step(admin_panel, when=lambda s: s.get("role") == "admin")
            flow.default(denied_panel)                      # runs when the guard fails
        """
        from teff.graph import Edge

        flow = self.flow
        flow._check_continuation()
        if not isinstance(node, Node):
            if not callable(node):
                raise TypeError(
                    "step() expects a Node instance or a function "
                    "(ctx, state) -> dict; got a value of type "
                    f"{type(node).__name__} — must be a Node or function"
                )
            node = make_function_node(node)
        flow._nodes.append(node)
        nid = flow._next_id(node, id)
        flow._node_ids.append(nid)
        flow._loop_decider = None
        prev = flow._last_added
        if prev is not None:
            flow._edges.append(Edge(source_id=prev, target_id=nid, condition=when))
        elif when is not None:
            raise ValueError("step(when=...) requires a preceding node to branch from")
        if when is not None:
            flow._guarded_step = prev
        else:
            flow._guarded_step = None
        flow._last_added = nid
        return flow

    def llm(self, node=None, id=None, *, memory=None, **config):
        """Add an :class:`~teff.node.llm.LLM` chat node.

        Pass a pre-built ``LLM`` instance to reuse a shared node, or give
        keyword config that is forwarded to the ``LLM`` constructor::

            flow.llm(model="gpt-4", system="You are helpful", output_key="answer")
            flow.llm(LLM(model="gpt-4", parse=True, output_key="data"))

        *memory* enables long-term memory injection on this node.  Passing
        both an instance and config kwargs raises ``TypeError``.  *id*
        optionally names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = LLM(memory=memory, **config)
        else:
            if config or memory:
                raise TypeError(
                    "llm() accepts either an LLM instance or config kwargs, not both"
                )
            if not isinstance(node, LLM):
                raise TypeError("llm() expects an LLM instance")
        return self.step(node, id=id)

    def transform(self, node=None, id=None, **config):
        """Add a :class:`~teff.node.transform.Transform` node.

        Pass a pre-built ``Transform`` instance or keyword config that is
        forwarded to the ``Transform`` constructor::

            flow.transform(action="uppercase", input_key="text", output_key="shout")
            flow.transform(Transform(action="value", value="done", output_key="status"))

        Passing both an instance and config kwargs raises ``TypeError``.
        *id* optionally names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = Transform(**config)
        else:
            if config:
                raise TypeError(
                    "transform() accepts either a Transform instance or config kwargs, not both"
                )
            if not isinstance(node, Transform):
                raise TypeError("transform() expects a Transform instance")
        return self.step(node, id=id)

    def context_builder(self, node=None, id=None, **config):
        """Add a :class:`~teff.node.context.ContextBuilder` node.

        Composes a plain-text ``input`` for an agent from shared state —
        each configured section rendered as ``<label>:\\n<value>`` plus the
        latest user message, clearing scratch keys before the agent runs::

            flow.context_builder(
                sections={"plan": "Plan", "summary": "Summary"},
                messages_key="messages",
                output_key="input",
            )

        Pass a pre-built ``ContextBuilder`` instance or keyword config that
        is forwarded to the ``ContextBuilder`` constructor.  Passing both an
        instance and config kwargs raises ``TypeError``.  *id* optionally
        names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = ContextBuilder(**config)
        else:
            if config:
                raise TypeError(
                    "context_builder() accepts either a ContextBuilder "
                    "instance or config kwargs, not both"
                )
            if not isinstance(node, ContextBuilder):
                raise TypeError("context_builder() expects a ContextBuilder instance")
        return self.step(node, id=id)

    def append_assistant(self, node=None, id=None, **config):
        """Add a :class:`~teff.node.context.AppendAssistant` node.

        Appends an agent's response (``state[output_key]``) back to the
        shared conversation as an ``assistant`` message::

            flow.append_assistant(output_key="draft", messages_key="messages")

        Pass a pre-built ``AppendAssistant`` instance or keyword config that
        is forwarded to the ``AppendAssistant`` constructor.  Passing both an
        instance and config kwargs raises ``TypeError``.  *id* optionally
        names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = AppendAssistant(**config)
        else:
            if config:
                raise TypeError(
                    "append_assistant() accepts either an AppendAssistant "
                    "instance or config kwargs, not both"
                )
            if not isinstance(node, AppendAssistant):
                raise TypeError(
                    "append_assistant() expects an AppendAssistant instance"
                )
        return self.step(node, id=id)

    def supervisor(self, node=None, id=None, **config):
        """Add a :class:`~teff.node.supervisor.Supervisor` decider node.

        Pass a pre-built ``Supervisor`` instance to reuse a shared node, or
        keyword config that is forwarded to the ``Supervisor`` constructor::

            flow.supervisor(
                model="llama3.1:8b",
                provider="ollama",
                sections=AGENT_SECTIONS,
                route_keys={"planner": "plan", "reviewer": "review"},
                done_keys={"plan", "review"},
            ).route("next_agent", ...)

        Passing both an instance and config kwargs raises ``TypeError``.
        *id* optionally names the node in the compiled graph.

        Returns ``self`` for chaining.
        """
        if node is None:
            node = Supervisor(**config)
        else:
            if config:
                raise TypeError(
                    "supervisor() accepts either a Supervisor instance or config kwargs, not both"
                )
            if not isinstance(node, Supervisor):
                raise TypeError("supervisor() expects a Supervisor instance")
        return self.step(node, id=id)
