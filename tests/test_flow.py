import pytest


class TestFlow:
    def test_compile_linear(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Transform

        flow = Flow("test")
        flow.step(
            Transform({"action": "uppercase", "input_key": "text", "output_key": "out"})
        )
        g = flow.compile()
        assert g.entry_point == "transform_1"
        r = asyncio.run(g.run(state={"text": "hello"}))
        assert r["out"] == "HELLO"

    def test_empty_flow_raises(self):
        from teff.flow import Flow

        with pytest.raises(ValueError, match="no nodes"):
            Flow("x").compile()

    def test_branch_routing(self):
        import asyncio

        from teff.flow import Case, Flow
        from teff.node import Node

        class CN(Node):
            type = "cn"

            async def execute(self, ctx, state):
                state["mode"] = "a"
                return state

        class AN(Node):
            type = "an"

            async def execute(self, ctx, state):
                state["result"] = "A"
                return state

        flow = Flow("t").step(CN({})).branch("mode", Case("a").add(AN({})))
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["result"] == "A"

    def test_default_fallback(self):
        import asyncio

        from teff.flow import Case, Flow
        from teff.node import Node

        class CN(Node):
            type = "cn"

            async def execute(self, ctx, state):
                state["mode"] = "unknown"
                return state

        class AN(Node):
            type = "an"

            async def execute(self, ctx, state):
                state["result"] = "A"
                return state

        class FN(Node):
            type = "fn"

            async def execute(self, ctx, state):
                state["result"] = "default"
                return state

        flow = (
            Flow("t").step(CN({})).branch("mode", Case("a").add(AN({}))).default(FN({}))
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["result"] == "default"


class TestStep:
    def test_step_accepts_node_instance(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class MyNode(Node):
            type = "my"

            async def execute(self, ctx, state):
                state["x"] = 42
                return state

        flow = Flow("test").step(MyNode())
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["x"] == 42

    def test_step_rejects_string(self):
        from teff.flow import Flow

        flow = Flow("test")
        with pytest.raises(TypeError, match="must be a Node or function"):
            flow.step("transform")  # type: ignore[arg-type]

    def test_step_rejects_non_node(self):
        from teff.flow import Flow

        flow = Flow("test")
        with pytest.raises(TypeError, match="must be a Node or function"):
            flow.step({"action": "uppercase"})  # type: ignore[arg-type]

    def test_step_with_transform_node(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Transform

        flow = Flow("default").step(
            Transform(action="uppercase", input_key="text", output_key="out")
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "hi"}))
        assert r["out"] == "HI"

    def test_step_chaining(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Transform

        flow = (
            Flow("chain")
            .step(Transform(action="trim", input_key="text", output_key="t"))
            .step(Transform(action="uppercase", input_key="t", output_key="out"))
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "  hi  "}))
        assert r["out"] == "HI"


class TestFlowLLM:
    def test_llm_accepts_config(self):
        from teff.flow import Flow
        from teff.node import LLM

        flow = Flow("t").llm(model="gpt-4", system="s", output_key="answer")
        g = flow.compile()
        assert g.entry_point == "llm_chat_1"
        node = g.nodes["llm_chat_1"]
        assert isinstance(node, LLM)
        assert node.config["model"] == "gpt-4"
        assert node.config["output_key"] == "answer"

    def test_llm_accepts_instance(self):
        from teff.flow import Flow
        from teff.node import LLM

        node = LLM(model="gpt-4", output_key="answer")
        flow = Flow("t").llm(node)
        g = flow.compile()
        assert g.nodes["llm_chat_1"] is node

    def test_llm_rejects_non_llm(self):
        from teff.flow import Flow

        with pytest.raises(TypeError, match="LLM instance"):
            Flow("t").llm("gpt-4")  # type: ignore[arg-type]

    def test_llm_rejects_instance_plus_config(self):
        from teff.flow import Flow
        from teff.node import LLM

        with pytest.raises(TypeError, match="not both"):
            Flow("t").llm(LLM(model="gpt-4"), output_key="x")


class TestFlowTransform:
    def test_transform_accepts_config(self):
        from teff.flow import Flow
        from teff.node import Transform

        flow = Flow("t").transform(action="uppercase", input_key="t", output_key="out")
        g = flow.compile()
        assert g.entry_point == "transform_1"
        assert isinstance(g.nodes["transform_1"], Transform)

    def test_transform_accepts_instance(self):
        from teff.flow import Flow
        from teff.node import Transform

        node = Transform(action="uppercase", input_key="t", output_key="out")
        flow = Flow("t").transform(node)
        g = flow.compile()
        assert g.nodes["transform_1"] is node

    def test_transform_rejects_non_transform(self):
        from teff.flow import Flow

        with pytest.raises(TypeError, match="Transform instance"):
            Flow("t").transform("uppercase")  # type: ignore[arg-type]

    def test_transform_rejects_instance_plus_config(self):
        from teff.flow import Flow
        from teff.node import Transform

        with pytest.raises(TypeError, match="not both"):
            Flow("t").transform(Transform(action="uppercase"), action="lowercase")

    def test_transform_runs(self):
        import asyncio

        from teff.flow import Flow

        flow = (
            Flow("t")
            .transform(action="trim", input_key="text", output_key="t")
            .transform(action="uppercase", input_key="t", output_key="out")
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "  hi  "}))
        assert r["out"] == "HI"


class TestReActAgentOverride:
    def test_react_accepts_agent_instance(self):
        from teff.flow import Flow
        from teff.node.agent import ReActAgent

        agent = ReActAgent(model="gpt-4", system="custom")
        flow = Flow("t").react(agent=agent)
        g = flow.compile()
        assert g.nodes[g.entry_point] is agent

    def test_react_accepts_agent_subclass(self):
        from teff.flow import Flow
        from teff.node.agent import ReActAgent

        class MyAgent(ReActAgent):
            pass

        flow = Flow("t").react(model="gpt-4", agent=MyAgent)
        g = flow.compile()
        node = g.nodes[g.entry_point]
        assert isinstance(node, MyAgent)
        assert node.config["model"] == "gpt-4"
        assert node.config["system"] == ""

    def test_react_agent_instance_ignores_model(self):
        from teff.flow import Flow
        from teff.node.agent import ReActAgent

        agent = ReActAgent(model="llama3.1:8b")
        flow = Flow("t").react(agent=agent, model="gpt-4")  # type: ignore[arg-type]
        g = flow.compile()
        assert g.nodes[g.entry_point] is agent
        assert agent.config["model"] == "llama3.1:8b"

    def test_react_requires_model_without_agent(self):
        from teff.flow import Flow

        with pytest.raises(TypeError, match="requires a model"):
            Flow("t").react()

    def test_react_rejects_wrong_agent_type(self):
        from teff.flow import Flow
        from teff.node import LLM

        with pytest.raises(TypeError, match="ReActAgent"):
            Flow("t").react(agent=LLM(model="gpt-4"))
        with pytest.raises(TypeError, match="ReActAgent"):
            Flow("t").react(agent=dict)  # type: ignore[arg-type]

    def test_react_agent_keeps_tool_exec_cycle(self):
        from teff.flow import Flow
        from teff.node.agent import ReActAgent

        flow = Flow("t").react(model="gpt-4", agent=ReActAgent(model="gpt-4"))
        g = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        assert g.nodes[g.entry_point].type == "react_agent"
        assert ("tool_exec_2", "react_agent_1", None) in edges


class TestSubFlow:
    def test_subflow_basic(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class AddOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                state["val"] = state.get("val", 0) + 1
                return state

        sub = Flow("counter").step(AddOne({}))

        parent = Flow("parent").step(AddOne({}))
        parent.add_flow(sub, max_iterations=5)
        g = parent.compile()

        r = asyncio.run(g.run({"val": 0}))
        assert r["val"] == 2

    def test_subflow_propagates_append_reducers(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class AppendMsg(Node):
            type = "am"

            async def execute(self, ctx, state):
                return {"msgs": [{"role": "user", "content": "hi"}]}

        sub = Flow("inner").step(AppendMsg({})).step(AppendMsg({}))
        parent = Flow("parent").add_flow(sub, output_map={"msgs": "result"})
        g = parent.compile()

        r = asyncio.run(g.run({}, reducers={"msgs": "append"}))
        assert r["result"] == [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "hi"},
        ]

    def test_subflow_appends_into_existing_list_once(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class AppendMsg(Node):
            type = "am2"

            async def execute(self, ctx, state):
                return {"msgs": [{"role": "assistant", "content": "reply"}]}

        sub = Flow("inner").step(AppendMsg({}))
        parent = Flow("parent").add_flow(sub)
        g = parent.compile()

        r = asyncio.run(
            g.run(
                {"msgs": [{"role": "user", "content": "hello"}]},
                reducers={"msgs": "append"},
            )
        )
        assert r["msgs"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_subflow_runs_twice_accumulates_once(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class AppendMsg(Node):
            type = "am3"

            async def execute(self, ctx, state):
                return {"msgs": [{"role": "assistant", "content": "reply"}]}

        sub = Flow("inner").step(AppendMsg({}))
        parent = Flow("parent").add_flow(sub)
        g = parent.compile()

        state = {"msgs": [{"role": "user", "content": "hello"}]}
        r1 = asyncio.run(g.run(dict(state), reducers={"msgs": "append"}))
        assert len(r1["msgs"]) == 2
        r2 = asyncio.run(g.run(dict(r1), reducers={"msgs": "append"}))
        assert [m["content"] for m in r2["msgs"]] == [
            "hello",
            "reply",
            "reply",
        ]

    def test_subflow_with_maps(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Transform

        sub = Flow("inner")
        sub.step(
            Transform({"action": "uppercase", "input_key": "x", "output_key": "y"})
        )

        parent = Flow("outer")
        parent.add_flow(sub, input_map={"text": "x"}, output_map={"y": "result"})
        g = parent.compile()

        r = asyncio.run(g.run({"text": "hello"}))
        assert r["result"] == "HELLO"

    def test_subflow_state_isolation(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class SetFoo(Node):
            type = "sf"

            async def execute(self, ctx, state):
                state["foo"] = "bar"
                return state

        sub = Flow("inner").step(SetFoo({}))
        parent = Flow("outer").step(SetFoo({}))
        parent.add_flow(sub)
        g = parent.compile()

        r = asyncio.run(g.run({}))
        assert r["foo"] == "bar"

    @pytest.mark.asyncio
    async def test_stream_forwards_subflow_events(self):
        from teff.flow import Flow
        from teff.node import Transform

        sub = Flow("inner").transform(action="uppercase", input_key="x", output_key="y")
        parent = Flow("outer")
        parent.step(Transform(action="trim", input_key="x", output_key="x"))
        parent.add_flow(sub)
        g = parent.compile()

        events = [ev async for ev in g.stream({"x": "  hi  "})]
        types = [ev.type for ev in events]

        # a single top-level lifecycle; nested run_start/run_end are stripped
        assert types.count("run_start") == 1
        assert types.count("run_end") == 1
        assert events[-1].data["status"] == "ok"

        # both the outer transform and the inner transform stream node events
        transform_starts = [
            ev
            for ev in events
            if ev.type == "node_start" and ev.node_type == "transform"
        ]
        assert len(transform_starts) == 2

        # the subflow node itself is reported too
        subflow_start = next(
            ev for ev in events if ev.type == "node_start" and ev.node_type == "subflow"
        )
        assert subflow_start.node_id == "subflow_2"


class TestCyclicGraph:
    @pytest.mark.asyncio
    async def test_simple_cycle_terminates_by_condition(self):
        from teff.graph import Edge, Graph
        from teff.node import Node

        class Counter(Node):
            type = "ct"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return state

        class Done(Node):
            type = "dn"

            async def execute(self, ctx, state):
                state["done"] = True
                return state

        # counter -> counter (loop, condition: n<3) -> done (unconditional)
        g = Graph(
            nodes={"ct": Counter({}), "dn": Done({})},
            edges=[
                Edge("ct", "ct", "n=1,2"),
                Edge("ct", "dn"),
            ],
            entry_point="ct",
        )
        r = await g.run(state={}, max_iterations=10)
        assert r["n"] == 3
        assert r["done"] is True

    @pytest.mark.asyncio
    async def test_max_iterations_raises(self):
        from teff.graph import Edge, Graph
        from teff.node import Node

        class InfLoop(Node):
            type = "il"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return state

        # self-loop with unconditional edge -> infinite
        g = Graph(
            nodes={"il": InfLoop({})},
            edges=[Edge("il", "il")],
            entry_point="il",
        )
        with pytest.raises(RuntimeError, match="max_iterations"):
            await g.run(state={}, max_iterations=5)

    @pytest.mark.asyncio
    async def test_max_iterations_linear_completes(self):
        from teff.graph import Edge, Graph
        from teff.node import Node

        class AddOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return state

        g = Graph(
            nodes={"a": AddOne({}), "b": AddOne({})},
            edges=[Edge("a", "b")],
            entry_point="a",
        )
        r = await g.run(state={}, max_iterations=10)
        assert r["n"] == 2


class TestRoute:
    def test_route_basic_loop(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            def __init__(self, values: list[str]):
                super().__init__()
                self.values = list(values)
                self.i = 0

            async def execute(self, ctx, state):
                state["next_agent"] = self.values[self.i % len(self.values)]
                self.i += 1
                return state

        class Mark(Node):
            type = "mark"

            def __init__(self, tag: str):
                super().__init__()
                self.tag = tag

            async def execute(self, ctx, state):
                state["log"] = state.get("log", []) + [self.tag]
                return state

        flow = (
            Flow("sup")
            .step(Decider(["planner", "finish"]))
            .route("next_agent", finish=Mark("final"), planner=Mark("planned"))
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={}, max_iterations=20))
        assert r["log"] == ["planned", "final"]

    def test_route_multiple_agents(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            def __init__(self, values: list[str]):
                super().__init__()
                self.values = list(values)
                self.i = 0

            async def execute(self, ctx, state):
                state["next_agent"] = self.values[self.i % len(self.values)]
                self.i += 1
                return state

        class Mark(Node):
            type = "mark"

            def __init__(self, tag: str):
                super().__init__()
                self.tag = tag

            async def execute(self, ctx, state):
                state["log"] = state.get("log", []) + [self.tag]
                return state

        flow = (
            Flow("sup")
            .step(Decider(["planner", "estimator", "finish"]))
            .route(
                "next_agent",
                finish=Mark("final"),
                planner=Mark("planned"),
                estimator=Mark("estimated"),
            )
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={}, max_iterations=20))
        assert r["log"] == ["planned", "estimated", "final"]

    def test_route_wiring(self):
        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            async def execute(self, ctx, state):
                return state

        class Mark(Node):
            type = "mark"

            async def execute(self, ctx, state):
                return state

        flow = (
            Flow("sup")
            .step(Decider({}))
            .route(
                "next_agent",
                finish=Mark({}),
                planner=Mark({}),
                qa=Mark({}),
            )
        )
        g = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        assert ("decider_1", "mark_2", "next_agent=finish") in edges
        assert ("decider_1", "mark_3", "next_agent=planner") in edges
        assert ("decider_1", "mark_4", "next_agent=qa") in edges
        assert ("mark_3", "decider_1", None) in edges
        assert ("mark_4", "decider_1", None) in edges
        assert ("mark_2", "decider_1", None) not in edges

    def test_route_loops_back_to_decider(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            def __init__(self, values: list[str]):
                super().__init__()
                self.values = list(values)
                self.i = 0

            async def execute(self, ctx, state):
                state["next_agent"] = self.values[self.i % len(self.values)]
                self.i += 1
                state["visits"] = state.get("visits", 0) + 1
                return state

        class Mark(Node):
            type = "mark"

            async def execute(self, ctx, state):
                state["log"] = state.get("log", []) + ["agent"]
                return state

        # decider visited twice -> supervisor loop ran once before finishing
        flow = (
            Flow("sup")
            .step(Decider(["planner", "finish"]))
            .route("next_agent", finish=Mark({}), planner=Mark({}))
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={}, max_iterations=20))
        assert r["visits"] == 2
        assert r["log"] == ["agent", "agent"]

    def test_route_finish_none_terminates(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            async def execute(self, ctx, state):
                state["next_agent"] = "finish"
                return state

        class Mark(Node):
            type = "mark"

            async def execute(self, ctx, state):
                state["log"] = state.get("log", []) + ["agent"]
                return state

        flow = Flow("sup").step(Decider({})).route("next_agent", planner=Mark({}))
        g = flow.compile()
        r = asyncio.run(g.run(state={}, max_iterations=20))
        assert "log" not in r

    def test_route_requires_decider(self):
        from teff.flow import Flow
        from teff.node import Node

        class Mark(Node):
            type = "mark"

            async def execute(self, ctx, state):
                return state

        with pytest.raises(ValueError, match="preceding"):
            Flow("x").route("next_agent", planner=Mark({}))

    def test_route_requires_agents(self):
        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            async def execute(self, ctx, state):
                return state

        with pytest.raises(ValueError, match="at least one agent"):
            Flow("x").step(Decider({})).route("next_agent")

    def test_route_finish_chain_continues(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            def __init__(self, values: list[str]):
                super().__init__()
                self.values = list(values)
                self.i = 0

            async def execute(self, ctx, state):
                state["next_agent"] = self.values[self.i % len(self.values)]
                self.i += 1
                return state

        class Mark(Node):
            type = "mark"

            def __init__(self, tag: str):
                super().__init__()
                self.tag = tag

            async def execute(self, ctx, state):
                state["log"] = state.get("log", []) + [self.tag]
                return state

        class After(Node):
            type = "after"

            async def execute(self, ctx, state):
                state["after"] = True
                return state

        flow = (
            Flow("t")
            .step(Decider(["planner", "finish"]))
            .route("next_agent", finish=Mark("final"), planner=Mark("planned"))
            .step(After({}))
        )
        g = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        r = asyncio.run(g.run(state={}, max_iterations=20))
        assert r["log"] == ["planned", "final"]
        assert r["after"] is True
        assert ("mark_2", "after_4", None) in edges

    def test_route_finish_none_blocks_chaining(self):
        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            async def execute(self, ctx, state):
                return state

        class Mark(Node):
            type = "mark"

            async def execute(self, ctx, state):
                return state

        flow = Flow("t").step(Decider({})).route("next_agent", planner=Mark({}))
        with pytest.raises(ValueError, match="finish=None"):
            flow.step(Mark({}))

    def test_route_accepts_chains(self):
        import asyncio

        from teff.flow import Flow
        from teff.node import Node

        class Decider(Node):
            type = "decider"

            def __init__(self, values: list[str]):
                super().__init__()
                self.values = list(values)
                self.i = 0

            async def execute(self, ctx, state):
                state["next_agent"] = self.values[self.i % len(self.values)]
                self.i += 1
                return state

        class Mark(Node):
            type = "mark"

            def __init__(self, tag: str):
                super().__init__()
                self.tag = tag

            async def execute(self, ctx, state):
                state["log"] = state.get("log", []) + [self.tag]
                return state

        flow = (
            Flow("sup")
            .step(Decider(["planner", "finish"]))
            .route(
                "next_agent",
                finish=[Mark("final-1"), Mark("final-2")],
                planner=[Mark("planned-1"), Mark("planned-2")],
            )
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={}, max_iterations=20))
        assert r["log"] == ["planned-1", "planned-2", "final-1", "final-2"]


class TestFlowToYaml:
    def test_react_flow_exports_and_round_trips(self, tmp_path):
        from teff.flow import Flow
        from teff.node import Transform
        from teff.yaml import load_workflow
        from teff.yaml_schema import validate_workflow_file

        flow = Flow("demo")
        flow.step(Transform(action="uppercase", input_key="t", output_key="u"))
        flow.react(model="gpt-4", use_tools="all", input_key="u", output_key="answer")
        text = flow.to_yaml()

        path = tmp_path / "wf.yaml"
        path.write_text(text, encoding="utf-8")
        assert validate_workflow_file(str(path)) == []

        graph, _, _, _ = load_workflow(str(path))
        types = {type(n).type for n in graph.nodes.values()}
        assert {"transform", "react_agent", "tool_exec"} <= types
        conditions = [e.condition for e in graph.edges if e.condition]
        assert any("_tool_call_name!=" in c for c in conditions)
        assert graph.entry_point.startswith("transform_")

    def test_workflow_to_yaml_includes_tools_and_state(self, tmp_path):
        from teff.flow import Flow
        from teff.node import Transform
        from teff.state.state import reducers_from_yaml_schema
        from teff.tool.registry import default_tool_registry
        from teff.yaml import load_workflow, workflow_to_yaml
        from teff.yaml_schema import validate_workflow_file

        flow = Flow("wf")
        flow.step(Transform(action="uppercase", input_key="t", output_key="u"))
        tool = default_tool_registry.create("calculator", {})

        text = workflow_to_yaml(
            flow.compile(),
            tools=[tool],
            initial={"status": "active"},
            reducers={"messages": "append"},
        )
        path = tmp_path / "wf.yaml"
        path.write_text(text, encoding="utf-8")
        assert validate_workflow_file(str(path)) == []

        graph, tools, initial, reducers = load_workflow(str(path))
        assert [t.name for t in tools] == ["calculator"]
        assert initial == {"status": "active"}
        assert reducers == {"messages": "append"}
        assert reducers_from_yaml_schema({"messages": {"reducer": "append"}}) == {
            "messages": "append"
        }

    def test_graph_to_yaml_back_compat(self):
        from teff.flow import Flow
        from teff.node import Transform
        from teff.yaml import graph_to_yaml

        flow = Flow()
        flow.step(Transform(action="trim", input_key="x", output_key="y"))
        text = graph_to_yaml(flow.compile())
        assert "steps:" in text
        assert "edges:" in text
        assert "transform" in text


class TestNodeIds:
    """User-supplied node ids (``id=``) across the Flow helpers."""

    def test_step_llm_transform_ids(self):
        from teff.flow import Flow

        flow = Flow()
        flow.llm(model="m", system="s", output_key="sent", id="classify")
        flow.transform(
            action="uppercase", input_key="sent", output_key="out", id="shout"
        )
        g = flow.compile()
        assert g.entry_point == "classify"
        assert list(g.nodes) == ["classify", "shout"]

    def test_branch_case_converge_ids(self):
        from teff.flow import Case, Flow
        from teff.node import Transform

        flow = Flow()
        flow.step(Transform(action="value", value="0", output_key="k"), id="start")
        flow.branch(
            "k",
            Case("a").add(
                Transform(action="value", value="A", output_key="r"), id="on_a"
            ),
            Case("b").add(
                Transform(action="value", value="B", output_key="r"), id="on_b"
            ),
        ).converge(
            Transform(action="uppercase", input_key="r", output_key="out"),
            id="merge",
        )
        g = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        assert ("start", "on_a", "k=a") in edges
        assert ("start", "on_b", "k=b") in edges
        assert ("on_a", "merge", None) in edges
        assert ("on_b", "merge", None) in edges
        assert list(g.nodes) == ["start", "on_a", "on_b", "merge"]

    def test_harness_id_prefixes_agent_and_tool(self):
        from teff.flow import Flow

        flow = Flow()
        flow.harness(model="m", output_key="answer", id="assistant")
        g = flow.compile()
        assert set(g.nodes) == {"assistant/agent", "assistant/tool"}
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        assert ("assistant/agent", "assistant/tool", "_tool_call_name!=") in edges

    def test_harness_uses_custom_tool_call_key_in_edge(self):
        from teff.flow import Flow

        flow = Flow()
        flow.harness(
            model="m", output_key="answer", config={"tool_call_key": "_my_call"}
        )
        g = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        assert any(cond == "_my_call!=" for _, _, cond in edges)

    def test_route_subflow_named_by_prefix(self):
        from teff.flow import Flow, agent_step
        from teff.node import LLM, Node

        class Decider(Node):
            type = "decider"

            async def execute(self, ctx, state):
                return {"next_agent": "planner"}

        flow = Flow()
        flow.step(Decider(), id="supervisor")
        flow.route(
            "next_agent",
            finish=LLM(model="m", output_key="done"),
            planner=agent_step("p", "plan", model="m", provider="openai", id="planner"),
        )
        g = flow.compile()
        assert "planner" in g.nodes
        sub = g.nodes["planner"]
        assert sub.type == "subflow"
        inner_ids = list(sub._graph.nodes)
        assert all(nid.startswith("planner/") for nid in inner_ids)
        assert sub._graph.entry_point == "planner/context_builder_1"

    def test_context_builder_and_append_assistant_helpers(self):
        from teff.flow import Flow

        flow = Flow()
        flow.context_builder(
            sections={"plan": "Plan", "summary": "Summary"},
            messages_key="messages",
            output_key="input",
            id="compose",
        )
        flow.append_assistant(output_key="draft", messages_key="messages", id="append")
        g = flow.compile()
        assert g.entry_point == "compose"
        assert list(g.nodes) == ["compose", "append"]
        assert g.nodes["compose"].type == "context_builder"
        assert g.nodes["append"].type == "append_assistant"
        assert g.nodes["compose"].config["sections"] == {
            "plan": "Plan",
            "summary": "Summary",
        }
        assert g.nodes["append"].config["output_key"] == "draft"

    def test_context_builder_helper_rejects_both(self):
        from teff.flow import Flow
        from teff.node.context import ContextBuilder

        flow = Flow()
        with pytest.raises(TypeError):
            flow.context_builder(
                ContextBuilder(sections={"k": "K"}), sections={"k2": "K2"}
            )

    def test_append_assistant_helper_rejects_wrong_type(self):
        from teff.flow import Flow
        from teff.node.context import ContextBuilder

        flow = Flow()
        with pytest.raises(TypeError):
            flow.append_assistant(ContextBuilder())

    def test_duplicate_id_raises(self):
        from teff.flow import Flow
        from teff.node import Transform

        flow = Flow()
        flow.step(Transform(action="value", value="1", output_key="a"), id="dup")
        with pytest.raises(ValueError, match="duplicate node id"):
            flow.step(Transform(action="value", value="2", output_key="b"), id="dup")

    def test_auto_ids_still_work_without_id(self):
        from teff.flow import Flow
        from teff.node import Transform

        flow = Flow()
        flow.step(Transform(action="value", value="1", output_key="a"))
        flow.step(Transform(action="value", value="2", output_key="b"))
        g = flow.compile()
        assert g.entry_point == "transform_1"
        assert list(g.nodes) == ["transform_1", "transform_2"]

    def test_map_and_parallel_accept_id(self):
        from teff.flow import Flow
        from teff.node import Transform

        flow = Flow()
        flow.step(Transform(action="value", value="x", output_key="items"), id="seed")
        flow.map(
            Transform(action="uppercase", input_key="item", output_key="o"),
            input_keys=["items"],
            output_key="out",
            id="fan",
        )
        flow.parallel(
            Transform(action="value", value="p", output_key="p1"),
            Transform(action="value", value="q", output_key="p2"),
            id="branch",
        )
        g = flow.compile()
        assert "seed" in g.nodes
        assert "fan" in g.nodes
        assert "branch" in g.nodes
