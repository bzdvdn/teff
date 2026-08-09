"""Tests for the flow.yaml authoring layer (two-layer compile).

Covers the compiler in :mod:`teff.flow.compiler` — the high-level idiom
surface that compiles down into an ordinary ``Graph`` / ``graph.yaml``.
"""

from __future__ import annotations

import asyncio
import textwrap

import pytest

from teff.errors import ConfigError


def compile_doc(yaml_text: str, name: str = "t"):
    """Parse inline YAML and compile it into a plain Graph (importing yaml)."""
    import yaml

    from teff.flow.compiler import flow_from_yaml

    data = yaml.safe_load(yaml_text)
    return flow_from_yaml(data).compile()


class TestLinearSteps:
    def test_llm_transform_chain(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: linear
                default_model: llama3.1:8b
                default_provider: ollama
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                steps:
                  - llm: {id: replier, model: llama3.1:8b, provider: ollama,
                          system: "You are helpful", output_key: answer}
                  - transform: {id: shout, action: uppercase, input_key: answer,
                                output_key: shout}
                """
            )
        )
        assert list(g.nodes) == ["replier", "shout"]
        assert g.entry_point == "replier"
        assert g.edges[0].source_id == "replier"
        assert g.edges[0].target_id == "shout"
        assert g.nodes["replier"].type == "llm_chat"
        assert g.nodes["shout"].type == "transform"

    def test_agent_step_top_level(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: agent
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - agent: {id: coder, system: "You code.", output_key: code}
                """
            )
        )
        (nid, node) = next(iter(g.nodes.items()))
        assert node.type == "subflow"
        assert nid.startswith("agent-coder")

    def test_context_builder_idiom(self):
        g = compile_doc(
            textwrap.dedent(
                """
                steps:
                  - context_builder:
                      id: compose
                      sections: {plan: "Plan", summary: "Summary"}
                      messages_key: messages
                      output_key: input
                      reset_keys: [scratch]
                """
            )
        )
        assert g.entry_point == "compose"
        node = g.nodes["compose"]
        assert node.type == "context_builder"
        assert node.config["sections"] == {"plan": "Plan", "summary": "Summary"}
        assert node.config["messages_key"] == "messages"
        assert node.config["output_key"] == "input"
        assert node.config["reset_keys"] == ["scratch"]

    def test_append_assistant_idiom(self):
        g = compile_doc(
            textwrap.dedent(
                """
                steps:
                  - append_assistant: {output_key: draft, messages_key: messages, id: show}
                """
            )
        )
        assert g.entry_point == "show"
        node = g.nodes["show"]
        assert node.type == "append_assistant"
        assert node.config["output_key"] == "draft"
        assert node.config["messages_key"] == "messages"

    def test_context_append_chain(self):
        g = compile_doc(
            textwrap.dedent(
                """
                steps:
                  - context_builder:
                      id: compose
                      sections: {plan: "Plan"}
                      messages_key: messages
                      output_key: input
                  - append_assistant: {output_key: draft, messages_key: messages}
                """
            )
        )
        assert g.entry_point == "compose"
        assert list(g.nodes) == ["compose", "append_assistant_2"]
        edges = {(e.source_id, e.target_id) for e in g.edges}
        assert ("compose", "append_assistant_2") in edges

    def test_unknown_step_raises(self):
        import yaml

        from teff.flow.compiler import flow_from_yaml

        with pytest.raises(ConfigError, match="unknown flow step"):
            flow_from_yaml(yaml.safe_load("steps:\n  - sleeper: {}"))


class TestTeam:
    def test_team_compiles_supervisor_route(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: router
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - team:
                      id: team
                      leader:
                        system: "You are a team lead. Route to coder or finish."
                        route_keys: {coder: code, talk: talk}
                        done_keys: [code, talk]
                        done_mode: any
                        fallback: talk
                      roles:
                        coder: {system: "You write code.", output_key: code}
                        talk: {system: "You chat.", output_key: talk}
                """
            )
        )
        first = g.entry_point
        assert g.nodes[first].type == "supervisor"
        agent_ids = [nid for nid, n in g.nodes.items() if n.type == "subflow"]
        assert len(agent_ids) == 2
        conds = {e.condition for e in g.edges if e.source_id == first}
        assert conds >= {"next_agent=coder", "next_agent=talk"}

    def test_team_requires_roles(self):
        with pytest.raises(ConfigError, match="roles"):
            compile_doc(
                textwrap.dedent(
                    """
                    name: bad
                    default_model: llama3.1:8b
                    default_provider: ollama
                    steps:
                      - team:
                          id: team
                          leader: {system: "lead"}
                    """
                )
            )


class TestFlowTeamPythonApi:
    def _flow(self):
        from teff.flow import Flow
        from teff.provider import ProviderRegistry

        return Flow(
            "team",
            providers=ProviderRegistry.from_presets("ollama"),
            default_provider="ollama",
            default_model="llama3.1:8b",
        )

    def test_team_accepts_agent_role(self):
        from teff.flow import AgentRole

        flow = self._flow()
        flow.team(
            "Route to coder or talk, then finish.",
            roles={
                "coder": AgentRole("You write code.", output_key="code"),
                "talk": AgentRole("You chat.", output_key="talk"),
            },
            fallback="talk",
        )
        g = flow.compile()
        first = g.entry_point
        assert g.nodes[first].type == "supervisor"
        subflows = {nid for nid, n in g.nodes.items() if n.type == "subflow"}
        assert {"coder", "talk"} <= subflows
        conds = {e.condition for e in g.edges if e.source_id == first}
        assert conds >= {"next_agent=coder", "next_agent=talk"}

    def test_team_agent_role_use_tools(self):
        from teff.flow import AgentRole

        flow = self._flow()
        flow.team(
            "Route or finish.",
            roles={
                "coder": AgentRole("You code.", output_key="code", use_tools=["shell"]),
            },
        )
        g = flow.compile()
        coder = next(n for n in g.nodes.values() if n.type == "subflow")
        inner = coder._graph
        harness = next(n for n in inner.nodes.values() if n.type == "react_agent")
        assert harness.config.get("use_tools") == ["shell"]

    def test_team_dict_recipe_parity(self):
        flow = self._flow()
        flow.team(
            "Route or finish.",
            roles={
                "coder": {"system": "You code.", "output_key": "code"},
            },
        )
        g = flow.compile()
        assert g.nodes[g.entry_point].type == "supervisor"
        assert any(n.type == "subflow" for n in g.nodes.values())

    def test_team_requires_model_and_provider(self):
        from teff.flow import AgentRole, Flow

        flow = Flow("bad")
        with pytest.raises(ValueError, match="model"):
            flow.team("x", roles={"coder": AgentRole("c", output_key="code")})


class TestSupervisorSupervise:
    def test_supervisor_idiom_compiles_decider(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: native
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - supervisor:
                      id: lead
                      system: "Route to coder or finish."
                      route_keys: {coder: code}
                      done_keys: [code]
                      fallback: coder
                      max_rounds: 5
                """
            )
        )
        first = g.entry_point
        assert first == "lead"
        assert g.nodes[first].type == "supervisor"
        assert g.nodes[first].config["route_keys"] == {"coder": "code"}
        assert g.nodes[first].config["done_keys"] == {"code"}
        assert g.nodes[first].config["fallback_agent"] == "coder"
        assert g.nodes[first].config["max_rounds"] == 5

    def test_supervisor_embedded_agents_wire_loop(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: native
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - supervisor:
                      id: lead
                      system: "Route to coder or finish."
                      route_keys: {coder: code}
                      done_keys: [code]
                      fallback: coder
                      agents:
                        coder: [agent_step: {id: coder, system: "You code.",
                                              output_key: code}]
                      finish:
                        - transform: {action: now, output_key: delivered_at}
                """
            )
        )
        lead = g.entry_point
        assert lead == "lead"
        assert g.nodes[lead].type == "supervisor"
        conds = {e.condition for e in g.edges if e.source_id == lead}
        assert conds == {"next_agent=coder", "next_agent=finish"}
        assert any(e.source_id == "coder" and e.target_id == lead for e in g.edges)
        assert any(n.type == "transform" for n in g.nodes.values())

    def test_supervisor_embedded_agents_require_mapping(self):
        with pytest.raises(ConfigError, match="agents"):
            compile_doc(
                textwrap.dedent(
                    """
                    name: bad
                    default_model: llama3.1:8b
                    default_provider: ollama
                    steps:
                      - supervisor:
                          id: lead
                          system: "Route."
                          agents: []
                    """
                )
            )

    def test_supervise_routes_existing_decider(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: native
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - supervisor:
                      id: lead
                      system: "Route to coder or finish."
                      route_keys: {coder: code}
                      done_keys: [code]
                      fallback: coder
                  - supervise:
                      key: next_agent
                      agents:
                        coder: [agent_step: {id: coder, system: "You code.",
                                              output_key: code}]
                      finish:
                        - transform: {action: now, output_key: delivered_at}
                """
            )
        )
        lead = g.entry_point
        assert lead == "lead"
        conds = {e.condition for e in g.edges if e.source_id == lead}
        assert conds == {"next_agent=coder", "next_agent=finish"}
        assert any(e.source_id == "coder" and e.target_id == lead for e in g.edges)
        assert any(n.type == "transform" for n in g.nodes.values())

    def test_supervise_requires_key(self):
        with pytest.raises(ConfigError, match="key"):
            compile_doc(
                textwrap.dedent(
                    """
                    name: bad
                    default_model: llama3.1:8b
                    default_provider: ollama
                    steps:
                      - supervise:
                          agents:
                            coder: [transform: {action: uppercase}]
                    """
                )
            )

    def test_supervise_requires_agents(self):
        with pytest.raises(ConfigError, match="agents"):
            compile_doc(
                textwrap.dedent(
                    """
                    name: bad
                    default_model: llama3.1:8b
                    default_provider: ollama
                    steps:
                      - supervise: {key: next_agent}
                    """
                )
            )


class TestMapLoopInterrupt:
    def test_map_compiles_processor(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: map
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - map:
                      input_keys: [items]
                      output_key: summaries
                      max_concurrency: 2
                      processor:
                        llm: {model: llama3.1:8b, provider: ollama,
                              prompt: "Summarize {item}", output_key: summary}
                """
            )
        )
        (node,) = [n for n in g.nodes.values() if n.type == "map"]
        cfg = node.config
        assert cfg["input_keys"] == ["items"]
        assert cfg["output_key"] == "summaries"
        assert cfg["processor"]["type"] == "llm_chat"

    def test_loop_requires_done_and_body(self):
        with pytest.raises(ConfigError, match="done"):
            compile_doc(
                textwrap.dedent(
                    """
                name: l
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - transform: {action: value, value: x, input_key: .}
                  - loop:
                      key: verdict
                      until: pass
                      body:
                        - llm: {}
                """
                )
            )

    def test_interrupt_with_strategy_expands(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: gate
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - interrupt:
                      id: approve
                      key: approved
                      prompt: "Go?"
                      strategy:
                        any_of: [yes, ok]
                        decision_key: decision
                        pass_value: approve
                        fail_value: rework
                """
            )
        )
        types = {n.type for n in g.nodes.values()}
        assert "interrupt" in types
        assert "validate" in types
        # the decision key routes onward
        assert g.nodes[g.entry_point].type == "interrupt"


class TestRoundTrip:
    def test_compiled_graph_roundtrips_through_yaml(self, tmp_path):
        from teff.flow.compiler import load_flow
        from teff.yaml import workflow_to_yaml

        src = tmp_path / "flow.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: rt
                default_model: llama3.1:8b
                default_provider: ollama
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                state:
                  schema:
                    messages: {reducer: append, type: list}
                  initial: {messages: [], note: hello}
                steps:
                  - transform: {id: upper, action: uppercase, input_key: text,
                                output_key: out}
                """
            )
        )
        graph, tools, initial, reducers = load_flow(str(src))
        text = workflow_to_yaml(
            graph, name="rt", tools=tools, initial=initial, reducers=reducers
        )
        assert "type: transform" in text

        out = tmp_path / "graph.yaml"
        out.write_text(text)
        from teff.yaml import load_workflow as load_graph

        g2, tools2, initial2, reducers2 = load_graph(str(out))
        assert list(g2.nodes) == list(graph.nodes)
        assert initial2 == initial

    def test_roundtrip_with_team_and_map(self, tmp_path):
        from teff.flow.compiler import load_flow
        from teff.yaml import workflow_to_yaml

        src = tmp_path / "f.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: mixed
                default_provider: ollama
                default_model: llama3.1:8b
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                steps:
                  - map:
                      input_keys: [items]
                      output_key: outs
                      processor:
                        llm: {model: llama3.1:8b, provider: ollama,
                              prompt: "go", output_key: s}
                  - team:
                      id: team
                      leader: {system: "Route: coder or finish."}
                      roles:
                        coder: {system: "code", output_key: code}
                """
            )
        )
        g1, _, _, _ = load_flow(str(src))
        text = workflow_to_yaml(g1, name="mixed")
        out = tmp_path / "g.yaml"
        out.write_text(text)

        from teff.yaml import load_workflow as load_graph

        g2, _, _, _ = load_graph(str(out))
        assert set(g2.nodes) == set(g1.nodes)

    def test_roundtrip_bounded_loop_executes_after_load(self, tmp_path):
        from teff.flow.compiler import load_flow
        from teff.yaml import load_workflow as load_graph
        from teff.yaml import workflow_to_yaml

        src = tmp_path / "bl.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: bounded
                default_model: llama3.1:8b
                default_provider: ollama
                providers:
                  - name: ollama
                    type: ollama
                    base_url: http://localhost:11434
                    chat_path: /api/chat
                steps:
                  - transform: {action: value, value: retry, output_key: verdict}
                  - loop:
                      key: verdict
                      until: pass
                      max_rounds: 2
                      body:
                        - transform: {action: value, value: retry, output_key: verdict}
                      done:
                        - transform: {id: finish, action: value, value: final,
                                      output_key: verdict}
                """
            )
        )
        g1, _, _, _ = load_flow(str(src))
        text = workflow_to_yaml(g1, name="bounded")
        out = tmp_path / "b.yaml"
        out.write_text(text)
        g2, _, _, _ = load_graph(str(out))
        assert set(g2.nodes) == set(g1.nodes)
        r = asyncio.run(g2.run({}))
        assert r["verdict"] == "final"


class TestExecutes:
    def test_linear_flow_runs_offline(self):
        import yaml

        from teff.flow.compiler import flow_from_yaml

        flow = flow_from_yaml(
            yaml.safe_load(
                textwrap.dedent(
                    """
                    name: offline
                    steps:
                      - transform: {action: uppercase, input_key: text, output_key: out}
                      - transform: {action: value, value: done, output_key: status}
                    """
                )
            )
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "hi"}))
        assert (r["out"], r["status"]) == ("HI", "done")


class TestComplexIdioms:
    def test_loop_with_done_and_body(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: lp
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - transform: {action: value, value: retry, output_key: verdict}
                  - loop:
                      key: verdict
                      until: pass
                      body:
                        - llm: {model: llama3.1:8b, provider: ollama,
                                prompt: "retry", output_key: verdict}
                      done:
                        - transform: {action: value, value: final, output_key: verdict}
                """
            )
        )
        conds = {e.condition for e in g.edges if e.condition}
        assert "verdict=pass" in conds
        assert "verdict!=pass" in conds

    def test_loop_with_max_rounds_compiles_to_bounded_node(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: lpm
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - transform: {action: value, value: retry, output_key: verdict}
                  - loop:
                      key: verdict
                      until: pass
                      max_rounds: 3
                      body:
                        - transform: {action: value, value: retry, output_key: verdict}
                      done:
                        - transform: {action: value, value: final, output_key: verdict}
                """
            )
        )
        loop = [n for n in g.nodes.values() if n.type == "loop"]
        assert len(loop) == 1
        assert loop[0].config["max_rounds"] == 3
        assert not any(e.condition for e in g.edges)

    def test_loop_invalid_max_rounds_raises(self):
        with pytest.raises(ConfigError, match="max_rounds"):
            compile_doc(
                textwrap.dedent(
                    """
                name: lm
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - transform: {action: value, value: retry, output_key: verdict}
                  - loop:
                      key: verdict
                      until: pass
                      max_rounds: 0
                      body:
                        - transform: {action: value, value: retry, output_key: verdict}
                      done:
                        - transform: {action: value, value: final, output_key: verdict}
                """
                )
            )

    def test_route_command_node(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: r
                steps:
                  - transform: {action: value, value: x, input_key: .}
                  - route:
                      when: status == "done"
                      goto: next
                  - transform: {id: next, action: uppercase, input_key: text,
                                output_key: out}
                """
            )
        )
        assert g.nodes[g.entry_point].type == "transform"

    def test_interrupt_without_strategy(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: gate
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - interrupt: {id: ask, key: approved, prompt: "Go?"}
                """
            )
        )
        assert g.nodes[g.entry_point].type == "interrupt"

    def test_loop_missing_body_raises(self):
        with pytest.raises(ConfigError, match="body"):
            compile_doc(
                textwrap.dedent(
                    """
                name: l
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - loop: {key: k, until: done, done: [{transform: {action: x}}]}
                """
                )
            )

    def test_branch_compiles_conditional_edges_and_converge(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: br
                steps:
                  - transform: {action: value, value: ok, output_key: status}
                  - branch:
                      key: status
                      cases:
                        - value: ok
                          steps:
                            - transform: {action: value, value: yes,
                                          output_key: reply}
                        - value: "no"
                          steps:
                            - transform: {action: value, value: nope,
                                          output_key: reply}
                      converge:
                        transform: {action: uppercase, input_key: reply,
                                    output_key: result}
                """
            )
        )
        conds = {e.condition for e in g.edges if e.condition}
        assert "status=ok" in conds
        assert "status=no" in conds
        converge = [e for e in g.edges if e.target_id != g.entry_point]
        assert any(e.condition is None for e in converge)

    def test_branch_default_edge(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: brd
                steps:
                  - transform: {action: value, value: "2", output_key: lines}
                  - branch:
                      key: lines
                      cases:
                        - value: "1"
                          steps:
                            - transform: {action: value, value: single,
                                          output_key: note}
                      default:
                        - transform: {action: value, value: multi,
                                      output_key: note}
                """
            )
        )
        conds = {e.condition for e in g.edges if e.condition}
        assert "lines=1" in conds
        assert "lines!=1" in conds

    def test_branch_missing_key_raises(self):
        with pytest.raises(ConfigError, match="branch requires a `key`"):
            compile_doc(
                textwrap.dedent(
                    """
                name: brk
                steps:
                  - transform: {action: value, value: x, output_key: k}
                  - branch: {cases: [{value: x, steps: [{transform: {action: value}}]}]}
                """
                )
            )

    def test_branch_empty_cases_raises(self):
        with pytest.raises(ConfigError, match="cases"):
            compile_doc(
                textwrap.dedent(
                    """
                name: brc
                steps:
                  - transform: {action: value, value: x, output_key: k}
                  - branch: {key: k, cases: []}
                """
                )
            )

    def test_parallel_missing_branches_raises(self):
        with pytest.raises(ConfigError, match="branches"):
            compile_doc(
                textwrap.dedent(
                    """
                name: p
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - parallel: {id: p, converge: {transform: {action: value,
                                                             value: 1}}}
                """
                )
            )

    def test_team_with_fallback_and_max_rounds(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: t
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - team:
                      id: t
                      leader: {system: "lead", fallback: chat, max_rounds: 3}
                      roles:
                        coder: {system: "code", output_key: code}
                        chat: {system: "chat", output_key: chat}
                """
            )
        )
        first = g.entry_point
        sup = g.nodes[first]
        assert sup.type == "supervisor"
        assert sup.config.get("max_rounds") == 3
        assert sup.config.get("fallback_agent") == "chat"

    def test_llm_requires_model_and_provider(self):
        with pytest.raises(ConfigError, match="model"):
            compile_doc(
                textwrap.dedent(
                    """
                    name: no
                    steps:
                      - llm: {prompt: "hi", output_key: a}
                    """
                )
            )

    def test_llm_inherits_defaults(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: inh
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - llm: {prompt: "hi", output_key: a}
                """
            )
        )
        (llm,) = [n for n in g.nodes.values() if n.type == "llm_chat"]
        assert llm.config["model"] == "llama3.1:8b"
        assert llm.config["provider"] == "ollama"

    def test_type_idiom_nested_processor(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: ty
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - map:
                      input_keys: [items]
                      output_key: outs
                      processor:
                        type: {type: transform, config: {action: uppercase,
                                                         input_key: item,
                                                         output_key: s}}
                """
            )
        )
        (m,) = [n for n in g.nodes.values() if n.type == "map"]
        assert m.config["processor"]["type"] == "transform"

    def test_parallel_with_converge(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: par
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - parallel:
                      branches:
                        - transform: {action: value, value: a, output_key: oa}
                        - transform: {action: value, value: b, output_key: ob}
                      converge: {transform: {action: concat, input_keys: [oa, ob],
                                            output_key: joined}}
                """
            )
        )
        assert any(n.type == "parallel" for n in g.nodes.values())
        para_id = next(nid for nid, n in g.nodes.items() if n.type == "parallel")
        assert any(e.source_id == para_id for e in g.edges)

    def test_parallel_converge_rejects_non_transform(self):
        with pytest.raises(ConfigError, match="Transform"):
            compile_doc(
                textwrap.dedent(
                    """
                name: par
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - parallel:
                      branches:
                        - transform: {action: value, value: a, output_key: oa}
                        - transform: {action: value, value: b, output_key: ob}
                      converge: {llm: {model: llama3.1:8b, provider: ollama}}
                """
                )
            )

    def test_agent_step_with_id(self):
        g = compile_doc(
            textwrap.dedent(
                """
                name: ag
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - agent: {id: coder, system: "code", output_key: code}
                """
            )
        )
        assert any(n.type == "subflow" for n in g.nodes.values())

    def test_unknown_team_route_key_is_fallback(self):
        compile_doc(
            textwrap.dedent(
                """
                name: u
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - team:
                      id: t
                      leader: {system: "lead", route_keys: {coder: code,
                                                            talk: t},
                               done_keys: [code]}
                      roles:
                        coder: {system: "code", output_key: code}
                        talk: {system: "chat"}
                """
            )
        )


class TestTwoLayerValidation:
    def test_validate_flow_ok(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: v
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - transform: {id: upper, action: uppercase, input_key: a,
                                output_key: b}
                  - loop:
                      key: v
                      until: pass
                      body:
                        - llm: {model: llama3.1:8b, provider: ollama,
                                prompt: "x", output_key: v}
                      done:
                        - interrupt: {key: ok, prompt: "Go?"}
                """
            )
        )
        assert not validate_flow_file(str(p))

    def test_validate_flow_context_append_ok(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: v
                steps:
                  - context_builder:
                      id: compose
                      sections: {plan: "Plan"}
                      output_key: input
                  - append_assistant: {output_key: draft}
                """
            )
        )
        assert not validate_flow_file(str(p))

    def test_validate_flow_unknown_idiom(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text("name: b\nsteps:\n  - fax: {}\n")
        errors = validate_flow_file(str(p))
        assert any("fax" in e["message"] for e in errors)

    def test_validate_flow_team_requires_roles(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text("name: b\nsteps:\n  - team: {leader: {system: hi}}\n")
        errors = validate_flow_file(str(p))
        assert any("roles" in e["message"] for e in errors)

    def test_validate_flow_supervisor_and_supervise_ok(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: v
                default_model: llama3.1:8b
                default_provider: ollama
                steps:
                  - supervisor:
                      id: lead
                      system: "Route or finish."
                      route_keys: {coder: code}
                      fallback: coder
                      agents:
                        coder: [agent_step: {system: "code",
                                              output_key: code}]
                      finish:
                        - transform: {action: now, output_key: delivered_at}
                  - supervise:
                      key: next_agent
                      agents:
                        talk: [agent_step: {system: "chat",
                                             output_key: talk}]
                """
            )
        )
        assert not validate_flow_file(str(p))

    def test_validate_flow_supervisor_agents_non_mapping(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text("name: b\nsteps:\n  - supervisor: {id: lead, agents: []}\n")
        msgs = {e["message"] for e in validate_flow_file(str(p))}
        assert "supervisor `agents:` must be a non-empty mapping" in msgs

    def test_validate_flow_supervise_requires_key_and_agents(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text("name: b\nsteps:\n  - supervise: {}\n")
        msgs = {e["message"] for e in validate_flow_file(str(p))}
        assert "supervise requires a `key:`" in msgs
        assert "supervise requires a non-empty `agents:` mapping" in msgs

    def test_validate_flow_loop_requires_keys(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "wf.yaml"
        p.write_text("name: b\nsteps:\n  - loop: {key: k}\n")
        errors = validate_flow_file(str(p))
        msgs = {e["message"] for e in errors}
        assert "loop requires an `until:`" in msgs
        assert "loop requires a `body:`" in msgs

    def test_validate_flow_missing_file(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml_schema import validate_flow_file

        with pytest.raises(ConfigError, match="not found"):
            validate_flow_file(str(tmp_path / "nope.yaml"))

    def test_validate_graph_does_not_call_flow_validator(self, tmp_path):
        from teff.yaml_schema import validate_workflow_file

        p = tmp_path / "g.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: g
                steps:
                  - id: t1
                    type: transform
                    config: {action: uppercase, input_key: a, output_key: b}
                edges:
                  - {from: t1, to: t1}
                """
            )
        )
        assert not validate_workflow_file(str(p))

    def test_looks_like_flow_detects_idioms_only(self):
        import yaml

        from teff.flow.compiler import looks_like_flow

        assert looks_like_flow(yaml.safe_load("steps:\n  - map: {}\n"))
        assert not looks_like_flow(
            yaml.safe_load("steps:\n  - id: a\n    type: transform\n")
        )


class TestMcpToolInYaml:
    def test_flow_compiles_mcp_tool(self, tmp_path):
        src = tmp_path / "flow.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: mcp
                default_model: llama3.1:8b
                default_provider: ollama
                tools:
                  - type: mcp
                    config: {id: demo, command: ["python", "server.py"]}
                steps:
                  - transform: {id: t, action: uppercase, input_key: a,
                                output_key: b}
                """
            )
        )
        from teff.flow.compiler import load_flow
        from teff.tool import McpToolGroup

        graph, tools, _initial, _reducers = load_flow(str(src))
        assert list(graph.nodes) == ["t"]
        (group,) = tools
        assert isinstance(group, McpToolGroup)
        assert group.id == "demo"
        assert group._command == ["python", "server.py"]

    def test_graph_loads_mcp_tool(self, tmp_path):
        src = tmp_path / "g.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: g
                tools:
                  - type: mcp
                    config: {id: drive, url: "http://localhost:8000/mcp"}
                steps:
                  - id: t1
                    type: transform
                    config: {action: uppercase, input_key: a, output_key: b}
                """
            )
        )
        from teff.tool import McpToolGroup
        from teff.yaml import load_workflow

        graph, tools, _initial, _reducers = load_workflow(str(src))
        (group,) = tools
        assert isinstance(group, McpToolGroup)
        assert group.id == "drive"
        assert group._url == "http://localhost:8000/mcp"

    def test_validate_flow_mcp_config_requires_url_or_command(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "bad.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: bad
                tools:
                  - type: mcp
                    config: {id: demo}
                steps:
                  - transform: {id: t, action: uppercase, input_key: a,
                                output_key: b}
                """
            )
        )
        errors = validate_flow_file(str(p))
        assert any("exactly one" in e["message"] for e in errors)
        assert any(e["path"] == "tools[0].config" for e in errors)

    def test_validate_graph_mcp_config_valid(self, tmp_path):
        from teff.yaml_schema import validate_workflow_file

        p = tmp_path / "ok.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: ok
                tools:
                  - type: mcp
                    config: {id: demo, command: ["uvx", "mcp-server-git"]}
                steps:
                  - id: t1
                    type: transform
                    config: {action: uppercase, input_key: a, output_key: b}
                """
            )
        )
        assert not validate_workflow_file(str(p))

    def test_flow_unknown_uses_mcp_in_use_tools(self, tmp_path):
        """mcp members appear to the agent as ``<id>__<tool>`` names."""
        src = tmp_path / "flow.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: mcp
                default_model: llama3.1:8b
                default_provider: ollama
                tools:
                  - type: mcp
                    config: {id: demo, command: ["python", "server.py"]}
                steps:
                  - agent: {id: bot, system: "You are helpful.",
                           use_tools: [demo__add]}
                """
            )
        )
        from teff.flow.compiler import load_flow

        graph, tools, _initial, _reducers = load_flow(str(src))
        from teff.tool import McpToolGroup

        assert isinstance(tools[0], McpToolGroup)
        assert tools[0].id == "demo"

    def test_flow_yaml_mcp_preset_builds_group(self, tmp_path):
        src = tmp_path / "flow.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: mcp
                default_model: llama3.1:8b
                default_provider: ollama
                tools:
                  - type: mcp
                    config:
                      preset: google_drive
                      env: {GOOGLE_DRIVE_REFRESH_TOKEN: "tok"}
                steps:
                  - transform: {id: t, action: uppercase, input_key: a,
                                output_key: b}
                """
            )
        )
        from teff.flow.compiler import load_flow
        from teff.tool import McpToolGroup

        _graph, tools, _initial, _reducers = load_flow(str(src))
        (group,) = tools
        assert isinstance(group, McpToolGroup)
        assert group.id == "google_drive"
        assert group._command[0] == "npx"
        assert group._env["GOOGLE_DRIVE_REFRESH_TOKEN"] == "tok"
        assert group._env["GOOGLE_DRIVE_CLIENT_ID"] == ""

    def test_flow_yaml_mcp_preset_merges_env(self, tmp_path):
        src = tmp_path / "flow.yaml"
        src.write_text(
            textwrap.dedent(
                """
                name: mcp
                tools:
                  - type: mcp
                    config:
                      preset: gmail
                      id: mail
                steps:
                  - id: t1
                    type: transform
                    config: {action: uppercase, input_key: a, output_key: b}
                """
            )
        )
        from teff.tool import McpToolGroup
        from teff.yaml import load_workflow

        _graph, tools, _initial, _reducers = load_workflow(str(src))
        (group,) = tools
        assert isinstance(group, McpToolGroup)
        assert group.id == "mail"
        assert group._command == ["npx", "-y", "@google/mcp-server-gmail"]

    def test_validate_flow_mcp_preset_valid(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "ok.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: ok
                tools:
                  - type: mcp
                    config: {preset: google_calendar}
                steps:
                  - transform: {id: t, action: uppercase, input_key: a,
                                output_key: b}
                """
            )
        )
        assert not validate_flow_file(str(p))

    def test_validate_flow_mcp_unknown_preset(self, tmp_path):
        from teff.yaml_schema import validate_flow_file

        p = tmp_path / "bad.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: bad
                tools:
                  - type: mcp
                    config: {preset: nope}
                steps:
                  - transform: {id: t, action: uppercase, input_key: a,
                                output_key: b}
                """
            )
        )
        errors = validate_flow_file(str(p))
        assert any("unknown mcp preset" in e["message"] for e in errors)
        assert any(e["path"] == "tools[0].config.preset" for e in errors)

    def test_validate_graph_mcp_preset_valid(self, tmp_path):
        from teff.yaml_schema import validate_workflow_file

        p = tmp_path / "ok.yaml"
        p.write_text(
            textwrap.dedent(
                """
                name: ok
                tools:
                  - type: mcp
                    config: {preset: google_drive}
                steps:
                  - id: t1
                    type: transform
                    config: {action: uppercase, input_key: a, output_key: b}
                """
            )
        )
        assert not validate_workflow_file(str(p))
