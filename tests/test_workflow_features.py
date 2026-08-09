import pytest

from teff.node import Transform

ROOT = __file__.rsplit("/tests/", 1)[0]
EXAMPLE = f"{ROOT}/examples/applications/gitlab-reviewer/workflow.yaml"
REPO_HEALTH = f"{ROOT}/examples/applications/repo-health/workflow.yaml"
PLUGINS_EXAMPLE = f"{ROOT}/examples/plugins"


class TestEnvInterpolation:
    def test_interpolates_env_in_tool_config(self, tmp_path, monkeypatch):
        from teff.yaml import load_workflow

        monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
        monkeypatch.setenv("GITLAB_TOKEN", "secret-token")
        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: env-workflow
tools:
  - type: gitlab_list_open_mrs
    config:
      url: "${GITLAB_URL}"
      token: "${GITLAB_TOKEN}"
steps:
  - id: s
    type: transform
    config: {action: trim, input_key: x, output_key: y}
"""
        )
        _, tools, _, _ = load_workflow(str(path))
        assert len(tools) == 1
        assert tools[0].url == "https://gitlab.example.com"
        assert tools[0].token == "secret-token"

    def test_missing_env_stays_as_placeholder(self, tmp_path, monkeypatch):
        from teff.yaml import load_workflow

        monkeypatch.delenv("GITLAB_URL", raising=False)
        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: env-workflow
tools:
  - type: gitlab_list_open_mrs
    config:
      url: "${GITLAB_URL}"
      token: "static"
steps:
  - id: s
    type: transform
    config: {action: trim, input_key: x, output_key: y}
"""
        )
        _, tools, _, _ = load_workflow(str(path))
        assert tools[0].url == "${GITLAB_URL}"
        assert tools[0].token == "static"

    def test_works_without_tools(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: plain
steps:
  - id: s
    type: transform
    config: {action: uppercase, input_key: t, output_key: o}
"""
        )
        graph, tools, initial, reducers = load_workflow(str(path))
        assert len(tools) == 0
        assert graph.entry_point == "s"

    def test_interpolates_env_in_step_and_state(self, tmp_path, monkeypatch):
        from teff.yaml import load_workflow

        monkeypatch.setenv("SYSTEM_HINT", "be brief")
        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: env-step
steps:
  - id: llm
    type: llm_chat
    config:
      model: gpt-4
      system: "hint: ${SYSTEM_HINT}"
state:
  initial:
    token: "${MISSING_TOKEN}"
"""
        )
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        graph, _, initial, _ = load_workflow(str(path))
        assert graph.nodes["llm"].config["system"] == "hint: be brief"
        assert initial["token"] == "${MISSING_TOKEN}"


class TestNodeTypePreserved:
    def test_yaml_load_keeps_real_node_type(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: types
steps:
  - id: first
    type: transform
    config: {action: uppercase, input_key: t, output_key: o}
"""
        )
        graph, _, _, _ = load_workflow(str(path))
        assert graph.nodes["first"].type == "transform"


class TestJsonGet:
    @pytest.mark.asyncio
    async def test_extracts_field_from_state_dict(self):
        node = Transform(
            action="json_get", input_key="payload", field="verdict", output_key="v"
        )
        out = await node.execute(None, {"payload": {"verdict": "approve", "score": 3}})
        assert out == {"v": "approve"}

    @pytest.mark.asyncio
    async def test_missing_field_raises(self):
        node = Transform(
            action="json_get", input_key="payload", field="nope", output_key="v"
        )
        with pytest.raises(KeyError, match="nope"):
            await node.execute(None, {"payload": {"verdict": "approve"}})

    @pytest.mark.asyncio
    async def test_non_dict_raises(self):
        node = Transform(
            action="json_get", input_key="payload", field="v", output_key="v"
        )
        with pytest.raises(ValueError, match="dict"):
            await node.execute(None, {"payload": "not a dict"})

    @pytest.mark.asyncio
    async def test_raw_keeps_list(self):
        node = Transform(
            action="json_get",
            input_key="payload",
            field="steps",
            output_key="steps",
            raw=True,
        )
        out = await node.execute(
            None, {"payload": {"steps": ["a", "b"], "verdict": "approve"}}
        )
        assert out == {"steps": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_default_stringifies_list(self):
        node = Transform(
            action="json_get",
            input_key="payload",
            field="steps",
            output_key="steps",
        )
        out = await node.execute(None, {"payload": {"steps": ["a", "b"]}})
        assert out == {"steps": "['a', 'b']"}


class TestAppendTransform:
    @pytest.mark.asyncio
    async def test_appends_template_to_list(self):
        node = Transform(
            action="append",
            template="Chapter 1: A hero named {hero} sets out.",
            output_key="chapters",
        )
        out = await node.execute(None, {"hero": "Ada", "chapters": []})
        assert out == {"chapters": ["Chapter 1: A hero named Ada sets out."]}

    @pytest.mark.asyncio
    async def test_creates_list_if_absent(self):
        node = Transform(
            action="append",
            template="They come to the town of {setting}.",
            output_key="chapters",
        )
        out = await node.execute(None, {"setting": "Bellmore"})
        assert out == {"chapters": ["They come to the town of Bellmore."]}

    @pytest.mark.asyncio
    async def test_accumulates_across_calls(self):
        node = Transform(
            action="append",
            template="They come to the town of {setting}.",
            output_key="chapters",
        )
        state = {"setting": "Bellmore", "chapters": []}
        await node.execute(None, state)
        state["setting"] = "Lakeside"
        out = await node.execute(None, state)
        assert out == {
            "chapters": [
                "They come to the town of Bellmore.",
                "They come to the town of Lakeside.",
            ]
        }

    @pytest.mark.asyncio
    async def test_uses_input_key_without_template(self):
        node = Transform(action="append", input_key="note", output_key="notes")
        out = await node.execute(None, {"note": "hello"})
        assert out == {"notes": ["hello"]}

    @pytest.mark.asyncio
    async def test_missing_template_key_raises(self):
        node = Transform(action="append", template="Ref {hero}", output_key="chapters")
        with pytest.raises(KeyError):
            await node.execute(None, {"chapters": []})


class TestNumericConditions:
    def _run(self, condition: str, state: dict) -> bool:
        from teff.graph.conditions import evaluate

        return evaluate(condition, state)

    def test_gte(self):
        assert self._run("diff_lines>=2", {"diff_lines": "2"})
        assert self._run("diff_lines>=2", {"diff_lines": 3})
        assert not self._run("diff_lines>=2", {"diff_lines": 1})

    def test_lte(self):
        assert self._run("diff_lines<=10", {"diff_lines": "10"})
        assert not self._run("diff_lines<=10", {"diff_lines": 11})

    def test_gt_lt(self):
        assert self._run("x>5", {"x": "6"})
        assert not self._run("x>5", {"x": 5})
        assert self._run("x<5", {"x": 4})
        assert not self._run("x<5", {"x": 5})

    def test_missing_key_is_false(self):
        assert not self._run("diff_lines>0", {})

    def test_non_numeric_is_false(self):
        assert not self._run("x>5", {"x": "abc"})

    def test_string_conditions_still_work(self):
        assert self._run("verdict=approve", {"verdict": "approve"})
        assert self._run("verdict!=approve", {"verdict": "comment"})


class TestGitlabReviewerExample:
    def test_example_loads(self):
        from teff.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(EXAMPLE)
        names = {t.name for t in tools}
        assert {
            "gitlab_list_open_mrs",
            "gitlab_get_mr_changes",
            "gitlab_post_note",
            "gitlab_approve",
            "send_telegram",
            "kv_store",
        } <= names
        assert graph.entry_point == "agent-reviewer"
        assert set(graph.nodes) == {"agent-reviewer"}
        assert initial["project_ids"] == ["group/repo1", "group/repo2"]

    def test_example_validates(self):
        from teff.yaml_schema import validate_workflow_file

        assert validate_workflow_file(EXAMPLE) == []


GITHUB_EXAMPLE = f"{ROOT}/examples/applications/github-reviewer/workflow.yaml"


class TestGithubReviewerExample:
    def test_example_loads(self):
        from teff.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(GITHUB_EXAMPLE)
        names = {t.name for t in tools}
        assert {
            "github_list_open_prs",
            "github_get_pr_changes",
            "github_post_comment",
            "github_approve",
            "send_telegram",
            "kv_store",
        } <= names
        assert graph.entry_point == "agent-reviewer"
        assert set(graph.nodes) == {"agent-reviewer"}
        assert initial["repo_ids"] == ["owner/repo1", "owner/repo2"]

    def test_example_validates(self):
        from teff.yaml_schema import validate_workflow_file

        assert validate_workflow_file(GITHUB_EXAMPLE) == []


class TestRepoHealthExample:
    def test_example_loads(self):
        from teff.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(REPO_HEALTH)
        names = {t.name for t in tools}
        assert {
            "git",
            "csv_query",
            "redis",
            "lock",
            "wait_for",
            "send_telegram",
        } <= names
        assert graph.entry_point == "agent-agent"
        assert set(graph.nodes) == {"agent-agent"}
        assert initial["priority_csv"] == "data/priority.csv"

    def test_example_validates(self):
        from teff.yaml_schema import validate_workflow_file

        assert validate_workflow_file(REPO_HEALTH) == []

    def test_flow_py_compiles_equivalent_structure(self):
        import importlib.util

        path = f"{ROOT}/examples/applications/repo-health/flow.py"
        spec = importlib.util.spec_from_file_location("repo_health_flow", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        graph = mod.build_flow().compile()
        node_types = {node.type for node in graph.nodes.values()}
        assert node_types == {"context_builder", "react_agent", "tool_exec"}

        tools = mod.build_tools()
        names = {t.name for t in tools}
        assert {
            "git",
            "csv_query",
            "redis",
            "lock",
            "wait_for",
            "send_telegram",
        } <= names


class TestPluginsExample:
    def test_workflow_validates(self):
        from teff.yaml_schema import validate_workflow_file

        assert validate_workflow_file(f"{PLUGINS_EXAMPLE}/workflow.yaml") == []

    def test_example_runs_offline(self):
        import asyncio
        import json

        from teff.yaml import load_workflow

        graph, tools, initial, reducers = load_workflow(
            f"{PLUGINS_EXAMPLE}/workflow.yaml"
        )
        result = asyncio.run(graph.run(state=initial, tools=tools, reducers=reducers))
        assert result["slug"] == "hello-world-teff-plugins"
        assert result["count"] == "4"
        assert json.loads(result["report"]) == {
            "slug": "hello-world-teff-plugins",
            "count": "4",
        }

    def test_class_based_workflow_validates_and_runs(self):
        import asyncio

        from teff.yaml import load_workflow
        from teff.yaml_schema import validate_workflow_file

        path = f"{PLUGINS_EXAMPLE}/workflow-classes.yaml"
        assert validate_workflow_file(path) == []
        graph, tools, initial, reducers = load_workflow(path)
        result = asyncio.run(graph.run(state=initial, tools=tools, reducers=reducers))
        assert result["shouted"] == "HELLO PLUGINS"


class TestContextBuilderListRendering:
    @pytest.mark.asyncio
    async def test_renders_lists_one_per_line(self):
        from teff.node import ContextBuilder

        node = ContextBuilder(
            sections={"project_ids": "Projects to review"},
            messages_key="messages",
            output_key="input",
        )
        out = await node.execute(
            None, {"project_ids": ["group/a", "group/b"], "messages": []}
        )
        assert out["input"] == "Projects to review:\ngroup/a\ngroup/b"


class TestDaemonOnce:
    YAML = """\
name: daemon-workflow
state:
  initial:
    text: "a\\nb\\nc"
steps:
  - id: bump
    type: transform
    config: {action: count_lines, input_key: text, output_key: lines}
"""

    def test_daemon_once_runs_single_tick(self, tmp_path, capsys):
        from teff.cli import daemon as daemon_cmd

        path = tmp_path / "wf.yaml"
        path.write_text(self.YAML)
        daemon_cmd(
            str(path),
            interval=0,
            once=True,
            trace=False,
            checkpoint=None,
            checkpoint_id="daemon",
            checkpoint_owner="test",
            node_timeout=None,
            max_iterations=None,
        )
        captured = capsys.readouterr()
        assert '"lines": "3"' in captured.out


class TestDeclarativeRetry:
    def test_step_retry_block_wraps_node(self, tmp_path):
        from teff.node.retry import Retry
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: retry-wf
steps:
  - id: s
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
    retry:
      max_retries: 5
      delay: 0.5
      backoff: 2.0
edges: []
"""
        )
        graph, _tools, _state, _reducers = load_workflow(str(path))
        node = graph.nodes["s"]
        assert isinstance(node, Retry)
        assert node._max_retries == 5
        assert node._delay == 0.5
        assert node._backoff == 2.0

    def test_step_retry_disabled_leaves_node_plain(self, tmp_path):
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: retry-wf
steps:
  - id: s
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
    retry:
      enabled: false
edges: []
"""
        )
        graph, _tools, _state, _reducers = load_workflow(str(path))
        assert type(graph.nodes["s"]).__name__ == "Transform"

    @pytest.mark.asyncio
    async def test_declarative_retry_retries_and_succeeds(self, tmp_path):
        import asyncio

        from teff.node import Node
        from teff.yaml import load_workflow

        attempt = 0
        original_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            await original_sleep(0)

        asyncio.sleep = fake_sleep
        try:
            from teff.node.registry import default_registry

            class Flaky(Node):
                type = "flaky"

                async def execute(self, ctx, state):
                    nonlocal attempt
                    attempt += 1
                    if attempt < 2:
                        raise ValueError("nope")
                    state["done"] = True
                    return state

            default_registry.register("flaky", lambda cfg: Flaky(cfg))
            path = tmp_path / "wf.yaml"
            path.write_text(
                """\
name: retry-wf
steps:
  - id: s
    type: flaky
    config: {}
    retry:
      max_retries: 3
      delay: 1.0
edges: []
"""
            )
            graph, _tools, _state, _reducers = load_workflow(str(path))
            result = await graph.run({})
        finally:
            asyncio.sleep = original_sleep
        assert result["done"] is True
        assert attempt == 2

    def test_retry_block_validates(self, tmp_path):
        from teff.yaml_schema import validate_workflow_file

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: retry-wf
steps:
  - id: s
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
    retry:
      max_retries: 0
edges: []
"""
        )
        errors = validate_workflow_file(str(path))
        assert errors


class TestSubFlowYaml:
    SUBFLOW_YAML = """\
name: subflow-wf
steps:
  - id: greet
    type: transform
    config: {action: trim, input_key: text, output_key: text}
  - id: inner
    type: subflow
    config:
      input_map: {text: x}
      output_map: {y: result}
      graph:
        steps:
          - id: up
            type: transform
            config: {action: uppercase, input_key: x, output_key: y}
edges:
  - from: greet
    to: inner
"""

    def test_subflow_nested_graph_loads(self, tmp_path):
        from teff.flow.sub_flow import SubFlow
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(self.SUBFLOW_YAML)
        graph, _tools, _state, _reducers = load_workflow(str(path))
        node = graph.nodes["inner"]
        assert isinstance(node, SubFlow)
        assert set(node._graph.nodes) == {"up"}
        assert node._graph.entry_point == "up"

    def test_subflow_nested_graph_runs(self, tmp_path):
        import asyncio

        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(self.SUBFLOW_YAML)
        graph, _tools, _state, _reducers = load_workflow(str(path))
        result = asyncio.run(graph.run(state={"text": "  hi  "}))
        assert result["text"] == "hi"
        assert result["result"] == "HI"

    def test_subflow_round_trips(self, tmp_path):
        import asyncio

        from teff.yaml import from_yaml, workflow_to_yaml

        path = tmp_path / "wf.yaml"
        path.write_text(self.SUBFLOW_YAML)
        graph, _tools, _state, _reducers = from_yaml(str(path)), [], {}, {}
        dumped = workflow_to_yaml(graph)
        reloaded = from_yaml(dumped)
        node = reloaded.nodes["inner"]
        assert node.config["graph"]["steps"][0]["config"]["action"] == "uppercase"
        result = asyncio.run(reloaded.run(state={"text": "  hi  "}))
        assert result["result"] == "HI"

    def test_subflow_build_agent_step(self, tmp_path):
        from teff.flow.sub_flow import SubFlow
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: agent-wf
steps:
  - id: a
    type: subflow
    config:
      id_prefix: chat
      build:
        type: agent_step
        system: You are helpful
        output_key: answer
        model: fake-model
        provider: fake
edges: []
"""
        )
        graph, _tools, _state, _reducers = load_workflow(str(path))
        node = graph.nodes["a"]
        assert isinstance(node, SubFlow)
        assert node.config["build"]["output_key"] == "answer"

    def test_subflow_build_rejects_recipe_providers(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: agent-wf
steps:
  - id: a
    type: subflow
    config:
      build:
        type: agent_step
        system: You are helpful
        output_key: answer
        model: fake-model
        provider: fake
        providers:
          - name: fake
            type: openai_compatible
edges: []
"""
        )
        with pytest.raises(ConfigError, match="top-level `providers:`"):
            load_workflow(str(path))

    def test_subflow_requires_graph_or_build(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: bad-wf
steps:
  - id: x
    type: subflow
    config: {}
edges: []
"""
        )
        with pytest.raises(ConfigError, match="subflow requires config.graph"):
            load_workflow(str(path))

    def test_subflow_unknown_build_recipe(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: bad-wf
steps:
  - id: x
    type: subflow
    config:
      build: {type: nope}
edges: []
"""
        )
        with pytest.raises(ConfigError, match="unknown subflow build recipe"):
            load_workflow(str(path))
