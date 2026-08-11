import contextlib
from typing import TypedDict

import pytest


def _build_linear_graph():
    from teff.flow import Flow
    from teff.node import Transform

    flow = Flow("ckpt")
    flow.step(
        Transform({"action": "uppercase", "input_key": "text", "output_key": "a"})
    )
    flow.step(Transform({"action": "lowercase", "input_key": "a", "output_key": "b"}))
    return flow.compile()


class TestCheckpointBase:
    def test_roundtrip_dict(self):
        from teff.checkpoint import Checkpoint, checkpoint_from_dict, checkpoint_to_dict

        cp = Checkpoint(state={"a": 1}, next_node_id="node_2", iteration=3)
        assert checkpoint_from_dict(checkpoint_to_dict(cp)) == cp

    def test_checkpoint_id_required(self):
        import asyncio

        from teff.checkpoint import SQLiteCheckpointer

        g = _build_linear_graph()
        with pytest.raises(ValueError, match="checkpoint_id"):
            asyncio.run(
                g.run(
                    state={"text": "hi"}, checkpointer=SQLiteCheckpointer("/tmp/x.db")
                )
            )

    def test_missing_dep_raises(self):
        import importlib.util

        if importlib.util.find_spec("asyncpg") is not None:
            pytest.skip("asyncpg is installed")
        from teff.checkpoint.pg import PGCheckpointer

        with pytest.raises(ImportError):
            PGCheckpointer("postgresql://localhost/x")


class TestJSONFileCheckpointer:
    def test_save_load_delete(self, tmp_path):
        import asyncio

        from teff.checkpoint import Checkpoint, JSONFileCheckpointer

        ck = JSONFileCheckpointer(str(tmp_path))
        asyncio.run(
            ck.save("t1", Checkpoint(state={"a": 1}, next_node_id="n", iteration=2))
        )
        cp = asyncio.run(ck.load("t1"))
        assert cp is not None
        assert cp.state == {"a": 1}
        assert cp.next_node_id == "n"
        assert cp.iteration == 2

        asyncio.run(ck.delete("t1"))
        assert asyncio.run(ck.load("t1")) is None

    def test_load_missing(self, tmp_path):
        import asyncio

        from teff.checkpoint import JSONFileCheckpointer

        ck = JSONFileCheckpointer(str(tmp_path))
        assert asyncio.run(ck.load("missing")) is None

    def test_sanitizes_path(self, tmp_path):
        import asyncio

        from teff.checkpoint import Checkpoint, JSONFileCheckpointer

        ck = JSONFileCheckpointer(str(tmp_path))
        asyncio.run(
            ck.save("a/b", Checkpoint(state={}, next_node_id=None, iteration=0))
        )
        assert asyncio.run(ck.load("a/b")) is not None


class TestSQLiteCheckpointer:
    def test_save_load_overwrite(self, tmp_path):
        import asyncio

        from teff.checkpoint import Checkpoint, SQLiteCheckpointer

        ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        try:
            asyncio.run(
                ck.save(
                    "t1", Checkpoint(state={"n": 1}, next_node_id="n1", iteration=1)
                )
            )
            asyncio.run(
                ck.save(
                    "t1", Checkpoint(state={"n": 2}, next_node_id="n2", iteration=2)
                )
            )
            cp = asyncio.run(ck.load("t1"))
            assert cp is not None
            assert cp.state == {"n": 2}
            assert cp.next_node_id == "n2"
            assert cp.iteration == 2

            asyncio.run(ck.delete("t1"))
            assert asyncio.run(ck.load("t1")) is None
        finally:
            ck.close()

    def test_independent_ids(self, tmp_path):
        import asyncio

        from teff.checkpoint import Checkpoint, SQLiteCheckpointer

        ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        try:
            asyncio.run(
                ck.save("a", Checkpoint(state={"x": 1}, next_node_id=None, iteration=0))
            )
            asyncio.run(
                ck.save("b", Checkpoint(state={"y": 2}, next_node_id=None, iteration=0))
            )
            assert asyncio.run(ck.load("a")).state == {"x": 1}
            assert asyncio.run(ck.load("b")).state == {"y": 2}
        finally:
            ck.close()


def _cp(state: dict):
    from teff.checkpoint import Checkpoint

    return Checkpoint(state=state, next_node_id=None, iteration=0)


@pytest.mark.parametrize("kind", ["file", "sqlite"])
class TestOwnerIsolation:
    @pytest.fixture(autouse=True)
    def _make(self, kind, tmp_path):
        import asyncio

        if kind == "file":
            from teff.checkpoint import JSONFileCheckpointer

            self.ck = JSONFileCheckpointer(str(tmp_path))
        else:
            from teff.checkpoint import SQLiteCheckpointer

            self.ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        self.run = lambda coro: asyncio.run(coro)
        yield
        close = getattr(self.ck, "close", None)
        if close is not None:
            close()

    async def test_same_id_different_owners_isolated(self):
        await self.ck.save("chat-1", _cp({"user": "alice"}), owner="alice")
        await self.ck.save("chat-1", _cp({"user": "bob"}), owner="bob")

        alice = await self.ck.load("chat-1", owner="alice")
        bob = await self.ck.load("chat-1", owner="bob")
        assert alice.state == {"user": "alice"}
        assert bob.state == {"user": "bob"}

        # default owner sees nothing
        assert await self.ck.load("chat-1") is None

    async def test_list_returns_owner_checkpoints(self):
        await self.ck.save("a", _cp({}), owner="alice")
        await self.ck.save("b", _cp({}), owner="alice")
        await self.ck.save("c", _cp({}), owner="bob")

        assert await self.ck.list("alice") == ["a", "b"]
        assert await self.ck.list("bob") == ["c"]
        assert await self.ck.list() == []

    async def test_delete_scoped_to_owner(self):
        await self.ck.save("x", _cp({}), owner="alice")
        await self.ck.save("x", _cp({}), owner="bob")

        await self.ck.delete("x", owner="alice")
        assert await self.ck.load("x", owner="alice") is None
        assert await self.ck.load("x", owner="bob") is not None

    async def test_overwrite_scoped_to_owner(self):
        await self.ck.save("x", _cp({"v": 1}), owner="alice")
        await self.ck.save("x", _cp({"v": 2}), owner="alice")
        assert (await self.ck.load("x", owner="alice")).state == {"v": 2}


class TestSQLiteOwnerMigration:
    def test_migrates_legacy_single_owner_table(self, tmp_path):
        import asyncio
        import sqlite3

        path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO checkpoints VALUES ('old-1', '{\"x\": 1}', NULL, 2)")
        conn.commit()
        conn.close()

        from teff.checkpoint import SQLiteCheckpointer

        ck = SQLiteCheckpointer(path)
        try:
            assert asyncio.run(ck.load("old-1")).state == {"x": 1}
            assert asyncio.run(ck.list()) == ["old-1"]
            # save into the migrated owner-scoped schema works
            asyncio.run(ck.save("new", _cp({"y": 2}), owner="alice"))
            assert asyncio.run(ck.load("new", owner="alice")).state == {"y": 2}
        finally:
            ck.close()


class TestGraphOwnerPassThrough:
    @pytest.mark.parametrize("kind", ["file", "sqlite"])
    @pytest.mark.asyncio
    async def test_run_scopes_checkpoints(self, kind, tmp_path):
        if kind == "file":
            from teff.checkpoint import JSONFileCheckpointer

            ck = JSONFileCheckpointer(str(tmp_path))
        else:
            from teff.checkpoint import SQLiteCheckpointer

            ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        try:
            g = _build_linear_graph()
            state = {"text": "hi"}
            await g.run(
                state=state,
                checkpointer=ck,
                checkpoint_id="run-1",
                owner="alice",
            )
            await g.run(
                state=state,
                checkpointer=ck,
                checkpoint_id="run-1",
                owner="bob",
            )

            assert await ck.list("alice") == ["run-1"]
            assert await ck.list("bob") == ["run-1"]

            # same id but different owners do not share state
            await ck.save("chat-9", _cp({"text": "alice-secret"}), owner="alice")
            await ck.save("chat-9", _cp({"text": "bob-secret"}), owner="bob")
            alice = await g.run(
                state={"text": "ignored"},
                checkpointer=ck,
                checkpoint_id="chat-9",
                owner="alice",
            )
            # resume used alice's saved state, not bob's
            assert alice["text"] == "alice-secret"
        finally:
            close = getattr(ck, "close", None)
            if close is not None:
                close()


@pytest.fixture
def checkpointer(request, tmp_path):
    kind = request.param
    if kind == "file":
        from teff.checkpoint import JSONFileCheckpointer

        return JSONFileCheckpointer(str(tmp_path))
    if kind == "sqlite":
        from teff.checkpoint import SQLiteCheckpointer

        return SQLiteCheckpointer(str(tmp_path / "ck.db"))


def _make_age(checkpointer, tmp_path):
    if type(checkpointer).__name__ == "JSONFileCheckpointer":
        import os

        def age(cid, owner="default"):
            os.utime(
                checkpointer._path(cid, owner),
                (checkpointer._path(cid, owner).stat().st_mtime - 100.0,) * 2,
            )

        return age

    def age(cid, owner="default"):
        checkpointer._conn.execute(
            "UPDATE checkpoints SET updated_at = ? WHERE owner = ? AND checkpoint_id = ?",
            (
                checkpointer._conn.execute(
                    "SELECT COALESCE(updated_at, 0) FROM checkpoints "
                    "WHERE owner = ? AND checkpoint_id = ?",
                    (owner, cid),
                ).fetchone()[0]
                - 100.0,
                owner,
                cid,
            ),
        )
        checkpointer._conn.commit()

    return age


@pytest.mark.parametrize("checkpointer", ["file", "sqlite"], indirect=True)
class TestCheckpointCleanup:
    async def test_noop_without_args(self, checkpointer, tmp_path):
        await checkpointer.save("a", _cp({}))
        await checkpointer.save("b", _cp({}))
        assert await checkpointer.cleanup() == 0
        assert await checkpointer.list() == ["a", "b"]

    async def test_keep_last(self, checkpointer, tmp_path):
        age = _make_age(checkpointer, tmp_path)
        await checkpointer.save("a", _cp({}))
        await checkpointer.save("b", _cp({}))
        await checkpointer.save("c", _cp({}))
        age("a")
        assert await checkpointer.cleanup(keep_last=2) == 1
        assert await checkpointer.list() == ["b", "c"]

    async def test_max_age_keeps_fresh(self, checkpointer, tmp_path):
        await checkpointer.save("old", _cp({}))
        await checkpointer.save("new", _cp({}))
        assert await checkpointer.cleanup(max_age=50.0) == 0
        assert await checkpointer.list() == ["new", "old"]

    async def test_max_age_removes_old(self, checkpointer, tmp_path):
        age = _make_age(checkpointer, tmp_path)
        await checkpointer.save("old", _cp({}))
        await checkpointer.save("new", _cp({}))
        age("old")
        assert await checkpointer.cleanup(max_age=50.0) == 1
        assert await checkpointer.list() == ["new"]

    async def test_keep_last_and_max_age_combined(self, checkpointer, tmp_path):
        age = _make_age(checkpointer, tmp_path)
        await checkpointer.save("old", _cp({}))
        await checkpointer.save("mid", _cp({}))
        await checkpointer.save("new", _cp({}))
        age("old")
        age("mid")
        assert await checkpointer.cleanup(max_age=50.0, keep_last=2) == 2
        assert await checkpointer.list() == ["new"]

    async def test_owner_scoped_cleanup(self, checkpointer, tmp_path):
        age = _make_age(checkpointer, tmp_path)
        await checkpointer.save("a", _cp({}), owner="alice")
        await checkpointer.save("b", _cp({}), owner="alice")
        await checkpointer.save("c", _cp({}), owner="bob")
        await checkpointer.save("d", _cp({}), owner="bob")
        age("c", owner="bob")
        assert await checkpointer.cleanup(owner="bob", keep_last=1) == 1
        assert await checkpointer.list("alice") == ["a", "b"]
        assert await checkpointer.list("bob") == ["d"]

    async def test_all_owners(self, checkpointer, tmp_path):
        age = _make_age(checkpointer, tmp_path)
        await checkpointer.save("a", _cp({}), owner="alice")
        await checkpointer.save("b", _cp({}), owner="bob")
        age("a", owner="alice")
        age("b", owner="bob")
        assert await checkpointer.cleanup(keep_last=0) == 2
        assert await checkpointer.list("alice") == []
        assert await checkpointer.list("bob") == []


@pytest.mark.parametrize("checkpointer", ["file", "sqlite"], indirect=True)
class TestCheckpointResume:
    async def test_resume_from_checkpoint(self, checkpointer):
        """Fresh run writes checkpoints; re-running with same id resumes."""
        g = _build_linear_graph()

        state = {"text": "Hello World"}
        result = await g.run(
            state=state,
            checkpointer=checkpointer,
            checkpoint_id="run-1",
        )
        assert result["a"] == "HELLO WORLD"
        assert result["b"] == "hello world"

        # Terminal checkpoint: next_node_id is None -> resume returns state
        cp = await checkpointer.load("run-1")
        assert cp is not None
        assert cp.next_node_id is None

        again = await g.run(
            state={"text": "IGNORED"},
            checkpointer=checkpointer,
            checkpoint_id="run-1",
        )
        assert again["a"] == "HELLO WORLD"
        assert again["b"] == "hello world"

    async def test_untouched_new_id_ignores_state(self, checkpointer):
        g = _build_linear_graph()
        result = await g.run(
            state={"text": "abc"},
            checkpointer=checkpointer,
            checkpoint_id="fresh",
        )
        assert result["a"] == "ABC"

    async def test_crash_between_nodes_resumes_next(self, checkpointer):
        """Simulate a crash: save a checkpoint manually and resume."""
        from teff.checkpoint import Checkpoint

        g = _build_linear_graph()
        await checkpointer.save(
            "run-2",
            Checkpoint(
                state={"text": "xx", "a": "XX"}, next_node_id=None, iteration=99
            ),
        )
        # next_node_id None -> completed, returns saved state
        result = await g.run(
            state={"text": "ignored"},
            checkpointer=checkpointer,
            checkpoint_id="run-2",
        )
        assert result["a"] == "XX"

    async def test_resume_mid_graph(self, checkpointer):
        """A checkpoint pointing at node 2 skips node 1 on resume."""
        from teff.checkpoint import Checkpoint

        g = _build_linear_graph()
        node_ids = list(g.nodes)
        await checkpointer.save(
            "run-3",
            Checkpoint(
                state={"text": "Hi", "a": "HI"},
                next_node_id=node_ids[1],
                iteration=1,
            ),
        )
        result = await g.run(
            state={"text": "ignored"},
            checkpointer=checkpointer,
            checkpoint_id="run-3",
        )
        assert result["a"] == "HI"
        assert result["b"] == "hi"

    async def test_state_instance_keeps_schema(self, checkpointer):
        from teff.state import State

        class S(TypedDict):
            text: str
            a: str

        g = _build_linear_graph()
        st = State(S, {"text": "Hello"})
        result = await g.run(
            state=st,
            checkpointer=checkpointer,
            checkpoint_id="state-run",
        )
        assert isinstance(result, State)
        assert result["a"] == "HELLO"

    async def test_error_edge_checkpoint_points_to_fallback(self, checkpointer):
        """After a node fails and routes via __error__, resume goes to fallback."""
        from teff.graph import Edge, Graph
        from teff.node import Node

        class Crash(Node):
            type = "cr"

            async def execute(self, ctx, state):
                raise ValueError("crash")

        class Fallback(Node):
            type = "fb"

            async def execute(self, ctx, state):
                state["handled"] = True
                return state

        g = Graph(
            nodes={"a": Crash({}), "b": Fallback({})},
            edges=[Edge("a", "b", "__error__")],
            entry_point="a",
        )
        result = await g.run(
            state={},
            checkpointer=checkpointer,
            checkpoint_id="err-run",
        )
        assert result["handled"] is True

        # checkpoint points at the fallback node (completed run -> None)
        cp = await checkpointer.load("err-run")
        assert cp is not None
        assert cp.next_node_id is None

    async def test_no_error_edge_keeps_checkpoint_at_failed_node(self, checkpointer):
        from teff.graph import Graph
        from teff.node import Node

        class Crash(Node):
            type = "cr"

            async def execute(self, ctx, state):
                raise ValueError("crash")

        g = Graph(
            nodes={"a": Crash({})},
            edges=[],
            entry_point="a",
        )
        with pytest.raises(ValueError, match="crash"):
            await g.run(
                state={"kept": True},
                checkpointer=checkpointer,
                checkpoint_id="fail-run",
            )

        # checkpoint still points at the failed node, so a resume retries it
        cp = await checkpointer.load("fail-run")
        assert cp is not None
        assert cp.next_node_id == "a"
        assert cp.state == {"kept": True}


class TestSQLiteHistoryCheckpointer:
    def test_save_load_history(self, tmp_path):
        import asyncio

        from teff.checkpoint import Checkpoint, SQLiteHistoryCheckpointer

        ck = SQLiteHistoryCheckpointer(str(tmp_path / "ck.db"))
        try:
            asyncio.run(
                ck.save(
                    "run-1", Checkpoint(state={"n": 1}, next_node_id="n1", iteration=1)
                )
            )
            asyncio.run(
                ck.save(
                    "run-1", Checkpoint(state={"n": 2}, next_node_id="n2", iteration=2)
                )
            )
            asyncio.run(
                ck.save(
                    "run-1", Checkpoint(state={"n": 3}, next_node_id=None, iteration=3)
                )
            )

            hist = asyncio.run(ck.history("run-1"))
            assert hist == [(1, "n1"), (2, "n2"), (3, None)]

            past = asyncio.run(ck.load_at("run-1", 1))
            assert past is not None
            assert past.state == {"n": 1}
            assert past.next_node_id == "n1"
            assert past.iteration == 1

            cur = asyncio.run(ck.load("run-1"))
            assert cur is not None
            assert cur.state == {"n": 3}
        finally:
            ck.close()

    def test_load_at_missing(self, tmp_path):
        import asyncio

        from teff.checkpoint import SQLiteHistoryCheckpointer

        ck = SQLiteHistoryCheckpointer(str(tmp_path / "ck.db"))
        try:
            assert asyncio.run(ck.load_at("run-1", 99)) is None
            assert asyncio.run(ck.history("run-1")) == []
        finally:
            ck.close()

    def test_time_travel_replay(self, tmp_path):
        """Rewind to a past checkpoint, edit state, replay from a branch id."""
        import asyncio

        from teff.checkpoint import SQLiteHistoryCheckpointer

        g = _build_linear_graph()
        ck = SQLiteHistoryCheckpointer(str(tmp_path / "ck.db"))
        try:
            result = asyncio.run(
                g.run(
                    state={"text": "Hello World"},
                    checkpointer=ck,
                    checkpoint_id="story",
                )
            )
            assert result["b"] == "hello world"

            timeline = asyncio.run(ck.history("story"))
            assert timeline  # at least the pre-node snapshots

            past = asyncio.run(ck.load_at("story", 1))
            assert past is not None
            past.state["a"] = "HELLO DRAFTFLOW"
            asyncio.run(ck.save("story-branch", past))

            branch = asyncio.run(
                g.run(
                    state={"text": "ignored"},
                    checkpointer=ck,
                    checkpoint_id="story-branch",
                )
            )
            assert branch["b"] == "hello draftflow"
        finally:
            ck.close()


class TestPGHistoryCheckpointer:
    @pytest.fixture(autouse=True)
    def _maybe_skip(self):
        import importlib.util
        import os

        if os.environ.get("TEFF_TEST_PG_DSN") is None:
            pytest.skip("set TEFF_TEST_PG_DSN to run PostgreSQL checkpoint tests")
        if importlib.util.find_spec("asyncpg") is None:
            pytest.skip("asyncpg is not installed")

    @pytest.fixture
    def pg(self):
        import os

        from teff.checkpoint import PGHistoryCheckpointer

        return PGHistoryCheckpointer(os.environ["TEFF_TEST_PG_DSN"])

    async def test_save_load_history(self, pg):
        from teff.checkpoint import Checkpoint

        await pg.save(
            "run-1", Checkpoint(state={"n": 1}, next_node_id="n1", iteration=1)
        )
        await pg.save(
            "run-1", Checkpoint(state={"n": 2}, next_node_id="n2", iteration=2)
        )
        hist = await pg.history("run-1")
        assert hist == [(1, "n1"), (2, "n2")]

        past = await pg.load_at("run-1", 1)
        assert past is not None
        assert past.state == {"n": 1}
        assert past.next_node_id == "n1"


class TestPGCheckpointCleanup:
    @pytest.fixture(autouse=True)
    def _maybe_skip(self):
        import importlib.util
        import os

        if os.environ.get("TEFF_TEST_PG_DSN") is None:
            pytest.skip("set TEFF_TEST_PG_DSN to run PostgreSQL checkpoint tests")
        if importlib.util.find_spec("asyncpg") is None:
            pytest.skip("asyncpg is not installed")

    @pytest.fixture
    def pg(self):
        import os

        from teff.checkpoint.pg import PGCheckpointer

        ck = PGCheckpointer(os.environ["TEFF_TEST_PG_DSN"])
        return ck

    async def test_cleanup_keep_last(self, pg):
        await pg.save("a", _cp({}))
        await pg.save("b", _cp({}))
        assert await pg.cleanup(keep_last=1) == 1
        assert await pg.list() == ["b"]


class _FakePGConn:
    """Tiny in-memory stand-in for an ``asyncpg`` connection.

    Understands exactly the SQL shapes issued by the PG checkpointer and the
    PG history checkpointer, so the PG backends can be unit-tested without a
    live PostgreSQL server.  Rows live in a plain dict shared by every
    connection of the same checkpointer instance.
    """

    def __init__(self, state):
        self._state = state

    async def execute(self, sql: str, *args) -> str | None:
        if sql.lstrip().startswith("CREATE TABLE"):
            return None
        if "checkpoint_history" in sql:
            if "INSERT" in sql or "ON CONFLICT" in sql:
                owner, cid, iteration, state_json, next_node_id = args
                hist = self._state.setdefault("history", {})
                key = (owner, cid)
                hist.setdefault(key, {})[iteration] = {
                    "state": state_json,
                    "next_node_id": next_node_id,
                }
                return None
        elif "INSERT" in sql or "ON CONFLICT" in sql:
            owner, cid, state_json, next_node_id, iteration, updated_at = args
            self._state.setdefault("checkpoints", {})[(owner, cid)] = {
                "state": state_json,
                "next_node_id": next_node_id,
                "iteration": iteration,
                "updated_at": updated_at,
            }
            return None
        if "COALESCE(updated_at, 0) < $2" in sql:
            owner, cutoff = args
            to_drop = [
                key
                for key, row in self._state.setdefault("checkpoints", {}).items()
                if key[0] == owner and (row["updated_at"] or 0) < cutoff
            ]
            for key in to_drop:
                del self._state["checkpoints"][key]
            return f"DELETE {len(to_drop)}"
        if "DELETE FROM" in sql:
            owner, cid = args
            removed = self._state.setdefault("checkpoints", {}).pop((owner, cid), None)
            return f"DELETE {1 if removed else 0}"
        return None

    async def fetch(self, sql: str, *args) -> list:
        if "SELECT DISTINCT owner" in sql:
            owners = {key[0] for key in self._state.setdefault("checkpoints", {})}
            return [{"owner": o} for o in sorted(owners)]
        if "checkpoint_history" in sql:
            owner, cid = args
            hist = self._state.setdefault("history", {}).get((owner, cid), {})
            return [
                {"iteration": it, "next_node_id": row["next_node_id"]}
                for it, row in sorted(hist.items())
            ]
        if "ORDER BY COALESCE(updated_at, 0) DESC OFFSET" in sql:
            owner, keep_last = args
            rows = sorted(
                self._state.setdefault("checkpoints", {}).items(),
                key=lambda kv: kv[1]["updated_at"] or 0,
                reverse=True,
            )
            return [{"checkpoint_id": key[1]} for key, _ in rows[keep_last:]]
        owner = args[0]
        return [
            {"checkpoint_id": key[1]}
            for key in sorted(self._state.setdefault("checkpoints", {}))
            if key[0] == owner
        ]

    async def fetchrow(self, sql: str, *args) -> dict | None:
        if "checkpoint_history" in sql:
            owner, cid, iteration = args
            row = (
                self._state.setdefault("history", {})
                .get((owner, cid), {})
                .get(iteration)
            )
            if row is None:
                return None
            return {"state": row["state"], "next_node_id": row["next_node_id"]}
        owner, cid = args
        row = self._state.setdefault("checkpoints", {}).get((owner, cid))
        if row is None:
            return None
        return {
            "state": row["state"],
            "next_node_id": row["next_node_id"],
            "iteration": row["iteration"],
        }

    async def close(self) -> None:
        pass


def _mock_pg(monkeypatch, checkpointer_cls, dsn="postgresql://mock/mock"):
    import importlib.util

    if importlib.util.find_spec("asyncpg") is None:
        pytest.skip("asyncpg is not installed")
    import asyncpg

    state = {}

    async def _connect(dsn):
        return _FakePGConn(state)

    class _FakePool:
        @contextlib.asynccontextmanager
        async def acquire(self):
            yield _FakePGConn(state)

        async def close(self) -> None:
            pass

    async def _create_pool(dsn, **kwargs):
        return _FakePool()

    monkeypatch.setattr(asyncpg, "connect", _connect)
    monkeypatch.setattr(asyncpg, "create_pool", _create_pool)
    return checkpointer_cls(dsn)


class TestPGCheckpointerMocked:
    """PG checkpointer logic without a live server (asyncpg faked)."""

    @pytest.fixture
    def pg(self, monkeypatch):
        from teff.checkpoint.pg import PGCheckpointer

        return _mock_pg(monkeypatch, PGCheckpointer)

    async def test_save_load_roundtrip(self, pg):
        await pg.save("run-1", _cp({"n": 1}), owner="u1")
        await pg.save("run-1", _cp({"n": 2}), owner="u1")
        cp = await pg.load("run-1", owner="u1")
        assert cp.state == {"n": 2}
        assert await pg.load("run-1", owner="other") is None

    async def test_delete_and_list(self, pg):
        await pg.save("a", _cp({}), owner="u1")
        await pg.save("b", _cp({}), owner="u1")
        assert await pg.list(owner="u1") == ["a", "b"]
        await pg.delete("a", owner="u1")
        assert await pg.list(owner="u1") == ["b"]

    async def test_cleanup_by_max_age(self, pg):
        await pg.save("old", _cp({}), owner="u1")
        await pg.save("new", _cp({}), owner="u1")
        assert await pg.cleanup(max_age=0) == 2
        assert await pg.list(owner="u1") == []

    async def test_cleanup_keep_last(self, pg):
        await pg.save("a", _cp({}), owner="u1")
        await pg.save("b", _cp({}), owner="u1")
        assert await pg.cleanup(keep_last=1) == 1
        assert await pg.list(owner="u1") == ["b"]


class TestPGHistoryCheckpointerMocked:
    @pytest.fixture
    def pg(self, monkeypatch):
        from teff.checkpoint import PGHistoryCheckpointer

        return _mock_pg(monkeypatch, PGHistoryCheckpointer)

    async def test_save_history_load_at(self, pg):
        from teff.checkpoint import Checkpoint

        await pg.save(
            "run-1",
            Checkpoint(state={"n": 1}, next_node_id="n1", iteration=1),
            owner="u1",
        )
        await pg.save(
            "run-1",
            Checkpoint(state={"n": 2}, next_node_id="n2", iteration=2),
            owner="u1",
        )
        assert await pg.history("run-1", owner="u1") == [(1, "n1"), (2, "n2")]

        past = await pg.load_at("run-1", 1, owner="u1")
        assert past.state == {"n": 1}
        assert past.next_node_id == "n1"
        assert await pg.load_at("run-1", 99, owner="u1") is None
        assert await pg.history("run-1", owner="other") == []
