"""SQLite checkpointing — stdlib only, shared file format with the RAG store."""

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

from teff.checkpoint.base import DEFAULT_OWNER, Checkpoint, Checkpointer


class SQLiteCheckpointer(Checkpointer):
    """Store checkpoints in a SQLite database.

    Uses one row per ``(owner, checkpoint_id)`` pair — a composite primary
    key, so the same ID can belong to different owners (users/tenants)
    without colliding.  Each ``save`` is a single ``INSERT .. ON CONFLICT
    REPLACE`` transaction, so a crash leaves either the old or the new row,
    never a mix.  Existing single-owner databases are migrated in place:
    their rows move under :data:`~teff.checkpoint.DEFAULT_OWNER`, and an
    ``updated_at`` column is added for TTL cleanup.

    All database work runs in a worker thread (``asyncio.to_thread``) behind
    a lock, so checkpoint saves never block the event loop — important when
    many parallel branches checkpoint through the same store.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._migrate()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                owner TEXT NOT NULL DEFAULT 'default',
                checkpoint_id TEXT NOT NULL,
                state TEXT NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL,
                updated_at REAL,
                PRIMARY KEY (owner, checkpoint_id)
            )
            """
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Migrate a legacy single-owner table to the owner-scoped schema."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ).fetchone()
        if row is None:
            return
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(checkpoints)")]
        if "owner" not in cols:
            self._conn.execute("ALTER TABLE checkpoints RENAME TO checkpoints_legacy")
            self._conn.execute(
                """
                CREATE TABLE checkpoints (
                    owner TEXT NOT NULL DEFAULT 'default',
                    checkpoint_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    next_node_id TEXT,
                    iteration INTEGER NOT NULL,
                    updated_at REAL,
                    PRIMARY KEY (owner, checkpoint_id)
                )
                """
            )
            self._conn.execute(
                """
                INSERT INTO checkpoints (owner, checkpoint_id, state, next_node_id, iteration)
                SELECT 'default', checkpoint_id, state, next_node_id, iteration
                FROM checkpoints_legacy
                """
            )
            self._conn.execute("DROP TABLE checkpoints_legacy")
            self._conn.commit()
            return
        if "updated_at" not in cols:
            self._conn.execute("ALTER TABLE checkpoints ADD COLUMN updated_at REAL")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    async def _run(self, fn, *args, **kwargs):
        """Run a sync DB call in a worker thread, serialised by the lock."""
        def _call():
            with self._lock:
                return fn(*args, **kwargs)

        return await asyncio.to_thread(_call)

    async def save(
        self,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        def _save():
            self._conn.execute(
                """
                INSERT INTO checkpoints (owner, checkpoint_id, state, next_node_id, iteration, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner, checkpoint_id) DO UPDATE SET
                    state = excluded.state,
                    next_node_id = excluded.next_node_id,
                    iteration = excluded.iteration,
                    updated_at = excluded.updated_at
                """,
                (
                    owner,
                    checkpoint_id,
                    json.dumps(checkpoint.state, ensure_ascii=False),
                    checkpoint.next_node_id,
                    checkpoint.iteration,
                    time.time(),
                ),
            )
            self._conn.commit()

        await self._run(_save)

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        def _load():
            return self._conn.execute(
                "SELECT state, next_node_id, iteration FROM checkpoints "
                "WHERE owner = ? AND checkpoint_id = ?",
                (owner, checkpoint_id),
            ).fetchone()

        row = await self._run(_load)
        if row is None:
            return None
        return Checkpoint(
            state=json.loads(row[0]),
            next_node_id=row[1],
            iteration=row[2],
        )

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        def _delete():
            self._conn.execute(
                "DELETE FROM checkpoints WHERE owner = ? AND checkpoint_id = ?",
                (owner, checkpoint_id),
            )
            self._conn.commit()

        await self._run(_delete)

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""

        def _list():
            return self._conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE owner = ? ORDER BY checkpoint_id",
                (owner,),
            ).fetchall()

        rows = await self._run(_list)
        return [r[0] for r in rows]

    async def cleanup(
        self,
        *,
        owner: str | None = None,
        max_age: float | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Delete stale checkpoints; returns how many were removed."""

        def _cleanup():
            if max_age is None and keep_last is None:
                return 0
            removed = 0
            now = time.time()
            if owner is not None:
                owners = [owner]
            else:
                owners = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT DISTINCT owner FROM checkpoints"
                    ).fetchall()
                ]
            for own in owners:
                if max_age is not None:
                    cur = self._conn.execute(
                        "DELETE FROM checkpoints WHERE owner = ? AND "
                        "COALESCE(updated_at, 0) < ?",
                        (own, now - max_age),
                    )
                    removed += cur.rowcount
                if keep_last is not None:
                    stale = [
                        r[0]
                        for r in self._conn.execute(
                            "SELECT checkpoint_id FROM checkpoints WHERE owner = ? "
                            "ORDER BY COALESCE(updated_at, 0) DESC LIMIT -1 OFFSET ?",
                            (own, keep_last),
                        ).fetchall()
                    ]
                    for cid in stale:
                        self._conn.execute(
                            "DELETE FROM checkpoints WHERE owner = ? AND checkpoint_id = ?",
                            (own, cid),
                        )
                        removed += 1
            self._conn.commit()
            return removed

        return await self._run(_cleanup)
