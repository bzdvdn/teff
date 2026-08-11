"""Time-travel checkpoints: keep every per-iteration snapshot.

A plain :class:`~teff.checkpoint.SQLiteCheckpointer` / ``PGCheckpointer``
overwrites its row on every ``save``, so only the latest snapshot of a run
survives.  Time travel needs the full history: one checkpoint *per node
execution* (keyed by ``iteration``), so a run can be rewound to any earlier
moment, edited, and replayed.

This module adds a ``checkpoint_history`` table holding every snapshot a
checkpointer ever saved for a ``(owner, checkpoint_id)``.  The shared
:class:`_HistoryMixin` implements ``save``/``history``/``load_at``; each
backend (:class:`SQLiteHistoryCheckpointer`, :class:`PGHistoryCheckpointer`)
mixes it with the plain checkpointer and supplies the storage dialect.
"""

from __future__ import annotations

import json

from teff.checkpoint.base import DEFAULT_OWNER, Checkpoint
from teff.checkpoint.pg import PGCheckpointer
from teff.checkpoint.sqlite import SQLiteCheckpointer


class _HistoryMixin:
    """Shared time-travel behaviour for checkpointer backends.

    Mix this with a plain checkpointer (e.g.
    :class:`teff.checkpoint.sqlite.SQLiteCheckpointer`); the mixin's
    ``save`` forwards to the parent via ``super()`` and then appends to the
    history table.  Subclasses implement:

    * ``_history_ensure()`` — create the ``checkpoint_history`` table.
    * ``_history_insert(owner, checkpoint_id, checkpoint)`` — append a row.
    * ``_history_rows(owner, checkpoint_id)`` — rows of
      ``(iteration, next_node_id)`` ordered by iteration.
    * ``_history_row_at(owner, checkpoint_id, iteration)`` — the stored
      ``(state, next_node_id)`` for a specific iteration, or ``None``.
    """

    async def save(
        self, checkpoint_id: str, checkpoint: Checkpoint, *, owner: str = DEFAULT_OWNER
    ) -> None:
        await super().save(  # type: ignore[misc]
            checkpoint_id, checkpoint, owner=owner
        )
        await self._history_insert(owner, checkpoint_id, checkpoint)

    async def history(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> list[tuple[int, str | None]]:
        """Return ``(iteration, next_node_id)`` for every saved snapshot."""
        rows = await self._history_rows(owner, checkpoint_id)
        return [(r[0], r[1]) for r in rows]

    async def load_at(
        self, checkpoint_id: str, iteration: int, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        """Return the checkpoint saved at *iteration*, or ``None``."""
        row = await self._history_row_at(owner, checkpoint_id, iteration)
        if row is None:
            return None
        return Checkpoint(
            state=json.loads(row[0]),
            next_node_id=row[1],
            iteration=iteration,
        )

    def _history_ensure(self) -> None:
        """Create the ``checkpoint_history`` table (implemented by backends)."""
        raise NotImplementedError

    async def _history_insert(
        self, owner: str, checkpoint_id: str, checkpoint: Checkpoint
    ) -> None:
        raise NotImplementedError

    async def _history_rows(
        self, owner: str, checkpoint_id: str
    ) -> list[tuple[int, str | None]]:
        raise NotImplementedError

    async def _history_row_at(
        self, owner: str, checkpoint_id: str, iteration: int
    ) -> tuple | None:
        raise NotImplementedError


_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS checkpoint_history (
    owner TEXT NOT NULL DEFAULT 'default',
    checkpoint_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    state TEXT NOT NULL,
    next_node_id TEXT,
    PRIMARY KEY (owner, checkpoint_id, iteration)
)
"""


class SQLiteHistoryCheckpointer(_HistoryMixin, SQLiteCheckpointer):
    """SQLite checkpointer that also keeps the full per-step history.

    A drop-in for :class:`teff.checkpoint.SQLiteCheckpointer` that, on every
    ``save``, additionally appends the snapshot to a ``checkpoint_history``
    table — so the current checkpoint can be overwritten without losing the
    earlier ones.  ``history`` / ``load_at`` expose the timeline for time
    travel.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str):
        super().__init__(path)
        self._history_ensure()

    def _history_ensure(self) -> None:
        with self._lock:
            self._conn.execute(_HISTORY_DDL)
            self._conn.commit()

    async def _history_insert(
        self, owner: str, checkpoint_id: str, checkpoint: Checkpoint
    ) -> None:
        def _insert():
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO checkpoint_history "
                    "(owner, checkpoint_id, iteration, state, next_node_id) VALUES (?, ?, ?, ?, ?)",
                    (
                        owner,
                        checkpoint_id,
                        checkpoint.iteration,
                        json.dumps(checkpoint.state, ensure_ascii=False),
                        checkpoint.next_node_id,
                    ),
                )
                self._conn.commit()

        import asyncio

        await asyncio.to_thread(_insert)

    async def _history_rows(
        self, owner: str, checkpoint_id: str
    ) -> list[tuple[int, str | None]]:
        def _rows():
            with self._lock:
                rows = self._conn.execute(
                    "SELECT iteration, next_node_id FROM checkpoint_history "
                    "WHERE owner = ? AND checkpoint_id = ? ORDER BY iteration",
                    (owner, checkpoint_id),
                ).fetchall()
                return [(r[0], r[1]) for r in rows]

        import asyncio

        return await asyncio.to_thread(_rows)

    async def _history_row_at(
        self, owner: str, checkpoint_id: str, iteration: int
    ) -> tuple | None:
        def _row_at():
            with self._lock:
                return self._conn.execute(
                    "SELECT state, next_node_id FROM checkpoint_history "
                    "WHERE owner = ? AND checkpoint_id = ? AND iteration = ?",
                    (owner, checkpoint_id, iteration),
                ).fetchone()

        import asyncio

        return await asyncio.to_thread(_row_at)


_HISTORY_DDL_PG = """
CREATE TABLE IF NOT EXISTS checkpoint_history (
    owner TEXT NOT NULL DEFAULT 'default',
    checkpoint_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    state JSONB NOT NULL,
    next_node_id TEXT,
    PRIMARY KEY (owner, checkpoint_id, iteration)
)
"""


class PGHistoryCheckpointer(_HistoryMixin, PGCheckpointer):
    """PostgreSQL checkpointer that also keeps the full per-step history.

    Requires ``asyncpg`` (install via ``teff[pg-checkpoint]``).  Mirrors
    :class:`PGCheckpointer` but appends every ``save`` to a
    ``checkpoint_history`` table, exposing ``history`` / ``load_at`` for
    time travel in production.

    Args:
        dsn: PostgreSQL connection string.
        table: Table name for the *current* checkpoints (default
            ``"checkpoints"``); the history table is ``checkpoint_history``.
    """

    async def _ensure_pool(self):
        """Create the pool plus the history table alongside the base tables."""
        pool = await super()._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(_HISTORY_DDL_PG)
        return pool

    async def _history_insert(
        self, owner: str, checkpoint_id: str, checkpoint: Checkpoint
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoint_history
                (owner, checkpoint_id, iteration, state, next_node_id)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                ON CONFLICT(owner, checkpoint_id, iteration) DO UPDATE SET
                    state = EXCLUDED.state,
                    next_node_id = EXCLUDED.next_node_id
                """,
                owner,
                checkpoint_id,
                checkpoint.iteration,
                json.dumps(checkpoint.state, ensure_ascii=False),
                checkpoint.next_node_id,
            )

    async def _history_rows(
        self, owner: str, checkpoint_id: str
    ) -> list[tuple[int, str | None]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT iteration, next_node_id FROM checkpoint_history "
                "WHERE owner = $1 AND checkpoint_id = $2 ORDER BY iteration",
                owner,
                checkpoint_id,
            )
            return [(r["iteration"], r["next_node_id"]) for r in rows]

    async def _history_row_at(
        self, owner: str, checkpoint_id: str, iteration: int
    ) -> tuple | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state, next_node_id FROM checkpoint_history "
                "WHERE owner = $1 AND checkpoint_id = $2 AND iteration = $3",
                owner,
                checkpoint_id,
                iteration,
            )
            if row is None:
                return None
            return (row["state"], row["next_node_id"])
