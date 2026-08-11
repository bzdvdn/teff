"""PostgreSQL checkpointing — requires ``asyncpg`` (``teff[pg-checkpoint]``)."""

import json
import time

from teff.checkpoint.base import DEFAULT_OWNER, Checkpoint, Checkpointer


class PGCheckpointer(Checkpointer):
    """Store checkpoints in a PostgreSQL table.

    Requires ``asyncpg`` (install via ``teff[pg-checkpoint]``). The
    table ``checkpoints`` is created lazily on first use.  Connections are
    drawn from a lazily-created async connection pool, so checkpoint saves
    reuse warm connections instead of paying the handshake per operation.

    Args:
        dsn: PostgreSQL connection string.
        table: Table name (default ``"checkpoints"``).
        pool_size: Maximum pooled connections (default 5).
    """

    def __init__(self, dsn: str, table: str = "checkpoints", pool_size: int = 5):
        import importlib.util

        if importlib.util.find_spec("asyncpg") is None:
            raise ImportError("install asyncpg for PGCheckpointer")
        self._dsn = dsn
        self._table = table
        self._pool_size = max(1, pool_size)
        self._pool = None

    async def _ensure_pool(self):
        """Lazily create the connection pool and the table."""
        if self._pool is None:
            import asyncpg

            pool = await asyncpg.create_pool(
                self._dsn, min_size=1, max_size=self._pool_size
            )
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        owner TEXT NOT NULL DEFAULT 'default',
                        checkpoint_id TEXT NOT NULL,
                        state JSONB NOT NULL,
                        next_node_id TEXT,
                        iteration INTEGER NOT NULL,
                        updated_at DOUBLE PRECISION,
                        PRIMARY KEY (owner, checkpoint_id)
                    )
                    """
                )
            self._pool = pool
        return self._pool

    async def close(self) -> None:
        """Close the connection pool (idempotent)."""
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    async def save(
        self,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table} (owner, checkpoint_id, state, next_node_id, iteration, updated_at)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6)
                ON CONFLICT(owner, checkpoint_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    next_node_id = EXCLUDED.next_node_id,
                    iteration = EXCLUDED.iteration,
                    updated_at = EXCLUDED.updated_at
                """,
                owner,
                checkpoint_id,
                json.dumps(checkpoint.state, ensure_ascii=False),
                checkpoint.next_node_id,
                checkpoint.iteration,
                time.time(),
            )

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT state, next_node_id, iteration FROM {self._table} "
                f"WHERE owner = $1 AND checkpoint_id = $2",
                owner,
                checkpoint_id,
            )
            if row is None:
                return None
            return Checkpoint(
                state=json.loads(row["state"]),
                next_node_id=row["next_node_id"],
                iteration=row["iteration"],
            )

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE owner = $1 AND checkpoint_id = $2",
                owner,
                checkpoint_id,
            )

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT checkpoint_id FROM {self._table} "
                f"WHERE owner = $1 ORDER BY checkpoint_id",
                owner,
            )
            return [r["checkpoint_id"] for r in rows]

    async def cleanup(
        self,
        *,
        owner: str | None = None,
        max_age: float | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Delete stale checkpoints; returns how many were removed."""
        if max_age is None and keep_last is None:
            return 0
        removed = 0
        now = time.time()
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if owner is not None:
                owners = [owner]
            else:
                rows = await conn.fetch(f"SELECT DISTINCT owner FROM {self._table}")
                owners = [r["owner"] for r in rows]
            for own in owners:
                if max_age is not None:
                    result = await conn.execute(
                        f"DELETE FROM {self._table} WHERE owner = $1 AND "
                        f"COALESCE(updated_at, 0) < $2",
                        own,
                        now - max_age,
                    )
                    tag = result.split(" ", 1)
                    removed += int(tag[1]) if len(tag) == 2 else 0
                if keep_last is not None:
                    stale = await conn.fetch(
                        f"SELECT checkpoint_id FROM {self._table} WHERE owner = $1 "
                        f"ORDER BY COALESCE(updated_at, 0) DESC OFFSET $2",
                        own,
                        keep_last,
                    )
                    for row in stale:
                        await conn.execute(
                            f"DELETE FROM {self._table} "
                            f"WHERE owner = $1 AND checkpoint_id = $2",
                            own,
                            row["checkpoint_id"],
                        )
                        removed += 1
        return removed
