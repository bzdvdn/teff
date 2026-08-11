"""SQLite vector store — file persistence with zero extra dependencies."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from teff.rag.base import VectorStore, cosine_similarity, finalize_results


class SQLiteVectorStore(VectorStore):
    """File-persistent vector store backed by SQLite (stdlib only).

    Vectors are stored as JSON blobs in a local ``.db`` file. Search is a
    brute-force cosine similarity scan over all rows — suitable for small
    to medium collections where you want persistence without installing a
    heavy vector database.

    All database work runs in a worker thread behind a lock, so searches and
    writes never block the event loop.

    Args:
        path: Path to the SQLite database file.
        dim: Vector dimensionality (used as a sanity check on add).
    """

    def __init__(self, path: str = "./vectors.db", dim: int | None = None):
        self.path = path
        self.dim = dim
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "id TEXT PRIMARY KEY, vector TEXT NOT NULL, metadata TEXT NOT NULL)"
        )

    async def _run(self, fn, *args, **kwargs):
        """Run a sync DB call in a worker thread, serialised by the lock."""
        def _call():
            with self._lock:
                return fn(*args, **kwargs)

        return await asyncio.to_thread(_call)

    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        rows = []
        for vid, vec, meta in vectors:
            if self.dim is not None and len(vec) != self.dim:
                msg = f"vector for '{vid}' has dim {len(vec)}, expected {self.dim}"
                raise ValueError(msg)
            rows.append((vid, json.dumps(vec), json.dumps(meta, ensure_ascii=False)))

        def _add():
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

        await self._run(_add)

    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        def _search():
            rows = self._conn.execute(
                "SELECT id, vector, metadata FROM vectors"
            ).fetchall()
            candidates = [
                (
                    vid,
                    cosine_similarity(query, json.loads(vec_json)),
                    json.loads(meta_json),
                )
                for vid, vec_json, meta_json in rows
            ]
            return finalize_results(candidates, k, filter, hybrid, query_text)

        return await self._run(_search)

    async def delete(self, ids: list[str]) -> None:
        def _delete():
            self._conn.executemany(
                "DELETE FROM vectors WHERE id = ?", [(i,) for i in ids]
            )
            self._conn.commit()

        await self._run(_delete)

    async def count(self) -> int:
        def _count():
            return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

        return await self._run(_count)

    def count_sync(self) -> int:
        """Synchronous count — lets a fresh process adopt an existing store."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        def _entries():
            rows = self._conn.execute(
                "SELECT id, metadata FROM vectors ORDER BY id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [(r[0], json.loads(r[1])) for r in rows]

        return await self._run(_entries)

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        if not ids:
            return []

        def _get():
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT id, metadata FROM vectors WHERE id IN ({marks})", ids
            ).fetchall()
            return [(r[0], json.loads(r[1])) for r in rows]

        return await self._run(_get)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        def _update():
            row = self._conn.execute(
                "SELECT metadata FROM vectors WHERE id = ?", (id,)
            ).fetchone()
            if row is None:
                return
            merged = {**json.loads(row[0]), **metadata}
            self._conn.execute(
                "UPDATE vectors SET metadata = ? WHERE id = ?",
                (json.dumps(merged, ensure_ascii=False), id),
            )
            self._conn.commit()

        await self._run(_update)

    async def clear(self) -> None:
        def _clear():
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()

        await self._run(_clear)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
