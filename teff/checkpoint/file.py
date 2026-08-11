"""JSON-file checkpointing — zero dependencies, atomic via tempfile + rename."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

from teff.checkpoint.base import (
    DEFAULT_OWNER,
    Checkpoint,
    Checkpointer,
    checkpoint_from_dict,
    checkpoint_to_dict,
)


class JSONFileCheckpointer(Checkpointer):
    """Store checkpoints as one JSON file per (owner, checkpoint ID).

    Writes go to a temp file in the same directory and are atomically
    renamed over the target, so a crash never leaves a corrupt file.
    Each *owner* gets its own subdirectory, so IDs only need to be
    unique within an owner.  See :class:`~teff.checkpoint.Checkpointer`
    for how to pick an owner.
    """

    def __init__(self, directory: str, suffix: str = ".json"):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._suffix = suffix

    def _path(self, checkpoint_id: str, owner: str = DEFAULT_OWNER) -> Path:
        safe = checkpoint_id.replace(os.sep, "_").replace("/", "_")
        owner_dir = self._directory / self._safe_owner(owner)
        owner_dir.mkdir(parents=True, exist_ok=True)
        return owner_dir / f"{safe}{self._suffix}"

    @staticmethod
    def _safe_owner(owner: str) -> str:
        return owner.replace(os.sep, "_").replace("/", "_").replace(".", "_")

    async def save(
        self,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        *,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        target = self._path(checkpoint_id, owner)
        tmp = target.with_suffix(f"{self._suffix}.tmp")

        def _save() -> None:
            tmp.write_text(
                json.dumps(checkpoint_to_dict(checkpoint), ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, target)

        await asyncio.to_thread(_save)

    async def load(
        self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER
    ) -> Checkpoint | None:
        path = self._path(checkpoint_id, owner)

        def _load() -> Checkpoint | None:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return checkpoint_from_dict(data)

        return await asyncio.to_thread(_load)

    async def delete(self, checkpoint_id: str, *, owner: str = DEFAULT_OWNER) -> None:
        path = self._path(checkpoint_id, owner)

        def _delete() -> None:
            if path.exists():
                path.unlink()

        await asyncio.to_thread(_delete)

    async def list(self, owner: str = DEFAULT_OWNER) -> list[str]:
        """Return all checkpoint IDs persisted for *owner*."""
        base = self._directory / self._safe_owner(owner)

        def _list() -> list[str]:
            if not base.exists():
                return []
            return sorted(
                p.name[: -len(self._suffix)]
                for p in base.glob(f"*{self._suffix}")
                if not p.name.endswith(f"{self._suffix}.tmp")
            )

        return await asyncio.to_thread(_list)

    def _owners(self) -> List[str]:
        if not self._directory.exists():
            return []
        return sorted(p.name for p in self._directory.iterdir() if p.is_dir())

    def _owner_checkpoints(self, owner: str) -> List[Tuple[str, float]]:
        """Return ``(checkpoint_id, mtime)`` pairs for one owner."""
        base = self._directory / self._safe_owner(owner)
        if not base.exists():
            return []
        pairs = []
        for p in base.glob(f"*{self._suffix}"):
            if p.name.endswith(f"{self._suffix}.tmp"):
                continue
            pairs.append((p.name[: -len(self._suffix)], p.stat().st_mtime))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs

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

        def _cleanup() -> int:
            removed = 0
            owners = [owner] if owner is not None else self._owners()
            now = time.time()
            for own in owners:
                pairs = self._owner_checkpoints(own)
                to_delete: List[str] = []
                for idx, (cid, mtime) in enumerate(pairs):
                    if max_age is not None and now - mtime > max_age:
                        to_delete.append(cid)
                    elif keep_last is not None and idx >= keep_last:
                        to_delete.append(cid)
                for cid in to_delete:
                    path = self._path(cid, own)
                    if path.exists():
                        path.unlink()
                        removed += 1
            return removed

        return await asyncio.to_thread(_cleanup)
