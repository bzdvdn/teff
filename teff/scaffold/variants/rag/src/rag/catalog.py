"""Document catalog — a small RAG store over ``data/documents/``.

Documents are embedded lazily on the first search, so building the catalog
never touches the network; only an actual query requires a configured
embedding provider.

HOW TO EXTEND
    * Drop ``.txt`` / ``.md`` / ``.csv`` files into ``data/documents/`` and
      they are indexed automatically on the next search (or by the Celery
      beat re-ingest when the ``celery`` variant is enabled).
    * Swap the vector store: build it in ``src/rag/wiring.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from teff.rag.stores import InMemoryVectorStore
from teff.rag.tool import load_documents_csv, load_documents_txt


@dataclass
class IngestReport:
    """Outcome of one ingestion run.

    Attributes:
        queued: Documents known to the catalog (parsed, not yet embedded).
        added:  Documents newly embedded+stored in this call.
        batches: Embedding HTTP calls made this call (``ceil(added / batch_size)``).
        stored: Rows currently resident in the vector store.
    """

    queued: int = 0
    added: int = 0
    batches: int = 0
    stored: int = 0


class DocumentCatalog:
    """Retrieval over a folder of text/CSV documents.

    Args:
        embedder: Anything exposing ``async embed(text) -> list[float]``.
        store: :class:`~teff.rag.VectorStore` (defaults to in-memory).
        top_k: Default number of results per search.
    """

    def __init__(self, embedder, store=None, top_k: int = 3):
        self.embedder = embedder
        self.store = store or InMemoryVectorStore(dim=768)
        self.top_k = top_k
        self._docs: list[tuple[str, dict]] = []
        self._ingested = 0

    def add_file(self, path: str) -> int:
        """Queue one text/CSV file (no network) and return the rows added.

        CSV rows use the ``text`` column as the embeddable document; all
        other columns become metadata.  Text files become one document per
        file, with the full text stored as metadata so search results can
        render snippets.
        """
        if path.endswith(".csv"):
            docs = load_documents_csv(path, text_column="text")
        else:
            docs = load_documents_txt(path)
        enriched = [(text, {**meta, "text": text}) for text, meta in docs]
        before = len(self._docs)
        self._docs.extend(enriched)
        return len(self._docs) - before

    def add_documents(self, docs: list[tuple[str, dict]]) -> None:
        """Queue raw ``(text, metadata)`` documents."""
        self._docs.extend((text, {**meta, "text": text}) for text, meta in docs)

    @property
    def size(self) -> int:
        """Documents currently known to the catalog (parsed or embedded)."""
        return len(self._docs)

    @property
    def stored(self) -> int:
        """Rows resident in the vector store (embedded)."""
        return self._ingested

    def resume(self) -> None:
        """Adopt rows a durable store already holds (e.g. indexed by a worker).

        A fresh process starts with ``_ingested == 0``, which would re-embed
        everything on the first search.  If the backing store is persistent
        and already populated, treat those rows as embedded so we never
        duplicate work across processes.  Tolerant of stores without a
        synchronous ``count`` (in-memory stores simply stay unchanged).
        """
        sync = getattr(self.store, "count_sync", None)
        if sync is None:
            return
        try:
            self._ingested = min(int(sync()), len(self._docs))
        except ValueError:
            pass

    async def _ensure_seeded(self) -> None:
        # Lazy fallback: a search against an empty store triggers ingestion,
        # so the graph works even when nobody pre-loaded via CLI/API.
        if self._ingested < len(self._docs):
            await self.ingest()

    async def ingest(self, batch_size: int = 250) -> IngestReport:
        """Embed queued-but-not-yet-stored documents into the vector store.

        Documents are embedded in *batch_size* chunks (one ``embed_many``
        call per chunk) and appended to the store.  Safe to call repeatedly:
        already-embedded rows are skipped.
        """
        pending = self._docs[self._ingested :]
        batches = 0
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            vectors = await self.embedder.embed_many([text for text, _ in chunk])
            await self.store.add(
                [
                    (f"doc_{self._ingested + start + i}", vectors[i], meta)
                    for i, (_text, meta) in enumerate(chunk)
                ]
            )
            batches += 1
        self._ingested += len(pending)
        return IngestReport(
            queued=len(self._docs),
            added=len(pending),
            batches=batches,
            stored=self._ingested,
        )

    async def rebuild(self, batch_size: int = 250) -> IngestReport:
        """Clear the store and re-embed every queued document (full refresh)."""
        await self.store.clear()
        self._ingested = 0
        return await self.ingest(batch_size=batch_size)

    async def search(self, query: str, top_k: int | None = None) -> str:
        """Search the catalog; return formatted ranked snippets."""
        await self._ensure_seeded()
        query_vector = await self.embedder.embed(query)
        results = await self.store.search(query_vector, k=top_k or self.top_k)
        if not results:
            return "Nothing found in the document catalog."
        lines = [
            f"[{i}] (score: {score:.3f}) {meta.get('text', doc_id)[:200]}"
            for i, (doc_id, score, meta) in enumerate(results, start=1)
        ]
        return "\n\n".join(lines)

    async def find_similar(self, text: str, top_k: int = 3) -> str:
        """Find documents similar to *text*, reporting similarity scores."""
        await self._ensure_seeded()
        query_vector = await self.embedder.embed(text)
        results = await self.store.search(query_vector, k=top_k)
        if not results:
            return "Nothing found."
        lines = [
            f"{i}. (similarity: {score:.2f}) {meta.get('text', doc_id)[:120]}"
            for i, (doc_id, score, meta) in enumerate(results, start=1)
        ]
        return "\n".join(lines)
