import pytest


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_abc_enforces_contract(self):
        from teff.rag import VectorStore

        with pytest.raises(TypeError):

            class Bad(VectorStore):
                pass

            Bad()

    @pytest.mark.asyncio
    async def test_inmemory_store_and_search(self):
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        await store.add([("d1", [1, 0, 0, 0], {"text": "hello"})])
        results = await store.search([1, 0, 0, 0], k=1)
        assert results[0][0] == "d1"
        assert results[0][1] > 0.99

    @pytest.mark.asyncio
    async def test_empty_search_returns_empty(self):
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        results = await store.search([1, 0, 0, 0], k=5)
        assert results == []

    def test_match_filter_semantics(self):
        from teff.rag.base import match_filter

        meta = {"category": "news", "tags": ["a", "b"], "views": 3}
        assert match_filter(meta, None)
        assert match_filter(meta, {})
        assert match_filter(meta, {"category": "news"})
        assert not match_filter(meta, {"category": "tech"})
        assert match_filter(meta, {"views": 3})
        assert not match_filter(meta, {"views": 4})
        assert not match_filter(meta, {"missing": "x"})
        assert match_filter(meta, {"tags": ["a"]})
        assert not match_filter(meta, {"tags": ["x"]})
        assert match_filter(meta, {"$and": [{"category": "news"}, {"views": 3}]})
        assert not match_filter(meta, {"$and": [{"category": "news"}, {"views": 4}]})
        assert match_filter(meta, {"$or": [{"category": "tech"}, {"views": 3}]})
        assert not match_filter(meta, {"$or": [{"category": "tech"}, {"views": 4}]})
        assert match_filter(
            meta,
            {"$and": [{"category": "news"}, {"$or": [{"views": 4}, {"views": 3}]}]},
        )

    @pytest.mark.asyncio
    async def test_inmemory_filter_search(self):
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        await store.add(
            [
                ("d1", [1, 0, 0, 0], {"category": "news", "text": "a"}),
                ("d2", [0.9, 0.1, 0, 0], {"category": "tech", "text": "b"}),
                ("d3", [0.8, 0.2, 0, 0], {"category": "news", "text": "c"}),
            ]
        )
        res = await store.search([1, 0, 0, 0], k=5, filter={"category": "news"})
        assert {r[0] for r in res} == {"d1", "d3"}
        res = await store.search(
            [1, 0, 0, 0],
            k=5,
            filter={"$or": [{"category": "news"}, {"category": "tech"}]},
        )
        assert len(res) == 3

    @pytest.mark.asyncio
    async def test_inmemory_hybrid_ranking(self):
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        await store.add(
            [
                ("d1", [1, 0, 0, 0], {"text": "a recipe for sourdough bread"}),
                ("d2", [0.99, 0.01, 0, 0], {"text": "python python python"}),
            ]
        )
        plain = await store.search([1, 0, 0, 0], k=5)
        assert plain[0][0] == "d1"
        hybrid = await store.search([1, 0, 0, 0], k=5, hybrid=True, query_text="python")
        assert hybrid[0][0] == "d2"

    @pytest.mark.asyncio
    async def test_inmemory_extended_ops(self):
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        await store.add(
            [
                ("a", [1, 0, 0, 0], {"category": "x"}),
                ("b", [0, 1, 0, 0], {"category": "y"}),
                ("c", [0, 0, 1, 0], {"category": "z"}),
            ]
        )
        assert await store.count() == 3
        lst = await store.entries(limit=2, offset=1)
        assert len(lst) == 2
        got = await store.get(["a", "nope"])
        assert [i for i, _ in got] == ["a"]
        await store.update_metadata("a", {"extra": 1})
        got = await store.get(["a"])
        assert got[0][1] == {"category": "x", "extra": 1}
        await store.clear()
        assert await store.count() == 0
        assert await store.search([1, 0, 0, 0], k=5) == []


class TestEmbedder:
    def test_ollama_needs_no_api_key(self):
        from teff.rag import Embedder

        e = Embedder(provider="ollama", model="nomic-embed-text")
        assert e._api_key == ""
        assert e._base_url == "http://localhost:11434/v1"

    def test_openai_requires_api_key(self, monkeypatch):
        from teff.rag import Embedder

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="API key"):
            Embedder(provider="openai")

    @pytest.mark.parametrize(
        "provider,env,model",
        [
            ("mistral", "MISTRAL_API_KEY", "mistral-embed"),
            ("voyage", "VOYAGE_API_KEY", "voyage-3"),
            ("jina", "JINA_API_KEY", "jina-embeddings-v3"),
            (
                "together",
                "TOGETHER_API_KEY",
                "togethercomputer/m2-bert-80M-8k-retrieval",
            ),
            ("groq", "GROQ_API_KEY", "nomic-embed-text-v1.5"),
        ],
    )
    def test_new_provider_defaults(self, monkeypatch, provider, env, model):
        from teff.rag import Embedder

        monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
        monkeypatch.delenv(f"{provider.upper()}_BASE_URL", raising=False)
        monkeypatch.setenv(env, "test-key")
        e = Embedder(provider=provider)
        assert e.model == model
        assert e._api_key == "test-key"

    @pytest.mark.parametrize(
        "provider,env",
        [
            ("mistral", "MISTRAL_API_KEY"),
            ("voyage", "VOYAGE_API_KEY"),
            ("jina", "JINA_API_KEY"),
            ("together", "TOGETHER_API_KEY"),
            ("groq", "GROQ_API_KEY"),
        ],
    )
    def test_new_provider_requires_api_key(self, monkeypatch, provider, env):
        from teff.rag import Embedder

        monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
        monkeypatch.delenv(env, raising=False)
        monkeypatch.delenv(f"{provider.upper()}_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="API key"):
            Embedder(provider=provider)

    def test_from_config_uses_explicit_values(self, monkeypatch):
        from teff.rag.embedder import embedder_from_config

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        e = embedder_from_config(
            {
                "embedder": {
                    "provider": "ollama",
                    "model": "my-model",
                    "base_url": "http://explicit:11434/v1",
                }
            }
        )
        assert e.model == "my-model"
        assert e._base_url == "http://explicit:11434/v1"

    def test_from_config_inherits_provider_base_url(self, monkeypatch):
        from teff.provider import Provider, ProviderRegistry
        from teff.rag.embedder import embedder_from_config

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        reg = ProviderRegistry()
        reg.register(
            Provider(
                name="my-openai",
                type="openai_compatible",
                base_url="http://embed:8080/v1",
            )
        )
        e = embedder_from_config({"embedder": {"provider": "my-openai"}}, providers=reg)
        assert e._base_url == "http://embed:8080/v1"

    def test_from_config_ollama_adds_v1(self, monkeypatch):
        from teff.provider import ProviderRegistry
        from teff.rag.embedder import embedder_from_config

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        reg = ProviderRegistry.from_presets("ollama")
        e = embedder_from_config({"embedder": {"provider": "ollama"}}, providers=reg)
        # ollama's provider base_url has no /v1; the embedder appends it.
        assert e._base_url == "http://localhost:11434/v1"

    def test_from_config_ollama_remote_host_inherited(self, monkeypatch):
        from teff.provider import Provider, ProviderRegistry
        from teff.rag.embedder import embedder_from_config

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        reg = ProviderRegistry()
        reg.register(
            Provider(
                name="ollama-remote",
                type="ollama",
                base_url="http://ollama-host:11434",
            )
        )
        e = embedder_from_config(
            {"embedder": {"provider": "ollama-remote"}}, providers=reg
        )
        assert e._base_url == "http://ollama-host:11434/v1"

    def test_from_config_explicit_base_url_wins(self, monkeypatch):
        from teff.provider import ProviderRegistry
        from teff.rag.embedder import embedder_from_config

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        reg = ProviderRegistry.from_presets("ollama")
        e = embedder_from_config(
            {
                "embedder": {
                    "provider": "ollama",
                    "base_url": "http://explicit:11434/v1",
                }
            },
            providers=reg,
        )
        assert e._base_url == "http://explicit:11434/v1"

    def test_rag_tool_config_resolves_default_model(self, monkeypatch):
        from teff.rag import RAGTool

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        rag = RAGTool(
            {
                "embedder": {"provider": "mistral"},
                "store": {"type": "in_memory", "dim": 8},
            }
        )
        assert rag.embedder.model == "mistral-embed"
        assert rag.embedder.provider == "mistral"


class TestSQLiteVectorStore:
    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        db = tmp_path / "nested" / "dir" / "v.db"
        s = SQLiteVectorStore(path=str(db), dim=3)
        assert db.is_file()
        await s.add([("d1", [1.0, 0, 0], {"text": "hello"})])
        results = await s.search([1.0, 0, 0], k=1)
        assert results[0][0] == "d1"
        s.close()

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        db = str(tmp_path / "v.db")
        s1 = SQLiteVectorStore(path=db, dim=3)
        await s1.add([("d1", [1.0, 0, 0], {"text": "hello"})])

        s2 = SQLiteVectorStore(path=db, dim=3)
        results = await s2.search([1.0, 0, 0], k=1)
        assert results[0][0] == "d1"
        assert results[0][2]["text"] == "hello"
        s1.close()
        s2.close()

    @pytest.mark.asyncio
    async def test_ranked_search(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        await s.add(
            [
                ("a", [1.0, 0, 0], {}),
                ("b", [0.0, 1.0, 0], {}),
            ]
        )
        results = await s.search([1.0, 0, 0], k=2)
        assert [r[0] for r in results] == ["a", "b"]
        s.close()

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        with pytest.raises(ValueError, match="dim"):
            await s.add([("d1", [1.0, 0], {})])
        s.close()

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        await s.add([("d1", [1.0, 0, 0], {})])
        await s.delete(["d1"])
        assert await s.search([1.0, 0, 0], k=5) == []
        s.close()

    @pytest.mark.asyncio
    async def test_filter_search(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=4)
        await s.add(
            [
                ("d1", [1, 0, 0, 0], {"category": "news", "text": "alpha"}),
                ("d2", [0.9, 0.1, 0, 0], {"category": "tech", "text": "beta"}),
            ]
        )
        res = await s.search([1, 0, 0, 0], k=5, filter={"category": "news"})
        assert [r[0] for r in res] == ["d1"]
        res = await s.search([1, 0, 0, 0], k=5, filter={"category": ["tech", "sports"]})
        assert [r[0] for r in res] == ["d2"]
        res = await s.search(
            [1, 0, 0, 0],
            k=5,
            filter={"$or": [{"category": "news"}, {"category": "tech"}]},
        )
        assert len(res) == 2
        s.close()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=4)
        await s.add(
            [
                ("d1", [1, 0, 0, 0], {"text": "a recipe for sourdough bread"}),
                ("d2", [0.99, 0.01, 0, 0], {"text": "python python python"}),
            ]
        )
        plain = await s.search([1, 0, 0, 0], k=5)
        assert plain[0][0] == "d1"
        hybrid = await s.search([1, 0, 0, 0], k=5, hybrid=True, query_text="python")
        assert hybrid[0][0] == "d2"
        s.close()

    @pytest.mark.asyncio
    async def test_extended_ops(self, tmp_path):
        from teff.rag.stores import SQLiteVectorStore

        s = SQLiteVectorStore(path=str(tmp_path / "v.db"), dim=3)
        await s.add(
            [
                ("a", [1, 0, 0], {"category": "x"}),
                ("b", [0, 1, 0], {"category": "y"}),
            ]
        )
        assert await s.count() == 2
        assert [i for i, _ in await s.entries(limit=1, offset=0)] == ["a"]
        assert [i for i, _ in await s.get(["a", "b", "z"])] == ["a", "b"]
        await s.update_metadata("a", {"extra": 1})
        got = await s.get(["a"])
        assert got[0][1] == {"category": "x", "extra": 1}
        await s.clear()
        assert await s.count() == 0
        s.close()

    def test_rag_tool_with_sqlite_store(self, tmp_path):
        from teff.rag import RAGTool

        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {"type": "sqlite", "path": str(tmp_path / "rag.db"), "dim": 8},
                "documents": [{"id": "d1", "text": "hello"}],
            }
        )
        assert type(rag.store).__name__ == "SQLiteVectorStore"

    def test_unknown_store_type_raises(self):
        from teff.rag import RAGTool

        with pytest.raises(ValueError, match="unsupported store type"):
            RAGTool(
                {
                    "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                    "store": {"type": "bogus"},
                }
            )

    def test_external_store_missing_dep_raises(self):
        import importlib.util

        from teff.rag import RAGTool

        deps = {
            "chroma": "chromadb",
            "qdrant": "qdrant_client",
            "pgvector": "asyncpg",
        }
        for stype, dep in deps.items():
            if importlib.util.find_spec(dep) is not None:
                pytest.skip(f"{dep} is installed")
            with pytest.raises(ImportError):
                RAGTool(
                    {
                        "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                        "store": {"type": stype},
                    }
                )

    def test_sqlite_store_from_config(self, tmp_path):
        from teff.rag import RAGTool

        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {
                    "type": "sqlite",
                    "path": str(tmp_path / "rag.db"),
                    "dim": 8,
                },
            }
        )
        assert type(rag.store).__name__ == "SQLiteVectorStore"


class TestFAISSVectorStore:
    def test_import_or_skip(self):
        pytest.importorskip("faiss")

    @pytest.mark.asyncio
    async def test_search_filter_and_extended_ops(self, tmp_path):
        pytest.importorskip("faiss")
        from teff.rag.stores import FAISSVectorStore

        store = FAISSVectorStore(dim=4, path=str(tmp_path / "idx.bin"))
        await store.add(
            [
                ("a", [1, 0, 0, 0], {"category": "news", "text": "alpha"}),
                ("b", [0, 1, 0, 0], {"category": "tech", "text": "beta"}),
                ("c", [0.8, 0.2, 0, 0], {"category": "news", "text": "gamma"}),
            ]
        )
        res = await store.search([1, 0, 0, 0], k=5)
        assert res[0][0] == "a"
        assert res[0][1] > 0.99
        res = await store.search([1, 0, 0, 0], k=5, filter={"category": "news"})
        assert {r[0] for r in res} == {"a", "c"}
        hybrid = await store.search([0, 1, 0, 0], k=5, hybrid=True, query_text="beta")
        assert hybrid[0][0] == "b"
        assert await store.count() == 3
        assert {i for i, _ in await store.entries()} == {"a", "b", "c"}
        assert [i for i, _ in await store.get(["a", "z"])] == ["a"]
        await store.update_metadata("a", {"topic": "x"})
        got = await store.get(["a"])
        assert got[0][1]["topic"] == "x"
        await store.delete(["b"])
        assert [i for i, _ in await store.get(["b"])] == []
        await store.clear()
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_path):
        pytest.importorskip("faiss")
        from teff.rag.stores import FAISSVectorStore

        path = str(tmp_path / "idx.bin")
        s1 = FAISSVectorStore(dim=3, path=path)
        await s1.add([("d1", [1.0, 0, 0], {"text": "hello"})])

        s2 = FAISSVectorStore(dim=3, path=path)
        results = await s2.search([1.0, 0, 0], k=1)
        assert results[0][0] == "d1"
        assert results[0][2]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self):
        pytest.importorskip("faiss")
        from teff.rag.stores import FAISSVectorStore

        store = FAISSVectorStore(dim=3)
        with pytest.raises(ValueError, match="dim"):
            await store.add([("d1", [1.0, 0], {})])


class TestLanceVectorStore:
    def test_import_or_skip(self):
        pytest.importorskip("lancedb")

    @pytest.mark.asyncio
    async def test_search_filter_and_extended_ops(self, tmp_path):
        pytest.importorskip("lancedb")
        from teff.rag.stores import LanceVectorStore

        store = LanceVectorStore(path=str(tmp_path / "lance"), table="vectors", dim=4)
        await store.add(
            [
                ("a", [1, 0, 0, 0], {"category": "news", "text": "alpha"}),
                ("b", [0, 1, 0, 0], {"category": "tech", "text": "beta"}),
                ("c", [0.8, 0.2, 0, 0], {"category": "news", "text": "gamma"}),
            ]
        )
        res = await store.search([1, 0, 0, 0], k=5)
        assert res[0][0] == "a"
        assert res[0][1] > 0.99
        res = await store.search([1, 0, 0, 0], k=5, filter={"category": "news"})
        assert {r[0] for r in res} == {"a", "c"}
        hybrid = await store.search([0, 1, 0, 0], k=5, hybrid=True, query_text="beta")
        assert hybrid[0][0] == "b"
        assert await store.count() == 3
        assert {i for i, _ in await store.entries()} == {"a", "b", "c"}
        assert [i for i, _ in await store.get(["a", "z"])] == ["a"]
        await store.update_metadata("a", {"topic": "x"})
        got = await store.get(["a"])
        assert got[0][1]["topic"] == "x"
        await store.delete(["b"])
        assert [i for i, _ in await store.get(["b"])] == []
        await store.clear()
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_dim_mismatch_raises(self, tmp_path):
        pytest.importorskip("lancedb")
        from teff.rag.stores import LanceVectorStore

        store = LanceVectorStore(path=str(tmp_path / "lance"), dim=3)
        with pytest.raises(ValueError, match="dim"):
            await store.add([("d1", [1.0, 0], {})])


class TestMilvusVectorStore:
    def test_import_or_skip(self):
        pytest.importorskip("pymilvus")
        try:
            import milvus_lite  # noqa: F401
        except ImportError:
            pytest.skip("milvus-lite not installed")

    @pytest.mark.asyncio
    async def test_search_filter_and_extended_ops(self, tmp_path):
        pytest.importorskip("pymilvus")
        try:
            import milvus_lite  # noqa: F401
        except ImportError:
            pytest.skip("milvus-lite not installed")
        from teff.rag.stores import MilvusVectorStore

        store = MilvusVectorStore(
            uri=str(tmp_path / "milvus.db"),
            collection="test",
            dim=4,
        )
        await store.clear()
        await store.add(
            [
                ("a", [1, 0, 0, 0], {"category": "news", "text": "alpha"}),
                ("b", [0, 1, 0, 0], {"category": "tech", "text": "beta"}),
                ("c", [0.8, 0.2, 0, 0], {"category": "news", "text": "gamma"}),
            ]
        )
        res = await store.search([1, 0, 0, 0], k=5)
        assert res[0][0] == "a"
        res = await store.search([1, 0, 0, 0], k=5, filter={"category": "news"})
        assert {r[0] for r in res} == {"a", "c"}
        hybrid = await store.search([0, 1, 0, 0], k=5, hybrid=True, query_text="beta")
        assert hybrid[0][0] == "b"
        assert await store.count() == 3
        assert {i for i, _ in await store.entries()} == {"a", "b", "c"}
        assert [i for i, _ in await store.get(["a", "z"])] == ["a"]
        await store.update_metadata("a", {"topic": "x"})
        got = await store.get(["a"])
        assert got[0][1]["topic"] == "x"
        await store.delete(["b"])
        assert [i for i, _ in await store.get(["b"])] == []
        await store.clear()
        assert await store.count() == 0


class TestWeaviatePineconeConfig:
    def test_pinecone_missing_api_key_raises(self, monkeypatch):
        pytest.importorskip("pinecone")
        from teff.rag.stores import PineconeVectorStore

        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            PineconeVectorStore(api_key="")

    def test_new_store_types_mapped_in_config(self, monkeypatch):
        from teff.rag import RAGTool

        monkeypatch.setattr(
            "teff.rag.stores.FAISSVectorStore", lambda **kw: "faiss-store"
        )
        monkeypatch.setattr(
            "teff.rag.stores.LanceVectorStore", lambda **kw: "lance-store"
        )
        monkeypatch.setattr(
            "teff.rag.stores.MilvusVectorStore", lambda **kw: "milvus-store"
        )
        monkeypatch.setattr(
            "teff.rag.stores.WeaviateVectorStore", lambda **kw: "weaviate-store"
        )
        monkeypatch.setattr(
            "teff.rag.stores.PineconeVectorStore", lambda **kw: "pinecone-store"
        )

        embedder = {"provider": "ollama", "model": "nomic-embed-text"}
        for stype, expected in [
            ("faiss", "faiss-store"),
            ("lance", "lance-store"),
            ("lancedb", "lance-store"),
            ("milvus", "milvus-store"),
            ("weaviate", "weaviate-store"),
            ("pinecone", "pinecone-store"),
        ]:
            rag = RAGTool({"embedder": embedder, "store": {"type": stype, "dim": 8}})
            assert rag.store == expected


class TestChromaVectorStore:
    def test_import_or_skip(self):
        pytest.importorskip("chromadb")

    @pytest.mark.asyncio
    async def test_filter_search_and_extended_ops(self, tmp_path):
        from teff.rag.stores import ChromaVectorStore

        store = ChromaVectorStore(path=str(tmp_path / "chroma"), collection="test")
        await store.add(
            [
                ("a", [1, 0, 0, 0], {"category": "news", "text": "alpha"}),
                ("b", [0, 1, 0, 0], {"category": "tech", "text": "beta"}),
            ]
        )
        res = await store.search([1, 0, 0, 0], k=5, filter={"category": "news"})
        assert [r[0] for r in res] == ["a"]
        res = await store.search([1, 0, 0, 0], k=5, filter={"category": ["tech"]})
        assert [r[0] for r in res] == ["b"]
        assert await store.count() == 2
        assert {i for i, _ in await store.entries()} == {"a", "b"}
        assert [i for i, _ in await store.get(["a", "z"])] == ["a"]
        await store.update_metadata("a", {"topic": "x"})
        got = await store.get(["a"])
        assert got[0][1]["category"] == "news"
        assert got[0][1]["topic"] == "x"
        await store.clear()
        assert await store.count() == 0


class TestRAGTool:
    def test_constructs_with_all_deps(self):
        from teff.rag import Chunker, Embedder, RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        embedder = Embedder.__new__(Embedder)
        embedder._api_key = "test"
        embedder._base_url = "http://test"
        embedder.provider = "test"
        embedder.model = "test"
        chunker = Chunker(strategy="fixed", chunk_size=50)
        rag = RAGTool(store=store, embedder=embedder, chunker=chunker)
        assert rag.name == "rag"
        assert "search" in rag.description.lower()

    def test_rag_tool_custom_name(self):
        from teff.rag import RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(768)
        rag = RAGTool(store=store, embedder=None, name="rag_docs")
        assert rag.name == "rag_docs"

        cfg = {
            "name": "kb_main",
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": {"type": "in_memory", "dim": 768},
        }
        rag = RAGTool(cfg)
        assert rag.name == "kb_main"

    def test_config_with_inline_documents(self, tmp_path):
        from teff.rag import RAGTool

        cfg = {
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": {"type": "in_memory", "dim": 8},
            "documents": [
                {"id": "d1", "topic": "a", "text": "first doc text"},
                {"id": "d2", "topic": "b", "text": "second doc text"},
            ],
        }
        rag = RAGTool(cfg)
        assert rag.store.dim == 8
        assert rag.embedder.provider == "ollama"
        assert rag._documents == [
            ("first doc text", {"id": "d1", "topic": "a"}),
            ("second doc text", {"id": "d2", "topic": "b"}),
        ]

    def test_config_with_csv_file(self, tmp_path):
        from teff.rag import RAGTool

        csv_path = tmp_path / "docs.csv"
        csv_path.write_text('id,topic,text\nd1,a,hello world\nd2,b,"two, words"\n')
        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {"type": "in_memory", "dim": 8},
                "documents": str(csv_path),
            }
        )
        assert rag._documents == [
            ("hello world", {"id": "d1", "topic": "a"}),
            ("two, words", {"id": "d2", "topic": "b"}),
        ]

    def test_config_with_csv_dict(self, tmp_path):
        from teff.rag import RAGTool

        csv_path = tmp_path / "docs.tsv"
        csv_path.write_text("id\tcontent\nd1\thello\n")
        rag = RAGTool(
            {
                "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                "store": {"type": "in_memory", "dim": 8},
                "documents": {
                    "file": str(csv_path),
                    "text_column": "content",
                    "delimiter": "\t",
                },
            }
        )
        assert rag._documents == [("hello", {"id": "d1"})]

    def test_lazy_seeding_seeds_once(self, tmp_path):
        from teff.rag import RAGTool
        from teff.rag.stores import InMemoryVectorStore

        class FakeEmbedder:
            async def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0, 0.0]

            async def embed_many(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            documents=[("hello world", {"id": "d1"})],
        )
        import asyncio

        async def go():
            await rag._ensure_seeded()
            await rag._ensure_seeded()

        asyncio.run(go())
        assert len(store._vectors) == 1

    def test_mixed_loader_config(self, tmp_path):
        from teff.rag import RAGTool

        (tmp_path / "a.txt").write_text("first text file")
        (tmp_path / "b.txt").write_text("second text file")
        csv_path = tmp_path / "c.csv"
        csv_path.write_text("id,text\nd1,csv row\n")
        cfg = {
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": {"type": "in_memory", "dim": 8},
            "documents": [
                {"type": "txt", "path": str(tmp_path / "*.txt")},
                {"type": "csv", "path": str(csv_path)},
            ],
        }
        rag = RAGTool(cfg)
        texts = [t for t, _ in rag._documents]
        assert "first text file" in texts
        assert "second text file" in texts
        assert "csv row" in texts

    def test_unknown_document_type_raises(self):
        from teff.rag import RAGTool

        with pytest.raises(ValueError, match="unsupported document type"):
            RAGTool(
                {
                    "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
                    "store": {"type": "in_memory", "dim": 8},
                    "documents": [{"type": "nope", "path": "x"}],
                }
            )

    def test_config_rag_options(self):
        from teff.rag import RAGTool

        cfg = {
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": {"type": "in_memory", "dim": 8},
            "filters": {"topic": "a"},
            "similarity_threshold": 0.7,
            "max_tokens": 512,
            "hybrid": True,
            "parent_chunks": True,
            "parent_retrieval": True,
        }
        rag = RAGTool(cfg)
        assert rag._filters == {"topic": "a"}
        assert rag._threshold == 0.7
        assert rag._max_tokens == 512
        assert rag._hybrid is True
        assert rag._parent_chunks is True
        assert rag._parent_retrieval is True

    def test_constructor_options(self):
        from teff.rag import RAGTool
        from teff.rag.stores import InMemoryVectorStore

        rag = RAGTool(
            store=InMemoryVectorStore(4),
            embedder=None,
            filter={"t": "a"},
            similarity_threshold=0.5,
            max_tokens=100,
            hybrid=True,
            parent_chunks=True,
            parent_retrieval=True,
        )
        assert rag._filters == {"t": "a"}
        assert rag._threshold == 0.5
        assert rag._max_tokens == 100
        assert rag._hybrid and rag._parent_chunks and rag._parent_retrieval

    def test_arun_similarity_threshold(self):
        import asyncio

        from teff.rag import Chunker, RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=_FakeEmbedder(),
            chunker=Chunker(strategy="fixed", chunk_size=50),
        )
        asyncio.run(rag.add_document("hello world", {"id": "d1"}))
        assert asyncio.run(rag.arun("find this", k=5, similarity_threshold=1.5)) == ""
        res = asyncio.run(rag.arun("find this", k=5, similarity_threshold=0.5))
        assert "hello world" in res

    def test_arun_max_tokens(self):
        import asyncio

        from teff.rag import Chunker, RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=_FakeEmbedder(),
            chunker=Chunker(strategy="fixed", chunk_size=1000),
        )
        asyncio.run(rag.add_document("x" * 400, {"id": "d1"}))
        res = asyncio.run(rag.arun("q", k=5, max_tokens=10))
        assert ("x" * 40) in res
        assert ("x" * 41) not in res

    def test_arun_uses_configured_filter(self):
        import asyncio

        from teff.rag import Chunker, RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=_FakeEmbedder(),
            chunker=Chunker(strategy="fixed", chunk_size=1000),
            filter={"topic": "a"},
        )
        asyncio.run(
            rag.add_documents(
                [
                    ("doc one", {"id": "d1", "topic": "a"}),
                    ("doc two", {"id": "d2", "topic": "b"}),
                ]
            )
        )
        res = asyncio.run(rag.arun("doc", k=5))
        assert "doc one" in res and "doc two" not in res
        res2 = asyncio.run(rag.arun("doc", k=5, filter={"topic": "b"}))
        assert "doc two" in res2 and "doc one" not in res2

    def test_parent_chunks_and_retrieval(self):
        import asyncio

        from teff.rag import Chunker, RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=_FakeEmbedder(),
            chunker=Chunker(strategy="fixed", chunk_size=10),
            parent_chunks=True,
        )
        asyncio.run(rag.add_document("parent content here", {"id": "d1"}))
        res = asyncio.run(rag.arun("q", k=5, parent_retrieval=True))
        assert "parent content here" in res
        res2 = asyncio.run(rag.arun("q", k=5))
        assert "parent content here" not in res2
        assert "parent con" in res2 or "tent here" in res2

    def test_arun_hybrid_ranking(self):
        import asyncio

        from teff.rag import Chunker, RAGTool
        from teff.rag.stores import InMemoryVectorStore

        store = InMemoryVectorStore(dim=4)
        rag = RAGTool(
            store=store,
            embedder=_FakeEmbedder(),
            chunker=Chunker(strategy="fixed", chunk_size=1000),
            hybrid=True,
        )
        asyncio.run(
            rag.add_documents(
                [
                    ("a recipe for sourdough bread", {"id": "d1"}),
                    ("python python python", {"id": "d2"}),
                ]
            )
        )
        res = asyncio.run(rag.arun("python", k=5))
        assert "python python python" in res
        assert res.index("python python python") < res.index("sourdough")


class TestDocumentLoaders:
    def test_csv_loader(self, tmp_path):
        from teff.rag.tool import load_documents_csv

        p = tmp_path / "d.csv"
        p.write_text("id,topic,text\nd1,a,hello world\n")
        assert load_documents_csv(str(p)) == [
            ("hello world", {"id": "d1", "topic": "a"})
        ]

    def test_txt_loader_single_and_glob(self, tmp_path):
        from teff.rag.tool import load_documents_txt

        (tmp_path / "one.txt").write_text("first")
        (tmp_path / "two.txt").write_text("second")
        assert load_documents_txt(str(tmp_path / "one.txt")) == [
            ("first", {"id": "one", "path": str(tmp_path / "one.txt")})
        ]
        assert len(load_documents_txt(str(tmp_path / "*.txt"))) == 2

    def test_pdf_loader(self, tmp_path):
        from teff.rag.tool import load_documents_pdf

        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        p = tmp_path / "d.pdf"
        with open(p, "wb") as f:
            w.write(f)
        assert load_documents_pdf(str(p)) == []

    def test_excel_loader(self, tmp_path):
        from teff.rag.tool import load_documents_excel

        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["id", "text"])
        ws.append(["e1", "row one"])
        wb.save(tmp_path / "d.xlsx")
        assert load_documents_excel(str(tmp_path / "d.xlsx")) == [
            ("row one", {"id": "e1"})
        ]

    def test_excel_loader_custom_column(self, tmp_path):
        from teff.rag.tool import load_documents_excel

        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["id", "content"])
        ws.append(["e1", "row one"])
        wb.save(tmp_path / "d.xlsx")
        docs = load_documents_excel(str(tmp_path / "d.xlsx"), text_column="content")
        assert docs == [("row one", {"id": "e1"})]

    def test_pdf_without_pypdf_raises_helpful_error(self, monkeypatch):
        import builtins

        from teff.rag.tool import load_documents_pdf

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf" or name.startswith("pypdf."):
                raise ImportError("no pypdf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="rag-pdf"):
            load_documents_pdf("whatever.pdf")


class TestPDFTool:
    def test_blank_pdf_returns_no_text(self, tmp_path):
        from teff.rag import PDFTool

        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        p = tmp_path / "d.pdf"
        with open(p, "wb") as f:
            w.write(f)
        assert PDFTool().run(str(p)) == "no text found in pdf"

    def test_missing_path_raises(self):
        from teff.rag import PDFTool

        with pytest.raises(ValueError, match="path is required"):
            PDFTool().run(path="")

    def test_config_max_chars_default(self):
        from teff.rag import PDFTool

        assert PDFTool().max_chars == 50000
        assert PDFTool({"max_chars": 10}).max_chars == 10

    def test_schema_requires_path(self):
        from teff.harness import tool_to_schema
        from teff.rag import PDFTool

        schema = tool_to_schema(PDFTool())
        assert "path" in schema["function"]["parameters"]["required"]

    def test_returns_pages_with_truncation(self, monkeypatch):
        from teff.rag import PDFTool

        docs = [("hello page", {"page": 1}), ("world page", {"page": 2})]
        monkeypatch.setattr("teff.rag.pdf_tool.load_documents_pdf", lambda path: docs)
        result = PDFTool().run("fake.pdf")
        assert "--- page 1 ---\nhello page" in result
        assert "--- page 2 ---\nworld page" in result
        short = PDFTool().run("fake.pdf", max_chars=15)
        assert len(short) <= 15


class TestImageTool:
    class _FakeVisionClient:
        def __init__(self, *a, **k):
            self.sent = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **k):
            return False

        async def post(self, url, **kwargs):
            self.sent = {
                "url": url,
                "headers": kwargs.get("headers"),
                "json": kwargs.get("json"),
            }

            class R:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "INVOICE TOTAL 100"}}]}

            return R()

    def _png(self, tmp_path, name="shot.png"):
        p = tmp_path / name
        p.write_bytes(b"\x89PNG\r\n\x1a\nfakepngdata")
        return p

    def test_defaults_ollama_llava(self, monkeypatch):
        from teff.rag import ImageTool

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        t = ImageTool({})
        assert t.provider == "ollama"
        assert t.model == "llava"
        assert t.base_url == "http://localhost:11434/v1"

    def test_config_overrides(self, monkeypatch):
        from teff.rag import ImageTool

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        t = ImageTool(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "https://x/v1",
                "api_key": "k",
            }
        )
        assert t.api_key == "k"
        assert t.model == "gpt-4o"
        assert t.base_url == "https://x/v1"

    def test_arun_posts_base64_image(self, monkeypatch, tmp_path):
        import asyncio
        import base64

        import httpx

        from teff.rag import ImageTool

        p = self._png(tmp_path)
        fake = self._FakeVisionClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)

        result = asyncio.run(ImageTool({}).arun(str(p)))
        assert result == "INVOICE TOTAL 100"
        payload = fake.sent["json"]
        assert payload["model"] == "llava"
        content = payload["messages"][0]["content"]
        assert content[0]["type"] == "text"
        image_url = content[1]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        raw = image_url.split(",", 1)[1]
        assert base64.b64decode(raw) == b"\x89PNG\r\n\x1a\nfakepngdata"

    def test_arun_uses_custom_prompt(self, monkeypatch, tmp_path):
        import asyncio

        import httpx

        from teff.rag import ImageTool

        p = self._png(tmp_path)
        fake = self._FakeVisionClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)

        asyncio.run(ImageTool({}).arun(str(p), prompt="Describe the chart"))
        text_block = fake.sent["json"]["messages"][0]["content"][0]
        assert text_block["text"] == "Describe the chart"

    def test_max_chars_truncates(self, monkeypatch, tmp_path):
        import asyncio

        import httpx

        from teff.rag import ImageTool

        p = self._png(tmp_path)
        fake = self._FakeVisionClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)
        result = asyncio.run(ImageTool({}).arun(str(p), max_chars=4))
        assert result == "INVO"

    def test_requires_path(self):
        import asyncio

        from teff.rag import ImageTool

        with pytest.raises(ValueError, match="path is required"):
            asyncio.run(ImageTool({}).arun(""))

    def test_missing_file_raises(self, tmp_path):
        import asyncio

        from teff.rag import ImageTool

        with pytest.raises(FileNotFoundError):
            asyncio.run(ImageTool({}).arun(str(tmp_path / "nope.png")))

    def test_http_error_raises(self, monkeypatch, tmp_path):
        import asyncio

        import httpx

        from teff.rag import ImageTool

        p = self._png(tmp_path)

        class Boom:
            def raise_for_status(self):
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a, **k):
                return False

            async def post(self, *a, **k):
                return Boom()

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: Client())
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(ImageTool({}).arun(str(p)))

    def test_schema_requires_path(self):
        from teff.harness import tool_to_schema
        from teff.rag import ImageTool

        schema = tool_to_schema(ImageTool({}))
        assert "path" in schema["function"]["parameters"]["required"]
