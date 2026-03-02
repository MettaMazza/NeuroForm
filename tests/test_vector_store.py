"""Tests for VectorStore — in-memory semantic memory with Ollama embeddings."""
import pytest
from unittest.mock import patch, MagicMock
from neuroform.memory.vector_store import VectorStore, VectorEntry, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1, 2], [1]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


class TestVectorEntry:
    def test_creation(self):
        e = VectorEntry("hello world", [1.0, 2.0])
        assert e.text == "hello world"
        assert e.embedding == [1.0, 2.0]
        assert e.valid is True
        assert e.metadata == {}

    def test_creation_with_metadata(self):
        e = VectorEntry("hello", [1.0], metadata={"user_id": "u1"})
        assert e.metadata["user_id"] == "u1"


class TestVectorStore:
    @pytest.fixture
    def store(self):
        return VectorStore(model="test-model", max_entries=100)

    def test_init(self, store):
        assert store.model == "test-model"
        assert store.max_entries == 100
        assert store.entries == []

    @patch("neuroform.memory.vector_store.ollama")
    def test_embed(self, mock_ollama, store):
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        result = store.embed("hello world")
        assert result == [0.1, 0.2, 0.3]

    @patch("neuroform.memory.vector_store.ollama")
    def test_embed_old_api(self, mock_ollama, store):
        mock_ollama.embed.return_value = {"embedding": [0.4, 0.5]}
        result = store.embed("hello")
        assert result == [0.4, 0.5]

    @patch("neuroform.memory.vector_store.ollama")
    def test_embed_failure(self, mock_ollama, store):
        mock_ollama.embed.side_effect = Exception("connection error")
        result = store.embed("hello")
        assert result == []

    @patch("neuroform.memory.vector_store.ollama")
    def test_store(self, mock_ollama, store):
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}
        store.store("hello world", user_id="u1")
        assert len(store.entries) == 1
        assert store.entries[0].text == "hello world"

    def test_store_trivial_skipped(self, store):
        store.store("hi", user_id="u1")  # Too short (<10 chars)
        assert len(store.entries) == 0

    @patch("neuroform.memory.vector_store.ollama")
    def test_store_eviction(self, mock_ollama):
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}
        store = VectorStore(model="test", max_entries=3)
        for i in range(5):
            store.store(f"message number {i} with extra padding", user_id="u1")
        assert len(store.entries) == 3

    @patch("neuroform.memory.vector_store.ollama")
    def test_retrieve(self, mock_ollama, store):
        # Store a few entries manually
        store.entries = [
            VectorEntry("Maria is my developer", [1.0, 0.0, 0.0]),
            VectorEntry("I like cats a lot", [0.0, 1.0, 0.0]),
        ]
        # Query embed returns something close to first entry
        mock_ollama.embed.return_value = {"embeddings": [[0.9, 0.1, 0.0]]}
        results = store.retrieve("who is Maria", max_results=1)
        assert len(results) == 1
        assert "Maria" in results[0]["text"]

    def test_retrieve_empty(self, store):
        results = store.retrieve("anything")
        assert results == []

    @patch("neuroform.memory.vector_store.ollama")
    def test_retrieve_scope_filtering(self, mock_ollama, store):
        store.entries = [
            VectorEntry("public memory", [1.0, 0.0], metadata={"scope": "PUBLIC"}),
            VectorEntry("private memory", [0.9, 0.1], metadata={"scope": "PRIVATE"}),
        ]
        mock_ollama.embed.return_value = {"embeddings": [[1.0, 0.0]]}
        results = store.retrieve("memory", scope="PUBLIC")
        assert all("private" not in r["text"].lower() for r in results)

    def test_invalidate_by_content(self, store):
        store.entries = [
            VectorEntry("Maria is cool", [1.0]),
            VectorEntry("cats are nice", [0.5]),
        ]
        store.invalidate_by_content(["Maria"], reason="test")
        assert store.entries[0].valid is False
        assert store.entries[1].valid is True

    def test_snapshot(self, store):
        snap = store.snapshot()
        assert snap["total_entries"] == 0
        assert snap["model"] == "test-model"

    @patch("neuroform.memory.vector_store.ollama")
    def test_retrieve_skips_invalid(self, mock_ollama, store):
        entry = VectorEntry("invalid memory", [1.0, 0.0])
        entry.valid = False
        store.entries = [entry]
        mock_ollama.embed.return_value = {"embeddings": [[1.0, 0.0]]}
        results = store.retrieve("memory")
        assert results == []

    def test_embed_ollama_none(self):
        """When ollama is not installed, embed should return []."""
        import neuroform.memory.vector_store as vs_mod
        original = vs_mod.ollama
        vs_mod.ollama = None
        try:
            store = VectorStore(model="test")
            result = store.embed("hello world")
            assert result == []
        finally:
            vs_mod.ollama = original

    def test_ollama_import_error(self):
        """Simulate ollama not being installed at import time."""
        import sys
        import importlib
        # Temporarily remove ollama from sys.modules
        ollama_mod = sys.modules.pop("ollama", None)
        # Also hide from import
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("No module named 'ollama'")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = mock_import
        try:
            import neuroform.memory.vector_store as vs_mod
            importlib.reload(vs_mod)
            assert vs_mod.ollama is None
        finally:
            builtins.__import__ = original_import
            if ollama_mod is not None:
                sys.modules["ollama"] = ollama_mod
            import neuroform.memory.vector_store as vs_mod2
            importlib.reload(vs_mod2)

    @patch("neuroform.memory.vector_store.ollama")
    def test_embed_non_dict_result(self, mock_ollama, store):
        """Non-dict embed result should return []."""
        mock_ollama.embed.return_value = "not a dict"
        result = store.embed("hello")
        assert result == []

    @patch("neuroform.memory.vector_store.ollama")
    def test_store_with_metadata(self, mock_ollama, store):
        """Store with custom metadata should merge."""
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}
        store.store("hello world extra", user_id="u1",
                    metadata={"custom": "value"})
        assert store.entries[0].metadata["custom"] == "value"
        assert store.entries[0].metadata["user_id"] == "u1"

    @patch("neuroform.memory.vector_store.ollama")
    def test_retrieve_embed_failure(self, mock_ollama, store):
        """If embed fails during retrieve, return empty."""
        store.entries = [VectorEntry("test memory", [1.0, 0.0])]
        mock_ollama.embed.return_value = {"embeddings": []}  # empty
        results = store.retrieve("test")
        assert results == []

    @patch("neuroform.memory.vector_store.ollama")
    def test_retrieve_user_scope_filter(self, mock_ollama, store):
        """Retrieve should skip other users' private entries."""
        store.entries = [
            VectorEntry("shared memory", [1.0, 0.0],
                       metadata={"scope": "PRIVATE", "user_id": "u99"}),
            VectorEntry("my memory", [0.9, 0.1],
                       metadata={"scope": "PUBLIC", "user_id": "u1"}),
        ]
        mock_ollama.embed.return_value = {"embeddings": [[1.0, 0.0]]}
        results = store.retrieve("memory", scope="PRIVATE", user_id="u1")
        # u99's private entry should be skipped
        for r in results:
            if r["metadata"]["scope"] == "PRIVATE":
                assert r["metadata"]["user_id"] == "u1"
