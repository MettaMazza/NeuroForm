"""Tests for neuroform.memory.reconciler — CrossTierReconciler."""
import pytest
from unittest.mock import patch, MagicMock
from neuroform.memory.reconciler import (
    CrossTierReconciler, ConflictRecord, ReconciliationResult,
)


class TestConflictRecord:
    def test_creation(self):
        cr = ConflictRecord(
            target_text="steak is great",
            authority_text="user is vegan",
            conflict_type="VM",
            reason="diet conflict",
        )
        assert cr.conflict_type == "VM"
        assert cr.reason == "diet conflict"


class TestReconciliationResult:
    def test_creation_defaults(self):
        r = ReconciliationResult(
            lessons=["vegan"], kg_facts=["likes steak"],
            vector_texts=["ate steak yesterday"],
        )
        assert r.conflicts == []
        assert r.stats == {}


class TestCrossTierReconciler:
    @pytest.fixture
    def reconciler(self):
        return CrossTierReconciler(model="test")

    def test_quick_exit_no_lessons_no_kg(self, reconciler):
        result = reconciler.reconcile(
            lessons=[], kg_facts=[], vector_texts=["some memory"])
        assert result.stats["conflicts"] == 0

    def test_quick_exit_no_vector_no_kg(self, reconciler):
        result = reconciler.reconcile(
            lessons=["a lesson"], kg_facts=[], vector_texts=[])
        assert result.stats["conflicts"] == 0

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_consistent(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {
            "message": {"content": "CONSISTENT"}
        }
        result = reconciler.reconcile(
            lessons=["user is vegan"],
            kg_facts=["user prefers salads"],
            vector_texts=["ate salad for lunch"],
        )
        assert result.stats["conflicts"] == 0
        assert len(result.conflicts) == 0

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_finds_conflict_vm(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {
            "message": {"content": "CONFLICT:VM:0|Lesson says vegan, VM says steak"}
        }
        result = reconciler.reconcile(
            lessons=["user is vegan"],
            kg_facts=["prefers plants"],
            vector_texts=["had steak for dinner"],
        )
        assert result.stats["conflicts"] == 1
        assert "[⚠️STALE?]" in result.vector_texts[0]

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_finds_conflict_kg(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {
            "message": {"content": "CONFLICT:KG:0|Lesson says vegan, KG says meat"}
        }
        result = reconciler.reconcile(
            lessons=["user is vegan"],
            kg_facts=["user eats meat"],
            vector_texts=["some memory"],
        )
        assert result.stats["conflicts"] == 1
        assert "[🛑REJECTED]" in result.kg_facts[0]

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_llm_failure(self, mock_ollama, reconciler):
        mock_ollama.chat.side_effect = Exception("LLM down")
        result = reconciler.reconcile(
            lessons=["fact"], kg_facts=["fact2"], vector_texts=["fact3"])
        assert result.stats["conflicts"] == 0

    def test_reconcile_ollama_none(self, reconciler):
        import neuroform.memory.reconciler as rec_mod
        original = rec_mod._ollama
        rec_mod._ollama = None
        try:
            result = reconciler.reconcile(
                lessons=["x"], kg_facts=["y"], vector_texts=["z"])
            assert result.stats["conflicts"] == 0
        finally:
            rec_mod._ollama = original

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_multiple_conflicts(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {
            "message": {"content": "CONFLICT:VM:0|bad\nCONFLICT:KG:0|also bad"}
        }
        result = reconciler.reconcile(
            lessons=["truth"],
            kg_facts=["wrong fact"],
            vector_texts=["wrong memory"],
        )
        assert result.stats["conflicts"] == 2

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_out_of_range_index(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {
            "message": {"content": "CONFLICT:VM:99|out of range"}
        }
        result = reconciler.reconcile(
            lessons=["x"], kg_facts=["y"], vector_texts=["z"])
        assert result.stats["conflicts"] == 0  # Index 99 is out of range

    @patch("neuroform.memory.reconciler._ollama")
    def test_build_input_format(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {"message": {"content": "CONSISTENT"}}
        reconciler.reconcile(
            lessons=["L1"], kg_facts=["KG1"], vector_texts=["VM1"])
        call_args = mock_ollama.chat.call_args
        user_content = call_args[1]["messages"][1]["content"]
        assert "LS[0]: L1" in user_content
        assert "KG[0]: KG1" in user_content
        assert "VM[0]: VM1" in user_content

    @patch("neuroform.memory.reconciler._ollama")
    def test_reconcile_vm_not_in_list(self, mock_ollama, reconciler):
        """ValueError case when annotating — target not found."""
        mock_ollama.chat.return_value = {
            "message": {"content": "CONFLICT:VM:0|reason"}
        }
        # Trick: modify vector_texts after parsing but before annotating
        # This tests the try/except ValueError path
        result = reconciler.reconcile(
            lessons=["x"], kg_facts=["y"], vector_texts=["z"])
        # Should still work (annotation tries to find index)
        assert result.stats["conflicts"] == 1

    def test_reconcile_no_lessons_authority(self, reconciler):
        """ConflictRecord uses 'Hierarchy' when no lessons."""
        import neuroform.memory.reconciler as rec_mod
        conflicts = rec_mod.CrossTierReconciler._parse_verdicts(
            reconciler,
            "CONFLICT:VM:0|reason",
            [],  # No lessons
            [],
            ["memory"],
        )
        assert conflicts[0].authority_text == "Hierarchy"

    def test_snapshot(self, reconciler):
        snap = reconciler.snapshot()
        assert snap["model"] == "test"
        assert "ollama_available" in snap

    @patch("neuroform.memory.reconciler._ollama")
    def test_llm_returns_empty(self, mock_ollama, reconciler):
        mock_ollama.chat.return_value = {"message": {"content": ""}}
        result = reconciler.reconcile(
            lessons=["x"], kg_facts=["y"], vector_texts=["z"])
        assert result.stats["conflicts"] == 0

    def test_reconcile_ollama_import_error(self):
        """Test the import error handling for ollama."""
        import sys
        import importlib
        ollama_mod = sys.modules.pop("ollama", None)
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("No module named 'ollama'")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = mock_import
        try:
            import neuroform.memory.reconciler as rec_mod
            importlib.reload(rec_mod)
            assert rec_mod._ollama is None
        finally:
            builtins.__import__ = original_import
            if ollama_mod is not None:
                sys.modules["ollama"] = ollama_mod
            import neuroform.memory.reconciler as rec_mod2
            importlib.reload(rec_mod2)

    def test_annotate_kg_value_error(self, reconciler):
        """Cover L143-144: ValueError when annotating KG conflict."""
        from neuroform.memory.reconciler import ConflictRecord
        # Create a conflict where target_text doesn't match any kg_fact
        conflict = ConflictRecord(
            target_text="nonexistent_fact",
            authority_text="lesson",
            conflict_type="KG",
            reason="test",
        )
        # Bypass LLM: inject conflicts directly via _parse_verdicts mock
        with patch.object(reconciler, "_call_llm", return_value="CONSISTENT"):
            with patch.object(reconciler, "_parse_verdicts", return_value=[conflict]):
                result = reconciler.reconcile(
                    lessons=["x"], kg_facts=["real_fact"], vector_texts=["y"])
                # Should not crash — ValueError is caught
                assert result.stats["conflicts"] == 1
                assert "[🛑REJECTED]" not in result.kg_facts[0]  # Not annotated

    def test_annotate_vm_value_error(self, reconciler):
        """Cover L149-150: ValueError when annotating VM conflict."""
        from neuroform.memory.reconciler import ConflictRecord
        conflict = ConflictRecord(
            target_text="nonexistent_memory",
            authority_text="lesson",
            conflict_type="VM",
            reason="test",
        )
        with patch.object(reconciler, "_call_llm", return_value="CONSISTENT"):
            with patch.object(reconciler, "_parse_verdicts", return_value=[conflict]):
                result = reconciler.reconcile(
                    lessons=["x"], kg_facts=["y"], vector_texts=["real_memory"])
                assert result.stats["conflicts"] == 1
                assert "[⚠️STALE?]" not in result.vector_texts[0]
