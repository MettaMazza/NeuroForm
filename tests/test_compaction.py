"""Tests for the auto-compaction engine."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from neuroform.memory.context_stream import ContextStream, CompactionSummary, COMPACTION_KEEP_RECENT
from neuroform.memory.compaction import (
    compact_context,
    format_turns_for_compaction,
    _fallback_summary,
)


class TestFormatTurns:
    def test_format_empty(self):
        result = format_turns_for_compaction([])
        assert result == ""

    def test_format_single_turn(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        cs.add_turn("u1", "hello world", "hi there", user_name="Maria")
        result = format_turns_for_compaction(cs.buffer)
        assert "Maria (u1): hello world" in result
        assert "Bot: hi there" in result

    def test_format_multiple_turns(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        cs.add_turn("u1", "msg1", "reply1", user_name="Alice")
        cs.add_turn("u2", "msg2", "reply2", user_name="Bob")
        result = format_turns_for_compaction(cs.buffer)
        assert "Alice (u1): msg1" in result
        assert "Bob (u2): msg2" in result


class TestFallbackSummary:
    def test_fallback_basic(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        for i in range(5):
            cs.add_turn("u1", f"message {i}", f"reply {i}", user_name="User")
        result = _fallback_summary(cs.buffer)
        assert "Summary of 5" in result
        assert "message 0" in result

    def test_fallback_truncates_long(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        for i in range(30):
            cs.add_turn("u1", f"message {i}", f"reply {i}", user_name="User")
        result = _fallback_summary(cs.buffer)
        assert "Summary of 30" in result
        # Should cap at 20 messages
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) <= 20


class TestCompactContext:
    @pytest.fixture
    def stream(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        # Very low threshold to trigger compaction easily
        cs = ContextStream(persist_path=path, compaction_threshold=100)
        for i in range(COMPACTION_KEEP_RECENT + 20):
            cs.add_turn("u1", f"message {i} with content", f"reply {i} with content",
                       user_name="TestUser")
        return cs

    @pytest.mark.asyncio
    async def test_compact_when_not_needed(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        cs.add_turn("u1", "hello", "hi")

        mock_vs = MagicMock()
        mock_llm = MagicMock()
        result = await compact_context(cs, mock_vs, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_compact_when_needed(self, stream):
        mock_vs = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate_raw.return_value = "Summary of conversation about testing."

        result = await compact_context(stream, mock_vs, mock_llm, user_id="u1")

        # Should return a CompactionSummary
        assert result is not None
        assert result.turns_compacted == 20
        assert "Summary of conversation" in result.summary

        # Buffer should now have only COMPACTION_KEEP_RECENT turns
        assert stream.turn_count == COMPACTION_KEEP_RECENT

        # Vector store should have been called
        assert mock_vs.store.called

    @pytest.mark.asyncio
    async def test_compact_llm_failure_uses_fallback(self, stream):
        mock_vs = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate_raw.side_effect = Exception("LLM unavailable")

        result = await compact_context(stream, mock_vs, mock_llm, user_id="u1")

        # Should still succeed with fallback summary
        assert result is not None
        assert "Summary of 20" in result.summary

    @pytest.mark.asyncio
    async def test_compact_vector_store_failure(self, stream):
        mock_vs = MagicMock()
        mock_vs.store.side_effect = Exception("Vector store down")
        mock_llm = MagicMock()
        mock_llm.generate_raw.return_value = "Good summary"

        result = await compact_context(stream, mock_vs, mock_llm, user_id="u1")

        # Should still succeed despite vector store failure
        assert result is not None
        assert result.summary == "Good summary"
        assert stream.turn_count == COMPACTION_KEEP_RECENT

    @pytest.mark.asyncio
    async def test_compact_no_turns_to_compact(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path, compaction_threshold=10)
        # Add fewer turns than COMPACTION_KEEP_RECENT
        for i in range(5):
            cs.add_turn("u1", "m" * 100, "r" * 100)  # Large to exceed threshold
        mock_vs = MagicMock()
        mock_llm = MagicMock()

        result = await compact_context(cs, mock_vs, mock_llm)
        assert result is None  # Nothing to compact even though over threshold

    @pytest.mark.asyncio
    async def test_compact_embeds_individual_turns(self, stream):
        mock_vs = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate_raw.return_value = "Summary"

        await compact_context(stream, mock_vs, mock_llm, user_id="u1")

        # Should have embedded: 1 summary + 20 individual turns = 21 calls
        assert mock_vs.store.call_count == 21

    @pytest.mark.asyncio
    async def test_compact_preserves_scope(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path, compaction_threshold=100)
        for i in range(COMPACTION_KEEP_RECENT + 5):
            cs.add_turn("u1", f"msg {i}", f"reply {i}", scope="PRIVATE")

        mock_vs = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate_raw.return_value = "Private summary"

        result = await compact_context(cs, mock_vs, mock_llm, user_id="u1", scope="PRIVATE")
        assert result.scope == "PRIVATE"
