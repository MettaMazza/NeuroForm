"""Tests for ContextStream — Token-based persistent conversation memory."""
import json
import os
import tempfile
import pytest
from neuroform.memory.context_stream import (
    ContextStream, Turn, CompactionSummary,
    estimate_tokens, estimate_turn_tokens,
    COMPACTION_KEEP_RECENT,
)


class TestTurn:
    def test_turn_fields(self):
        t = Turn(
            user_id="u1", user_name="Maria",
            user_message="hello", bot_message="hi",
            timestamp=1.0, scope="PUBLIC"
        )
        assert t.user_id == "u1"
        assert t.user_name == "Maria"
        assert t.user_message == "hello"
        assert t.bot_message == "hi"
        assert t.scope == "PUBLIC"

    def test_turn_defaults(self):
        t = Turn(
            user_id="u1", user_name="Unknown",
            user_message="x", bot_message="y",
            timestamp=2.0
        )
        assert t.scope == "PUBLIC"
        assert t.channel_id is None
        assert t.metadata is None


class TestCompactionSummary:
    def test_compaction_summary_fields(self):
        cs = CompactionSummary(
            summary="User discussed cats",
            turns_compacted=10,
            tokens_before=5000,
            tokens_after=1000,
            timestamp=1.0,
            user_id="u1",
            scope="PUBLIC",
        )
        assert cs.turns_compacted == 10
        assert cs.tokens_before == 5000


class TestTokenEstimation:
    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_none(self):
        assert estimate_tokens(None) == 0

    def test_estimate_tokens_short(self):
        # "hello" = 5 chars → ~1 token
        assert estimate_tokens("hello") >= 1

    def test_estimate_tokens_known(self):
        # 400 chars → ~100 tokens
        text = "x" * 400
        result = estimate_tokens(text)
        assert result == 100

    def test_estimate_turn_tokens(self):
        t = Turn(
            user_id="u1", user_name="Maria",
            user_message="hello world",
            bot_message="hi there",
            timestamp=1.0
        )
        tokens = estimate_turn_tokens(t)
        assert tokens > 0


class TestContextStream:
    @pytest.fixture
    def stream(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        return ContextStream(persist_path=path)

    def test_init_empty(self, stream):
        assert stream.turn_count == 0
        assert stream.buffer == []

    def test_add_turn(self, stream):
        stream.add_turn("u1", "hello", "hi", user_name="Maria")
        assert stream.turn_count == 1
        turn = stream.buffer[0]
        assert turn.user_id == "u1"
        assert turn.user_name == "Maria"
        assert turn.user_message == "hello"
        assert turn.bot_message == "hi"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        s1 = ContextStream(persist_path=path)
        s1.add_turn("u1", "hello", "world", user_name="Maria")
        s1.add_turn("u2", "foo", "bar", user_name="John")

        # Reload from disk
        s2 = ContextStream(persist_path=path)
        assert s2.turn_count == 2
        assert s2.buffer[0].user_name == "Maria"
        assert s2.buffer[1].user_name == "John"

    def test_get_context(self, stream):
        stream.add_turn("u1", "hello", "hi", user_name="Maria")
        stream.add_turn("u1", "how are you", "doing well", user_name="Maria")
        ctx = stream.get_context()
        assert "Maria: hello" in ctx
        assert "Bot: hi" in ctx
        assert "Maria: how are you" in ctx

    def test_get_context_empty(self, stream):
        ctx = stream.get_context()
        assert ctx == "No conversation history."

    def test_get_context_max_turns(self, stream):
        for i in range(60):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        ctx = stream.get_context(max_turns=3)
        assert "msg57" in ctx
        assert "msg58" in ctx
        assert "msg59" in ctx
        assert "msg0" not in ctx

    def test_get_context_50_turns_default(self, stream):
        """Default injection should be 50 turns."""
        for i in range(60):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        ctx = stream.get_context()
        # First 10 should be excluded (60 - 50 = 10)
        assert "msg9" not in ctx
        assert "msg10" in ctx
        assert "msg59" in ctx

    def test_get_context_by_tokens(self, stream):
        """Token-budget selection."""
        for i in range(100):
            stream.add_turn("u1", f"message number {i}", f"reply number {i}")
        ctx = stream.get_context(max_tokens=500)
        # Should contain some recent turns but not all 100
        assert "message number 99" in ctx
        assert "message number 0" not in ctx

    def test_scope_filtering(self, stream):
        stream.add_turn("u1", "public msg", "pub reply", scope="PUBLIC")
        stream.add_turn("u1", "private msg", "priv reply", scope="PRIVATE")
        ctx = stream.get_context(target_scope="PUBLIC")
        assert "public msg" in ctx
        assert "private msg" not in ctx

    def test_scope_private_with_user(self, stream):
        stream.add_turn("u1", "public msg", "pub reply", scope="PUBLIC")
        stream.add_turn("u1", "private msg", "priv reply", scope="PRIVATE")
        ctx = stream.get_context(target_scope="PRIVATE", user_id="u1")
        assert "public msg" in ctx
        assert "private msg" in ctx

    def test_channel_filtering(self, stream):
        stream.add_turn("u1", "ch1 msg", "ch1 reply", channel_id=100)
        stream.add_turn("u1", "ch2 msg", "ch2 reply", channel_id=200)
        ctx = stream.get_context(channel_id=100)
        assert "ch1 msg" in ctx
        assert "ch2 msg" not in ctx

    def test_get_recent_turns(self, stream):
        for i in range(5):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        recent = stream.get_recent_turns(2)
        assert len(recent) == 2
        assert recent[0].user_message == "msg3"

    def test_get_user_facts(self, stream):
        stream.add_turn("u1", "I am Maria", "noted", user_name="Maria")
        stream.add_turn("u2", "I am John", "noted", user_name="John")
        facts = stream.get_user_facts("u1")
        assert facts == ["I am Maria"]

    def test_search(self, stream):
        stream.add_turn("u1", "I love cats", "cats are great")
        stream.add_turn("u1", "dogs are ok", "sure")
        matches = stream.search("cats")
        assert len(matches) == 1
        assert "cats" in matches[0].user_message

    def test_get_conversation_history(self, stream):
        stream.add_turn("u1", "hello", "hi")
        history = stream.get_conversation_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hello"}
        assert history[1] == {"role": "assistant", "content": "hi"}

    def test_get_conversation_history_50_default(self, stream):
        """History should default to 50 turns."""
        for i in range(60):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        history = stream.get_conversation_history()
        # 50 turns = 100 entries (user + assistant)
        assert len(history) == 100
        assert history[0]["content"] == "msg10"

    def test_total_tokens_property(self, stream):
        stream.add_turn("u1", "hello world", "hi there")
        tokens = stream.total_tokens
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_needs_compaction_false(self, stream):
        stream.add_turn("u1", "hello", "hi")
        assert stream.needs_compaction is False

    def test_needs_compaction_true(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        # Very low threshold to trigger compaction
        s = ContextStream(persist_path=path, compaction_threshold=100)
        for i in range(50):
            s.add_turn("u1", f"message {i} with lots of content", f"reply {i} also with content")
        assert s.needs_compaction is True

    def test_turns_compat_property(self, stream):
        """_turns property should expose buffer for PHUD compat."""
        stream.add_turn("u1", "test", "reply")
        assert stream._turns is stream.buffer

    def test_snapshot(self, stream):
        snap = stream.snapshot()
        assert snap["turn_count"] == 0
        assert snap["total_tokens"] == 0
        assert snap["needs_compaction"] is False
        assert snap["compaction_count"] == 0

    def test_snapshot_with_data(self, stream):
        stream.add_turn("u1", "first", "reply1")
        stream.add_turn("u1", "last", "reply2")
        snap = stream.snapshot()
        assert snap["turn_count"] == 2
        assert snap["total_tokens"] > 0
        assert snap["oldest"] is not None
        assert snap["newest"] is not None

    def test_get_turns_for_compaction_empty(self, stream):
        for i in range(10):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        turns = stream.get_turns_for_compaction()
        assert turns == []  # Under COMPACTION_KEEP_RECENT

    def test_get_turns_for_compaction_returns_old(self, stream):
        for i in range(COMPACTION_KEEP_RECENT + 10):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        turns = stream.get_turns_for_compaction()
        assert len(turns) == 10

    def test_apply_compaction(self, stream):
        for i in range(COMPACTION_KEEP_RECENT + 20):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")

        summary = CompactionSummary(
            summary="Test summary",
            turns_compacted=20,
            tokens_before=10000,
            tokens_after=5000,
            timestamp=1.0,
            user_id="u1",
            scope="PUBLIC",
        )
        stream.apply_compaction(summary)

        assert stream.turn_count == COMPACTION_KEEP_RECENT
        assert len(stream.compaction_summaries) == 1
        assert stream.compaction_summaries[0].summary == "Test summary"

    def test_compaction_summary_in_context(self, stream):
        """After compaction, the summary should appear in context output."""
        stream.add_turn("u1", "hello", "hi")
        summary = CompactionSummary(
            summary="User discussed cats and dogs",
            turns_compacted=10,
            tokens_before=5000,
            tokens_after=1000,
            timestamp=1.0,
            user_id="u1",
            scope="PUBLIC",
        )
        stream.compaction_summaries.append(summary)
        ctx = stream.get_context()
        assert "User discussed cats and dogs" in ctx

    def test_clear(self, stream):
        stream.add_turn("u1", "hello", "hi")
        stream.compaction_summaries.append(CompactionSummary(
            summary="old", turns_compacted=1, tokens_before=100,
            tokens_after=0, timestamp=1.0, user_id="u1", scope="PUBLIC",
        ))
        stream.clear()
        assert stream.turn_count == 0
        assert len(stream.compaction_summaries) == 0

    def test_persistence_with_compaction(self, tmp_path):
        """Compaction summaries should persist and reload."""
        path = str(tmp_path / "wm.jsonl")
        s1 = ContextStream(persist_path=path)
        s1.add_turn("u1", "hello", "hi")
        s1.compaction_summaries.append(CompactionSummary(
            summary="persisted summary",
            turns_compacted=5,
            tokens_before=3000,
            tokens_after=1000,
            timestamp=1.0,
            user_id="u1",
            scope="PUBLIC",
        ))
        s1._save_to_disk()

        s2 = ContextStream(persist_path=path)
        assert len(s2.compaction_summaries) == 1
        assert s2.compaction_summaries[0].summary == "persisted summary"
        assert s2.turn_count == 1

    def test_private_scope_no_user_id(self, stream):
        """PRIVATE scope without user_id should only return PUBLIC turns."""
        stream.add_turn("u1", "public msg", "pub reply", scope="PUBLIC")
        stream.add_turn("u1", "private msg", "priv reply", scope="PRIVATE")
        ctx = stream.get_context(target_scope="PRIVATE", user_id=None)
        assert "public msg" in ctx
        assert "private msg" not in ctx

    def test_search_max_results_cap(self, stream):
        """Search should stop at max_results."""
        for i in range(10):
            stream.add_turn("u1", f"cats are great {i}", f"agreed {i}")
        matches = stream.search("cats", max_results=3)
        assert len(matches) == 3

    def test_save_to_disk_error(self, stream):
        """Save errors should be caught gracefully."""
        stream.persist_path = "/dev/null/impossible/path.jsonl"
        stream.add_turn("u1", "msg", "reply")  # Should not crash

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        s = ContextStream(persist_path=path)
        assert s.turn_count == 0

    def test_load_corrupt_file(self, tmp_path):
        path = str(tmp_path / "corrupt.jsonl")
        with open(path, "w") as f:
            f.write("not valid json\n")
        s = ContextStream(persist_path=path)
        assert s.turn_count == 0

    def test_unlimited_storage(self, stream):
        """No turn cap — can store thousands of turns."""
        for i in range(600):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        assert stream.turn_count == 600  # No truncation

    def test_select_by_tokens(self, stream):
        """Internal _select_by_tokens selects within budget."""
        for i in range(20):
            stream.add_turn("u1", f"message {i}", f"reply {i}")
        # Token budget of 50 should give us a few recent turns
        selected = stream._select_by_tokens(stream.buffer, 50)
        assert len(selected) > 0
        assert len(selected) < 20
        # Most recent should be last
        assert selected[-1].user_message == "message 19"
