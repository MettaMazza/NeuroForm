"""Tests for ContextStream — 500-turn persistent working memory."""
import json
import os
import tempfile
import pytest
from neuroform.memory.context_stream import ContextStream, Turn


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


class TestContextStream:
    @pytest.fixture
    def stream(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        return ContextStream(max_turns=500, persist_path=path)

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
        s1 = ContextStream(max_turns=500, persist_path=path)
        s1.add_turn("u1", "hello", "world", user_name="Maria")
        s1.add_turn("u2", "foo", "bar", user_name="John")

        # Reload from disk
        s2 = ContextStream(max_turns=500, persist_path=path)
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
        for i in range(10):
            stream.add_turn("u1", f"msg{i}", f"reply{i}")
        ctx = stream.get_context(max_turns=3)
        assert "msg7" in ctx
        assert "msg8" in ctx
        assert "msg9" in ctx
        assert "msg0" not in ctx

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

    def test_consolidation(self, tmp_path):
        path = str(tmp_path / "wm.jsonl")
        archive = str(tmp_path / "archive.jsonl")
        import neuroform.memory.context_stream as cs_mod
        original_archive = cs_mod.ARCHIVE_PATH
        cs_mod.ARCHIVE_PATH = archive

        try:
            s = ContextStream(max_turns=10, persist_path=path)
            for i in range(15):
                s.add_turn("u1", f"msg{i}", f"reply{i}")
            # Should have consolidated
            assert s.turn_count < 15
            # Archive file should exist
            assert os.path.exists(archive)
        finally:
            cs_mod.ARCHIVE_PATH = original_archive

    def test_snapshot(self, stream):
        snap = stream.snapshot()
        assert snap["turn_count"] == 0
        assert snap["max_turns"] == 500

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        s = ContextStream(max_turns=500, persist_path=path)
        assert s.turn_count == 0

    def test_load_corrupt_file(self, tmp_path):
        path = str(tmp_path / "corrupt.jsonl")
        with open(path, "w") as f:
            f.write("not valid json\n")
        s = ContextStream(max_turns=500, persist_path=path)
        assert s.turn_count == 0

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

    def test_archive_error(self, tmp_path):
        """Archive errors should be caught gracefully."""
        import neuroform.memory.context_stream as cs_mod
        original_archive = cs_mod.ARCHIVE_PATH
        cs_mod.ARCHIVE_PATH = "/dev/null/impossible/archive.jsonl"
        try:
            path = str(tmp_path / "wm.jsonl")
            s = ContextStream(max_turns=5, persist_path=path)
            for i in range(10):
                s.add_turn("u1", f"msg{i}", f"reply{i}")
            # Should not crash even though archive fails
        finally:
            cs_mod.ARCHIVE_PATH = original_archive

    def test_snapshot_with_data(self, stream):
        """Snapshot should show oldest/newest when data exists."""
        stream.add_turn("u1", "first", "reply1")
        stream.add_turn("u1", "last", "reply2")
        snap = stream.snapshot()
        assert snap["turn_count"] == 2
        assert snap["oldest"] is not None
        assert snap["newest"] is not None
