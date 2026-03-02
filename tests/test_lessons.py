"""Tests for LessonManager — structured fact persistence."""
import json
import os
import pytest
from neuroform.memory.lessons import LessonManager


class TestLessonManager:
    @pytest.fixture
    def lm(self, tmp_path):
        path = str(tmp_path / "lessons.json")
        return LessonManager(persist_path=path)

    def test_init_empty(self, lm):
        assert lm.count() == 0
        assert lm.lessons == []

    def test_add_lesson(self, lm):
        result = lm.add_lesson("Maria is the developer", user_id="u1")
        assert result is True
        assert lm.count() == 1

    def test_add_duplicate(self, lm):
        lm.add_lesson("Maria is the developer")
        result = lm.add_lesson("Maria is the developer")
        assert result is False
        assert lm.count() == 1

    def test_add_duplicate_case_insensitive(self, lm):
        lm.add_lesson("Maria is the developer")
        result = lm.add_lesson("MARIA IS THE DEVELOPER")
        assert result is False
        assert lm.count() == 1

    def test_get_all_lessons(self, lm):
        lm.add_lesson("fact one", user_id="u1")
        lm.add_lesson("fact two", user_id="u2")
        lessons = lm.get_all_lessons()
        assert len(lessons) == 2
        assert "fact one" in lessons
        assert "fact two" in lessons

    def test_get_lessons_user_filtered(self, lm):
        lm.add_lesson("public fact", user_id="u1", scope="PUBLIC")
        lm.add_lesson("private fact", user_id="u1", scope="PRIVATE")
        lm.add_lesson("other private", user_id="u2", scope="PRIVATE")

        # u1 sees public + their private (when scope allows)
        lessons = lm.get_all_lessons(user_id="u1", scope="PRIVATE")
        assert "public fact" in lessons
        assert "private fact" in lessons
        assert "other private" not in lessons

    def test_remove_lesson(self, lm):
        lm.add_lesson("to be removed")
        result = lm.remove_lesson("to be removed")
        assert result is True
        assert lm.count() == 0

    def test_remove_nonexistent(self, lm):
        result = lm.remove_lesson("does not exist")
        assert result is False

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "lessons.json")
        lm1 = LessonManager(persist_path=path)
        lm1.add_lesson("Maria is the developer", user_id="u1")
        lm1.add_lesson("User likes cats", user_id="u1")

        # Reload
        lm2 = LessonManager(persist_path=path)
        assert lm2.count() == 2
        lessons = lm2.get_all_lessons()
        assert "Maria is the developer" in lessons

    def test_snapshot(self, lm):
        snap = lm.snapshot()
        assert snap["total_lessons"] == 0

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        lm = LessonManager(persist_path=path)
        assert lm.count() == 0

    def test_load_corrupt_file(self, tmp_path):
        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("not json")
        lm = LessonManager(persist_path=path)
        assert lm.count() == 0

    def test_save_error(self, lm):
        """Save errors should be caught gracefully."""
        lm.persist_path = "/dev/null/impossible/lessons.json"
        lm.add_lesson("test fact")  # Should not crash despite save error
        assert lm.count() == 1  # Still in memory
