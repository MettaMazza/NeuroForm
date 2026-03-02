"""
Tier 4: Lesson Manager — Structured Fact Persistence
======================================================

Stores verified facts as structured lessons with user scoping.
Ported from ErnOS V3/V4 LessonManager.

Facts like "Maria is my developer" become lessons that persist across
sessions and are injected into every prompt. This is the most reliable
memory tier — facts here are treated as ground truth.
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

PERSIST_PATH = os.path.join(os.getcwd(), "memory", "core", "lessons.json")


class LessonManager:
    """
    Structured fact storage with JSON persistence.

    - Each lesson has: fact, user_id, scope, timestamp
    - Facts persist across restarts
    - Filtered retrieval by user_id and scope
    - Deduplication on add
    """

    def __init__(self, persist_path: str = PERSIST_PATH):
        self.persist_path = persist_path
        self.lessons: List[Dict[str, Any]] = []
        self._load()
        logger.info(f"LessonManager initialized: {len(self.lessons)} lessons loaded")

    def add_lesson(self, fact: str, user_id: str = "",
                   scope: str = "PUBLIC") -> bool:
        """
        Add a new lesson. Returns True if added, False if duplicate.
        """
        # Dedup check
        fact_lower = fact.lower().strip()
        for existing in self.lessons:
            if existing["fact"].lower().strip() == fact_lower:
                logger.debug(f"Duplicate lesson skipped: {fact[:50]}")
                return False

        import time
        lesson = {
            "fact": fact,
            "user_id": str(user_id),
            "scope": scope,
            "timestamp": time.time(),
        }
        self.lessons.append(lesson)
        self._save()
        logger.info(f"Lesson added: {fact[:80]}")
        return True

    def get_all_lessons(self, user_id: Optional[str] = None,
                        scope: str = "PUBLIC") -> List[str]:
        """
        Get all lesson facts, filtered by user/scope (ground rule).

        Visibility:
        - PUBLIC lessons: always visible
        - PRIVATE lessons: only when user_id matches AND scope allows
        """
        results = []
        for lesson in self.lessons:
            lesson_scope = lesson.get("scope", "PUBLIC")
            lesson_uid = lesson.get("user_id", "")
            # PUBLIC lessons are always visible
            if lesson_scope == "PUBLIC":
                results.append(lesson["fact"])
            # PRIVATE lessons: only if requesting user matches AND scope allows
            elif scope in ("PRIVATE", "CORE_PRIVATE"):
                if user_id and lesson_uid == str(user_id):
                    results.append(lesson["fact"])
        return results

    def remove_lesson(self, fact: str) -> bool:
        """Remove a lesson by exact fact text."""
        fact_lower = fact.lower().strip()
        for i, lesson in enumerate(self.lessons):
            if lesson["fact"].lower().strip() == fact_lower:
                self.lessons.pop(i)
                self._save()
                logger.info(f"Lesson removed: {fact[:80]}")
                return True
        return False

    def count(self) -> int:
        return len(self.lessons)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_lessons": len(self.lessons),
            "persist_path": self.persist_path,
        }

    def _save(self):
        """Persist lessons to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.lessons, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save lessons: {e}")

    def _load(self):
        """Load lessons from JSON file on startup."""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.lessons = json.load(f)
                logger.info(f"Loaded {len(self.lessons)} lessons from disk")
        except Exception as e:
            logger.warning(f"Failed to load lessons: {e}")
            self.lessons = []
