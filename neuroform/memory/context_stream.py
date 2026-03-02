"""
Tier 1: ContextStream — Persistent Working Memory
====================================================

500-turn rolling buffer with JSONL persistence, scope filtering,
and user_name tracking. Ported from ErnOS V3/V4 WorkingMemory.

Unlike the original 7-item WorkingMemory, this persists to disk and
survives restarts. Turns carry user_id, user_name, scope, and channel_id
for proper filtering.
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

PERSIST_DIR = os.path.join(os.getcwd(), "memory", "core")
PERSIST_PATH = os.path.join(PERSIST_DIR, "working_memory.jsonl")
ARCHIVE_PATH = os.path.join(PERSIST_DIR, "archive_working_memory.jsonl")


@dataclass
class Turn:
    """A single conversation turn with full metadata."""
    user_id: str
    user_name: str
    user_message: str
    bot_message: str
    timestamp: float
    scope: str = "PUBLIC"
    channel_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ContextStream:
    """
    Rolling buffer of recent conversation turns with JSONL persistence.

    - 500-turn capacity (vs the old 7-item WM)
    - Persists to disk on every write
    - Scope/channel/user filtering for context retrieval
    - Consolidation: archives oldest turns when buffer overflows
    """

    def __init__(self, max_turns: int = 500, persist_path: str = PERSIST_PATH):
        self.max_turns = max_turns
        self.persist_path = persist_path
        self.buffer: List[Turn] = []
        self._load_from_disk()
        logger.info(f"ContextStream initialized: {len(self.buffer)} turns loaded, "
                     f"capacity={max_turns}")

    def add_turn(
        self,
        user_id: str,
        user_message: str,
        bot_message: str,
        user_name: str = "Unknown",
        scope: str = "PUBLIC",
        channel_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a conversation turn."""
        turn = Turn(
            user_id=str(user_id),
            user_name=user_name,
            user_message=user_message,
            bot_message=bot_message,
            timestamp=datetime.now().timestamp(),
            scope=scope,
            channel_id=channel_id,
            metadata=metadata or {},
        )
        self.buffer.append(turn)

        if len(self.buffer) > self.max_turns:
            self._consolidate()
        else:
            self._save_to_disk()

    def get_context(
        self,
        target_scope: str = "PUBLIC",
        user_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        max_turns: int = 20,
    ) -> str:
        """
        Get formatted conversation history filtered by scope/user/channel.

        Returns a string suitable for system prompt injection.
        """
        relevant = self.buffer

        # Scope filtering
        if target_scope in ("PRIVATE", "CORE_PRIVATE"):
            if user_id:
                relevant = [
                    t for t in relevant
                    if t.scope == "PUBLIC" or (t.scope == target_scope and t.user_id == str(user_id))
                ]
            else:
                relevant = [t for t in relevant if t.scope == "PUBLIC"]
        else:
            relevant = [t for t in relevant if t.scope == "PUBLIC"]

        # Channel filtering
        if channel_id is not None:
            relevant = [
                t for t in relevant
                if t.channel_id is None or t.channel_id == channel_id
            ]

        if not relevant:
            return "No conversation history."

        # Take the most recent turns
        recent = relevant[-max_turns:]

        lines = []
        for t in recent:
            ts = datetime.fromtimestamp(t.timestamp).strftime("%H:%M")
            lines.append(f"[{ts}] {t.user_name}: {t.user_message}")
            lines.append(f"[{ts}] Bot: {t.bot_message}")
        return "\n".join(lines)

    def get_recent_turns(self, n: int = 5) -> List[Turn]:
        """Return the last N turns (unfiltered)."""
        return self.buffer[-n:]

    def get_user_facts(self, user_id: str) -> List[str]:
        """
        Extract all messages from a specific user across the full buffer.
        Useful for foundation context and building user profiles.
        """
        return [
            t.user_message for t in self.buffer
            if t.user_id == str(user_id)
        ]

    def search(self, query: str, max_results: int = 5) -> List[Turn]:
        """Simple keyword search across all turns."""
        query_lower = query.lower()
        matches = []
        for t in reversed(self.buffer):
            if query_lower in t.user_message.lower() or query_lower in t.bot_message.lower():
                matches.append(t)
                if len(matches) >= max_results:
                    break
        return matches

    def get_conversation_history(self, max_turns: int = 12) -> List[Dict[str, str]]:
        """
        Get conversation history in LLM message format.
        Backwards compat with the old WorkingMemory interface.
        """
        history = []
        for t in self.buffer[-max_turns:]:
            history.append({"role": "user", "content": t.user_message})
            history.append({"role": "assistant", "content": t.bot_message})
        return history

    @property
    def turn_count(self) -> int:
        return len(self.buffer)

    def snapshot(self) -> Dict[str, Any]:
        """Return diagnostic snapshot."""
        return {
            "turn_count": len(self.buffer),
            "max_turns": self.max_turns,
            "persist_path": self.persist_path,
            "oldest": self.buffer[0].timestamp if self.buffer else None,
            "newest": self.buffer[-1].timestamp if self.buffer else None,
        }

    # ─── Persistence ───────────────────────────────────────────

    def _save_to_disk(self):
        """Persist current buffer to JSONL."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            lines = [json.dumps(asdict(t)) for t in self.buffer]
            with open(self.persist_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
        except Exception as e:
            logger.error(f"Failed to persist ContextStream: {e}")

    def _load_from_disk(self):
        """Load turns from JSONL on startup."""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.buffer.append(Turn(**data))
                logger.info(f"Loaded {len(self.buffer)} turns from disk")
        except Exception as e:
            logger.warning(f"Failed to load ContextStream from disk: {e}")

    def _consolidate(self):
        """Archive oldest turns when buffer exceeds capacity."""
        overflow = len(self.buffer) - self.max_turns
        to_archive = self.buffer[:overflow + 50]
        self.buffer = self.buffer[overflow + 50:]

        self._archive(to_archive)
        self._save_to_disk()

        logger.info(f"Consolidated {len(to_archive)} turns "
                     f"(buffer: {len(self.buffer)}/{self.max_turns})")

    def _archive(self, turns: List[Turn]):
        """Append archived turns to long-term storage file."""
        try:
            os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
            with open(ARCHIVE_PATH, "a", encoding="utf-8") as f:
                for t in turns:
                    f.write(json.dumps(asdict(t)) + "\n")
        except Exception as e:
            logger.error(f"Failed to archive turns: {e}")
