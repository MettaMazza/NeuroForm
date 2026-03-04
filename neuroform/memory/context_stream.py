"""
Tier 1: ContextStream — Token-Based Conversation Memory
=========================================================

Unlimited conversation buffer with JSONL persistence, scope filtering,
token estimation, and auto-compaction support. When the transcript
exceeds the token budget, older turns are compacted into vector-embedded
per-user per-scope searchable memory.

Inspired by ErnOS compaction architecture. No arbitrary turn caps.
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

# Token estimation: ~4 chars per token (conservative)
CHARS_PER_TOKEN = 4

# Default compaction threshold (tokens)
DEFAULT_COMPACTION_THRESHOLD = 110_000

# How many recent turns to keep after compaction
COMPACTION_KEEP_RECENT = 50


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


@dataclass
class CompactionSummary:
    """A persisted summary of compacted conversation history."""
    summary: str
    turns_compacted: int
    tokens_before: int
    tokens_after: int
    timestamp: float
    user_id: str
    scope: str


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length. ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_turn_tokens(turn: Turn) -> int:
    """Estimate tokens for a single turn (both user + bot messages)."""
    total_chars = len(turn.user_message) + len(turn.bot_message)
    # Add overhead for metadata, names, formatting
    total_chars += len(turn.user_name) + 20
    return max(1, total_chars // CHARS_PER_TOKEN)


class ContextStream:
    """
    Token-aware conversation memory with unlimited storage.

    - No arbitrary turn cap — stores ALL turns on disk
    - Token estimation for context window management
    - Scope/channel/user filtering for context retrieval
    - Auto-compaction support: when tokens exceed threshold,
      older turns are summarized and vector-embedded
    """

    def __init__(self, persist_path: str = PERSIST_PATH,
                 compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
                 max_turns: int = 0):
        """
        Args:
            persist_path: Path to JSONL persistence file
            compaction_threshold: Token count that triggers compaction (default 110k)
            max_turns: Legacy compat — ignored if 0 (unlimited)
        """
        self.persist_path = persist_path
        self.compaction_threshold = compaction_threshold
        self.buffer: List[Turn] = []
        self.compaction_summaries: List[CompactionSummary] = []
        self._load_from_disk()
        logger.info(f"ContextStream initialized: {len(self.buffer)} turns loaded, "
                     f"~{self.total_tokens} tokens, "
                     f"compaction threshold={compaction_threshold}")

    # ─── Public API ────────────────────────────────────────────

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
        """Record a conversation turn and persist to disk."""
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
        self._save_to_disk()

    @property
    def total_tokens(self) -> int:
        """Estimate total tokens across all stored turns."""
        return sum(estimate_turn_tokens(t) for t in self.buffer)

    @property
    def needs_compaction(self) -> bool:
        """Check if the transcript exceeds the compaction threshold."""
        return self.total_tokens > self.compaction_threshold

    def get_context(
        self,
        target_scope: str = "PUBLIC",
        user_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        max_turns: int = 50,
        max_tokens: int = 0,
    ) -> str:
        """
        Get formatted conversation history filtered by scope/user/channel.

        Args:
            target_scope: PUBLIC or PRIVATE
            user_id: Filter by user (for PRIVATE scope)
            channel_id: Filter by channel
            max_turns: Maximum turns to return (default 50)
            max_tokens: If > 0, limit by token budget instead of turn count

        Returns a string suitable for system prompt injection.
        """
        relevant = self._filter_turns(target_scope, user_id, channel_id)

        if not relevant:
            return "No conversation history."

        # Select turns: by token budget or by count
        if max_tokens > 0:
            recent = self._select_by_tokens(relevant, max_tokens)
        else:
            recent = relevant[-max_turns:]

        # Prepend any compaction summaries for continuity
        lines = []
        for cs in self.compaction_summaries:
            if cs.scope == target_scope or target_scope == "PUBLIC":
                if not user_id or cs.user_id == user_id:
                    lines.append(f"[Previous conversation summary: {cs.summary}]")

        for t in recent:
            ts = datetime.fromtimestamp(t.timestamp).strftime("%H:%M")
            lines.append(f"[{ts}] {t.user_name}: {t.user_message}")
            lines.append(f"[{ts}] Bot: {t.bot_message}")
        return "\n".join(lines)

    def get_conversation_history(self, max_turns: int = 50) -> List[Dict[str, str]]:
        """
        Get conversation history in LLM message format.
        Returns list of {"role": "user"/"assistant", "content": "..."} dicts.
        """
        history = []
        for t in self.buffer[-max_turns:]:
            history.append({"role": "user", "content": t.user_message})
            history.append({"role": "assistant", "content": t.bot_message})
        return history

    def get_recent_turns(self, n: int = 5) -> List[Turn]:
        """Return the last N turns (unfiltered)."""
        return self.buffer[-n:]

    def get_user_facts(self, user_id: str) -> List[str]:
        """Extract all messages from a specific user across the full buffer."""
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

    @property
    def turn_count(self) -> int:
        return len(self.buffer)

    # Backwards compat: expose buffer as _turns for PHUD access
    @property
    def _turns(self) -> List[Turn]:
        return self.buffer

    def snapshot(self) -> Dict[str, Any]:
        """Return diagnostic snapshot."""
        return {
            "turn_count": len(self.buffer),
            "total_tokens": self.total_tokens,
            "compaction_threshold": self.compaction_threshold,
            "needs_compaction": self.needs_compaction,
            "compaction_count": len(self.compaction_summaries),
            "persist_path": self.persist_path,
            "oldest": self.buffer[0].timestamp if self.buffer else None,
            "newest": self.buffer[-1].timestamp if self.buffer else None,
        }

    # ─── Compaction API ────────────────────────────────────────

    def get_turns_for_compaction(self) -> List[Turn]:
        """
        Return turns that should be compacted (everything except
        the most recent COMPACTION_KEEP_RECENT turns).
        """
        if len(self.buffer) <= COMPACTION_KEEP_RECENT:
            return []
        return self.buffer[:-COMPACTION_KEEP_RECENT]

    def apply_compaction(self, summary: CompactionSummary):
        """
        Apply a compaction result: remove compacted turns from buffer,
        store the summary for future context injection.
        """
        turns_to_keep = self.buffer[-COMPACTION_KEEP_RECENT:]
        compacted_count = len(self.buffer) - len(turns_to_keep)

        self.compaction_summaries.append(summary)
        self.buffer = turns_to_keep
        self._save_to_disk()

        logger.info(f"Compaction applied: removed {compacted_count} turns, "
                     f"kept {len(turns_to_keep)}, "
                     f"~{self.total_tokens} tokens remaining")

    def clear(self):
        """Clear all turns and compaction summaries (for /new command)."""
        self.buffer.clear()
        self.compaction_summaries.clear()
        self._save_to_disk()
        logger.info("ContextStream cleared (fresh start)")

    # ─── Internal ──────────────────────────────────────────────

    def _filter_turns(self, target_scope: str, user_id: Optional[str],
                      channel_id: Optional[int]) -> List[Turn]:
        """Filter turns by scope, user, and channel."""
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

        return relevant

    def _select_by_tokens(self, turns: List[Turn], max_tokens: int) -> List[Turn]:
        """Select as many recent turns as fit within a token budget."""
        selected = []
        token_count = 0
        for t in reversed(turns):
            t_tokens = estimate_turn_tokens(t)
            if token_count + t_tokens > max_tokens:
                break
            selected.append(t)
            token_count += t_tokens
        selected.reverse()
        return selected

    # ─── Persistence ───────────────────────────────────────────

    def _save_to_disk(self):
        """Persist current buffer + compaction summaries to JSONL."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                # Write compaction summaries first (prefixed with type marker)
                for cs in self.compaction_summaries:
                    entry = {"_type": "compaction", **asdict(cs)}
                    f.write(json.dumps(entry) + "\n")
                # Write turns
                for t in self.buffer:
                    entry = {"_type": "turn", **asdict(t)}
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist ContextStream: {e}")

    def _load_from_disk(self):
        """Load turns and compaction summaries from JSONL on startup."""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        entry_type = data.pop("_type", "turn")
                        if entry_type == "compaction":
                            self.compaction_summaries.append(CompactionSummary(**data))
                        else:
                            self.buffer.append(Turn(**data))
                logger.info(f"Loaded {len(self.buffer)} turns + "
                             f"{len(self.compaction_summaries)} compaction summaries from disk")
        except Exception as e:
            logger.warning(f"Failed to load ContextStream from disk: {e}")
