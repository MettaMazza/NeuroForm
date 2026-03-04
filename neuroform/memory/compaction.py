"""
Auto-Compaction Engine — Token-Based Context Window Management
================================================================

When the ContextStream transcript exceeds the token threshold (default 110k),
this module:
1. Takes all turns except the 50 most recent
2. Summarizes them via LLM call
3. Vector-embeds the summary + individual messages into the VectorStore
   (per-user, per-scope)
4. Persists a CompactionSummary in the ContextStream
5. Removes compacted turns from the in-memory buffer

Inspired by ErnOS compaction architecture.
"""
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from neuroform.memory.context_stream import (
    ContextStream, CompactionSummary, estimate_turn_tokens, COMPACTION_KEEP_RECENT,
)

if TYPE_CHECKING:
    from neuroform.memory.vector_store import VectorStore
    from neuroform.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ─── Compaction Prompt ─────────────────────────────────────────

COMPACTION_PROMPT = """You are summarizing a conversation transcript into a concise memory summary.
Preserve ALL of the following:
- User names, roles, and relationships
- Key facts stated by users (preferences, identities, decisions)
- Important context and topics discussed
- Commitments, promises, and action items
- Emotional tone and relationship dynamics

Be concise but LOSE NOTHING IMPORTANT. This summary replaces the original transcript.
Write in past tense, as a factual record.

TRANSCRIPT:
{transcript}

SUMMARY:"""


def format_turns_for_compaction(turns) -> str:
    """Format turns into a readable transcript for the LLM to summarize."""
    lines = []
    for t in turns:
        ts = datetime.fromtimestamp(t.timestamp).strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{ts}] {t.user_name} ({t.user_id}): {t.user_message}")
        lines.append(f"[{ts}] Bot: {t.bot_message}")
    return "\n".join(lines)


async def compact_context(
    context_stream: ContextStream,
    vector_store: "VectorStore",
    llm_client: "OllamaClient",
    model: str = "gemma3:4b",
    user_id: Optional[str] = None,
    scope: str = "PUBLIC",
) -> Optional[CompactionSummary]:
    """
    Run auto-compaction if the transcript exceeds the token threshold.

    1. Summarize old turns via LLM
    2. Vector-embed the summary + individual messages
    3. Apply the compaction to the ContextStream

    Returns the CompactionSummary if compaction was performed, None otherwise.
    """
    if not context_stream.needs_compaction:
        logger.debug("Compaction not needed — under threshold")
        return None

    turns_to_compact = context_stream.get_turns_for_compaction()
    if not turns_to_compact:
        logger.debug("No turns to compact")
        return None

    tokens_before = context_stream.total_tokens
    logger.info(f"Starting compaction: {len(turns_to_compact)} turns, "
                 f"~{tokens_before} tokens")

    # ── Step 1: Summarize via LLM ──
    transcript = format_turns_for_compaction(turns_to_compact)
    prompt = COMPACTION_PROMPT.format(transcript=transcript)

    try:
        summary_text = llm_client.generate_raw(prompt, model=model)
    except Exception as e:
        logger.error(f"Compaction LLM call failed: {e}")
        # Fallback: create a simple concatenated summary
        summary_text = _fallback_summary(turns_to_compact)

    # ── Step 2: Vector-embed for searchability ──
    try:
        # Embed the summary itself
        vector_store.store(
            text=summary_text,
            metadata={
                "type": "compaction_summary",
                "turns_compacted": len(turns_to_compact),
                "scope": scope,
                "user_id": user_id or "unknown",
            },
            scope=scope,
            user_id=user_id,
        )

        # Embed individual messages for granular search
        for turn in turns_to_compact:
            combined = f"{turn.user_name}: {turn.user_message}\nBot: {turn.bot_message}"
            vector_store.store(
                text=combined,
                metadata={
                    "type": "compacted_turn",
                    "user_id": turn.user_id,
                    "user_name": turn.user_name,
                    "scope": turn.scope,
                    "timestamp": turn.timestamp,
                },
                scope=turn.scope,
                user_id=turn.user_id,
            )
    except Exception as e:
        logger.error(f"Vector embedding during compaction failed: {e}")
        # Continue anyway — the summary is still valuable

    # ── Step 3: Apply compaction ──
    summary = CompactionSummary(
        summary=summary_text,
        turns_compacted=len(turns_to_compact),
        tokens_before=tokens_before,
        tokens_after=context_stream.total_tokens - sum(
            estimate_turn_tokens(t) for t in turns_to_compact
        ),
        timestamp=datetime.now().timestamp(),
        user_id=user_id or "unknown",
        scope=scope,
    )

    context_stream.apply_compaction(summary)

    logger.info(f"Compaction complete: {len(turns_to_compact)} turns → summary, "
                 f"~{context_stream.total_tokens} tokens remaining")

    return summary


def _fallback_summary(turns) -> str:
    """Create a basic summary without LLM when the LLM call fails."""
    user_messages = []
    for t in turns:
        user_messages.append(f"- {t.user_name}: {t.user_message[:100]}")
        if len(user_messages) >= 20:
            break

    return (
        f"Summary of {len(turns)} conversation turns. "
        f"Key messages:\n" + "\n".join(user_messages)
    )
