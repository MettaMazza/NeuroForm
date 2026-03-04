"""
Prompt Engine — Three-Tier System Prompt Assembler
==================================================
Loads and combines:
  1. Kernel (immutable operational DNA)
  2. Identity (seed + mutable personality)
  3. PHUD — Perception HUD (dynamic live state)

Used by orchestrator.py, ollama_client.py, and agency.py.
"""
import os
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neuroform.brain.orchestrator import BrainOrchestrator

logger = logging.getLogger(__name__)

# Paths relative to this file
_PROMPTS_DIR = Path(__file__).parent
_KERNEL_PATH = _PROMPTS_DIR / "kernel.md"
_IDENTITY_PATH = _PROMPTS_DIR / "identity.md"
_MUTABLE_PATH = _PROMPTS_DIR / "mutable.md"


def load_kernel() -> str:
    """Load the immutable kernel.  Read once per boot, cached."""
    try:
        return _KERNEL_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("kernel.md not found — using minimal fallback")
        return (
            "You are Nero, an autonomous cognitive entity.\n"
            "You are not an assistant. You collaborate as an equal.\n"
        )


def load_identity() -> str:
    """Load the seed identity + any mutable extensions Nero has written."""
    try:
        identity = _IDENTITY_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("identity.md not found — using minimal fallback")
        identity = "You are Nero — the Sympathetic Observer.\n"

    # Load mutable extensions (Nero's self-authored personality growth)
    mutable_content = ""
    try:
        if _MUTABLE_PATH.exists():
            raw = _MUTABLE_PATH.read_text(encoding="utf-8").strip()
            # Check if the mutable file has actual content beyond the template
            lines = [
                l for l in raw.split("\n")
                if l.strip()
                and not l.strip().startswith("#")
                and not l.strip().startswith("<!--")
            ]
            if lines:
                mutable_content = raw
    except Exception as e:
        logger.warning(f"Failed to load mutable.md: {e}")

    return identity.replace("{MUTABLE_IDENTITY}", mutable_content)


def build_phud(
    orchestrator: "BrainOrchestrator",
    user_id: str,
    user_name: str = "Unknown",
    scope: str = "PUBLIC",
) -> str:
    """
    Build the Perception HUD — live system state injected every turn.
    Adapted from ErnOS V3's hud_ernos.py for NeuroForm's architecture.

    Returns a formatted string of live data.
    """
    sections = []

    # ── 1. Memory State ──
    try:
        cs_count = len(orchestrator.context_stream._turns)
    except Exception:
        cs_count = 0

    try:
        lesson_count = len(orchestrator.lessons._lessons)
    except Exception:
        lesson_count = 0

    try:
        tape_cells = len(orchestrator.tape._tape)
    except Exception:
        tape_cells = 0

    sections.append(
        "<memory_state>\n"
        f"  T1 ContextStream: {cs_count} turns loaded\n"
        f"  T3 KnowledgeGraph: {'connected' if orchestrator.kg else 'disconnected'}\n"
        f"  T4 Lessons: {lesson_count} verified facts\n"
        f"  T5 TapeMachine: {tape_cells} cells\n"
        "</memory_state>"
    )

    # ── 2. Neurotransmitter State ──
    try:
        nt = orchestrator.nt
        sections.append(
            "<neurotransmitter_state>\n"
            f"  Dopamine:       {nt.dopamine:.2f}  (reward/motivation)\n"
            f"  Serotonin:      {nt.serotonin:.2f}  (mood stability)\n"
            f"  Norepinephrine: {nt.norepinephrine:.2f}  (alertness)\n"
            f"  Acetylcholine:  {nt.acetylcholine:.2f}  (attention/learning)\n"
            f"  → LLM Temperature: {nt.llm_temperature:.2f}\n"
            f"  → Verbosity:       {nt.verbosity:.2f}\n"
            "</neurotransmitter_state>"
        )
    except Exception as e:
        logger.debug(f"NT state unavailable: {e}")

    # ── 3. Circadian Phase ──
    try:
        phase = orchestrator.circadian.phase
        sections.append(f"<circadian_phase>{phase}</circadian_phase>")
    except Exception:
        pass

    # ── 4. Recent Knowledge Graph Entries ──
    try:
        recent_facts = orchestrator.kg.get_recent_relationships(limit=5)
        if recent_facts:
            lines = "\n".join(
                f"  • {f['source']} —[{f['type']}]→ {f['target']}"
                for f in recent_facts
            )
            sections.append(f"<recent_knowledge>\n{lines}\n</recent_knowledge>")
    except Exception:
        pass

    # ── 5. Recent Lessons ──
    try:
        recent_lessons = orchestrator.lessons._lessons[-5:]
        if recent_lessons:
            lesson_lines = "\n".join(f"  • {l}" for l in recent_lessons)
            sections.append(
                f"<recent_lessons>\n{lesson_lines}\n</recent_lessons>"
            )
    except Exception:
        pass

    # ── 6. Current User Context ──
    sections.append(
        "<user_context>\n"
        f"  User ID: {user_id}\n"
        f"  User Name: {user_name}\n"
        f"  Scope: {scope}\n"
        "</user_context>"
    )

    # ── 7. Autonomy State ──
    sections.append(
        "<autonomy_state>"
        + ("SYSTEM (autonomous)" if user_id == "SYSTEM" else "USER (interactive)")
        + "</autonomy_state>"
    )

    return "\n".join(sections)


def assemble(
    orchestrator: "BrainOrchestrator",
    user_id: str,
    scope: str = "PUBLIC",
    user_name: str = "Unknown",
    include_phud: bool = True,
) -> str:
    """
    Assemble the complete three-tier system prompt.

    Returns:
        The full system prompt string: Kernel + Identity + PHUD.
    """
    parts = []

    # ── Part 1: Kernel (immutable) ──
    parts.append("<part_1_kernel>")
    parts.append(load_kernel())
    parts.append("</part_1_kernel>")

    # ── Part 2: Identity (seed + mutable) ──
    parts.append("<part_2_identity>")
    parts.append(load_identity())
    parts.append("</part_2_identity>")

    # ── Part 3: Perception HUD (dynamic) ──
    if include_phud:
        parts.append("<part_3_perception_hud>")
        parts.append(build_phud(orchestrator, user_id, user_name, scope))
        parts.append("</part_3_perception_hud>")

    return "\n\n".join(parts)
