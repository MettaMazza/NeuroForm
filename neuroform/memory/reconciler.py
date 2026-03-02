"""
Cross-Tier Memory Reconciler
===============================

Post-retrieval conflict detection between Lessons (T4), Knowledge Graph (T3),
and Vector Store (T2). Ported from ErnOS V3 (src/memory/reconciler.py).

Uses an Ollama LLM call for authority-weighted analysis:
  - T4 Lessons: ABSOLUTE AUTHORITY
  - T3 KG Facts: High Authority
  - T2 Vector Memories: Low Authority

Detects contradictions and annotates stale/rejected entries so the main
inference model can make informed epistemic judgments.

Made synchronous (ErnOS V3 was async) to match NeuroForm's pipeline.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import ollama as _ollama
except ImportError:
    _ollama = None  # type: ignore


# ─── LLM Prompt ──────────────────────────────────────────────────

RECONCILIATION_PROMPT = """You are a memory reconciliation auditor for an AI system.
You are given three sets of retrieved context about the same topic:

1. TIER 4 LESSONS — Universal truths, verified user lessons (ABSOLUTE AUTHORITY).
2. TIER 3 KNOWLEDGE GRAPH FACTS — Structured, curated facts (High Authority).
3. TIER 2 VECTOR MEMORIES — Semantic conversational history (Low Authority).

Your job: Identify cases where lower-tier data contradicts higher-tier data.

CONTRADICTION RULES:
1. T4 SUPREMACY: If Vector/KG contradicts a Lesson, the Lesson is correct.
2. KG OVER VECTOR: If Vector contradicts KG (and no Lesson applies), KG wins.
3. TEMPORAL: Newer KG facts supersede older Vector memories.

RESPOND WITH ONE LINE PER FINDING:
- If NO conflicts: CONSISTENT
- For each conflict: CONFLICT:<tier>:<index>|<reason in under 15 words>
  (tier is "VM" for vector or "KG" for knowledge graph)

Examples:
CONSISTENT
CONFLICT:VM:0|Lesson says user is vegan, VM claims steak preference
CONFLICT:KG:1|Lesson dictates privacy; KG fact reveals PII"""


@dataclass
class ConflictRecord:
    """A detected cross-tier disagreement."""
    target_text: str
    authority_text: str
    conflict_type: str   # "VM" or "KG"
    reason: str = ""


@dataclass
class ReconciliationResult:
    """Output of the reconciliation step."""
    lessons: List[str]
    kg_facts: List[str]
    vector_texts: List[str]
    conflicts: List[ConflictRecord] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


class CrossTierReconciler:
    """
    Ollama-powered conflict detection between Lessons (T4), KG (T3),
    and Vector Store (T2).

    1. Formats all tiers into a structured hierarchy.
    2. Sends to Ollama for authority-weighted analysis.
    3. Parses structured verdicts.
    4. Annotates conflicting entries with [⚠️STALE?] or [🛑REJECTED].
    """

    def __init__(self, model: str = "gemma3:4b"):
        self.model = model
        logger.info(f"CrossTierReconciler initialized (model={model})")

    def reconcile(
        self,
        lessons: Optional[List[str]] = None,
        kg_facts: Optional[List[str]] = None,
        vector_texts: Optional[List[str]] = None,
    ) -> ReconciliationResult:
        """Run cross-tier reconciliation."""
        lessons = lessons or []
        kg_facts = kg_facts or []
        vector_texts = vector_texts or []

        # Quick exit: nothing to compare
        if not (lessons or kg_facts) or not (kg_facts or vector_texts):
            return ReconciliationResult(
                lessons=lessons,
                kg_facts=kg_facts,
                vector_texts=vector_texts,
                stats={
                    "lessons": len(lessons), "kg": len(kg_facts),
                    "vector": len(vector_texts), "conflicts": 0,
                },
            )

        # Build prompt
        reconciliation_input = self._build_input(lessons, kg_facts, vector_texts)

        # Call LLM
        verdict_text = self._call_llm(reconciliation_input)
        if not verdict_text:
            return ReconciliationResult(
                lessons=lessons, kg_facts=kg_facts,
                vector_texts=vector_texts,
                stats={
                    "lessons": len(lessons), "kg": len(kg_facts),
                    "vector": len(vector_texts), "conflicts": 0,
                },
            )

        # Parse
        conflicts = self._parse_verdicts(
            verdict_text, lessons, kg_facts, vector_texts)

        # Annotate
        annotated_kg = list(kg_facts)
        annotated_vectors = list(vector_texts)

        for conflict in conflicts:
            if conflict.conflict_type == "KG":
                try:
                    idx = kg_facts.index(conflict.target_text)
                    annotated_kg[idx] = f"[🛑REJECTED] {annotated_kg[idx]}"
                except ValueError:
                    pass
            else:
                try:
                    idx = vector_texts.index(conflict.target_text)
                    annotated_vectors[idx] = f"[⚠️STALE?] {annotated_vectors[idx]}"
                except ValueError:
                    pass

        return ReconciliationResult(
            lessons=lessons,
            kg_facts=annotated_kg,
            vector_texts=annotated_vectors,
            conflicts=conflicts,
            stats={
                "lessons": len(lessons), "kg": len(kg_facts),
                "vector": len(vector_texts), "conflicts": len(conflicts),
            },
        )

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama for reconciliation. Returns empty string on failure."""
        if _ollama is None:
            logger.warning("Reconciler: ollama not installed")
            return ""
        try:
            result = _ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": RECONCILIATION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.warning(f"Reconciler: LLM call failed ({e})")
            return ""

    def _build_input(
        self, lessons: List[str], kg_facts: List[str],
        vector_texts: List[str],
    ) -> str:
        """Build the structured reconciliation prompt."""
        parts = ["TIER 4 LESSONS (AUTHORITY):"]
        for i, lesson in enumerate(lessons):
            parts.append(f"  LS[{i}]: {lesson}")

        parts.append("\nTIER 3 KG FACTS:")
        for i, fact in enumerate(kg_facts):
            parts.append(f"  KG[{i}]: {fact}")

        parts.append("\nTIER 2 VECTOR MEMORIES:")
        for i, vm in enumerate(vector_texts):
            parts.append(f"  VM[{i}]: {vm[:400]}")

        parts.append("\nAnalyze hierarchy for contradictions.")
        return "\n".join(parts)

    def _parse_verdicts(
        self, verdict_text: str, lessons: List[str],
        kg_facts: List[str], vector_texts: List[str],
    ) -> List[ConflictRecord]:
        """Parse LLM verdict lines into ConflictRecords."""
        conflicts = []
        for line in verdict_text.strip().split("\n"):
            match = re.match(
                r"CONFLICT:(VM|KG):(\d+)\|(.+)", line, re.IGNORECASE)
            if match:
                tier = match.group(1).upper()
                idx = int(match.group(2))
                reason = match.group(3).strip()

                target_list = vector_texts if tier == "VM" else kg_facts
                if 0 <= idx < len(target_list):
                    conflicts.append(ConflictRecord(
                        target_text=target_list[idx],
                        authority_text=lessons[0] if lessons else "Hierarchy",
                        conflict_type=tier,
                        reason=reason,
                    ))
        return conflicts

    def snapshot(self) -> dict:
        """Diagnostic snapshot."""
        return {
            "model": self.model,
            "ollama_available": _ollama is not None,
        }
