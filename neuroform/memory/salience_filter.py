"""
Reticular Activating System — Attention & Salience Filtering
=============================================================

Biological basis: The RAS filters overwhelming sensory input, allowing only
salient stimuli to reach conscious awareness. Without it, the brain
would be flooded with irrelevant data.

Computational analogue: A salience scorer that ranks candidate graph context
by relevance before injecting it into the LLM prompt. Uses lightweight
embedding similarity, strength weighting, and recency to build a
token-efficient context window.
"""
import logging
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)


class SalienceScorer:
    """
    The Reticular Activating System.

    Filters and ranks graph context candidates by salience before
    they enter the attention window (Working Memory). Prevents the
    LLM from being overwhelmed with irrelevant old memories.
    """

    def __init__(self, attention_budget: int = 10,
                 recency_weight: float = 0.3,
                 strength_weight: float = 0.4,
                 relevance_weight: float = 0.3):
        self.attention_budget = attention_budget
        self.recency_weight = recency_weight
        self.strength_weight = strength_weight
        self.relevance_weight = relevance_weight

    def score_candidates(self, message: str,
                         candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score and rank graph context candidates by salience.

        Each candidate is expected to have:
        - source, target, relationship, strength
        - Optionally: last_fired (timestamp)

        Returns the top-K candidates within the attention budget,
        each augmented with a 'salience_score' field.
        """
        scored = []
        message_tokens = set(message.lower().split())

        for candidate in candidates:
            score = self._compute_salience(message_tokens, candidate)
            candidate_copy = dict(candidate)
            candidate_copy["salience_score"] = score
            scored.append(candidate_copy)

        scored.sort(key=lambda x: x["salience_score"], reverse=True)
        return scored[:self.attention_budget]

    def _compute_salience(self, message_tokens: set,
                          candidate: Dict[str, Any]) -> float:
        """
        Compute a salience score for a single candidate.
        Combines keyword relevance, connection strength, and recency.
        """
        # 1. Keyword relevance (simple token overlap)
        source = str(candidate.get("source", "")).lower()
        target = str(candidate.get("target", "")).lower()
        rel = str(candidate.get("relationship", "")).lower().replace("_", " ")

        candidate_tokens = set(source.split() + target.split() + rel.split())
        overlap = len(message_tokens & candidate_tokens)
        max_possible = max(len(message_tokens), 1)
        relevance = min(overlap / max_possible, 1.0)

        # 2. Strength (normalized loosely)
        strength = min(float(candidate.get("strength", 1.0)) / 5.0, 1.0)

        # 3. Recency (based on last_fired timestamp)
        last_fired = candidate.get("last_fired")
        if last_fired and isinstance(last_fired, (int, float)):
            age_seconds = (time.time() * 1000 - last_fired) / 1000
            recency = 1.0 / (1.0 + age_seconds / 3600)
        else:
            recency = 0.5  # Default for missing timestamps

        return (relevance * self.relevance_weight +
                strength * self.strength_weight +
                recency * self.recency_weight)

    def filter_by_budget(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple budget filter without scoring — just truncate to budget."""
        return candidates[:self.attention_budget]
