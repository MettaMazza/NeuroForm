"""
Basal Ganglia — Habit Formation & Procedural Memory
=====================================================

Biological basis: The basal ganglia converts conscious, effortful actions into
automatic habits through repeated reinforcement. Once habituated, behaviors
run without prefrontal involvement (Yin & Knowlton, 2006).

Computational analogue: Track invocation frequency on PROCEDURAL layer nodes.
When a procedure exceeds a threshold, it becomes a cached "muscle memory"
template that bypasses full LLM reasoning for significant latency savings.
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class HabitCache:
    """
    The Basal Ganglia — Procedural Memory Cache.

    Tracks how often procedures are invoked. When frequency exceeds
    a threshold, the procedure is promoted to a cached template
    that short-circuits the full LLM inference loop.
    """

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self._invocations: Dict[str, int] = {}
        self._cache: Dict[str, str] = {}

    def record_invocation(self, procedure_key: str, response: str) -> bool:
        """
        Record a procedure invocation. If the threshold is exceeded,
        cache the response as a habit template.

        Returns True if the procedure was promoted to a habit.
        """
        self._invocations[procedure_key] = self._invocations.get(procedure_key, 0) + 1
        count = self._invocations[procedure_key]

        if count >= self.threshold and procedure_key not in self._cache:
            self._cache[procedure_key] = response
            logger.info(f"Habit formed: '{procedure_key}' cached after {count} invocations")
            return True

        return False

    def get_cached_response(self, procedure_key: str) -> Optional[str]:
        """
        Check if a cached habit exists for the given procedure.
        Returns the cached template or None.
        """
        return self._cache.get(procedure_key)

    def is_habituated(self, procedure_key: str) -> bool:
        """Check if a procedure has been promoted to a habit."""
        return procedure_key in self._cache

    def get_invocation_count(self, procedure_key: str) -> int:
        """Get the current invocation count for a procedure."""
        return self._invocations.get(procedure_key, 0)

    def clear_habit(self, procedure_key: str):
        """Remove a habit from the cache (de-habituate)."""
        self._cache.pop(procedure_key, None)
        self._invocations.pop(procedure_key, None)

    def snapshot(self) -> Dict[str, Any]:
        """Diagnostic snapshot of the habit system."""
        return {
            "habits": list(self._cache.keys()),
            "total_procedures_tracked": len(self._invocations),
            "threshold": self.threshold,
            "invocations": dict(self._invocations),
        }
