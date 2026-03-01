"""
Circadian Rhythms — Time-Based Behavioral Modulation
=====================================================

Biological basis: The suprachiasmatic nucleus drives 24-hour cycles
affecting alertness, memory consolidation, and mood. The brain operates
differently at 3am vs 3pm.

Computational analogue: Time-aware parameter modulation that adjusts
system behavior based on the current hour — triggering dream consolidation
at night, increasing attention during peak hours, and enabling creative
exploration during low-activity periods.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CircadianProfile:
    """
    Time-based behavioral modulation.

    Maps hours of the day to parameter multipliers that affect
    the neurotransmitter state, dream consolidation triggers,
    and attention budgets.
    """

    # Default profile: inspired by typical circadian alertness curves
    DEFAULT_PROFILE = {
        # hour: (alertness, consolidation_trigger, exploration_boost)
        0: (0.2, True, 0.3),   1: (0.15, True, 0.4),   2: (0.1, True, 0.5),
        3: (0.1, True, 0.5),   4: (0.15, True, 0.4),   5: (0.2, True, 0.3),
        6: (0.4, False, 0.2),  7: (0.6, False, 0.1),   8: (0.8, False, 0.0),
        9: (0.9, False, 0.0),  10: (1.0, False, 0.0),  11: (0.95, False, 0.0),
        12: (0.8, False, 0.1), 13: (0.7, False, 0.15), 14: (0.75, False, 0.1),
        15: (0.85, False, 0.0), 16: (0.9, False, 0.0), 17: (0.8, False, 0.1),
        18: (0.7, False, 0.15), 19: (0.6, False, 0.2), 20: (0.5, False, 0.25),
        21: (0.4, False, 0.3), 22: (0.3, True, 0.35),  23: (0.25, True, 0.35),
    }

    def __init__(self, custom_profile: Optional[Dict[int, tuple]] = None):
        self.profile = custom_profile or self.DEFAULT_PROFILE

    def get_modulation(self, hour: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the current circadian modulation parameters.

        Args:
            hour: Override hour (0-23). If None, uses current system time.

        Returns:
            Dict with alertness, should_consolidate, and exploration_boost.
        """
        if hour is None:
            hour = datetime.now().hour

        hour = hour % 24
        alertness, consolidate, exploration = self.profile.get(hour, (0.5, False, 0.2))

        return {
            "hour": hour,
            "alertness": alertness,
            "should_consolidate": consolidate,
            "exploration_boost": exploration,
        }

    def should_dream_now(self, hour: Optional[int] = None) -> bool:
        """Check if the current time phase should trigger dream consolidation."""
        return self.get_modulation(hour)["should_consolidate"]

    def get_alertness(self, hour: Optional[int] = None) -> float:
        """Get the current alertness level (0.0 - 1.0)."""
        return self.get_modulation(hour)["alertness"]

    def get_exploration_boost(self, hour: Optional[int] = None) -> float:
        """Get the current exploration/creativity boost (0.0 - 0.5)."""
        return self.get_modulation(hour)["exploration_boost"]

    def apply_to_neurotransmitters(self, nt_state, hour: Optional[int] = None):
        """
        Modulate a NeurotransmitterState based on circadian phase.
        """
        mod = self.get_modulation(hour)

        # Alertness → norepinephrine
        nt_state.norepinephrine = mod["alertness"]
        # Exploration → dopamine boost
        nt_state.dopamine = min(1.0, 0.5 + mod["exploration_boost"])
        # Night phase → higher acetylcholine (learning during consolidation)
        if mod["should_consolidate"]:
            nt_state.acetylcholine = 0.8

        logger.debug(f"Circadian modulation applied for hour {mod['hour']}")
