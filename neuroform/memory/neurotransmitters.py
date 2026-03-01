"""
Neurotransmitter Systems — Behavioral Modulation
==================================================

Biological basis: Dopamine, serotonin, norepinephrine, and acetylcholine
modulate global brain behavior — not specific memories, but the *mode*
of processing (Schultz, 1998).

Computational analogue: A dynamic state object that adjusts system parameters
like LLM temperature, decay rates, attention budgets, and learning rates
based on contextual signals.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class NeurotransmitterState:
    """
    Global behavioral modulation through simulated neurotransmitter levels.

    Each transmitter level (0.0-1.0) modulates a different system parameter:
    - Dopamine: exploration_rate → LLM temperature
    - Serotonin: stability → response verbosity/patience
    - Norepinephrine: alertness → attention budget tightness
    - Acetylcholine: learning_rate → plasticity speed
    """

    def __init__(self):
        self.dopamine: float = 0.5
        self.serotonin: float = 0.5
        self.norepinephrine: float = 0.5
        self.acetylcholine: float = 0.5

    def set_level(self, transmitter: str, level: float):
        """Set a neurotransmitter level (clamped to 0.0-1.0)."""
        level = max(0.0, min(1.0, level))
        if hasattr(self, transmitter):
            setattr(self, transmitter, level)
            logger.debug(f"NT: {transmitter} → {level:.2f}")
        else:
            logger.warning(f"Unknown neurotransmitter: {transmitter}")

    @property
    def llm_temperature(self) -> float:
        """Dopamine modulates exploration: high DA = more creative/divergent."""
        return 0.3 + (self.dopamine * 0.7)  # Range: 0.3 - 1.0

    @property
    def decay_rate(self) -> float:
        """Acetylcholine modulates learning: high ACh = faster plasticity."""
        return 0.05 + (self.acetylcholine * 0.15)  # Range: 0.05 - 0.20

    @property
    def attention_budget(self) -> int:
        """Norepinephrine modulates focus: high NE = tighter salience filter."""
        return max(3, int(15 - self.norepinephrine * 10))  # Range: 5 - 15

    @property
    def verbosity(self) -> float:
        """Serotonin modulates patience: high 5-HT = more detailed responses."""
        return 0.5 + (self.serotonin * 0.5)  # Range: 0.5 - 1.0

    def modulate_from_sentiment(self, sentiment: float):
        """
        Adjust neurotransmitter levels based on detected user sentiment.
        sentiment: -1.0 (very negative) to +1.0 (very positive)
        """
        if sentiment > 0.3:
            self.dopamine = min(1.0, self.dopamine + 0.1)
            self.serotonin = min(1.0, self.serotonin + 0.05)
        elif sentiment < -0.3:
            self.norepinephrine = min(1.0, self.norepinephrine + 0.1)
            self.serotonin = max(0.0, self.serotonin - 0.05)
        # Neutral sentiment: no change

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "acetylcholine": self.acetylcholine,
            "derived": {
                "llm_temperature": self.llm_temperature,
                "decay_rate": self.decay_rate,
                "attention_budget": self.attention_budget,
                "verbosity": self.verbosity,
            }
        }

    def reset(self):
        """Reset all levels to baseline (0.5)."""
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.norepinephrine = 0.5
        self.acetylcholine = 0.5
