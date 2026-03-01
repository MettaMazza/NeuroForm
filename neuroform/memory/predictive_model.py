"""
Cerebellum — Predictive Error Correction
==========================================

Biological basis: The cerebellum continuously generates predictions about
expected outcomes. When reality diverges from prediction, error signals
update internal models (Wolpert et al., 1998).

Computational analogue: Before the LLM responds, generate a prediction of
the user's likely intent from graph context. After the response, compare
the prediction against the actual user follow-up. High prediction error
weakens the context pathways; low error strengthens them.
"""
import logging
from typing import Dict, Any, List, Optional
import json
import ollama
from neuroform.memory.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class PredictiveModel:
    """
    The Cerebellum — Predictive Error Correction.

    Generates intent predictions before responses and evaluates error
    against actual follow-ups. Feeds error signals back into the
    neuroplasticity loop as STRENGTHEN/DECAY signals.
    """

    def __init__(self, kg: KnowledgeGraph, model: str = "llama3"):
        self.kg = kg
        self.model = model
        self._last_prediction: Optional[str] = None
        self._last_context_sources: List[str] = []

    def predict_intent(self, context: str, conversation_history: str) -> str:
        """
        Given graph context and conversation history, predict what the
        user likely wants next.
        """
        prompt = f"""Based on this context and conversation, predict in one sentence 
what the user is most likely to say or ask next.

Context: {context}
Conversation: {conversation_history}

Prediction (one sentence):"""
        try:
            response = ollama.chat(model=self.model, messages=[
                {"role": "system", "content": "You predict user intent in one sentence."},
                {"role": "user", "content": prompt}
            ])
            prediction = response["message"]["content"].strip()
            self._last_prediction = prediction
            return prediction
        except Exception as e:
            logger.warning(f"Cerebellum prediction failed: {e}")
            self._last_prediction = None
            return ""

    def evaluate_error(self, predicted: str, actual: str) -> float:
        """
        Compare the prediction against the actual user follow-up.
        Returns a normalized error score (0.0 = perfect match, 1.0 = complete mismatch).

        Uses simple token overlap as a lightweight proxy for semantic similarity.
        """
        if not predicted or not actual:
            return 0.5  # No data → neutral error

        pred_tokens = set(predicted.lower().split())
        actual_tokens = set(actual.lower().split())

        if not pred_tokens or not actual_tokens:
            return 0.5

        overlap = len(pred_tokens & actual_tokens)
        max_len = max(len(pred_tokens), len(actual_tokens))

        similarity = overlap / max_len
        return 1.0 - similarity  # Error = 1 - similarity

    def generate_feedback_signal(self, error: float,
                                  context_sources: List[str]) -> List[Dict[str, Any]]:
        """
        Convert prediction error into neuroplasticity feedback signals.

        Low error → STRENGTHEN context sources (good predictions)
        High error → DECAY context sources (misleading context)
        """
        signals = []
        for source in context_sources:
            if error < 0.3:
                signals.append({"action": "STRENGTHEN", "target": source, "amount": 0.2})
            elif error > 0.7:
                signals.append({"action": "DECAY", "target": source, "amount": 0.15})

        return signals

    def set_context_sources(self, sources: List[str]):
        """Record which context sources were used for the current prediction."""
        self._last_context_sources = sources

    @property
    def last_prediction(self) -> Optional[str]:
        return self._last_prediction

    @property
    def last_context_sources(self) -> List[str]:
        return self._last_context_sources
