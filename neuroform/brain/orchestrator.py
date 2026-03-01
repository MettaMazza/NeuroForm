"""
BrainOrchestrator — Central Nervous System
=============================================

Coordinates all 9 brain systems in a neuroscience-accurate processing
pipeline. Every incoming message flows through the full cognitive loop.
"""
import logging
import time
from typing import Dict, Any, Optional, List

from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.memory.working_memory import WorkingMemory
from neuroform.memory.amygdala import Amygdala
from neuroform.memory.salience_filter import SalienceScorer
from neuroform.memory.habit_cache import HabitCache
from neuroform.memory.neurotransmitters import NeurotransmitterState
from neuroform.memory.predictive_model import PredictiveModel
from neuroform.memory.default_mode_network import DefaultModeNetwork
from neuroform.memory.circadian import CircadianProfile
from neuroform.memory.neuroplasticity import AutonomousNeuroplasticity
from neuroform.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class BrainOrchestrator:
    """
    The Central Nervous System — coordinates all brain modules.

    Processing pipeline per message:
      1. Circadian → NT modulation
      2. NT → LLM temperature, attention budget
      3. Salience → filter graph context
      4. Working Memory → attention-gated buffer
      5. Habit Cache → check for cached response
      6. Predictive Model → predict user intent
      7. OllamaClient → LLM inference
      8. Amygdala → emotional tagging
      9. Predictive Model → evaluate prediction error
      10. Neuroplasticity → apply feedback + baseline decay
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model: str = "llama3",
        working_memory: Optional[WorkingMemory] = None,
        amygdala: Optional[Amygdala] = None,
        salience: Optional[SalienceScorer] = None,
        habit_cache: Optional[HabitCache] = None,
        neurotransmitters: Optional[NeurotransmitterState] = None,
        predictive_model: Optional[PredictiveModel] = None,
        dmn: Optional[DefaultModeNetwork] = None,
        circadian: Optional[CircadianProfile] = None,
        neuroplasticity: Optional[AutonomousNeuroplasticity] = None,
    ):
        self.kg = kg
        self.model = model

        # Initialize all 9 brain systems (with sensible defaults)
        self.working_memory = working_memory or WorkingMemory(capacity=7)
        self.amygdala = amygdala or Amygdala()
        self.salience = salience or SalienceScorer(attention_budget=10)
        self.habit_cache = habit_cache or HabitCache(threshold=5)
        self.nt = neurotransmitters or NeurotransmitterState()
        self.predictive_model = predictive_model or PredictiveModel(kg, model=model)
        self.dmn = dmn or DefaultModeNetwork(kg, model=model)
        self.circadian = circadian or CircadianProfile()
        self.neuroplasticity = neuroplasticity or AutonomousNeuroplasticity(
            kg, model=model, amygdala=self.amygdala
        )

        # Wire OllamaClient with WM + Amygdala
        self.client = OllamaClient(
            kg, model=model,
            working_memory=self.working_memory,
            amygdala=self.amygdala,
        )

        # Track state for predictive model evaluation
        self._last_user_message: Optional[str] = None
        self._message_count: int = 0
        self._last_active: float = time.time()

        logger.info("BrainOrchestrator initialized with all 9 systems")

    def process(self, user_id: str, message: str) -> str:
        """
        Full cognitive processing pipeline.

        Returns the brain's response to the user.
        """
        self._message_count += 1
        self._last_active = time.time()

        # ──── 1. Circadian → Neurotransmitter modulation ────
        self.circadian.apply_to_neurotransmitters(self.nt)
        logger.debug(f"NT state: temp={self.nt.llm_temperature:.2f}, "
                     f"budget={self.nt.attention_budget}")

        # ──── 2. Evaluate previous prediction (if exists) ────
        if self.predictive_model.last_prediction and self._last_user_message:
            error = self.predictive_model.evaluate_error(
                self.predictive_model.last_prediction, message
            )
            sources = self.predictive_model.last_context_sources
            signals = self.predictive_model.generate_feedback_signal(error, sources)
            if signals:
                self._apply_feedback_signals(signals)
            logger.debug(f"Prediction error: {error:.3f}, signals: {len(signals)}")

        # ──── 3. Fetch and filter graph context ────
        raw_context = self.kg.query_context("User", layer=GraphLayer.NARRATIVE)
        if raw_context:
            filtered = self.salience.score_candidates(message, raw_context)
            # Override attention budget from NT state
            budget = self.nt.attention_budget
            filtered = filtered[:budget]
        else:
            filtered = []

        # ──── 4. Inject filtered context into Working Memory ────
        if filtered:
            self.working_memory.add_graph_context(filtered)

        # ──── 5. Check Habit Cache ────
        habit_key = self._compute_habit_key(message)
        cached = self.habit_cache.get_cached_response(habit_key)
        if cached:
            logger.info(f"Habit cache hit for '{habit_key}' — bypassing LLM")
            self.working_memory.add_conversation_turn("user", message)
            self.working_memory.add_conversation_turn("assistant", cached)
            self._last_user_message = message
            return cached

        # ──── 6. Predict user intent (before LLM) ────
        context_str = self.working_memory.build_context_string()
        history_str = self._build_history_string()
        try:
            prediction = self.predictive_model.predict_intent(context_str, history_str)
            # Track which context sources fed the prediction
            source_ids = [f"{c.get('source', '')}_{c.get('target', '')}" for c in filtered]
            self.predictive_model.set_context_sources(source_ids)
            logger.debug(f"Intent prediction: {prediction[:60]}...")
        except Exception as e:
            logger.warning(f"Prediction skipped: {e}")

        # ──── 7. LLM Inference (via OllamaClient) ────
        response = self.client.chat_with_memory(user_id, message)

        # ──── 8. Record habit invocation ────
        self.habit_cache.record_invocation(habit_key, response)

        # ──── 9. NT modulation from response sentiment (simple heuristic) ────
        sentiment = self._estimate_sentiment(message)
        self.nt.modulate_from_sentiment(sentiment)

        self._last_user_message = message
        return response

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a snapshot of all brain system states."""
        return {
            "message_count": self._message_count,
            "neurotransmitters": self.nt.to_dict(),
            "circadian": self.circadian.get_modulation(),
            "working_memory": self.working_memory.snapshot(),
            "habit_cache": self.habit_cache.snapshot(),
            "last_prediction": self.predictive_model.last_prediction,
        }

    def _apply_feedback_signals(self, signals: List[Dict[str, Any]]):
        """Apply STRENGTHEN/DECAY signals from the predictive model."""
        for signal in signals:
            action = signal.get("action")
            target = signal.get("target", "")
            amount = signal.get("amount", 0.1)

            if not target or "_" not in target:
                continue

            parts = target.split("_", 1)
            source_name, target_name = parts[0], parts[1]

            try:
                with self.kg.driver.session() as session:
                    if action == "STRENGTHEN":
                        session.run("""
                            MATCH ({name: $source})-[r]->({name: $target})
                            SET r.strength = COALESCE(r.strength, 1.0) + $amount
                        """, source=source_name, target=target_name, amount=amount)
                    elif action == "DECAY":
                        session.run("""
                            MATCH ({name: $source})-[r]->({name: $target})
                            SET r.strength = COALESCE(r.strength, 1.0) - $amount
                        """, source=source_name, target=target_name, amount=amount)
            except Exception as e:
                logger.warning(f"Feedback signal failed: {e}")

    def _compute_habit_key(self, message: str) -> str:
        """Derive a habit key from the message (normalized first 5 tokens)."""
        tokens = message.lower().strip().split()[:5]
        return "_".join(tokens) if tokens else "empty"

    def _build_history_string(self) -> str:
        """Build a compact history string for the predictive model."""
        history = self.working_memory.get_conversation_history()[-4:]
        return " | ".join(f"{t['role']}: {t['content'][:50]}" for t in history)

    @staticmethod
    def _estimate_sentiment(message: str) -> float:
        """Simple keyword-based sentiment estimation."""
        positive = {"love", "great", "happy", "thanks", "awesome", "wonderful",
                    "amazing", "enjoy", "excellent", "good", "like", "appreciate"}
        negative = {"hate", "sad", "angry", "terrible", "awful", "bad",
                    "frustrated", "annoyed", "upset", "disappointed", "horrible"}

        tokens = set(message.lower().split())
        pos_count = len(tokens & positive)
        neg_count = len(tokens & negative)

        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
