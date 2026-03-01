"""
BackgroundScheduler — Autonomous Background Tasks
====================================================

Runs Dream Consolidation, DMN introspection, and neuroplasticity
decay on configurable intervals, triggered by circadian state.
"""
import logging
import time
import threading
from typing import Optional

from neuroform.memory.dream_consolidation import DreamConsolidation
from neuroform.memory.default_mode_network import DefaultModeNetwork
from neuroform.memory.neuroplasticity import AutonomousNeuroplasticity
from neuroform.memory.circadian import CircadianProfile
from neuroform.memory.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    """
    Runs background brain tasks on a timer:
    - Dream Consolidation: when circadian says should_consolidate
    - DMN Introspection: after idle_timeout seconds of inactivity
    - Baseline Decay: every decay_interval seconds
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model: str = "llama3",
        dream: Optional[DreamConsolidation] = None,
        dmn: Optional[DefaultModeNetwork] = None,
        neuroplasticity: Optional[AutonomousNeuroplasticity] = None,
        circadian: Optional[CircadianProfile] = None,
        idle_timeout: float = 300.0,
        decay_interval: float = 600.0,
        tick_interval: float = 60.0,
    ):
        self.kg = kg
        self.model = model

        self.dream = dream or DreamConsolidation(kg, model=model)
        self.dmn = dmn or DefaultModeNetwork(kg, model=model)
        self.neuroplasticity = neuroplasticity or AutonomousNeuroplasticity(
            kg, model=model
        )
        self.circadian = circadian or CircadianProfile()

        self.idle_timeout = idle_timeout
        self.decay_interval = decay_interval
        self.tick_interval = tick_interval

        self._last_active: float = time.time()
        self._last_decay: float = time.time()
        self._last_dream: float = 0.0
        self._last_dmn: float = 0.0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Counters for diagnostics
        self.dream_runs: int = 0
        self.dmn_runs: int = 0
        self.decay_runs: int = 0

    def record_activity(self):
        """Call this when a message is processed to reset idle timer."""
        self._last_active = time.time()

    def tick(self) -> dict:
        """
        Run one background tick. Returns a dict of what happened.
        Call this periodically (e.g. every 60s).
        """
        now = time.time()
        results = {"dream": None, "dmn": None, "decay": None}

        # 1. Dream Consolidation (circadian-triggered, max once per hour)
        if self.circadian.should_dream_now() and (now - self._last_dream) > 3600:
            try:
                result = self.dream.consolidate()
                results["dream"] = result
                self._last_dream = now
                self.dream_runs += 1
                logger.info(f"Dream consolidation: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"Dream consolidation error: {e}")
                results["dream"] = {"status": "error", "error": str(e)}

        # 2. DMN Introspection (idle-triggered, max once per 30 min)
        idle_duration = now - self._last_active
        if idle_duration > self.idle_timeout and (now - self._last_dmn) > 1800:
            try:
                result = self.dmn.introspect()
                results["dmn"] = result
                self._last_dmn = now
                self.dmn_runs += 1
                logger.info(f"DMN introspection: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"DMN introspection error: {e}")
                results["dmn"] = {"status": "error", "error": str(e)}

        # 3. Baseline Decay (periodic)
        if (now - self._last_decay) > self.decay_interval:
            try:
                self.neuroplasticity.apply_baseline_decay()
                results["decay"] = {"status": "applied"}
                self._last_decay = now
                self.decay_runs += 1
                logger.info("Baseline decay applied")
            except Exception as e:
                logger.error(f"Baseline decay error: {e}")
                results["decay"] = {"status": "error", "error": str(e)}

        return results

    def start(self):  # pragma: no cover
        """Start the background scheduler in a daemon thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Background scheduler started (tick={self.tick_interval}s)")

    def stop(self):
        """Stop the background scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Background scheduler stopped")

    def _run_loop(self):  # pragma: no cover
        """Internal loop — runs tick() at tick_interval."""
        while self._running:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Background tick error: {e}")
            time.sleep(self.tick_interval)

    @property
    def is_running(self) -> bool:
        return self._running

    def snapshot(self) -> dict:
        """Diagnostic snapshot of scheduler state."""
        return {
            "running": self._running,
            "dream_runs": self.dream_runs,
            "dmn_runs": self.dmn_runs,
            "decay_runs": self.decay_runs,
            "idle_seconds": time.time() - self._last_active,
            "since_last_decay": time.time() - self._last_decay,
        }
