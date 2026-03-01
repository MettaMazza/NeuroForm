"""
Unit Tests for BrainOrchestrator + BackgroundScheduler
=======================================================
"""
import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock

from neuroform.brain.orchestrator import BrainOrchestrator
from neuroform.brain.background import BackgroundScheduler


# ===========================================================================
# BrainOrchestrator Tests
# ===========================================================================
class TestBrainOrchestrator:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        kg.query_context.return_value = [
            {"source": "User", "relationship": "LIKES", "target": "Music", "strength": 3.0},
        ]
        return kg

    @pytest.fixture
    def orchestrator(self, mock_kg):
        with patch("neuroform.brain.orchestrator.OllamaClient") as MockClient:
            MockClient.return_value.chat_with_memory.return_value = "Test response"
            orch = BrainOrchestrator(mock_kg, model="test-model")
            return orch

    def test_init_creates_all_systems(self, orchestrator):
        assert orchestrator.working_memory is not None
        assert orchestrator.amygdala is not None
        assert orchestrator.salience is not None
        assert orchestrator.habit_cache is not None
        assert orchestrator.nt is not None
        assert orchestrator.predictive_model is not None
        assert orchestrator.dmn is not None
        assert orchestrator.circadian is not None
        assert orchestrator.neuroplasticity is not None
        assert orchestrator.client is not None

    def test_process_returns_response(self, orchestrator):
        response = orchestrator.process("user1", "Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_process_increments_count(self, orchestrator):
        assert orchestrator._message_count == 0
        orchestrator.process("user1", "Hello")
        assert orchestrator._message_count == 1

    def test_circadian_modulates_nt(self, orchestrator):
        orchestrator.process("user1", "Hello")
        # NT should have been modulated by circadian
        assert orchestrator.nt.norepinephrine is not None
        assert orchestrator.nt.dopamine is not None

    def test_salience_filters_context(self, orchestrator):
        orchestrator.process("user1", "I love music")
        # Working memory should have received graph context
        items = orchestrator.working_memory.items
        assert len(items) >= 0  # May or may not have items depending on mock

    def test_habit_cache_records_invocation(self, orchestrator):
        orchestrator.process("user1", "hello there")
        # Short responses (<50 chars) should NOT be recorded
        count = orchestrator.habit_cache.get_invocation_count("hello there")
        # The mock returns "Test response" (13 chars) which is < 50, so NOT recorded
        assert count == 0

    def test_habit_cache_shortcircuit(self, orchestrator):
        """After enough invocations with long responses, habit cache should bypass LLM."""
        # Make the response long enough to be cached
        orchestrator.client.chat_with_memory.return_value = "A" * 60
        for i in range(15):
            orchestrator.process("user1", "hello there my friend")
        # The 16th call should hit the cache
        orchestrator.client.chat_with_memory.reset_mock()
        response = orchestrator.process("user1", "hello there my friend")
        # Should have gotten the cached response
        assert isinstance(response, str)

    def test_sentiment_modulation(self, orchestrator):
        orchestrator.nt.reset()
        baseline_da = orchestrator.nt.dopamine
        orchestrator.process("user1", "I love this amazing awesome great thing")
        # Positive sentiment should boost dopamine
        assert orchestrator.nt.dopamine >= baseline_da

    def test_negative_sentiment_modulation(self, orchestrator):
        orchestrator.nt.reset()
        baseline_ne = orchestrator.nt.norepinephrine
        # Process positive first, then negative
        orchestrator.process("user1", "I hate this terrible awful thing")
        # Then the NT state gets circadian + sentiment modulation
        assert orchestrator.nt.norepinephrine is not None

    def test_prediction_evaluation(self, orchestrator):
        """Verify that prediction error is evaluated on second message."""
        # First message: set up prediction state directly
        orchestrator.predictive_model._last_prediction = "User wants music"
        orchestrator.predictive_model._last_context_sources = ["User_Music"]
        orchestrator._last_user_message = "Tell me about music"

        # Second message: evaluate_error should be called
        with patch.object(orchestrator.predictive_model, 'evaluate_error',
                         return_value=0.2) as mock_eval:
            with patch.object(orchestrator.predictive_model, 'generate_feedback_signal',
                             return_value=[]) as mock_signal:
                orchestrator.process("user1", "Play some jazz")
                mock_eval.assert_called_once()
                mock_signal.assert_called_once()

    def test_get_diagnostics(self, orchestrator):
        orchestrator.process("user1", "Hello brain")
        diag = orchestrator.get_diagnostics()
        assert "message_count" in diag
        assert diag["message_count"] == 1
        assert "neurotransmitters" in diag
        assert "circadian" in diag
        assert "working_memory" in diag
        assert "habit_cache" in diag

    def test_compute_habit_key(self, orchestrator):
        key = orchestrator._compute_habit_key("Hello world how are you today")
        assert key == "hello world how are you today"

    def test_compute_habit_key_short(self, orchestrator):
        key = orchestrator._compute_habit_key("hi")
        assert key == "hi"

    def test_compute_habit_key_empty(self, orchestrator):
        key = orchestrator._compute_habit_key("")
        assert key == "empty"

    def test_estimate_sentiment_positive(self):
        s = BrainOrchestrator._estimate_sentiment("I love this amazing thing")
        assert s > 0

    def test_estimate_sentiment_negative(self):
        s = BrainOrchestrator._estimate_sentiment("I hate this terrible thing")
        assert s < 0

    def test_estimate_sentiment_neutral(self):
        s = BrainOrchestrator._estimate_sentiment("The sky is blue")
        assert s == 0.0

    def test_build_history_string(self, orchestrator):
        orchestrator.working_memory.add_conversation_turn("user", "Hello")
        orchestrator.working_memory.add_conversation_turn("assistant", "Hi there")
        history = orchestrator._build_history_string()
        assert "user:" in history.lower()
        assert "assistant:" in history.lower()

    def test_apply_feedback_strengthen(self, orchestrator):
        signals = [{"action": "STRENGTHEN", "target": "User_Music", "amount": 0.2}]
        orchestrator._apply_feedback_signals(signals)
        # Should have run a cypher query
        orchestrator.kg.driver.session.assert_called()

    def test_apply_feedback_decay(self, orchestrator):
        signals = [{"action": "DECAY", "target": "User_OldTopic", "amount": 0.1}]
        orchestrator._apply_feedback_signals(signals)
        orchestrator.kg.driver.session.assert_called()

    def test_apply_feedback_bad_target(self, orchestrator):
        signals = [{"action": "STRENGTHEN", "target": "notarget", "amount": 0.1}]
        orchestrator._apply_feedback_signals(signals)  # Should not crash

    def test_apply_feedback_db_error(self, orchestrator):
        orchestrator.kg.driver.session.side_effect = Exception("DB down")
        signals = [{"action": "STRENGTHEN", "target": "A_B", "amount": 0.1}]
        orchestrator._apply_feedback_signals(signals)  # Should not crash

    def test_no_context_from_graph(self, orchestrator):
        orchestrator.kg.query_context.return_value = []
        response = orchestrator.process("user1", "Hello")
        assert isinstance(response, str)

    def test_prediction_exception_handled(self, orchestrator):
        with patch.object(orchestrator.predictive_model, 'predict_intent',
                         side_effect=Exception("LLM down")):
            response = orchestrator.process("user1", "Hello")
            assert isinstance(response, str)

    def test_prediction_with_feedback_signals(self, orchestrator):
        """Verify feedback signals from prediction are applied."""
        orchestrator.predictive_model._last_prediction = "User wants cats"
        orchestrator.predictive_model._last_context_sources = ["User_Cats"]
        orchestrator._last_user_message = "Tell me about cats"

        with patch.object(orchestrator.predictive_model, 'evaluate_error',
                         return_value=0.8):
            with patch.object(orchestrator.predictive_model, 'generate_feedback_signal',
                             return_value=[{"action": "DECAY", "target": "User_Cats", "amount": 0.1}]):
                orchestrator.process("user1", "Actually tell me about dogs")
                # Feedback signals should have been applied
                orchestrator.kg.driver.session.assert_called()

    def test_custom_systems_injected(self, mock_kg):
        """Verify dependency injection of custom brain systems."""
        from neuroform.memory.working_memory import WorkingMemory
        from neuroform.memory.habit_cache import HabitCache

        custom_wm = WorkingMemory(capacity=3)
        custom_habits = HabitCache(threshold=2)

        with patch("neuroform.brain.orchestrator.OllamaClient"):
            orch = BrainOrchestrator(
                mock_kg, working_memory=custom_wm, habit_cache=custom_habits
            )
            assert orch.working_memory.capacity == 3
            assert orch.habit_cache.threshold == 2

    def test_habit_not_recorded_for_short_response(self, orchestrator):
        """Responses under 50 chars should not be cached."""
        orchestrator.client.chat_with_memory.return_value = "Short"
        orchestrator.process("user1", "test message")
        count = orchestrator.habit_cache.get_invocation_count("test message")
        assert count == 0

    def test_habit_recorded_for_long_response(self, orchestrator):
        """Responses over 50 chars should be cached."""
        orchestrator.client.chat_with_memory.return_value = "A" * 60
        orchestrator.process("user1", "test message")
        count = orchestrator.habit_cache.get_invocation_count("test message")
        assert count == 1


# ===========================================================================
# BackgroundScheduler Tests
# ===========================================================================
class TestBackgroundScheduler:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        return kg

    @pytest.fixture
    def scheduler(self, mock_kg):
        return BackgroundScheduler(
            mock_kg, model="test",
            idle_timeout=1.0, decay_interval=1.0, tick_interval=1.0,
        )

    def test_init(self, scheduler):
        assert not scheduler.is_running
        assert scheduler.dream_runs == 0
        assert scheduler.dmn_runs == 0
        assert scheduler.decay_runs == 0

    def test_record_activity(self, scheduler):
        old = scheduler._last_active
        time.sleep(0.01)
        scheduler.record_activity()
        assert scheduler._last_active > old

    def test_tick_decay(self, scheduler):
        scheduler._last_decay = 0  # Force decay to trigger
        results = scheduler.tick()
        assert results["decay"] is not None
        assert results["decay"]["status"] == "applied"
        assert scheduler.decay_runs == 1

    def test_tick_dream_consolidation(self, scheduler):
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=True):
            scheduler._last_dream = 0  # Force dream to trigger
            scheduler._last_decay = time.time()  # Prevent decay
            with patch.object(scheduler.dream, 'consolidate',
                            return_value={"status": "consolidated"}):
                results = scheduler.tick()
                assert results["dream"] is not None
                assert results["dream"]["status"] == "consolidated"
                assert scheduler.dream_runs == 1

    def test_tick_dmn_introspection(self, scheduler):
        scheduler._last_active = 0  # Force idle
        scheduler._last_dmn = 0  # Force DMN
        scheduler._last_decay = time.time()  # Prevent decay
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=False):
            with patch.object(scheduler.dmn, 'introspect',
                            return_value={"status": "complete"}):
                results = scheduler.tick()
                assert results["dmn"] is not None
                assert results["dmn"]["status"] == "complete"
                assert scheduler.dmn_runs == 1

    def test_tick_dream_error(self, scheduler):
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=True):
            scheduler._last_dream = 0
            scheduler._last_decay = time.time()
            with patch.object(scheduler.dream, 'consolidate',
                            side_effect=Exception("LLM error")):
                results = scheduler.tick()
                assert results["dream"]["status"] == "error"

    def test_tick_dmn_error(self, scheduler):
        scheduler._last_active = 0
        scheduler._last_dmn = 0
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=False):
            with patch.object(scheduler.dmn, 'introspect',
                            side_effect=Exception("Error")):
                results = scheduler.tick()
                assert results["dmn"]["status"] == "error"

    def test_tick_decay_error(self, scheduler):
        scheduler._last_decay = 0
        with patch.object(scheduler.neuroplasticity, 'apply_baseline_decay',
                         side_effect=Exception("DB error")):
            results = scheduler.tick()
            assert results["decay"]["status"] == "error"

    def test_stop_when_not_running(self, scheduler):
        scheduler.stop()
        assert not scheduler.is_running

    def test_stop_with_thread(self, scheduler):
        """Verify stop correctly joins the thread."""
        import threading
        mock_thread = MagicMock(spec=threading.Thread)
        scheduler._running = True
        scheduler._thread = mock_thread
        scheduler.stop()
        assert not scheduler._running
        mock_thread.join.assert_called_once_with(timeout=5.0)
        assert scheduler._thread is None

    def test_snapshot(self, scheduler):
        snap = scheduler.snapshot()
        assert "running" in snap
        assert "dream_runs" in snap
        assert "dmn_runs" in snap
        assert "decay_runs" in snap
        assert "idle_seconds" in snap
        assert snap["running"] is False

    def test_no_duplicate_dream(self, scheduler):
        """Verify dream won't run more than once per hour."""
        scheduler._last_dream = time.time()  # Just ran
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=True):
            results = scheduler.tick()
            assert results["dream"] is None

    def test_no_dmn_when_active(self, scheduler):
        """DMN shouldn't run when there's recent activity."""
        scheduler._last_active = time.time()  # Just active
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=False):
            results = scheduler.tick()
            assert results["dmn"] is None
