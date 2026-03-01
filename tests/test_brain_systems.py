"""
Unit Tests for P4: SalienceScorer, P5: HabitCache,
P6: NeurotransmitterState, P7: PredictiveModel,
P8: DefaultModeNetwork, P9: CircadianProfile
==================================================
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from neuroform.memory.salience_filter import SalienceScorer
from neuroform.memory.habit_cache import HabitCache
from neuroform.memory.neurotransmitters import NeurotransmitterState
from neuroform.memory.predictive_model import PredictiveModel
from neuroform.memory.default_mode_network import DefaultModeNetwork
from neuroform.memory.circadian import CircadianProfile
from neuroform.memory.graph import KnowledgeGraph


# ===========================================================================
# P4: SalienceScorer Tests
# ===========================================================================
class TestSalienceScorer:

    def test_init_defaults(self):
        ss = SalienceScorer()
        assert ss.attention_budget == 10

    def test_init_custom(self):
        ss = SalienceScorer(attention_budget=5, recency_weight=0.5)
        assert ss.attention_budget == 5

    def test_score_candidates_returns_ranked(self):
        ss = SalienceScorer(attention_budget=3)
        candidates = [
            {"source": "User", "relationship": "LIKES", "target": "Python", "strength": 5.0},
            {"source": "User", "relationship": "SAW", "target": "Cat", "strength": 0.1},
            {"source": "User", "relationship": "CODES_IN", "target": "Python", "strength": 3.0},
        ]
        scored = ss.score_candidates("I want to write Python code", candidates)
        assert len(scored) <= 3
        # All should have salience_score
        for s in scored:
            assert "salience_score" in s
        # First should be highest score
        assert scored[0]["salience_score"] >= scored[-1]["salience_score"]

    def test_score_candidates_respects_budget(self):
        ss = SalienceScorer(attention_budget=2)
        candidates = [
            {"source": "A", "target": "B", "strength": 1.0},
            {"source": "C", "target": "D", "strength": 2.0},
            {"source": "E", "target": "F", "strength": 3.0},
        ]
        scored = ss.score_candidates("test", candidates)
        assert len(scored) == 2

    def test_score_candidates_empty(self):
        ss = SalienceScorer()
        scored = ss.score_candidates("test", [])
        assert scored == []

    def test_compute_salience_with_recency(self):
        ss = SalienceScorer()
        candidate = {
            "source": "User", "target": "Python",
            "strength": 3.0,
            "last_fired": time.time() * 1000  # Recent
        }
        score = ss._compute_salience({"python"}, candidate)
        assert score > 0

    def test_compute_salience_without_last_fired(self):
        ss = SalienceScorer()
        candidate = {"source": "A", "target": "B", "strength": 1.0}
        score = ss._compute_salience({"test"}, candidate)
        assert score > 0

    def test_compute_salience_non_numeric_last_fired(self):
        ss = SalienceScorer()
        candidate = {"source": "A", "target": "B", "strength": 1.0, "last_fired": "not a number"}
        score = ss._compute_salience({"test"}, candidate)
        assert score > 0

    def test_filter_by_budget(self):
        ss = SalienceScorer(attention_budget=2)
        candidates = [{"a": 1}, {"b": 2}, {"c": 3}]
        filtered = ss.filter_by_budget(candidates)
        assert len(filtered) == 2


# ===========================================================================
# P5: HabitCache Tests
# ===========================================================================
class TestHabitCache:

    def test_init_defaults(self):
        hc = HabitCache()
        assert hc.threshold == 5

    def test_record_invocation_below_threshold(self):
        hc = HabitCache(threshold=3)
        promoted = hc.record_invocation("greet", "Hello!")
        assert promoted is False
        assert hc.get_invocation_count("greet") == 1

    def test_record_invocation_at_threshold(self):
        hc = HabitCache(threshold=3)
        for _ in range(2):
            hc.record_invocation("greet", "Hello!")
        promoted = hc.record_invocation("greet", "Hello!")
        assert promoted is True
        assert hc.is_habituated("greet")

    def test_record_invocation_already_cached(self):
        hc = HabitCache(threshold=2)
        hc.record_invocation("greet", "Hi")
        hc.record_invocation("greet", "Hi")
        # Already cached, shouldn't promote again
        promoted = hc.record_invocation("greet", "Hi")
        assert promoted is False

    def test_get_cached_response(self):
        hc = HabitCache(threshold=1)
        hc.record_invocation("greet", "Hello!")
        assert hc.get_cached_response("greet") == "Hello!"
        assert hc.get_cached_response("unknown") is None

    def test_is_habituated(self):
        hc = HabitCache(threshold=1)
        assert hc.is_habituated("greet") is False
        hc.record_invocation("greet", "Hi")
        assert hc.is_habituated("greet") is True

    def test_get_invocation_count_unknown(self):
        hc = HabitCache()
        assert hc.get_invocation_count("unknown") == 0

    def test_clear_habit(self):
        hc = HabitCache(threshold=1)
        hc.record_invocation("greet", "Hi")
        hc.clear_habit("greet")
        assert hc.is_habituated("greet") is False
        assert hc.get_invocation_count("greet") == 0

    def test_clear_habit_nonexistent(self):
        hc = HabitCache()
        hc.clear_habit("nothing")  # Should not raise

    def test_snapshot(self):
        hc = HabitCache(threshold=1)
        hc.record_invocation("greet", "Hi")
        snap = hc.snapshot()
        assert "greet" in snap["habits"]
        assert snap["total_procedures_tracked"] == 1
        assert snap["threshold"] == 1
        assert snap["invocations"]["greet"] == 1


# ===========================================================================
# P6: NeurotransmitterState Tests
# ===========================================================================
class TestNeurotransmitterState:

    def test_init_defaults(self):
        ns = NeurotransmitterState()
        assert ns.dopamine == 0.5
        assert ns.serotonin == 0.5

    def test_set_level_clamping(self):
        ns = NeurotransmitterState()
        ns.set_level("dopamine", 5.0)
        assert ns.dopamine == 1.0
        ns.set_level("dopamine", -1.0)
        assert ns.dopamine == 0.0

    def test_set_level_unknown(self):
        ns = NeurotransmitterState()
        ns.set_level("unknown_thing", 0.5)  # Should not crash

    def test_llm_temperature(self):
        ns = NeurotransmitterState()
        ns.dopamine = 0.0
        assert ns.llm_temperature == pytest.approx(0.3)
        ns.dopamine = 1.0
        assert ns.llm_temperature == pytest.approx(1.0)

    def test_decay_rate(self):
        ns = NeurotransmitterState()
        ns.acetylcholine = 0.0
        assert ns.decay_rate == pytest.approx(0.05)
        ns.acetylcholine = 1.0
        assert ns.decay_rate == pytest.approx(0.20)

    def test_attention_budget(self):
        ns = NeurotransmitterState()
        budget = ns.attention_budget
        assert isinstance(budget, int)
        assert budget >= 3

    def test_verbosity(self):
        ns = NeurotransmitterState()
        ns.serotonin = 0.0
        assert ns.verbosity == pytest.approx(0.5)
        ns.serotonin = 1.0
        assert ns.verbosity == pytest.approx(1.0)

    def test_modulate_positive_sentiment(self):
        ns = NeurotransmitterState()
        old_da = ns.dopamine
        ns.modulate_from_sentiment(0.5)
        assert ns.dopamine > old_da

    def test_modulate_negative_sentiment(self):
        ns = NeurotransmitterState()
        old_ne = ns.norepinephrine
        ns.modulate_from_sentiment(-0.5)
        assert ns.norepinephrine > old_ne

    def test_modulate_neutral_sentiment(self):
        ns = NeurotransmitterState()
        old_da = ns.dopamine
        ns.modulate_from_sentiment(0.0)
        assert ns.dopamine == old_da

    def test_to_dict(self):
        ns = NeurotransmitterState()
        d = ns.to_dict()
        assert "dopamine" in d
        assert "derived" in d
        assert "llm_temperature" in d["derived"]

    def test_reset(self):
        ns = NeurotransmitterState()
        ns.dopamine = 1.0
        ns.reset()
        assert ns.dopamine == 0.5


# ===========================================================================
# P7: PredictiveModel Tests
# ===========================================================================
class TestPredictiveModel:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        return kg

    @patch("neuroform.memory.predictive_model.ollama.chat")
    def test_predict_intent(self, mock_chat, mock_kg):
        mock_chat.return_value = {"message": {"content": "User wants weather info"}}
        pm = PredictiveModel(mock_kg)
        pred = pm.predict_intent("Context here", "Chat history")
        assert "weather" in pred.lower()
        assert pm.last_prediction == pred

    @patch("neuroform.memory.predictive_model.ollama.chat")
    def test_predict_intent_exception(self, mock_chat, mock_kg):
        mock_chat.side_effect = Exception("LLM down")
        pm = PredictiveModel(mock_kg)
        pred = pm.predict_intent("ctx", "hist")
        assert pred == ""
        assert pm.last_prediction is None

    def test_evaluate_error_exact_match(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        error = pm.evaluate_error("hello world", "hello world")
        assert error == 0.0

    def test_evaluate_error_no_match(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        error = pm.evaluate_error("alpha beta", "gamma delta")
        assert error == 1.0

    def test_evaluate_error_partial(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        error = pm.evaluate_error("hello world", "hello there")
        assert 0.0 < error < 1.0

    def test_evaluate_error_empty(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        assert pm.evaluate_error("", "something") == 0.5
        assert pm.evaluate_error("something", "") == 0.5

    def test_evaluate_error_empty_tokens(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        # Single space produces empty tokens after split
        assert pm.evaluate_error(" ", " ") == 0.5

    def test_generate_feedback_low_error(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        signals = pm.generate_feedback_signal(0.1, ["Source1", "Source2"])
        assert all(s["action"] == "STRENGTHEN" for s in signals)

    def test_generate_feedback_high_error(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        signals = pm.generate_feedback_signal(0.9, ["Source1"])
        assert all(s["action"] == "DECAY" for s in signals)

    def test_generate_feedback_neutral_error(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        signals = pm.generate_feedback_signal(0.5, ["Source1"])
        assert len(signals) == 0

    def test_set_context_sources(self, mock_kg):
        pm = PredictiveModel(mock_kg)
        pm.set_context_sources(["A", "B"])
        assert pm.last_context_sources == ["A", "B"]


# ===========================================================================
# P8: DefaultModeNetwork Tests
# ===========================================================================
class TestDefaultModeNetwork:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        return kg

    @pytest.fixture
    def mock_kg_offline(self):
        kg = MagicMock()
        kg.driver = None
        return kg

    def test_introspect_offline(self, mock_kg_offline):
        dmn = DefaultModeNetwork(mock_kg_offline)
        result = dmn.introspect()
        assert result["status"] == "offline"

    @patch("neuroform.memory.default_mode_network.ollama.chat")
    def test_introspect_empty_graph(self, mock_chat, mock_kg):
        session = MagicMock()
        # All stats return 0
        single_result = MagicMock()
        single_result.single.return_value = {"c": 0}
        session.run.return_value = single_result
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        dmn = DefaultModeNetwork(mock_kg)
        result = dmn.introspect()
        assert result["status"] == "empty_graph"
        mock_chat.assert_not_called()

    @patch("neuroform.memory.default_mode_network.ollama.chat")
    def test_introspect_full_analysis(self, mock_chat, mock_kg):
        session = MagicMock()
        call_count = [0]
        
        def mock_run(q, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if "n.layer" in q:
                result.__iter__ = MagicMock(return_value=iter([{"layer": "SEMANTIC", "c": 5}]))
                return result
            result.single.return_value = {"c": 10 if call_count[0] == 1 else 2}
            return result
        
        session.run.side_effect = mock_run
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_chat.return_value = {"message": {"content": '```json\n["Finding 1"]\n```'}}

        dmn = DefaultModeNetwork(mock_kg)
        result = dmn.introspect()
        assert result["status"] == "complete"
        assert len(result["findings"]) >= 1

    @patch("neuroform.memory.default_mode_network.ollama.chat")
    def test_analyze_with_llm_exception(self, mock_chat, mock_kg):
        mock_chat.side_effect = Exception("LLM down")
        dmn = DefaultModeNetwork(mock_kg)
        findings = dmn._analyze_with_llm({"total_nodes": 5})
        assert findings == []

    def test_parse_findings_json_block(self, mock_kg):
        dmn = DefaultModeNetwork(mock_kg)
        result = dmn._parse_findings('```json\n["A", "B"]\n```')
        assert result == ["A", "B"]

    def test_parse_findings_generic_block(self, mock_kg):
        dmn = DefaultModeNetwork(mock_kg)
        result = dmn._parse_findings('```\n["A"]\n```')
        assert result == ["A"]

    def test_parse_findings_raw_json(self, mock_kg):
        dmn = DefaultModeNetwork(mock_kg)
        result = dmn._parse_findings('["X"]')
        assert result == ["X"]

    def test_parse_findings_bad_json(self, mock_kg):
        dmn = DefaultModeNetwork(mock_kg)
        result = dmn._parse_findings("not json")
        assert result == []

    def test_parse_findings_non_list(self, mock_kg):
        dmn = DefaultModeNetwork(mock_kg)
        result = dmn._parse_findings('{"key": "val"}')
        assert result == []

    def test_gather_stats_exception(self, mock_kg):
        session = MagicMock()
        session.run.side_effect = Exception("DB error")
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        dmn = DefaultModeNetwork(mock_kg)
        stats = dmn._gather_stats()
        # Should gracefully degrade
        assert stats["total_nodes"] == 0
        assert stats["layer_counts"] == []


# ===========================================================================
# P9: CircadianProfile Tests
# ===========================================================================
class TestCircadianProfile:

    def test_defaults(self):
        cp = CircadianProfile()
        mod = cp.get_modulation(hour=10)
        assert mod["alertness"] == 1.0  # Peak alertness hour
        assert mod["should_consolidate"] is False

    def test_night_hours_trigger_consolidation(self):
        cp = CircadianProfile()
        assert cp.should_dream_now(hour=2) is True
        assert cp.should_dream_now(hour=10) is False

    def test_get_alertness(self):
        cp = CircadianProfile()
        assert cp.get_alertness(hour=10) > cp.get_alertness(hour=3)

    def test_get_exploration_boost(self):
        cp = CircadianProfile()
        # Night hours have higher exploration
        assert cp.get_exploration_boost(hour=2) > cp.get_exploration_boost(hour=10)

    def test_custom_profile(self):
        custom = {i: (0.5, True, 0.1) for i in range(24)}
        cp = CircadianProfile(custom_profile=custom)
        mod = cp.get_modulation(hour=10)
        assert mod["alertness"] == 0.5
        assert mod["should_consolidate"] is True

    def test_modulation_wraps_hours(self):
        cp = CircadianProfile()
        mod = cp.get_modulation(hour=25)
        assert mod["hour"] == 1  # 25 % 24 = 1

    def test_modulation_none_uses_system_time(self):
        cp = CircadianProfile()
        mod = cp.get_modulation()  # No hour specified
        assert 0 <= mod["hour"] <= 23

    def test_apply_to_neurotransmitters_day(self):
        cp = CircadianProfile()
        ns = NeurotransmitterState()
        cp.apply_to_neurotransmitters(ns, hour=10)
        assert ns.norepinephrine == 1.0  # Peak alertness

    def test_apply_to_neurotransmitters_night(self):
        cp = CircadianProfile()
        ns = NeurotransmitterState()
        cp.apply_to_neurotransmitters(ns, hour=2)
        assert ns.acetylcholine == 0.8  # Learning boost during consolidation
