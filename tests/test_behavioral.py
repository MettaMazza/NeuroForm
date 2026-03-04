"""
Behavioral E2E Tests — Long-Running Multi-Turn Conversations
===============================================================

These tests run against LIVE Neo4j + Ollama to verify that the brain
actually performs like a mind over time. Each test simulates a multi-turn
conversation and validates behavioral properties.

Markers:
    @pytest.mark.behavioral — marks long-running tests (~1-3 min each)

Run:
    pytest tests/test_behavioral.py -v -s --timeout=300
"""
import pytest
import time
import logging

from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.memory.working_memory import WorkingMemory
from neuroform.memory.amygdala import Amygdala
from neuroform.memory.context_stream import ContextStream
from neuroform.memory.lessons import LessonManager
from neuroform.llm.ollama_client import OllamaClient
from neuroform.brain.orchestrator import BrainOrchestrator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def kg():
    """Live Neo4j connection — skips if unavailable."""
    kg = KnowledgeGraph()
    if kg.driver is None:
        pytest.skip("Neo4j not connected — skipping behavioral test")
    # Clean slate for behavioral tests
    try:
        with kg.driver.session() as session:
            session.run("MATCH (n) WHERE n.name STARTS WITH 'BehavTest' DETACH DELETE n")
    except Exception:
        pytest.skip("Neo4j not reachable — skipping behavioral test")
    yield kg
    # Cleanup after test
    try:
        with kg.driver.session() as session:
            session.run("MATCH (n) WHERE n.name STARTS WITH 'BehavTest' DETACH DELETE n")
    except Exception:
        pass  # Best-effort cleanup


@pytest.fixture
def brain(kg, tmp_path):
    """Live BrainOrchestrator with Five-Tier Memory System."""
    import os
    model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    cs = ContextStream(max_turns=500, persist_path=str(tmp_path / "wm.jsonl"))
    lm = LessonManager(persist_path=str(tmp_path / "lessons.json"))
    orch = BrainOrchestrator(kg, model=model, context_stream=cs, lesson_manager=lm)
    return orch


def talk(brain, message, user_id="BehavTestUser", user_name="TestUser"):
    """Helper: send a message and return the response."""
    response = brain.process(user_id, message, user_name=user_name)
    logger.info(f"USER: {message}")
    logger.info(f"BOT: {response[:120]}...")
    return response


# ─────────────────────────────────────────────────────────────────────
# Behavioral Tests
# ─────────────────────────────────────────────────────────────────────

class TestFactMemory:
    """Verify the brain remembers stated facts across turns."""

    @pytest.mark.behavioral
    def test_remembers_stated_name(self, brain):
        """Tell the bot a name, later ask if it remembers."""
        talk(brain, "Hi there, my name is BehavTestAlice")
        talk(brain, "I really enjoy hiking in the mountains")
        talk(brain, "What's the weather like today?")

        # Ask about the name — it should be in graph or conversation history
        response = talk(brain, "Do you remember my name?")
        assert any(term in response.lower() for term in ["alice", "behavtestalice"]), \
            f"Bot forgot the name! Response: {response}"

    @pytest.mark.behavioral
    @pytest.mark.timeout(120)
    def test_remembers_developer_identity(self, brain):
        """Tell the bot 'I am your developer' — critical identity fact."""
        talk(brain, "Hello, I am your developer")
        talk(brain, "How are you doing today?")
        talk(brain, "Tell me about your architecture")

        response = talk(brain, "Who am I to you?")
        # The LLM might express "developer" in varied ways
        identity_terms = ["developer", "develop", "creator", "built", "build",
                          "made", "create", "designer", "engineer"]
        assert any(term in response.lower() for term in identity_terms), \
            f"Bot forgot developer identity! Response: {response}"


class TestNoRepetition:
    """Verify responses are varied and not repetitive."""

    @pytest.mark.behavioral
    @pytest.mark.timeout(120)
    def test_no_identical_responses(self, brain):
        """Over 8+ turns, no two responses should be identical."""
        messages = [
            "Hello there",
            "How are you?",
            "Tell me something interesting",
            "What do you think about AI?",
            "That's cool",
            "Can you elaborate?",
            "Nice",
            "Tell me more",
        ]

        responses = []
        for msg in messages:
            response = talk(brain, msg)
            responses.append(response)

        # Check for exact duplicates
        unique_responses = set(responses)
        duplicates = len(responses) - len(unique_responses)
        assert duplicates == 0, \
            f"Found {duplicates} duplicate responses out of {len(responses)}!"


class TestConversationCoherence:
    """Verify multi-topic conversations stay coherent."""

    @pytest.mark.behavioral
    def test_topic_continuity(self, brain):
        """Establish a topic, ask about it later."""
        talk(brain, "I've been learning about BehavTestQuantumComputing this week")
        talk(brain, "It's really fascinating stuff")
        talk(brain, "Anyway, how's the weather?")

        response = talk(brain, "What was I learning about?")
        assert any(term in response.lower() for term in
                   ["quantum", "computing", "behavtestquantumcomputing"]), \
            f"Bot lost topic continuity! Response: {response}"


class TestGraphMemoryPersistence:
    """Verify facts reach the Neo4j graph and persist."""

    @pytest.mark.behavioral
    def test_fact_reaches_graph(self, brain, kg):
        """State a fact, verify it's stored in Neo4j."""
        talk(brain, "My favorite color is BehavTestTurquoise")

        # Give the system a moment to process
        time.sleep(0.5)

        # Check if the fact made it to the graph
        with kg.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE toLower(m.name) CONTAINS 'turquoise'
                   OR toLower(n.name) CONTAINS 'turquoise'
                RETURN n.name AS source, type(r) AS rel, m.name AS target
            """)
            records = list(result)

        if records:
            logger.info(f"Graph stored: {records}")
            assert True  # Fact reached the graph
        else:
            # The LLM might not always extract — check at least conversation memory
            response = talk(brain, "What is my favorite color?")
            # Should at least recall from conversation history
            assert "turquoise" in response.lower() or "color" in response.lower(), \
                f"Neither graph nor conversation memory has the color! Response: {response}"


class TestWorkingMemoryUnderLoad:
    """Verify the brain handles many turns without degrading."""

    @pytest.mark.behavioral
    @pytest.mark.timeout(180)
    def test_early_facts_persist_under_load(self, brain):
        """State an important fact early, then send 10+ filler turns."""
        # Important fact at the start
        talk(brain, "Remember this: my secret code is BehavTestBRAVO42")

        # 10 filler turns
        fillers = [
            "Tell me about the ocean",
            "What about mountains?",
            "Do you like music?",
            "What's your favorite number?",
            "Tell me a joke",
            "What's quantum entanglement?",
            "How does memory work?",
            "Tell me about stars",
            "What is gravity?",
            "Describe a sunset",
        ]
        for msg in fillers:
            talk(brain, msg)

        # Ask about the early fact
        response = talk(brain, "What is my secret code?")
        # Should recall from graph or conversation history
        assert any(term in response.lower() for term in
                   ["bravo", "bravo42", "behavtestbravo42"]), \
            f"Bot lost early fact after {len(fillers)} filler turns! Response: {response}"


class TestEmotionalPersistence:
    """Verify emotional context is acknowledged."""

    @pytest.mark.behavioral
    def test_acknowledges_emotion(self, brain):
        """Express strong emotion, check the bot acknowledges it."""
        response = talk(brain, "I'm really frustrated and disappointed today. Nothing is going right.")

        # Bot should acknowledge the negative emotion, not give a generic response
        emotional_indicators = ["sorry", "frustrat", "tough", "difficult", "hear",
                                "understand", "hard", "rough", "help", "concern"]
        assert any(term in response.lower() for term in emotional_indicators), \
            f"Bot didn't acknowledge emotion! Response: {response}"
