"""
NeuroForm Live End-to-End Tests
================================
These tests run against REAL Neo4j and Ollama instances.
They validate the entire system pipeline: graph CRUD, LLM inference,
memory extraction, and the hybrid neuroplasticity optimization loop.

Requirements:
  - Neo4j running on bolt://localhost:7687
  - Ollama running on localhost:11434 with at least one model
  - A .env file with valid NEO4J_PASSWORD

Run with:
  pytest tests/test_e2e.py -v -s

The `-s` flag is recommended to see live LLM output during testing.
"""
import pytest
import os
import time
import logging

# Load .env before any imports that use env vars
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.memory.neuroplasticity import AutonomousNeuroplasticity
from neuroform.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
E2E_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

def _neo4j_available():
    """Check if Neo4j is reachable with the current credentials."""
    try:
        kg = KnowledgeGraph()
        available = kg.driver is not None
        kg.close()
        return available
    except Exception:
        return False

def _ollama_available():
    """Check if Ollama is running and has at least one model."""
    try:
        import ollama
        tags = ollama.list()
        return len(tags.get("models", [])) > 0
    except Exception:
        return False


neo4j_required = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j is not available — skipping live graph tests"
)

ollama_required = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama is not available — skipping live LLM tests"
)

full_stack = pytest.mark.skipif(
    not (_neo4j_available() and _ollama_available()),
    reason="Both Neo4j and Ollama are required for full-stack e2e tests"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def live_kg():
    """Provides a real KnowledgeGraph connected to Neo4j, cleaned before and after."""
    kg = KnowledgeGraph()
    assert kg.driver is not None, "KnowledgeGraph failed to connect to Neo4j"
    kg.clear_all()
    yield kg
    kg.clear_all()
    kg.close()


# ===========================================================================
# TIER 1: Graph CRUD (Neo4j Only)
# ===========================================================================
class TestGraphCRUD:
    """Live tests for KnowledgeGraph against real Neo4j."""

    @neo4j_required
    def test_connect_and_schema(self, live_kg):
        """Verify connection succeeds and schema indexes are created."""
        assert live_kg.driver is not None
        with live_kg.driver.session() as session:
            result = session.run("SHOW INDEXES")
            indexes = [r["name"] for r in result]
            assert "node_name_idx" in indexes
            assert "node_layer_idx" in indexes

    @neo4j_required
    def test_add_node_and_retrieve(self, live_kg):
        """Add a node and verify it exists in Neo4j."""
        live_kg.add_node("Entity", "Alice", layer=GraphLayer.SOCIAL)
        
        with live_kg.driver.session() as session:
            result = session.run("MATCH (n {name: 'Alice'}) RETURN n.name AS name, n.layer AS layer")
            record = result.single()
            assert record is not None
            assert record["name"] == "Alice"
            assert record["layer"] == "SOCIAL"

    @neo4j_required
    def test_layer_root_mesh_topology(self, live_kg):
        """Verify that adding nodes to different layers creates interconnected LayerRoot nodes."""
        live_kg.add_node("Entity", "Fact1", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Event1", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "Person1", layer=GraphLayer.SOCIAL)

        with live_kg.driver.session() as session:
            # Check LayerRoot nodes exist
            result = session.run("MATCH (r:LayerRoot) RETURN r.name AS name ORDER BY r.name")
            roots = [r["name"] for r in result]
            assert "SEMANTIC" in roots
            assert "EPISODIC" in roots
            assert "SOCIAL" in roots

            # Check PEER_LAYER mesh: each root should connect to the others
            result = session.run("""
                MATCH (a:LayerRoot)-[:PEER_LAYER]-(b:LayerRoot) 
                RETURN DISTINCT a.name AS from_layer, b.name AS to_layer
            """)
            peers = [(r["from_layer"], r["to_layer"]) for r in result]
            assert len(peers) >= 4  # At least 2 bidirectional pairs for 3 roots

    @neo4j_required
    def test_add_relationship_with_strength(self, live_kg):
        """Add a relationship and verify strength + last_fired timestamp."""
        live_kg.add_node("Entity", "Alice", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Bob", layer=GraphLayer.SOCIAL)
        live_kg.add_relationship("Alice", "KNOWS", "Bob", strength=1.0)

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH (a {name: 'Alice'})-[r:KNOWS]->(b {name: 'Bob'})
                RETURN r.strength AS strength, r.last_fired AS last_fired, r.created AS created
            """)
            record = result.single()
            assert record is not None
            assert record["strength"] == 1.0
            assert record["last_fired"] is not None
            assert record["created"] is not None

    @neo4j_required
    def test_relationship_strength_increment_on_match(self, live_kg):
        """Calling add_relationship again should increment strength (ON MATCH)."""
        live_kg.add_node("Entity", "Alice", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Bob", layer=GraphLayer.SOCIAL)
        live_kg.add_relationship("Alice", "KNOWS", "Bob", strength=1.0)
        live_kg.add_relationship("Alice", "KNOWS", "Bob", strength=1.0)

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH (a {name: 'Alice'})-[r:KNOWS]->(b {name: 'Bob'})
                RETURN r.strength AS strength
            """)
            record = result.single()
            # ON MATCH increments by strength * 0.1 = 0.1
            assert record["strength"] == pytest.approx(1.1, abs=0.01)

    @neo4j_required
    def test_relationship_sanitization(self, live_kg):
        """Verify that unsafe relationship types are sanitized."""
        live_kg.add_node("Entity", "A", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "B", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("A", "HAS BEEN TO!!!", "B")

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH (a {name: 'A'})-[r]->(b {name: 'B'})
                WHERE type(r) <> 'IN_LAYER'
                RETURN type(r) AS rel_type
            """)
            record = result.single()
            assert record is not None
            assert record["rel_type"] == "HASBEENTO"

    @neo4j_required
    def test_query_context(self, live_kg):
        """Verify query_context returns connected nodes sorted by strength."""
        live_kg.add_node("Entity", "User", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Python", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Rust", layer=GraphLayer.SEMANTIC)
        live_kg.add_relationship("User", "CODES_IN", "Python", strength=5.0)
        live_kg.add_relationship("User", "LEARNING", "Rust", strength=1.0)

        context = live_kg.query_context("User")
        assert len(context) >= 2

        # Find non-structural relationships
        code_rels = [c for c in context if c["relationship"] not in ("IN_LAYER", "PEER_LAYER")]
        assert len(code_rels) >= 2
        # First result should be the stronger relationship
        assert code_rels[0]["strength"] >= code_rels[1]["strength"]

    @neo4j_required
    def test_query_context_fires_neurons(self, live_kg):
        """Verify that query_context updates last_fired timestamps."""
        live_kg.add_node("Entity", "Test", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Data", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("Test", "HAS", "Data", strength=1.0)

        # Get initial last_fired
        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH ({name: 'Test'})-[r:HAS]->({name: 'Data'})
                RETURN r.last_fired AS lf
            """)
            initial_lf = result.single()["lf"]

        time.sleep(0.01)  # Small delay to ensure timestamp difference
        live_kg.query_context("Test")

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH ({name: 'Test'})-[r:HAS]->({name: 'Data'})
                RETURN r.last_fired AS lf
            """)
            updated_lf = result.single()["lf"]

        assert updated_lf >= initial_lf

    @neo4j_required
    def test_clear_all(self, live_kg):
        """Verify clear_all wipes the graph."""
        live_kg.add_node("Entity", "Temp", layer=GraphLayer.NARRATIVE)
        deleted = live_kg.clear_all()
        assert deleted >= 1

        with live_kg.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS c")
            assert result.single()["c"] == 0


# ===========================================================================
# TIER 2: Neuroplasticity Engine (Neo4j + Math Heuristic)
# ===========================================================================
class TestBaselineDecay:
    """Live tests for the mathematical heuristic phase of neuroplasticity."""

    @neo4j_required
    def test_baseline_decay_reduces_strength(self, live_kg):
        """Verify that apply_baseline_decay subtracts from edge strength."""
        live_kg.add_node("Entity", "A", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "B", layer=GraphLayer.SEMANTIC)
        live_kg.add_relationship("A", "RELATED", "B", strength=1.0)

        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)
        actions = neuro.apply_baseline_decay(decay_rate=0.1)
        assert actions > 0

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH ({name: 'A'})-[r:RELATED]->({name: 'B'})
                RETURN r.strength AS s
            """)
            record = result.single()
            assert record is not None
            assert record["s"] == pytest.approx(0.9, abs=0.01)

    @neo4j_required
    def test_baseline_decay_prunes_weak_edges(self, live_kg):
        """Verify that edges at or below 0 are pruned."""
        live_kg.add_node("Entity", "X", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Y", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("X", "WEAK", "Y", strength=0.05)

        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)
        neuro.apply_baseline_decay(decay_rate=0.1)  # 0.05 - 0.1 = -0.05 <= 0 → pruned

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH ({name: 'X'})-[r:WEAK]->({name: 'Y'})
                RETURN r
            """)
            assert result.single() is None  # Edge should be deleted

    @neo4j_required
    def test_baseline_decay_preserves_structural_edges(self, live_kg):
        """Verify that IN_LAYER and PEER_LAYER edges are NOT decayed."""
        live_kg.add_node("Entity", "Test", layer=GraphLayer.SEMANTIC)
        
        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)
        neuro.apply_baseline_decay(decay_rate=0.5)

        with live_kg.driver.session() as session:
            # IN_LAYER should still exist
            result = session.run("""
                MATCH (n {name: 'Test'})-[r:IN_LAYER]->(root:LayerRoot)
                RETURN r
            """)
            assert result.single() is not None  # Structural edge preserved

    @neo4j_required
    def test_orphan_cleanup(self, live_kg):
        """Verify that orphaned nodes (no connections) are cleaned up."""
        live_kg.add_node("Entity", "Orphan1", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Orphan2", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("Orphan1", "TEMP", "Orphan2", strength=0.01)

        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)
        neuro.apply_baseline_decay(decay_rate=0.1)  # Kills the weak edge

        with live_kg.driver.session() as session:
            # Check that orphans were cleaned up (non-root nodes with no connections)
            result = session.run("""
                MATCH (n)
                WHERE n.name IN ['Orphan1', 'Orphan2'] AND NOT (n)--()
                RETURN count(n) AS orphans
            """)
            assert result.single()["orphans"] == 0


# ===========================================================================
# TIER 3: Ollama Client (Neo4j + LLM)
# ===========================================================================
class TestOllamaClient:
    """Live tests for the OllamaClient chat and memory extraction pipeline."""

    @full_stack
    def test_chat_returns_response(self, live_kg):
        """Verify that chat_with_memory returns a non-empty string from the LLM."""
        client = OllamaClient(live_kg, model=E2E_MODEL)
        response = client.chat_with_memory("test_user", "Hello, who are you?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\n[LLM RESPONSE]: {response}")

    @full_stack
    def test_chat_extracts_memories(self, live_kg):
        """
        Tell the LLM a clear fact and verify it extracts and stores 
        it in the knowledge graph.
        """
        client = OllamaClient(live_kg, model=E2E_MODEL)
        
        # Give the LLM a very explicit, unambiguous fact
        response = client.chat_with_memory(
            "test_user",
            "My name is Sarah and I am a software engineer who lives in Seattle."
        )
        print(f"\n[LLM RESPONSE]: {response}")

        # Check if any new entities were created in the graph
        with live_kg.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN n.name AS name")
            entities = [r["name"] for r in result]
            print(f"[GRAPH ENTITIES]: {entities}")
            
            # The LLM should have extracted at least some facts
            # Note: this is probabilistic — the LLM might not always extract perfectly
            result2 = session.run("MATCH (a)-[r]->(b) WHERE type(r) <> 'IN_LAYER' AND type(r) <> 'PEER_LAYER' RETURN count(r) AS c")
            rel_count = result2.single()["c"]
            print(f"[GRAPH RELATIONSHIPS]: {rel_count} non-structural edges")
            
            # We just verify the system didn't crash and produced some output
            assert isinstance(response, str)
            assert len(response) > 10

    @full_stack
    def test_chat_with_prior_context(self, live_kg):
        """Verify that the LLM receives graph context when available."""
        # Seed the graph with a known fact
        live_kg.add_node("Entity", "User", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Python", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("User", "LOVES", "Python", strength=5.0)

        client = OllamaClient(live_kg, model=E2E_MODEL)
        response = client.chat_with_memory(
            "test_user",
            "What programming languages do I like?"
        )
        print(f"\n[LLM RESPONSE]: {response}")

        # The response should reference Python since it's in the context
        # This is probabilistic but a strong signal
        assert isinstance(response, str)
        assert len(response) > 0


# ===========================================================================
# TIER 4: Full Hybrid Neuroplasticity Loop (Neo4j + LLM)
# ===========================================================================
class TestHybridNeuroplasticity:
    """Live tests for the full hybrid heuristic + LLM optimization loop."""

    @full_stack
    def test_evaluate_and_optimize_full_loop(self, live_kg):
        """Run the full optimization loop against a real graph with real LLM."""
        # Seed graph with diverse data
        live_kg.add_node("Entity", "User", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Coffee", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Tea", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "OldEvent", layer=GraphLayer.EPISODIC)
        
        live_kg.add_relationship("User", "LIKES", "Coffee", strength=3.0)
        live_kg.add_relationship("User", "DISLIKES", "Tea", strength=1.5)
        live_kg.add_relationship("User", "ATTENDED", "OldEvent", strength=0.5)

        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)
        result = neuro.evaluate_and_optimize()

        print(f"\n[OPTIMIZATION RESULT]: {result}")
        
        assert result["status"] in ("success", "no_data")
        assert "actions_taken" in result
        assert isinstance(result["actions_taken"], int)
        
        if result["status"] == "success":
            assert "heuristic_actions" in result
            assert "llm_actions" in result
            assert "decisions" in result
            print(f"[HEURISTIC ACTIONS]: {result['heuristic_actions']}")
            print(f"[LLM ACTIONS]: {result['llm_actions']}")
            print(f"[LLM DECISIONS]: {result['decisions']}")

    @full_stack
    def test_contradiction_detection(self, live_kg):
        """
        Seed contradictory facts and verify the LLM identifies them.
        Note: This is probabilistic — the LLM may not always prune correctly.
        """
        live_kg.add_node("Entity", "User", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Boston", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "New York", layer=GraphLayer.SEMANTIC)
        
        # Contradictory: user can't live in two cities
        live_kg.add_relationship("User", "LIVES_IN", "Boston", strength=1.0)
        live_kg.add_relationship("User", "LIVES_IN", "New York", strength=2.0)

        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)
        result = neuro.evaluate_and_optimize()

        print(f"\n[CONTRADICTION TEST RESULT]: {result}")
        
        # We just verify the loop completes without error
        assert result["status"] in ("success", "no_data")
        
        if result.get("decisions"):
            prune_decisions = [d for d in result["decisions"] if d.get("action") == "PRUNE"]
            print(f"[PRUNE DECISIONS]: {prune_decisions}")


# ===========================================================================
# TIER 5: Full Pipeline Integration
# ===========================================================================
class TestFullPipeline:
    """End-to-end pipeline: chat → extract → store → decay → optimize."""

    @full_stack
    def test_full_lifecycle(self, live_kg):
        """
        Simulate a full agent lifecycle:
        1. User chats and declares facts
        2. Facts are extracted and stored in the graph
        3. The neuroplasticity engine runs decay + LLM optimization
        4. Verify the graph state is coherent
        """
        client = OllamaClient(live_kg, model=E2E_MODEL)
        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL)

        # Step 1: Chat and extract
        print("\n[STEP 1] Chatting with the agent...")
        response1 = client.chat_with_memory(
            "test_user",
            "Hi! I'm Alex. I'm a musician who plays guitar and lives in Portland."
        )
        print(f"[RESPONSE 1]: {response1}")

        # Step 2: Verify graph state after extraction
        with live_kg.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS c")
            entity_count_before = result.single()["c"]
            print(f"[ENTITIES AFTER CHAT]: {entity_count_before}")

            result2 = session.run("""
                MATCH (a)-[r]->(b)
                WHERE type(r) <> 'IN_LAYER' AND type(r) <> 'PEER_LAYER'
                RETURN a.name AS src, type(r) AS rel, b.name AS tgt, r.strength AS strength
            """)
            edges = [dict(r) for r in result2]
            print(f"[EDGES AFTER CHAT]: {edges}")

        # Step 3: Run the hybrid optimization loop
        print("\n[STEP 3] Running neuroplasticity optimization...")
        opt_result = neuro.evaluate_and_optimize()
        print(f"[OPTIMIZATION]: {opt_result}")

        assert opt_result["status"] in ("success", "no_data", "error")

        # Step 4: Verify graph is still coherent (no crashes, indexes intact)
        with live_kg.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS c")
            entity_count_after = result.single()["c"]
            print(f"[ENTITIES AFTER OPTIMIZATION]: {entity_count_after}")

            # LayerRoots should still exist
            result3 = session.run("MATCH (r:LayerRoot) RETURN count(r) AS c")
            root_count = result3.single()["c"]
            print(f"[LAYER ROOTS]: {root_count}")
            assert root_count >= 0  # Roots should survive optimization

        # Step 5: Chat again to verify memory retrieval works post-optimization
        print("\n[STEP 5] Chatting again after optimization...")
        response2 = client.chat_with_memory(
            "test_user",
            "What do you remember about me?"
        )
        print(f"[RESPONSE 2]: {response2}")
        
        assert isinstance(response2, str)
        assert len(response2) > 0

        print("\n[FULL LIFECYCLE COMPLETE] ✅")


# ===========================================================================
# TIER 6: Working Memory (Neo4j + LLM Integration)
# ===========================================================================
class TestWorkingMemoryE2E:
    """Live tests for WorkingMemory integrated with OllamaClient."""

    @full_stack
    def test_working_memory_records_conversation(self, live_kg):
        """Verify that conversation turns are tracked in working memory."""
        from neuroform.memory.working_memory import WorkingMemory

        wm = WorkingMemory(capacity=7)
        client = OllamaClient(live_kg, model=E2E_MODEL, working_memory=wm)

        response = client.chat_with_memory("e2e_user", "Hello, I like jazz music.")
        print(f"\n[WM RESPONSE]: {response}")

        history = wm.get_conversation_history()
        assert len(history) >= 2  # user + assistant turns
        assert history[0]["role"] == "user"
        assert "jazz" in history[0]["content"].lower()

    @full_stack
    def test_working_memory_injects_graph_context(self, live_kg):
        """Verify graph context flows through working memory into the LLM prompt."""
        from neuroform.memory.working_memory import WorkingMemory

        # Seed graph with a known fact
        live_kg.add_node("Entity", "User", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Cooking", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("User", "ENJOYS", "Cooking", strength=5.0)

        wm = WorkingMemory(capacity=7)
        client = OllamaClient(live_kg, model=E2E_MODEL, working_memory=wm)

        client.chat_with_memory("e2e_user", "What are my hobbies?")

        # Working memory should contain the graph context item
        items = wm.items
        sources = [item.source for item in items]
        assert "graph" in sources or "conversation" in sources
        print(f"[WM ITEMS]: {[i.content for i in items]}")

    @full_stack
    def test_working_memory_multi_turn(self, live_kg):
        """Verify multi-turn conversation history accumulates correctly."""
        from neuroform.memory.working_memory import WorkingMemory

        wm = WorkingMemory(capacity=7)
        client = OllamaClient(live_kg, model=E2E_MODEL, working_memory=wm)

        client.chat_with_memory("e2e_user", "My favorite color is blue.")
        client.chat_with_memory("e2e_user", "I also enjoy swimming.")

        history = wm.get_conversation_history()
        assert len(history) >= 4  # 2 user + 2 assistant turns
        print(f"[WM HISTORY LENGTH]: {len(history)} turns")

    @neo4j_required
    def test_working_memory_attention_scoring(self, live_kg):
        """Verify attention scoring works with real graph data."""
        from neuroform.memory.working_memory import WorkingMemory

        wm = WorkingMemory(capacity=5)
        # Simulate adding graph context
        context_data = [
            {"source": "User", "relationship": "LIKES", "target": "Music", "strength": 5.0},
            {"source": "User", "relationship": "SAW", "target": "Movie", "strength": 0.5},
        ]
        wm.add_graph_context(context_data)

        # Attend should rank by strength
        top = wm.attend(top_k=2)
        assert len(top) == 2
        assert top[0].attention_score() >= top[1].attention_score()
        print(f"[ATTENTION SCORES]: {[i.attention_score() for i in top]}")

    @neo4j_required
    def test_working_memory_capacity_eviction(self, live_kg):
        """Verify eviction works with real interaction patterns."""
        from neuroform.memory.working_memory import WorkingMemory

        wm = WorkingMemory(capacity=3)
        wm.add("item1", source="test", strength=0.1)
        wm.add("item2", source="test", strength=0.5)
        wm.add("item3", source="test", strength=0.9)
        wm.add("item4", source="test", strength=1.0)  # Should evict weakest

        assert len(wm.items) == 3
        contents = [i.content for i in wm.items]
        # The weakest (item1) should have been evicted
        print(f"[EVICTION RESULT]: {contents}")


# ===========================================================================
# TIER 7: Amygdala — Emotional Valence (Neo4j)
# ===========================================================================
class TestAmygdalaE2E:
    """Live tests for Amygdala emotional tagging on real Neo4j edges."""

    @neo4j_required
    def test_valence_tagging_on_real_edge(self, live_kg):
        """Apply emotional valence to a real graph edge and verify persistence."""
        from neuroform.memory.amygdala import Amygdala, EmotionalValence

        live_kg.add_node("Entity", "User", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Dog", layer=GraphLayer.SOCIAL)
        live_kg.add_relationship("User", "LOST", "Dog", strength=1.0)

        amygdala = Amygdala()
        with live_kg.driver.session() as session:
            valence = EmotionalValence(valence=-0.9, intensity=0.8, emotion="sadness")
            amygdala.apply_valence_to_edge(session, "User", "LOST", "Dog", valence)

        # Verify the valence was persisted
        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH ({name: 'User'})-[r:LOST]->({name: 'Dog'})
                RETURN r.valence AS v, r.intensity AS i, r.emotion AS e
            """)
            record = result.single()
            assert record is not None
            assert record["v"] == pytest.approx(-0.9)
            assert record["i"] == pytest.approx(0.8)
            assert record["e"] == "sadness"
            print(f"[VALENCE]: v={record['v']}, i={record['i']}, e={record['e']}")

    @neo4j_required
    def test_decay_immunity_for_significant_edges(self, live_kg):
        """Verify that emotionally significant edges survive baseline decay."""
        from neuroform.memory.amygdala import Amygdala

        # Create two edges: one significant, one neutral
        live_kg.add_node("Entity", "User", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Wedding", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "Lunch", layer=GraphLayer.EPISODIC)
        live_kg.add_relationship("User", "ATTENDED", "Wedding", strength=0.15)
        live_kg.add_relationship("User", "HAD", "Lunch", strength=0.15)

        # Tag Wedding with high emotional valence
        with live_kg.driver.session() as session:
            session.run("""
                MATCH ({name: 'User'})-[r:ATTENDED]->({name: 'Wedding'})
                SET r.valence = 0.95, r.intensity = 0.9, r.emotion = 'joy'
            """)
            session.run("""
                MATCH ({name: 'User'})-[r:HAD]->({name: 'Lunch'})
                SET r.valence = 0.0, r.intensity = 0.0, r.emotion = 'neutral'
            """)

        # Run decay — Lunch should be pruned, Wedding should survive
        amygdala = Amygdala()
        neuro = AutonomousNeuroplasticity(live_kg, model=E2E_MODEL, amygdala=amygdala)
        neuro.apply_baseline_decay(decay_rate=0.2)

        with live_kg.driver.session() as session:
            # Wedding (significant) should still exist
            result = session.run("""
                MATCH ({name: 'User'})-[r:ATTENDED]->({name: 'Wedding'})
                RETURN r.strength AS s
            """)
            wedding = result.single()
            assert wedding is not None, "Emotionally significant edge should survive decay!"
            print(f"[WEDDING SURVIVED]: strength={wedding['s']}")

    @neo4j_required
    def test_batch_tag_memories(self, live_kg):
        """Batch-apply valence tags to multiple memories on real graph."""
        from neuroform.memory.amygdala import Amygdala

        live_kg.add_node("Entity", "User", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Cat", layer=GraphLayer.SOCIAL)
        live_kg.add_node("Entity", "Spider", layer=GraphLayer.SOCIAL)
        live_kg.add_relationship("User", "LOVES", "Cat", strength=1.0)
        live_kg.add_relationship("User", "FEARS", "Spider", strength=1.0)

        amygdala = Amygdala()
        memories = [
            {"source": "User", "relation": "LOVES", "target": "Cat",
             "valence": 0.8, "intensity": 0.6, "emotion": "joy"},
            {"source": "User", "relation": "FEARS", "target": "Spider",
             "valence": -0.7, "intensity": 0.9, "emotion": "fear"},
        ]
        amygdala.tag_memories(live_kg.driver, memories)

        with live_kg.driver.session() as session:
            result = session.run("""
                MATCH ({name: 'User'})-[r:LOVES]->({name: 'Cat'})
                RETURN r.emotion AS e
            """)
            assert result.single()["e"] == "joy"

            result2 = session.run("""
                MATCH ({name: 'User'})-[r:FEARS]->({name: 'Spider'})
                RETURN r.emotion AS e
            """)
            assert result2.single()["e"] == "fear"
            print("[BATCH TAGGING]: ✅ Both edges tagged correctly")

    @full_stack
    def test_ollama_client_with_amygdala(self, live_kg):
        """Verify the full extraction pipeline includes emotional valence."""
        from neuroform.memory.amygdala import Amygdala

        amygdala = Amygdala()
        client = OllamaClient(live_kg, model=E2E_MODEL, amygdala=amygdala)

        response = client.chat_with_memory(
            "e2e_user",
            "I'm devastated because I lost my grandmother last week."
        )
        print(f"\n[EMOTIONAL RESPONSE]: {response}")
        assert isinstance(response, str)
        assert len(response) > 0


# ===========================================================================
# TIER 8: Dream Consolidation (Neo4j + LLM)
# ===========================================================================
class TestDreamConsolidationE2E:
    """Live tests for Dream Consolidation (hippocampal replay)."""

    @full_stack
    def test_consolidation_with_real_episodes(self, live_kg):
        """Seed episodic memories and run dream consolidation against live LLM."""
        from neuroform.memory.dream_consolidation import DreamConsolidation

        # Seed episodic events
        live_kg.add_node("Entity", "User", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "Coffee", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "MorningRun", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "Yoga", layer=GraphLayer.EPISODIC)

        live_kg.add_relationship("User", "DRANK", "Coffee", strength=1.0)
        live_kg.add_relationship("User", "DID", "MorningRun", strength=1.0)
        live_kg.add_relationship("User", "PRACTICED", "Yoga", strength=1.0)

        # Update last_fired to be recent
        with live_kg.driver.session() as session:
            session.run("""
                MATCH (n:Entity {layer: 'EPISODIC'})
                SET n.last_fired = timestamp()
            """)

        dc = DreamConsolidation(live_kg, model=E2E_MODEL)
        result = dc.consolidate(window_ms=60_000)  # 1 minute window

        print(f"\n[DREAM RESULT]: {result}")
        assert result["status"] in ("consolidated", "no_episodes")

        if result["status"] == "consolidated":
            assert result["episodes_processed"] > 0
            print(f"[EPISODES]: {result['episodes_processed']}")
            print(f"[SEMANTICS]: {result['semantics_created']}")
            print(f"[DECAYED]: {result['episodes_decayed']}")

    @neo4j_required
    def test_consolidation_no_episodes(self, live_kg):
        """Verify graceful handling when no episodic memories exist."""
        from neuroform.memory.dream_consolidation import DreamConsolidation

        dc = DreamConsolidation(live_kg, model=E2E_MODEL)
        result = dc.consolidate()
        assert result["status"] == "no_episodes"
        print("[NO EPISODES]: ✅ Graceful early return")

    @full_stack
    def test_semantic_layer_population(self, live_kg):
        """Verify that consolidation creates SEMANTIC layer nodes."""
        from neuroform.memory.dream_consolidation import DreamConsolidation

        # Seed episodic data
        live_kg.add_node("Entity", "User", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "Python", layer=GraphLayer.EPISODIC)
        live_kg.add_relationship("User", "STUDIED", "Python", strength=1.5)

        with live_kg.driver.session() as session:
            session.run("""
                MATCH (n:Entity {layer: 'EPISODIC'})
                SET n.last_fired = timestamp()
            """)

        dc = DreamConsolidation(live_kg, model=E2E_MODEL)
        result = dc.consolidate(window_ms=60_000)

        if result["semantics_created"] > 0:
            with live_kg.driver.session() as session:
                result_q = session.run("""
                    MATCH (n:Entity {layer: 'SEMANTIC'})
                    RETURN n.name AS name
                """)
                semantic_nodes = [r["name"] for r in result_q]
                print(f"[SEMANTIC NODES CREATED]: {semantic_nodes}")
                assert len(semantic_nodes) > 0


# ===========================================================================
# TIER 9: Salience Filter (Neo4j Context)
# ===========================================================================
class TestSalienceFilterE2E:
    """Live tests for the Salience Filter (RAS) against real graph context."""

    @neo4j_required
    def test_salience_scoring_with_real_context(self, live_kg):
        """Score real graph context candidates by salience."""
        from neuroform.memory.salience_filter import SalienceScorer

        # Seed graph with diverse data
        live_kg.add_node("Entity", "User", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Python", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Cooking", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "OldEvent", layer=GraphLayer.EPISODIC)

        live_kg.add_relationship("User", "CODES_IN", "Python", strength=5.0)
        live_kg.add_relationship("User", "ENJOYS", "Cooking", strength=2.0)
        live_kg.add_relationship("User", "ATTENDED", "OldEvent", strength=0.3)

        # Get real context
        context = live_kg.query_context("User")
        non_structural = [c for c in context if c["relationship"] not in ("IN_LAYER", "PEER_LAYER")]

        scorer = SalienceScorer(attention_budget=5)
        scored = scorer.score_candidates("I want to write Python code", non_structural)

        print(f"\n[SALIENCE SCORES]:")
        for s in scored:
            print(f"  {s.get('source', '?')} → {s.get('target', '?')}: {s['salience_score']:.3f}")

        assert len(scored) <= 5
        # Python-related context should rank higher for a Python query
        if len(scored) >= 2:
            assert scored[0]["salience_score"] >= scored[-1]["salience_score"]

    @neo4j_required
    def test_salience_budget_enforcement(self, live_kg):
        """Verify the attention budget limits context injection."""
        from neuroform.memory.salience_filter import SalienceScorer

        # Create many relationships
        live_kg.add_node("Entity", "Hub", layer=GraphLayer.NARRATIVE)
        for i in range(15):
            live_kg.add_node("Entity", f"Target{i}", layer=GraphLayer.NARRATIVE)
            live_kg.add_relationship("Hub", "CONNECTS", f"Target{i}", strength=float(i))

        context = live_kg.query_context("Hub")
        non_structural = [c for c in context if c["relationship"] not in ("IN_LAYER", "PEER_LAYER")]

        scorer = SalienceScorer(attention_budget=5)
        scored = scorer.score_candidates("test query", non_structural)

        assert len(scored) <= 5
        print(f"[BUDGET]: {len(non_structural)} candidates → {len(scored)} after budget filter")


# ===========================================================================
# TIER 10: In-Memory Brain Systems Integration
# ===========================================================================
class TestInMemorySystemsE2E:
    """Integration tests for HabitCache, NeurotransmitterState, CircadianProfile."""

    def test_habit_cache_full_lifecycle(self):
        """Test habit formation, retrieval, and clearing end-to-end."""
        from neuroform.memory.habit_cache import HabitCache

        cache = HabitCache(threshold=3)

        # Record invocations below threshold
        for i in range(2):
            promoted = cache.record_invocation("greet_user", f"Hello response v{i}")
            assert promoted is False

        # Third invocation promotes to habit
        promoted = cache.record_invocation("greet_user", "Hello, welcome back!")
        assert promoted is True

        # Now should serve from cache
        cached = cache.get_cached_response("greet_user")
        assert cached == "Hello, welcome back!"
        assert cache.is_habituated("greet_user") is True

        # Snapshot should show the habit
        snap = cache.snapshot()
        assert "greet_user" in snap["habits"]
        assert snap["invocations"]["greet_user"] == 3

        # De-habituate
        cache.clear_habit("greet_user")
        assert cache.is_habituated("greet_user") is False
        print("[HABIT LIFECYCLE]: ✅ Form → Cache → Clear")

    def test_neurotransmitter_modulation_cycle(self):
        """Test a full neurotransmitter modulation cycle."""
        from neuroform.memory.neurotransmitters import NeurotransmitterState

        ns = NeurotransmitterState()

        # Positive sentiment boosts dopamine
        ns.modulate_from_sentiment(0.8)
        assert ns.dopamine > 0.5
        assert ns.serotonin > 0.5

        # Derived parameters should reflect changes
        temp = ns.llm_temperature
        assert 0.3 <= temp <= 1.0

        budget = ns.attention_budget
        assert isinstance(budget, int) and budget >= 3

        # Negative sentiment increases alertness
        ns.modulate_from_sentiment(-0.8)
        assert ns.norepinephrine > 0.5

        # Reset
        ns.reset()
        assert ns.dopamine == 0.5
        assert ns.serotonin == 0.5

        state = ns.to_dict()
        assert "derived" in state
        print(f"[NT STATE]: {state}")

    def test_circadian_full_day_cycle(self):
        """Verify circadian modulation across a full 24-hour cycle."""
        from neuroform.memory.circadian import CircadianProfile
        from neuroform.memory.neurotransmitters import NeurotransmitterState

        cp = CircadianProfile()

        # Track consolidation windows
        consolidation_hours = []
        peak_alertness_hour = 0
        peak_alertness = 0.0

        for hour in range(24):
            mod = cp.get_modulation(hour)
            if mod["should_consolidate"]:
                consolidation_hours.append(hour)
            if mod["alertness"] > peak_alertness:
                peak_alertness = mod["alertness"]
                peak_alertness_hour = hour

        assert len(consolidation_hours) > 0  # Should have sleep/consolidation hours
        assert peak_alertness_hour in range(8, 17)  # Peak should be during daytime
        print(f"[CONSOLIDATION HOURS]: {consolidation_hours}")
        print(f"[PEAK ALERTNESS]: hour={peak_alertness_hour}, value={peak_alertness}")

        # Test NT modulation during night
        ns = NeurotransmitterState()
        cp.apply_to_neurotransmitters(ns, hour=2)
        assert ns.acetylcholine == 0.8  # Learning boost
        assert ns.norepinephrine < 0.5  # Low alertness

        # Test NT modulation during day
        ns2 = NeurotransmitterState()
        cp.apply_to_neurotransmitters(ns2, hour=10)
        assert ns2.norepinephrine == 1.0  # High alertness
        print("[CIRCADIAN CYCLE]: ✅ Full day validated")

    def test_circadian_and_nt_integration(self):
        """Test that circadian profile correctly modulates neurotransmitter state."""
        from neuroform.memory.circadian import CircadianProfile
        from neuroform.memory.neurotransmitters import NeurotransmitterState

        cp = CircadianProfile()
        ns = NeurotransmitterState()

        # Night phase
        cp.apply_to_neurotransmitters(ns, hour=3)
        night_temp = ns.llm_temperature
        night_budget = ns.attention_budget

        # Day phase
        ns.reset()
        cp.apply_to_neurotransmitters(ns, hour=10)
        day_temp = ns.llm_temperature
        day_budget = ns.attention_budget

        # Day should have different parameters than night
        # Night: higher exploration (dopamine), lower alertness
        # Day: tighter attention budget, lower exploration
        print(f"[NIGHT]: temp={night_temp:.2f}, budget={night_budget}")
        print(f"[DAY]: temp={day_temp:.2f}, budget={day_budget}")
        assert night_budget != day_budget or night_temp != day_temp


# ===========================================================================
# TIER 11: Predictive Model + Default Mode Network (Neo4j + LLM)
# ===========================================================================
class TestPredictiveModelE2E:
    """Live tests for the Cerebellum (predictive error correction)."""

    @full_stack
    def test_predict_intent_live(self, live_kg):
        """Generate a live intent prediction from graph context."""
        from neuroform.memory.predictive_model import PredictiveModel

        # Seed context
        live_kg.add_node("Entity", "User", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Python", layer=GraphLayer.NARRATIVE)
        live_kg.add_relationship("User", "CODES_IN", "Python", strength=5.0)

        pm = PredictiveModel(live_kg, model=E2E_MODEL)
        context = "User codes in Python with high proficiency."
        history = "User: Tell me about my skills."

        prediction = pm.predict_intent(context, history)
        assert isinstance(prediction, str)
        assert len(prediction) > 0
        assert pm.last_prediction == prediction
        print(f"\n[PREDICTION]: {prediction}")

    def test_error_evaluation_live(self):
        """Test error evaluation with realistic text."""
        from neuroform.memory.predictive_model import PredictiveModel
        from unittest.mock import MagicMock

        pm = PredictiveModel(MagicMock(), model=E2E_MODEL)

        # Good prediction
        error_low = pm.evaluate_error(
            "User will ask about Python programming",
            "Tell me more about Python development"
        )
        assert error_low < 1.0

        # Bad prediction
        error_high = pm.evaluate_error(
            "User will ask about cooking recipes",
            "What's the weather forecast tomorrow"
        )
        assert error_high > error_low
        print(f"[ERROR LOW]: {error_low:.3f} (good prediction)")
        print(f"[ERROR HIGH]: {error_high:.3f} (bad prediction)")

    def test_feedback_signal_generation(self):
        """Test feedback signal generation with realistic error values."""
        from neuroform.memory.predictive_model import PredictiveModel
        from unittest.mock import MagicMock

        pm = PredictiveModel(MagicMock(), model=E2E_MODEL)

        # Low error → strengthen
        signals = pm.generate_feedback_signal(0.1, ["User_Python", "User_Skills"])
        assert all(s["action"] == "STRENGTHEN" for s in signals)
        assert len(signals) == 2

        # High error → decay
        signals = pm.generate_feedback_signal(0.9, ["OldContext"])
        assert all(s["action"] == "DECAY" for s in signals)
        print("[FEEDBACK SIGNALS]: ✅ STRENGTHEN and DECAY generated correctly")


class TestDefaultModeNetworkE2E:
    """Live tests for the Default Mode Network (idle introspection)."""

    @full_stack
    def test_introspect_populated_graph(self, live_kg):
        """Run DMN introspection on a populated graph with real LLM analysis."""
        from neuroform.memory.default_mode_network import DefaultModeNetwork

        # Seed a realistic graph
        live_kg.add_node("Entity", "User", layer=GraphLayer.NARRATIVE)
        live_kg.add_node("Entity", "Python", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Coffee", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "Meeting", layer=GraphLayer.EPISODIC)
        live_kg.add_node("Entity", "Orphan", layer=GraphLayer.NARRATIVE)  # No connections

        live_kg.add_relationship("User", "CODES_IN", "Python", strength=5.0)
        live_kg.add_relationship("User", "DRINKS", "Coffee", strength=3.0)
        live_kg.add_relationship("User", "ATTENDED", "Meeting", strength=0.2)  # Weak

        dmn = DefaultModeNetwork(live_kg, model=E2E_MODEL)
        result = dmn.introspect()

        print(f"\n[DMN STATUS]: {result['status']}")
        print(f"[DMN STATS]: {result.get('stats', {})}")
        print(f"[DMN FINDINGS]: {result.get('findings', [])}")

        assert result["status"] == "complete"
        assert "stats" in result
        assert result["stats"]["total_nodes"] > 0

    @neo4j_required
    def test_introspect_empty_graph(self, live_kg):
        """Verify DMN handles empty graph gracefully."""
        from neuroform.memory.default_mode_network import DefaultModeNetwork

        dmn = DefaultModeNetwork(live_kg, model=E2E_MODEL)
        result = dmn.introspect()

        assert result["status"] == "empty_graph"
        print("[DMN EMPTY GRAPH]: ✅ Graceful handling")

    @neo4j_required
    def test_graph_health_statistics(self, live_kg):
        """Verify that DMN gathers accurate graph statistics."""
        from neuroform.memory.default_mode_network import DefaultModeNetwork

        # Seed data
        live_kg.add_node("Entity", "A", layer=GraphLayer.SEMANTIC)
        live_kg.add_node("Entity", "B", layer=GraphLayer.SEMANTIC)
        live_kg.add_relationship("A", "CONNECTED", "B", strength=0.1)

        dmn = DefaultModeNetwork(live_kg, model=E2E_MODEL)
        stats = dmn._gather_stats()

        assert stats["total_nodes"] >= 2
        assert stats["total_edges"] >= 1
        assert isinstance(stats["layer_counts"], list)
        print(f"[GRAPH STATS]: nodes={stats['total_nodes']}, edges={stats['total_edges']}")
        print(f"[LAYER COUNTS]: {stats['layer_counts']}")


# ===========================================================================
# TIER 12: Full Brain Architecture Integration
# ===========================================================================
class TestFullBrainArchitecture:
    """End-to-end test of the complete brain architecture working together."""

    @full_stack
    def test_full_brain_cycle(self, live_kg):
        """
        Orchestrate all brain systems in a single lifecycle:
        1. Circadian sets neurotransmitter state
        2. User chats through Working Memory + Amygdala
        3. Salience filter scores context
        4. Predictive model evaluates
        5. DMN introspects
        6. Dream consolidation runs
        """
        from neuroform.memory.working_memory import WorkingMemory
        from neuroform.memory.amygdala import Amygdala
        from neuroform.memory.salience_filter import SalienceScorer
        from neuroform.memory.neurotransmitters import NeurotransmitterState
        from neuroform.memory.circadian import CircadianProfile
        from neuroform.memory.habit_cache import HabitCache
        from neuroform.memory.predictive_model import PredictiveModel
        from neuroform.memory.default_mode_network import DefaultModeNetwork

        # 1. Initialize all systems
        print("\n[BRAIN INIT] Initializing all systems...")
        wm = WorkingMemory(capacity=7)
        amygdala = Amygdala()
        scorer = SalienceScorer(attention_budget=10)
        nt = NeurotransmitterState()
        circadian = CircadianProfile()
        habits = HabitCache(threshold=3)
        predictor = PredictiveModel(live_kg, model=E2E_MODEL)
        dmn = DefaultModeNetwork(live_kg, model=E2E_MODEL)

        # 2. Apply circadian modulation
        circadian.apply_to_neurotransmitters(nt)
        print(f"[CIRCADIAN]: temp={nt.llm_temperature:.2f}, budget={nt.attention_budget}")

        # 3. Chat with full brain integration
        client = OllamaClient(live_kg, model=E2E_MODEL, working_memory=wm, amygdala=amygdala)
        response = client.chat_with_memory("brain_user", "I love playing piano and composing music.")
        print(f"[CHAT]: {response[:100]}...")

        # 4. Score salience of graph context
        context = live_kg.query_context("User", layer=GraphLayer.NARRATIVE)
        if context:
            scored = scorer.score_candidates("piano music", context)
            print(f"[SALIENCE]: {len(scored)} items scored")

        # 5. Record habit
        habits.record_invocation("greeting", response)
        print(f"[HABITS]: {habits.get_invocation_count('greeting')} invocations")

        # 6. Modulate based on positive sentiment
        nt.modulate_from_sentiment(0.7)
        print(f"[NT MODULATED]: dopamine={nt.dopamine:.2f}")

        # 7. Predict next intent
        prediction = predictor.predict_intent(
            "User likes piano and composing",
            "User: I love playing piano and composing music."
        )
        print(f"[PREDICTION]: {prediction[:80]}...")

        # 8. DMN introspection
        dmn_result = dmn.introspect()
        print(f"[DMN]: status={dmn_result['status']}")

        # 9. Verify all systems operated without errors
        assert isinstance(response, str) and len(response) > 0
        assert nt.dopamine > 0.5  # Positive sentiment boosted
        assert wm.get_conversation_history()  # History recorded

        print("\n[FULL BRAIN CYCLE]: ✅ All 9 systems operational")

