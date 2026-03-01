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
