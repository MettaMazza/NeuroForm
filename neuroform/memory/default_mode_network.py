"""
Default Mode Network — Self-Referential Processing
=====================================================

Biological basis: The DMN activates during idle states — mind-wandering,
self-reflection, future planning. It's the brain thinking about itself
when not engaged in external tasks (Raichle, 2015).

Computational analogue: An idle-time introspection loop that reviews the
graph structure, identifies missing connections, contradictions, and
knowledge gaps, and pre-seeds facts for anticipated future conversations.
"""
import logging
from typing import Dict, Any, List, Optional
import json
import ollama
from neuroform.memory.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class DefaultModeNetwork:
    """
    The DMN — Idle-Time Introspection.

    When the system has no active conversation, the DMN performs
    meta-analysis of the knowledge graph: structural health, gap
    detection, and proactive knowledge building.
    """

    def __init__(self, kg: KnowledgeGraph, model: str = "llama3"):
        self.kg = kg
        self.model = model

    def introspect(self) -> Dict[str, Any]:
        """
        Run a meta-analysis of the knowledge graph.

        Returns:
            A report dict with findings, suggestions, and graph health metrics.
        """
        if not self.kg.driver:
            return {"status": "offline", "findings": []}

        # 1. Gather graph statistics
        stats = self._gather_stats()
        if stats["total_nodes"] == 0:
            return {"status": "empty_graph", "findings": [], "stats": stats}

        # 2. Ask the LLM to analyze
        findings = self._analyze_with_llm(stats)

        return {"status": "complete", "findings": findings, "stats": stats}

    def _gather_stats(self) -> Dict[str, Any]:
        """Gather structural statistics from the knowledge graph."""
        queries = {
            "total_nodes": "MATCH (n:Entity) RETURN count(n) AS c",
            "total_edges": "MATCH ()-[r]->() WHERE type(r) <> 'IN_LAYER' AND type(r) <> 'PEER_LAYER' RETURN count(r) AS c",
            "layer_counts": "MATCH (n:Entity) RETURN n.layer AS layer, count(n) AS c ORDER BY c DESC",
            "weak_edges": "MATCH ()-[r]->() WHERE r.strength IS NOT NULL AND r.strength < 0.3 AND type(r) <> 'IN_LAYER' AND type(r) <> 'PEER_LAYER' RETURN count(r) AS c",
            "orphan_nodes": "MATCH (n:Entity) WHERE NOT (n)-->() AND NOT (n)<--() RETURN count(n) AS c",
        }

        stats = {}
        with self.kg.driver.session() as session:
            for key, q in queries.items():
                try:
                    result = session.run(q)
                    if key == "layer_counts":
                        stats[key] = [{
                            "layer": r["layer"],
                            "count": r["c"]
                        } for r in result]
                    else:
                        record = result.single()
                        stats[key] = record["c"] if record else 0
                except Exception as e:
                    logger.warning(f"DMN stats query failed ({key}): {e}")
                    stats[key] = 0 if key != "layer_counts" else []

        return stats

    def _analyze_with_llm(self, stats: Dict[str, Any]) -> List[str]:
        """Use the LLM to provide meta-analysis insights."""
        prompt = f"""You are a memory health analyst. Analyze this knowledge graph and identify issues.

Graph Statistics:
- Total entities: {stats.get('total_nodes', 0)}
- Total relationships: {stats.get('total_edges', 0)}
- Weak edges (strength < 0.3): {stats.get('weak_edges', 0)}
- Orphan nodes (no connections): {stats.get('orphan_nodes', 0)}
- Layer distribution: {json.dumps(stats.get('layer_counts', []))}

Return a JSON array of string findings/suggestions:
```json
["Finding 1: ...", "Suggestion: ..."]
```
Be concise. Return ONLY the JSON array."""

        try:
            response = ollama.chat(model=self.model, messages=[
                {"role": "system", "content": "You analyze knowledge graph health."},
                {"role": "user", "content": prompt}
            ])
            reply = response["message"]["content"]
            return self._parse_findings(reply)
        except Exception as e:
            logger.warning(f"DMN LLM analysis failed: {e}")
            return []

    def _parse_findings(self, text: str) -> List[str]:
        """Parse LLM findings from JSON."""
        try:
            if "```json" in text:
                json_block = text.split("```json")[-1].split("```")[0].strip()
            elif "```" in text:
                json_block = text.split("```")[1].split("```")[0].strip()
            else:
                json_block = text.strip()

            parsed = json.loads(json_block)
            if isinstance(parsed, list):
                return [str(f) for f in parsed]
            return []
        except json.JSONDecodeError:
            logger.warning("DMN: failed to parse LLM findings.")
            return []
