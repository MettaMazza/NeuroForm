"""
Dream Consolidation — Hippocampal Replay
========================================

Biological basis: During sleep, the hippocampus replays recent experiences at
accelerated speed, transferring fragile episodic memories into stable semantic
knowledge in the neocortex. The Synaptic Homeostasis Hypothesis (Tononi &
Cirelli, 2003) suggests that sleep also globally downscales synaptic strength,
keeping only the most consolidated traces.

Computational analogue: A background cron-style process that:
1. Pulls recent EPISODIC layer nodes from the graph
2. Feeds them to the LLM with a consolidation prompt
3. Creates distilled SEMANTIC layer nodes from the LLM output
4. Aggressively decays the original episodic nodes
"""
import logging
from typing import Dict, Any, List, Optional
import json
import ollama
from neuroform.memory.graph import KnowledgeGraph, GraphLayer

logger = logging.getLogger(__name__)


class DreamConsolidation:
    """
    The Hippocampal Replay Engine.

    Converts fragile episodic memories into stable semantic knowledge
    through LLM-driven distillation. Analogous to the brain's sleep
    consolidation process.
    """

    def __init__(self, kg: KnowledgeGraph, model: str = "gemma3:4b"):
        self.kg = kg
        self.model = model

    def consolidate(self, window_ms: int = 21_600_000) -> Dict[str, Any]:
        """
        Run a dream consolidation cycle.

        1. Fetch all EPISODIC nodes created within the time window
        2. Feed them to the LLM with a distillation prompt
        3. Create SEMANTIC nodes from the distilled output
        4. Aggressively decay the original episodic nodes

        Args:
            window_ms: Time window in milliseconds (default: 6 hours = 21,600,000ms)

        Returns:
            Summary dict with counts of episodes processed, semantics created, and episodes decayed.
        """
        if not self.kg.driver:
            return {"status": "offline", "episodes_processed": 0, "semantics_created": 0, "episodes_decayed": 0}

        # 1. Fetch recent episodic memories
        episodes = self._fetch_recent_episodes(window_ms)
        if not episodes:
            return {"status": "no_episodes", "episodes_processed": 0, "semantics_created": 0, "episodes_decayed": 0}

        # 2. Ask the LLM to distill them into semantic facts
        semantics = self._distill_episodes(episodes)

        # 3. Write the semantic nodes to the graph
        created = self._write_semantic_nodes(semantics)

        # 4. Aggressively decay the episodic nodes
        decayed = self._decay_episodes(episodes)

        return {
            "status": "consolidated",
            "episodes_processed": len(episodes),
            "semantics_created": created,
            "episodes_decayed": decayed,
        }

    def _fetch_recent_episodes(self, window_ms: int) -> List[Dict[str, Any]]:
        """Fetch episodic nodes created within the time window."""
        query = """
        MATCH (a:Entity {layer: 'EPISODIC'})-[r]->(b)
        WHERE type(r) <> 'IN_LAYER' AND type(r) <> 'PEER_LAYER'
        AND a.last_fired > (timestamp() - $window_ms)
        RETURN a.name AS source, type(r) AS relation, b.name AS target,
               r.strength AS strength
        LIMIT 50
        """
        with self.kg.driver.session() as session:
            result = session.run(query, window_ms=window_ms)
            return [
                {"source": r["source"], "relation": r["relation"],
                 "target": r["target"], "strength": r["strength"]}
                for r in result
            ]

    def _distill_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Use the LLM to distill raw episodic events into semantic generalizations."""
        episode_str = json.dumps(episodes, indent=2)
        prompt = f"""You are a memory consolidation engine. Below are raw episodic memories 
(specific events that occurred). Your job is to distill them into general semantic facts 
that should be retained long-term.

Raw episodic memories:
{episode_str}

For each meaningful pattern or fact you can extract, return a JSON array of semantic facts:
```json
[
  {{"source": "User", "relation": "PREFERS", "target": "Morning Coffee", "confidence": 0.9}},
  {{"source": "User", "relation": "WORKS_AT", "target": "Tech Company", "confidence": 0.8}}
]
```
Only extract facts that represent stable, generalizable truths — not one-time events.
If no generalizable facts can be extracted, return an empty array `[]`.
Return ONLY the JSON array.
"""
        try:
            response = ollama.chat(model=self.model, messages=[
                {"role": "system", "content": "You are an autonomous memory consolidation daemon."},
                {"role": "user", "content": prompt}
            ])
            reply = response['message']['content']
            return self._parse_semantics(reply)
        except Exception as e:
            logger.error(f"Dream consolidation LLM error: {e}")
            return []

    def _parse_semantics(self, text: str) -> List[Dict[str, str]]:
        """Parse the LLM's JSON output into a list of semantic facts."""
        try:
            if "```json" in text:
                json_block = text.split("```json")[-1].split("```")[0].strip()
            elif "```" in text:
                json_block = text.split("```")[1].split("```")[0].strip()
            else:
                json_block = text.strip()

            parsed = json.loads(json_block)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            logger.warning("Dream consolidation: failed to parse LLM semantic output.")
            return []

    def _write_semantic_nodes(self, semantics: List[Dict[str, str]]) -> int:
        """Write distilled semantic facts to the graph's SEMANTIC layer."""
        created = 0
        for fact in semantics:
            source = fact.get("source")
            rel = fact.get("relation")
            target = fact.get("target")

            if not all([source, rel, target]):
                continue

            try:
                self.kg.add_node("Entity", source, layer=GraphLayer.SEMANTIC)
                self.kg.add_node("Entity", target, layer=GraphLayer.SEMANTIC)
                self.kg.add_relationship(source, rel, target, strength=2.0)
                created += 1
                logger.info(f"Dream: consolidated {source} -[{rel}]-> {target}")
            except Exception as e:
                logger.warning(f"Dream: failed to write semantic node: {e}")

        return created

    def _decay_episodes(self, episodes: List[Dict[str, Any]]) -> int:
        """
        Aggressively decay the original episodic nodes after consolidation.
        This mimics the biological process of episodic memories fading 
        after being consolidated into semantic memory.
        """
        decayed = 0
        decay_query = """
        MATCH (a:Entity {name: $name, layer: 'EPISODIC'})-[r]->()
        WHERE type(r) <> 'IN_LAYER' AND type(r) <> 'PEER_LAYER'
        SET r.strength = r.strength - 0.5
        """
        with self.kg.driver.session() as session:
            # Get unique episode source names
            sources = set(ep["source"] for ep in episodes)
            for name in sources:
                try:
                    session.run(decay_query, name=name)
                    decayed += 1
                except Exception as e:
                    logger.warning(f"Dream: failed to decay episode {name}: {e}")

        return decayed
