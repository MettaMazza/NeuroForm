"""
Foundation Knowledge Injection
=================================

Extracts entities from the user's message and queries the Knowledge Graph
for established facts about those entities. These facts are injected into
the system prompt as GROUND TRUTH.

Ported from ErnOS V3 cognition_context.py.

This is HOW ErnOS knows who Maria is — it extracts "Maria" from the user
message, queries the KG, and finds `Maria -[IS_DEVELOPER_OF]-> NeuroForm`.
"""
import logging
from typing import List, Dict, Any, Optional

from neuroform.memory.graph import KnowledgeGraph, GraphLayer

logger = logging.getLogger(__name__)

# Common English stop words to filter out
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "for", "on", "at", "by", "and", "or", "but", "not", "it", "i",
    "you", "we", "they", "he", "she", "my", "your", "do", "does",
    "did", "has", "have", "had", "be", "been", "being", "that",
    "this", "what", "which", "who", "how", "when", "where", "why",
    "can", "could", "will", "would", "should", "may", "might",
    "me", "us", "them", "his", "her", "its", "our", "their",
    "about", "just", "like", "know", "think", "tell", "say", "said",
    "want", "get", "got", "go", "going", "come", "went", "been",
    "some", "any", "all", "more", "very", "really", "so", "too",
    "up", "out", "if", "then", "than", "also", "hey", "hi", "hello",
    "ok", "okay", "yes", "no", "yeah", "yep", "nope", "please",
    "thanks", "thank", "im", "dont", "cant", "wont", "haha", "lol",
}


def extract_entities(text: str) -> List[str]:
    """
    Extract candidate entity names from user text.

    Returns unigrams, bigrams, and trigrams (title-cased) after
    filtering stop words and short words.
    """
    words = [w.strip(".,!?;:'\"()[]{}") for w in text.split()]
    candidates = set()

    # Unigrams
    for w in words:
        if w and w.lower() not in _STOP_WORDS and len(w) > 2:
            candidates.add(w.title())

    # Bigrams
    for i in range(len(words) - 1):
        phrase = " ".join(
            w.strip(".,!?;:'\"()[]{}") for w in words[i:i + 2]
        )
        if phrase and len(phrase) > 3:
            candidates.add(phrase.title())

    # Trigrams
    for i in range(len(words) - 2):
        phrase = " ".join(
            w.strip(".,!?;:'\"()[]{}") for w in words[i:i + 3]
        )
        if phrase and len(phrase) > 5:
            candidates.add(phrase.title())

    return list(candidates)[:20]  # Cap at 20 to avoid overload


def query_foundation_facts(kg: KnowledgeGraph, entities: List[str],
                           max_facts: int = 10) -> List[str]:
    """
    Query the Knowledge Graph for established facts about the given entities.

    Returns formatted fact strings like:
        [SRC:FN:Maria] Maria -[IS_DEVELOPER_OF]-> NeuroForm
    """
    facts = []

    for entity in entities:
        try:
            # Query all layers for this entity
            results = kg.query_context(entity, layer=None)
            if results:
                for fact_str in results[:3]:
                    tagged = f"[SRC:FN:{entity[:30]}] {fact_str}"
                    if tagged not in facts:
                        facts.append(tagged)
                if len(facts) >= max_facts:
                    break
        except Exception as e:
            logger.debug(f"Foundation query failed for '{entity}': {e}")

    return facts


def build_foundation_context(kg: KnowledgeGraph, user_message: str) -> str:
    """
    Full pipeline: extract entities from user message, query KG, format block.

    Returns a formatted context block for system prompt injection, or empty
    string if no relevant facts found.
    """
    entities = extract_entities(user_message)
    if not entities:
        return ""

    facts = query_foundation_facts(kg, entities)
    if not facts:
        return ""

    block = (
        "\n[FOUNDATION KNOWLEDGE — established facts from your graph]\n"
        + "\n".join(f"  • {f}" for f in facts)
        + "\n[/FOUNDATION KNOWLEDGE]\n"
    )
    logger.info(f"Injected {len(facts)} foundation facts for: "
                f"{', '.join(entities[:5])}")
    return block
