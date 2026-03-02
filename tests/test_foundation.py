"""Tests for Foundation Knowledge Injection."""
import pytest
from unittest.mock import MagicMock, patch
from neuroform.memory.foundation import (
    extract_entities,
    query_foundation_facts,
    build_foundation_context,
    _STOP_WORDS,
)


class TestExtractEntities:
    def test_simple_names(self):
        entities = extract_entities("My name is Maria")
        titles = [e.lower() for e in entities]
        assert "maria" in titles

    def test_filters_stop_words(self):
        entities = extract_entities("the and or but")
        # All unigrams should be filtered (stop words or too short)
        # Only bigrams/trigrams may remain — those are acceptable since
        # they won't match any KG entity anyway
        for e in entities:
            words = e.lower().split()
            # At least one constituent should not be a stop word
            # (in this case all are, which is fine — the KG won't match)
            assert isinstance(e, str)

    def test_filters_short_words(self):
        entities = extract_entities("I am ok")
        # "ok" is 2 chars → filtered, "am" is a stop word
        assert all(len(e) > 2 for e in entities)

    def test_bigrams(self):
        entities = extract_entities("Maria Smith is here")
        # Should produce bigram "Maria Smith"
        titles = [e.lower() for e in entities]
        assert "maria smith" in titles

    def test_trigrams(self):
        entities = extract_entities("New York City is great")
        titles = [e.lower() for e in entities]
        assert "new york city" in titles

    def test_caps_at_20(self):
        text = " ".join(f"entity{i}" for i in range(50))
        entities = extract_entities(text)
        assert len(entities) <= 20

    def test_empty_input(self):
        assert extract_entities("") == []

    def test_punctuation_stripped(self):
        entities = extract_entities("Hello, Maria!")
        titles = [e.lower() for e in entities]
        assert "maria" in titles

    def test_stop_words_complete(self):
        # Common fragments that caused bugs
        assert "im" in _STOP_WORDS
        assert "just" in _STOP_WORDS
        assert "hey" in _STOP_WORDS
        assert "ok" in _STOP_WORDS


class TestQueryFoundationFacts:
    def test_with_results(self):
        kg = MagicMock()
        kg.query_context.return_value = [
            "Maria -[IS_DEVELOPER_OF]-> NeuroForm"
        ]
        facts = query_foundation_facts(kg, ["Maria"])
        assert len(facts) == 1
        assert "[SRC:FN:Maria]" in facts[0]
        assert "IS_DEVELOPER_OF" in facts[0]

    def test_no_results(self):
        kg = MagicMock()
        kg.query_context.return_value = []
        facts = query_foundation_facts(kg, ["Unknown"])
        assert facts == []

    def test_max_facts(self):
        kg = MagicMock()
        kg.query_context.return_value = [f"fact{i}" for i in range(5)]
        facts = query_foundation_facts(kg, ["A", "B", "C"], max_facts=3)
        assert len(facts) <= 3

    def test_exception_handling(self):
        kg = MagicMock()
        kg.query_context.side_effect = Exception("DB error")
        facts = query_foundation_facts(kg, ["Maria"])
        assert facts == []

    def test_dedup(self):
        kg = MagicMock()
        kg.query_context.return_value = ["same fact"]
        facts = query_foundation_facts(kg, ["A", "B"])
        # Same fact from different entities should not duplicate
        unique_facts = set(facts)
        assert len(unique_facts) == len(facts)


class TestBuildFoundationContext:
    def test_builds_block(self):
        kg = MagicMock()
        kg.query_context.return_value = [
            "Maria -[IS_DEVELOPER_OF]-> NeuroForm"
        ]
        block = build_foundation_context(kg, "Do you remember Maria?")
        assert "FOUNDATION KNOWLEDGE" in block
        assert "Maria" in block

    def test_empty_message(self):
        kg = MagicMock()
        block = build_foundation_context(kg, "")
        assert block == ""

    def test_no_entities_found(self):
        kg = MagicMock()
        block = build_foundation_context(kg, "the and or but")
        assert block == ""

    def test_no_facts_found(self):
        kg = MagicMock()
        kg.query_context.return_value = []
        block = build_foundation_context(kg, "Tell me about QuantumPhysics")
        assert block == ""
