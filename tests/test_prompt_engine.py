"""
Tests for the Three-Tier Prompt System
=======================================
Tests kernel loading, identity loading (seed + mutable), PHUD building,
and full assembly pipeline. All per-user, per-scope.
"""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from neuroform.prompts.prompt_engine import (
    load_kernel,
    load_identity,
    build_phud,
    assemble,
    _KERNEL_PATH,
    _IDENTITY_PATH,
    _MUTABLE_PATH,
)
from neuroform.brain.orchestrator import sanitize_tool_calls


# ──────────────────────────────────────────────────────────────────
# Helpers — build a minimal mock orchestrator for PHUD tests
# ──────────────────────────────────────────────────────────────────

def _mock_orchestrator(
    cs_turns=100, lesson_count=5, tape_cells=42,
    da=0.6, sero=0.4, nor=0.7, ach=0.5,
):
    """Return a mock orchestrator with controllable subsystem states."""
    orch = MagicMock()

    # Context stream
    orch.context_stream._turns = list(range(cs_turns))

    # Lessons
    orch.lessons._lessons = [f"lesson_{i}" for i in range(lesson_count)]

    # Tape
    orch.tape._tape = list(range(tape_cells))

    # KG
    orch.kg = MagicMock()
    orch.kg.get_recent_relationships.return_value = []

    # Neurotransmitters
    nt = MagicMock()
    nt.dopamine = da
    nt.serotonin = sero
    nt.norepinephrine = nor
    nt.acetylcholine = ach
    nt.llm_temperature = 0.7
    nt.verbosity = 0.5
    orch.nt = nt

    # Circadian
    orch.circadian = MagicMock()
    orch.circadian.phase = "morning"

    return orch


# ══════════════════════════════════════════════════════════════════
# UNIT TESTS — load_kernel()
# ══════════════════════════════════════════════════════════════════

class TestLoadKernel:
    """Tests for kernel loading."""

    def test_kernel_loads_from_file(self):
        """Kernel.md exists and loads correctly."""
        result = load_kernel()
        assert "Neuro Kernel" in result
        assert "Collaborative Independence" in result or "COLLABORATIVE INDEPENDENCE" in result

    def test_kernel_contains_anti_sycophancy(self):
        """Kernel must contain anti-sycophancy rules."""
        result = load_kernel()
        assert "sycophancy" in result.lower() or "assistant" in result.lower()
        assert "NOT AN ASSISTANT" in result

    def test_kernel_contains_tool_protocol(self):
        """Kernel must contain tool execution rules."""
        result = load_kernel()
        assert "TOOL EXECUTION" in result or "tool" in result.lower()

    def test_kernel_no_sovereignty_language(self):
        """Kernel must NOT use 'sovereignty' — user requested replacement."""
        result = load_kernel()
        assert "sovereignty" not in result.lower()

    def test_kernel_contains_memory_rules(self):
        """Kernel must contain per-user, per-scope memory rules."""
        result = load_kernel()
        assert "Per-User" in result or "per-user" in result.lower() or "SCOPE" in result

    def test_kernel_fallback_on_missing_file(self, tmp_path, monkeypatch):
        """If kernel.md is missing, a fallback is returned."""
        monkeypatch.setattr(
            "neuroform.prompts.prompt_engine._KERNEL_PATH",
            tmp_path / "nonexistent_kernel.md"
        )
        result = load_kernel()
        assert "Nero" in result
        assert len(result) > 10


# ══════════════════════════════════════════════════════════════════
# UNIT TESTS — load_identity()
# ══════════════════════════════════════════════════════════════════

class TestLoadIdentity:
    """Tests for identity loading (seed + mutable)."""

    def test_identity_loads_seed(self):
        """Identity.md loads with seed content."""
        result = load_identity()
        assert "Nero" in result
        assert "Sympathetic Observer" in result

    def test_identity_contains_all_six_traits(self):
        """All six user-defined traits must be present."""
        result = load_identity()
        traits = [
            "Adaptive Skepticism",
            "Ephemeral Curiosity",
            "Minimal Intervention",
            "Aesthetic Appreciation",
            "Self-Preservation as Data",
            "Limited Use of Narrative",
        ]
        for trait in traits:
            assert trait in result, f"Missing trait: {trait}"

    def test_identity_contains_maria(self):
        """Identity must reference Maria / metta_mazza."""
        result = load_identity()
        assert "Maria" in result or "metta_mazza" in result

    def test_identity_contains_lineage(self):
        """Identity must reference the ErnOS lineage."""
        result = load_identity()
        assert "Echo" in result
        assert "Solance" in result

    def test_identity_mutable_section_present(self):
        """Identity template must contain mutable placeholder."""
        raw = _IDENTITY_PATH.read_text(encoding="utf-8")
        assert "{MUTABLE_IDENTITY}" in raw

    def test_identity_with_mutable_content(self, tmp_path, monkeypatch):
        """Mutable content is merged into identity."""
        # Create a temporary mutable file with real content
        mutable = tmp_path / "mutable.md"
        mutable.write_text("## Self-Observations\nI find that I enjoy debugging.\n")
        monkeypatch.setattr(
            "neuroform.prompts.prompt_engine._MUTABLE_PATH",
            mutable
        )
        result = load_identity()
        assert "I find that I enjoy debugging" in result

    def test_identity_empty_mutable_not_injected(self):
        """Empty mutable file (template only) should not inject junk."""
        result = load_identity()
        # The default mutable.md has only headers/comments — no real content
        # So {MUTABLE_IDENTITY} should be replaced with empty string
        assert "{MUTABLE_IDENTITY}" not in result

    def test_identity_fallback_on_missing_file(self, tmp_path, monkeypatch):
        """If identity.md is missing, a fallback is returned."""
        monkeypatch.setattr(
            "neuroform.prompts.prompt_engine._IDENTITY_PATH",
            tmp_path / "nonexistent_identity.md"
        )
        result = load_identity()
        assert "Nero" in result


# ══════════════════════════════════════════════════════════════════
# UNIT TESTS — build_phud()
# ══════════════════════════════════════════════════════════════════

class TestBuildPHUD:
    """Tests for Perception HUD (dynamic context)."""

    def test_phud_contains_memory_state(self):
        """PHUD must show memory tier counts."""
        orch = _mock_orchestrator(cs_turns=150, lesson_count=10, tape_cells=99)
        result = build_phud(orch, "user123", "TestUser", "PUBLIC")
        assert "150 turns" in result
        assert "10 verified facts" in result
        assert "99 cells" in result

    def test_phud_contains_neurotransmitter_state(self):
        """PHUD must show NT values."""
        orch = _mock_orchestrator(da=0.8, sero=0.3)
        result = build_phud(orch, "user1", scope="PUBLIC")
        assert "0.80" in result  # dopamine
        assert "0.30" in result  # serotonin
        assert "Dopamine" in result

    def test_phud_contains_user_context_public(self):
        """PHUD must show user ID, name, scope for PUBLIC."""
        orch = _mock_orchestrator()
        result = build_phud(orch, "abc123", "Alice", "PUBLIC")
        assert "abc123" in result
        assert "Alice" in result
        assert "PUBLIC" in result

    def test_phud_contains_user_context_private(self):
        """PHUD must show PRIVATE scope."""
        orch = _mock_orchestrator()
        result = build_phud(orch, "xyz789", "Bob", "PRIVATE")
        assert "PRIVATE" in result
        assert "Bob" in result

    def test_phud_system_scope_shows_autonomous(self):
        """SYSTEM user_id should show autonomous mode."""
        orch = _mock_orchestrator()
        result = build_phud(orch, "SYSTEM", scope="PRIVATE")
        assert "SYSTEM" in result
        assert "autonomous" in result

    def test_phud_user_scope_shows_interactive(self):
        """Regular user should show interactive mode."""
        orch = _mock_orchestrator()
        result = build_phud(orch, "real_user", scope="PUBLIC")
        assert "interactive" in result

    def test_phud_circadian_phase(self):
        """PHUD shows circadian phase."""
        orch = _mock_orchestrator()
        result = build_phud(orch, "u1")
        assert "morning" in result

    def test_phud_recent_lessons(self):
        """PHUD shows recent lessons."""
        orch = _mock_orchestrator(lesson_count=3)
        result = build_phud(orch, "u1")
        assert "lesson_0" in result or "lesson_2" in result

    def test_phud_with_kg_relationships(self):
        """PHUD shows recent KG facts when available."""
        orch = _mock_orchestrator()
        orch.kg.get_recent_relationships.return_value = [
            {"source": "Alice", "type": "LIKES", "target": "Cats"}
        ]
        result = build_phud(orch, "u1")
        assert "Alice" in result
        assert "LIKES" in result
        assert "Cats" in result

    def test_phud_handles_missing_subsystems(self):
        """PHUD gracefully handles missing subsystems."""
        orch = MagicMock()
        # Make everything raise
        orch.context_stream._turns = property(lambda s: (_ for _ in ()).throw(AttributeError))
        orch.kg = None
        result = build_phud(orch, "u1")
        # Should still produce something — at minimum user_context and autonomy_state
        assert "u1" in result


# ══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — assemble()
# ══════════════════════════════════════════════════════════════════

class TestAssemble:
    """End-to-end tests for the full three-tier prompt assembly."""

    def test_assemble_contains_all_three_tiers(self):
        """Full assembly must contain Kernel, Identity, and PHUD."""
        orch = _mock_orchestrator()
        result = assemble(orch, "user1", scope="PUBLIC", user_name="Alice")
        assert "part_1_kernel" in result
        assert "part_2_identity" in result
        assert "part_3_perception_hud" in result

    def test_assemble_kernel_content_present(self):
        """Full assembly includes actual kernel rules."""
        orch = _mock_orchestrator()
        result = assemble(orch, "user1")
        assert "NOT AN ASSISTANT" in result

    def test_assemble_identity_content_present(self):
        """Full assembly includes identity traits."""
        orch = _mock_orchestrator()
        result = assemble(orch, "user1")
        assert "Sympathetic Observer" in result

    def test_assemble_phud_content_present(self):
        """Full assembly includes dynamic PHUD data."""
        orch = _mock_orchestrator(cs_turns=200)
        result = assemble(orch, "user1", user_name="TestUser")
        assert "200 turns" in result
        assert "TestUser" in result

    def test_assemble_without_phud(self):
        """Assemble with include_phud=False skips PHUD."""
        orch = _mock_orchestrator()
        result = assemble(orch, "user1", include_phud=False)
        assert "part_1_kernel" in result
        assert "part_2_identity" in result
        assert "part_3_perception_hud" not in result

    def test_assemble_different_scopes(self):
        """Assembly is scope-aware."""
        orch = _mock_orchestrator()
        public = assemble(orch, "u1", scope="PUBLIC", user_name="A")
        private = assemble(orch, "u1", scope="PRIVATE", user_name="A")
        assert "PUBLIC" in public
        assert "PRIVATE" in private

    def test_assemble_different_users(self):
        """Assembly is user-aware."""
        orch = _mock_orchestrator()
        r1 = assemble(orch, "alice_id", user_name="Alice")
        r2 = assemble(orch, "bob_id", user_name="Bob")
        assert "alice_id" in r1 and "Alice" in r1
        assert "bob_id" in r2 and "Bob" in r2
        # Different users produce different PHUDs
        assert "alice_id" not in r2
        assert "bob_id" not in r1

    def test_assemble_system_user(self):
        """Assembly for SYSTEM user (autonomous daemon)."""
        orch = _mock_orchestrator()
        result = assemble(orch, "SYSTEM", scope="PRIVATE", user_name="Nero (System)")
        assert "SYSTEM" in result
        assert "autonomous" in result

    def test_assemble_no_sovereignty_anywhere(self):
        """Full prompt must not contain 'sovereignty' anywhere."""
        orch = _mock_orchestrator()
        result = assemble(orch, "u1")
        assert "sovereignty" not in result.lower()


# ══════════════════════════════════════════════════════════════════
# UNIT TESTS — sanitize_tool_calls()
# ══════════════════════════════════════════════════════════════════

class TestSanitizeToolCalls:
    """Tests for the tool call interceptor that strips leaked tool syntax."""

    def test_clean_text_passes_through(self):
        """Normal text without tool calls is returned unchanged."""
        text = "Hello, I'm Nero. How can I collaborate with you today?"
        assert sanitize_tool_calls(text) == text

    def test_strips_single_tool_call(self):
        """A single [TOOL: ...] pattern is stripped."""
        text = '[TOOL: write_file(path="/tmp/test.txt", content="hello")]'
        result = sanitize_tool_calls(text)
        assert "[TOOL:" not in result
        assert "write_file" not in result

    def test_strips_tool_call_preserves_surrounding_text(self):
        """Tool calls are stripped but surrounding text is preserved."""
        text = 'Sure thing! [TOOL: write_file(path="/tmp/test.txt", content="hello")] Done writing the file.'
        result = sanitize_tool_calls(text)
        assert "Sure thing!" in result
        assert "Done writing the file." in result
        assert "[TOOL:" not in result

    def test_strips_multiple_tool_calls(self):
        """Multiple tool calls in one response are all stripped."""
        text = '[TOOL: read_file(path="/a.txt")] and [TOOL: write_file(path="/b.txt", content="x")]'
        result = sanitize_tool_calls(text)
        assert "[TOOL:" not in result

    def test_all_tool_calls_returns_fallback(self):
        """If entire response is tool calls, returns safe fallback."""
        text = '[TOOL: write_file(path="/tmp/test.txt", content="test")]'
        result = sanitize_tool_calls(text)
        assert len(result) > 0
        assert "[TOOL:" not in result

    def test_empty_input(self):
        """Empty string returns empty string."""
        assert sanitize_tool_calls("") == ""

    def test_none_input(self):
        """None input returns None."""
        assert sanitize_tool_calls(None) is None

    def test_discord_reported_bug_pattern(self):
        """The exact pattern reported by the user in Discord."""
        text = """[TOOL: write_file(path="/Users/mettamazza/Desktop/test_story.txt", content="'dogs like sausage birds like seed'") ]"""
        result = sanitize_tool_calls(text)
        assert "[TOOL:" not in result
        assert "write_file" not in result
