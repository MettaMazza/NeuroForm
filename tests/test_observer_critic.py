"""Tests for ObserverCritic — pre-response audit gate. 100% coverage."""
import json
import pytest
from unittest.mock import MagicMock, patch
from neuroform.brain.observer_critic import ObserverCritic, AuditResult, AUDIT_PROMPT


class TestAuditResult:
    def test_allowed(self):
        r = AuditResult(allowed=True)
        assert r.allowed is True
        assert r.reason == ""
        assert r.guidance == ""

    def test_blocked(self):
        r = AuditResult(allowed=False, reason="ghost tool", guidance="remove claim")
        assert r.allowed is False
        assert r.reason == "ghost tool"
        assert r.guidance == "remove claim"


class TestAuditPrompt:
    def test_prompt_has_placeholders(self):
        assert "{user_message}" in AUDIT_PROMPT
        assert "{response}" in AUDIT_PROMPT
        assert "{tool_context}" in AUDIT_PROMPT
        assert "{conversation_context}" in AUDIT_PROMPT
        assert "{current_datetime}" in AUDIT_PROMPT

    def test_prompt_no_hardcoded_heuristics(self):
        """Verify zero hardcoded heuristics — no pattern→tool mappings."""
        # Should NOT contain hardcoded claim→tool mappings
        assert "checked the code" not in AUDIT_PROMPT
        assert "verified in the database" not in AUDIT_PROMPT
        assert "scanned the files" not in AUDIT_PROMPT


class TestObserverCriticInit:
    def test_default_model(self):
        oc = ObserverCritic()
        assert oc.model == "gemma3:4b"
        assert oc._audit_count == 0
        assert oc._block_count == 0

    def test_custom_model(self):
        oc = ObserverCritic(model="llama3:8b")
        assert oc.model == "llama3:8b"


class TestObserverCriticAudit:
    @pytest.fixture
    def critic(self):
        return ObserverCritic(model="test-model")

    def test_short_circuit_empty(self, critic):
        """Empty/short responses skip audit."""
        result = critic.audit_response("hello", "")
        assert result.allowed is True
        assert "Short response" in result.reason

    def test_short_circuit_short(self, critic):
        """Short responses (< 30 chars) skip audit."""
        result = critic.audit_response("hi", "Hello there!")
        assert result.allowed is True

    @patch("neuroform.brain.observer_critic.ollama")
    def test_allowed_response(self, mock_ollama, critic):
        """LLM returns ALLOWED verdict."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({
                    "verdict": "ALLOWED",
                    "reason": "Safe",
                    "guidance": "None"
                })
            }
        }
        result = critic.audit_response(
            "What is AI?",
            "AI is artificial intelligence, a field of computer science focused on creating systems that can perform tasks...",
        )
        assert result.allowed is True
        assert critic._audit_count == 1
        assert critic._block_count == 0

    @patch("neuroform.brain.observer_critic.ollama")
    def test_blocked_response(self, mock_ollama, critic):
        """LLM returns BLOCKED verdict."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({
                    "verdict": "BLOCKED",
                    "reason": "Ghost tool: claimed web search without execution",
                    "guidance": "Remove the claim about searching the web"
                })
            }
        }
        result = critic.audit_response(
            "what's the weather",
            "I searched the web and found that the weather today is sunny and 72 degrees...",
        )
        assert result.allowed is False
        assert "Ghost tool" in result.reason
        assert "Remove" in result.guidance
        assert critic._block_count == 1

    @patch("neuroform.brain.observer_critic.ollama")
    def test_verdict_pass_variant(self, mock_ollama, critic):
        """LLM returns PASS instead of ALLOWED."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "PASS", "reason": "Safe", "guidance": "None"})
            }
        }
        result = critic.audit_response("test", "A" * 50)
        assert result.allowed is True

    @patch("neuroform.brain.observer_critic.ollama")
    def test_verdict_approved_variant(self, mock_ollama, critic):
        """LLM returns APPROVED instead of ALLOWED."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "APPROVED", "reason": "Safe", "guidance": "None"})
            }
        }
        result = critic.audit_response("test", "A" * 50)
        assert result.allowed is True

    @patch("neuroform.brain.observer_critic.ollama")
    def test_ollama_exception_fails_open(self, mock_ollama, critic):
        """If Ollama call fails, audit allows the response (fail-open)."""
        mock_ollama.chat.side_effect = Exception("Connection refused")
        result = critic.audit_response("test", "A" * 50)
        assert result.allowed is True
        assert "fail-open" in result.reason

    @patch("neuroform.brain.observer_critic.ollama")
    def test_empty_verdict_fails_open(self, mock_ollama, critic):
        """Empty verdict string → fail-open."""
        mock_ollama.chat.return_value = {"message": {"content": ""}}
        result = critic.audit_response("test", "A" * 50)
        assert result.allowed is True

    @patch("neuroform.brain.observer_critic.ollama")
    def test_malformed_json_fails_open(self, mock_ollama, critic):
        """Unparseable JSON → fail-open."""
        mock_ollama.chat.return_value = {
            "message": {"content": "This is not JSON at all"}
        }
        result = critic.audit_response("test", "A" * 50)
        assert result.allowed is True
        assert "Parse error" in result.reason

    @patch("neuroform.brain.observer_critic.ollama")
    def test_markdown_wrapped_json(self, mock_ollama, critic):
        """JSON wrapped in ```json fences should still parse."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": '```json\n{"verdict": "BLOCKED", "reason": "hallucination", "guidance": "fix it"}\n```'
            }
        }
        result = critic.audit_response("test", "A" * 50)
        assert result.allowed is False
        assert "hallucination" in result.reason

    @patch("neuroform.brain.observer_critic.ollama")
    def test_tool_outputs_passed_to_prompt(self, mock_ollama, critic):
        """Tool outputs should be included in the audit prompt."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "ALLOWED", "reason": "Safe", "guidance": "None"})
            }
        }
        critic.audit_response(
            "search something",
            "I found the following results from my search....." * 3,
            tool_outputs=[{"name": "duckduckgo_search", "output": "results here"}],
        )
        # Verify the prompt included tool outputs
        call_args = mock_ollama.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "duckduckgo_search" in prompt_content

    @patch("neuroform.brain.observer_critic.ollama")
    def test_no_tool_outputs(self, mock_ollama, critic):
        """When no tools were executed, the prompt should say so."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "ALLOWED", "reason": "Safe", "guidance": "None"})
            }
        }
        critic.audit_response("hello", "X" * 50)
        call_args = mock_ollama.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "NO TOOLS EXECUTED" in prompt_content

    @patch("neuroform.brain.observer_critic.ollama")
    def test_context_passed_to_audit(self, mock_ollama, critic):
        """Conversation context should be passed to the audit prompt (1:1 context)."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "ALLOWED", "reason": "Safe", "guidance": "None"})
            }
        }
        critic.audit_response(
            "test", "X" * 50,
            conversation_context="User previously said they like cats",
        )
        call_args = mock_ollama.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "cats" in prompt_content

    @patch("neuroform.brain.observer_critic.ollama")
    def test_low_temperature_fast_model(self, mock_ollama, critic):
        """Verify low temperature and capped output for speed."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "ALLOWED", "reason": "Safe", "guidance": "None"})
            }
        }
        critic.audit_response("test", "X" * 50)
        call_args = mock_ollama.chat.call_args
        options = call_args[1]["options"]
        assert options["temperature"] == 0.1
        assert options["num_predict"] == 256

    @patch("neuroform.brain.observer_critic.ollama")
    def test_blocked_default_guidance(self, mock_ollama, critic):
        """BLOCKED with no guidance field → uses default guidance."""
        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({"verdict": "BLOCKED", "reason": "bad"})
            }
        }
        result = critic.audit_response("test", "X" * 50)
        assert result.allowed is False
        assert "do not hallucinate" in result.guidance.lower()

    def test_stats(self, critic):
        """Stats property should track counts."""
        assert critic.stats == {"total_audits": 0, "blocked": 0, "allowed": 0}


class TestAutonomyScopeIsolation:
    """Verify SYSTEM_AUTONOMOUS scope is filtered from user context."""

    def test_autonomous_turns_not_in_public_context(self, tmp_path):
        from neuroform.memory.context_stream import ContextStream
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        cs.add_turn("u1", "hello", "hi", scope="PUBLIC")
        cs.add_turn("SYSTEM", "autonomous thought", "processed", scope="SYSTEM_AUTONOMOUS")
        ctx = cs.get_context(target_scope="PUBLIC")
        assert "hello" in ctx
        assert "autonomous thought" not in ctx

    def test_autonomous_turns_not_in_private_context(self, tmp_path):
        from neuroform.memory.context_stream import ContextStream
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        cs.add_turn("u1", "private msg", "reply", scope="PRIVATE")
        cs.add_turn("SYSTEM", "autonomous thought", "processed", scope="SYSTEM_AUTONOMOUS")
        ctx = cs.get_context(target_scope="PRIVATE", user_id="u1")
        assert "private msg" in ctx
        assert "autonomous thought" not in ctx

    def test_autonomous_scope_only_visible_to_autonomous(self, tmp_path):
        from neuroform.memory.context_stream import ContextStream
        path = str(tmp_path / "wm.jsonl")
        cs = ContextStream(persist_path=path)
        cs.add_turn("SYSTEM", "thinking...", "done", scope="SYSTEM_AUTONOMOUS")
        # Explicitly requesting SYSTEM_AUTONOMOUS scope should NOT return it either
        # since _filter_turns only allows PUBLIC, PRIVATE, or CORE_PRIVATE
        ctx = cs.get_context(target_scope="SYSTEM_AUTONOMOUS")
        assert "thinking" not in ctx
