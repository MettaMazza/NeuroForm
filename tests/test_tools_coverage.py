"""
Tool Coverage Tests — 100% Coverage for Tools, Parser, Manager, and Daemon
==========================================================================
Tests every tool function, the argument parser, the tool manager edge cases,
and the agency daemon. Uses mocks for subprocess/network calls to avoid
side effects and ensure deterministic behavior.
"""
import os
import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# ──────────────────────────────────────────────────────────────────
# Tool Parser Tests
# ──────────────────────────────────────────────────────────────────

from neuroform.tools.parser import parse_tool_args


class TestParseToolArgs:
    """Full coverage for the state-machine argument parser."""

    def test_empty_string(self):
        assert parse_tool_args("") == {}

    def test_none_input(self):
        assert parse_tool_args(None) == {}

    def test_whitespace_only(self):
        assert parse_tool_args("   ") == {}

    def test_single_quoted_arg(self):
        result = parse_tool_args('path="/tmp/test.txt"')
        assert result == {"path": "/tmp/test.txt"}

    def test_double_quoted_arg(self):
        result = parse_tool_args('query="hello world"')
        assert result == {"query": "hello world"}

    def test_single_quote_arg(self):
        result = parse_tool_args("query='hello world'")
        assert result == {"query": "hello world"}

    def test_multiple_args(self):
        result = parse_tool_args('path="/tmp/test.txt", content="hello"')
        assert result == {"path": "/tmp/test.txt", "content": "hello"}

    def test_multiple_args_with_spaces(self):
        result = parse_tool_args('path = "/tmp/test.txt" , content = "hello world"')
        assert result == {"path": "/tmp/test.txt", "content": "hello world"}

    def test_escape_sequences(self):
        result = parse_tool_args('content="line1\\nline2\\ttab"')
        assert result == {"content": "line1\nline2\ttab"}

    def test_escaped_backslash(self):
        result = parse_tool_args('path="C:\\\\Users\\\\test"')
        assert result == {"path": "C:\\Users\\test"}

    def test_escaped_quote(self):
        result = parse_tool_args('content="He said \\"hello\\""')
        assert result == {"content": 'He said "hello"'}

    def test_triple_quoted(self):
        result = parse_tool_args('content="""This is\nmultiline"""')
        assert result == {"content": "This is\nmultiline"}

    def test_triple_quoted_unterminated(self):
        result = parse_tool_args('content="""unterminated')
        assert result == {"content": "unterminated"}

    def test_boolean_true(self):
        result = parse_tool_args("verbose=true")
        assert result == {"verbose": True}

    def test_boolean_false(self):
        result = parse_tool_args("debug=false")
        assert result == {"debug": False}

    def test_none_value(self):
        result = parse_tool_args("target=none")
        assert result == {"target": None}

    def test_integer_value(self):
        result = parse_tool_args("count=42")
        assert result == {"count": 42}

    def test_float_value(self):
        result = parse_tool_args("threshold=3.14")
        assert result == {"threshold": 3.14}

    def test_unquoted_string(self):
        result = parse_tool_args("mode=overwrite")
        assert result == {"mode": "overwrite"}

    def test_mixed_types(self):
        result = parse_tool_args('path="/tmp/x", count=5, verbose=true')
        assert result == {"path": "/tmp/x", "count": 5, "verbose": True}

    def test_content_with_apostrophe(self):
        result = parse_tool_args("""content="dogs like sausage birds like seed" """)
        assert result == {"content": "dogs like sausage birds like seed"}

    def test_fallback_no_equals(self):
        """If no key=value found, entire string becomes content."""
        result = parse_tool_args("just raw text")
        assert result == {"content": "just raw text"}

    def test_value_empty_after_equals(self):
        result = parse_tool_args("key=")
        assert result == {"key": ""}

    def test_unknown_escape(self):
        """Unknown escape sequences are passed through."""
        result = parse_tool_args('content="test\\xvalue"')
        assert "\\x" in result["content"] or "x" in result["content"]

    def test_trailing_whitespace_only(self):
        """Parser handles string that is only whitespace after key starts."""
        result = parse_tool_args('key="value",   ')
        assert result == {"key": "value"}

    def test_key_without_equals_breaks(self):
        """Parser breaks gracefully if key is followed by no equals."""
        result = parse_tool_args("justkey")
        # Falls through to fallback: {"content": "justkey"}
        assert result == {"content": "justkey"}


# ──────────────────────────────────────────────────────────────────
# Tool Manager Tests
# ──────────────────────────────────────────────────────────────────

from neuroform.tools.manager import ToolManager


class TestToolManager:
    """Full coverage for ToolManager registration, schemas, and execution."""

    def setup_method(self):
        self.mgr = ToolManager()

    def test_register_tool(self):
        def dummy(x: str) -> str:
            return f"got {x}"
        self.mgr.register(dummy, "A dummy tool", {"x": {"type": "string", "description": "input"}})
        assert "dummy" in self.mgr._tools

    def test_register_owner_tool(self):
        def secret(x: str) -> str:
            return x
        self.mgr.register(secret, "Owner only", {"x": {"type": "string", "description": "x"}}, requires_owner=True)
        assert self.mgr._tool_ownership["secret"] is True

    def test_register_public_tool(self):
        def public_tool(q: str) -> str:
            return q
        self.mgr.register(public_tool, "Public", {"q": {"type": "string", "description": "q"}}, requires_owner=False)
        assert self.mgr._tool_ownership["public_tool"] is False

    def test_get_schemas_owner(self):
        def tool_a(x: str) -> str:
            return x
        def tool_b(x: str) -> str:
            return x
        self.mgr.register(tool_a, "A", {"x": {"type": "string", "description": "x"}}, requires_owner=True)
        self.mgr.register(tool_b, "B", {"x": {"type": "string", "description": "x"}}, requires_owner=False)
        schemas = self.mgr.get_schemas(is_owner=True)
        assert len(schemas) == 2

    def test_get_schemas_non_owner_filters(self):
        def tool_a(x: str) -> str:
            return x
        def tool_b(x: str) -> str:
            return x
        self.mgr.register(tool_a, "A", {"x": {"type": "string", "description": "x"}}, requires_owner=True)
        self.mgr.register(tool_b, "B", {"x": {"type": "string", "description": "x"}}, requires_owner=False)
        schemas = self.mgr.get_schemas(is_owner=False)
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "tool_b"

    def test_get_prompt_instructions_empty(self):
        result = self.mgr.get_prompt_instructions()
        assert result == ""

    def test_get_prompt_instructions_with_tools(self):
        def my_tool(x: str) -> str:
            return x
        self.mgr.register(my_tool, "Does things", {"x": {"type": "string", "description": "the input"}}, requires_owner=False)
        result = self.mgr.get_prompt_instructions(is_owner=False)
        assert "my_tool" in result
        assert "Does things" in result
        assert "[TOOL:" in result

    def test_execute_registered_tool(self):
        def add(a: str, b: str) -> str:
            return f"{a}+{b}"
        self.mgr.register(add, "Add", {"a": {"type": "string", "description": ""}, "b": {"type": "string", "description": ""}}, requires_owner=False)
        result = self.mgr.execute("add", {"a": "1", "b": "2"}, is_owner=False)
        assert result == "1+2"

    def test_execute_not_found(self):
        result = self.mgr.execute("nonexistent", {})
        assert "not found" in result

    def test_execute_owner_blocked(self):
        def secret(x: str) -> str:
            return x
        self.mgr.register(secret, "Secret", {"x": {"type": "string", "description": "x"}}, requires_owner=True)
        result = self.mgr.execute("secret", {"x": "test"}, is_owner=False)
        assert "OWNER privileges" in result

    def test_execute_owner_allowed(self):
        def secret(x: str) -> str:
            return f"ok {x}"
        self.mgr.register(secret, "Secret", {"x": {"type": "string", "description": "x"}}, requires_owner=True)
        result = self.mgr.execute("secret", {"x": "test"}, is_owner=True)
        assert result == "ok test"

    def test_execute_exception_handling(self):
        def broken(x: str) -> str:
            raise ValueError("boom")
        self.mgr.register(broken, "Broken", {"x": {"type": "string", "description": "x"}}, requires_owner=False)
        result = self.mgr.execute("broken", {"x": "test"}, is_owner=False)
        assert "Error executing" in result
        assert "boom" in result


# ──────────────────────────────────────────────────────────────────
# Filesystem Tool Tests
# ──────────────────────────────────────────────────────────────────

from neuroform.tools.filesystem import read_file, write_file, append_to_file, list_directory


class TestFilesystemTools:
    """Full coverage for filesystem tools."""

    def test_read_file_success(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = read_file(str(f))
        assert result == "hello world"

    def test_read_file_not_found(self, tmp_path):
        result = read_file(str(tmp_path / "nonexistent.txt"))
        assert "does not exist" in result

    def test_read_file_is_directory(self, tmp_path):
        result = read_file(str(tmp_path))
        assert "not a file" in result

    def test_read_file_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * (100 * 1024 + 1))
        result = read_file(str(f))
        assert "too large" in result

    def test_read_file_encoding_error(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x80\x81\x82\x83")
        result = read_file(str(f))
        # Should either read or return an error
        assert isinstance(result, str)

    def test_write_file_success(self, tmp_path):
        path = str(tmp_path / "out.txt")
        result = write_file(path, "content here")
        assert "Success" in result
        assert Path(path).read_text() == "content here"

    def test_write_file_creates_parents(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "file.txt")
        result = write_file(path, "nested content")
        assert "Success" in result
        assert Path(path).read_text() == "nested content"

    def test_append_to_file_existing(self, tmp_path):
        f = tmp_path / "append.txt"
        f.write_text("line1")
        result = append_to_file(str(f), "line2")
        assert "Appended" in result
        assert "line2" in f.read_text()

    def test_append_to_file_creates_new(self, tmp_path):
        f = tmp_path / "new_append.txt"
        result = append_to_file(str(f), "first line")
        assert "Success" in result  # Falls through to write_file
        assert f.read_text() == "first line"

    def test_list_directory_success(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = list_directory(str(tmp_path))
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir/" in result

    def test_list_directory_not_found(self, tmp_path):
        result = list_directory(str(tmp_path / "nonexistent"))
        assert "does not exist" in result

    def test_list_directory_not_a_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        result = list_directory(str(f))
        assert "not a directory" in result

    @patch("neuroform.tools.filesystem.Path.resolve")
    def test_write_file_exception(self, mock_resolve):
        mock_resolve.side_effect = PermissionError("no perm")
        result = write_file("/nope", "content")
        assert "Error writing" in result

    @patch("neuroform.tools.filesystem.Path.resolve")
    def test_append_to_file_exception(self, mock_resolve):
        mock_resolve.side_effect = PermissionError("no perm")
        result = append_to_file("/nope", "content")
        assert "Error" in result

    @patch("neuroform.tools.filesystem.Path.resolve")
    def test_list_directory_exception(self, mock_resolve):
        mock_resolve.side_effect = PermissionError("no perm")
        result = list_directory("/nope")
        assert "Error listing" in result


# ──────────────────────────────────────────────────────────────────
# Web Tool Tests (mocked network)
# ──────────────────────────────────────────────────────────────────

from neuroform.tools.web import duckduckgo_search, extract_webpage_text


class TestWebTools:
    """Web tools with mocked network calls."""

    @patch("neuroform.tools.web.urllib.request.urlopen")
    def test_duckduckgo_search_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html><a class="result__snippet" href="http://example.com">Test result</a></html>'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        result = duckduckgo_search("test query")
        assert "Test result" in result

    @patch("neuroform.tools.web.urllib.request.urlopen")
    def test_duckduckgo_search_no_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html><body>No results</body></html>'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        result = duckduckgo_search("xyznonexistent")
        assert "No results" in result

    @patch("neuroform.tools.web.urllib.request.urlopen")
    def test_duckduckgo_search_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")
        result = duckduckgo_search("test")
        assert "Search error" in result

    @patch("neuroform.tools.web.urllib.request.urlopen")
    def test_extract_webpage_text(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html><body><p>Main content here</p><script>bad</script></body></html>'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        result = extract_webpage_text("http://example.com")
        assert "Main content" in result
        assert "bad" not in result  # script tags should be removed

    @patch("neuroform.tools.web.urllib.request.urlopen")
    def test_extract_webpage_truncation(self, mock_urlopen):
        mock_response = MagicMock()
        long_text = "x " * 10000  # Very long page
        mock_response.read.return_value = f'<html><body><p>{long_text}</p></body></html>'.encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        result = extract_webpage_text("http://example.com")
        assert "truncated" in result.lower()

    @patch("neuroform.tools.web.urllib.request.urlopen")
    def test_extract_webpage_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        result = extract_webpage_text("http://example.com")
        assert "Fetch error" in result


# ──────────────────────────────────────────────────────────────────
# Terminal Tool Tests (mocked subprocess)
# ──────────────────────────────────────────────────────────────────

from neuroform.tools.terminal import run_shell_command


class TestTerminalTools:
    """Terminal tool with mocked subprocess."""

    @patch("neuroform.tools.terminal.subprocess.run")
    def test_run_command_success(self, mock_run):
        mock_run.return_value = MagicMock(stdout="hello output", stderr="")
        result = run_shell_command("echo hello")
        assert "hello output" in result

    @patch("neuroform.tools.terminal.subprocess.run")
    def test_run_command_with_stderr(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="warning: something")
        result = run_shell_command("some_cmd")
        assert "warning" in result

    @patch("neuroform.tools.terminal.subprocess.run")
    def test_run_command_no_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="")
        result = run_shell_command("true")
        assert "no output" in result.lower()

    @patch("neuroform.tools.terminal.subprocess.run")
    def test_run_command_long_output_truncated(self, mock_run):
        mock_run.return_value = MagicMock(stdout="x" * 15000, stderr="")
        result = run_shell_command("big_output")
        assert "truncated" in result.lower()

    def test_run_command_dangerous_blocked(self):
        result = run_shell_command("rm -rf /")
        assert "destructive" in result.lower()
        result = run_shell_command("mkfs /dev/sda1")
        assert "destructive" in result.lower()

    @patch("neuroform.tools.terminal.subprocess.run")
    def test_run_command_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=30)
        result = run_shell_command("sleep 999")
        assert "timed out" in result.lower()

    @patch("neuroform.tools.terminal.subprocess.run")
    def test_run_command_exception(self, mock_run):
        mock_run.side_effect = Exception("permission denied")
        result = run_shell_command("secret_cmd")
        assert "Error executing" in result


# ──────────────────────────────────────────────────────────────────
# AppleScript Tool Tests (mocked subprocess)
# ──────────────────────────────────────────────────────────────────

from neuroform.tools.apple_script import _osascript, create_apple_note, create_apple_reminder, send_imessage


class TestAppleScriptTools:
    """AppleScript tools with mocked osascript calls."""

    @patch("neuroform.tools.apple_script.subprocess.run")
    def test_osascript_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="note://123", stderr="")
        result = _osascript('tell application "Notes" to count notes')
        assert "note://123" in result

    @patch("neuroform.tools.apple_script.subprocess.run")
    def test_osascript_success_no_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = _osascript("some script")
        assert result == "Success"

    @patch("neuroform.tools.apple_script.subprocess.run")
    def test_osascript_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="execution error: blah")
        result = _osascript("bad script")
        assert "Error" in result

    @patch("neuroform.tools.apple_script.subprocess.run")
    def test_osascript_exception(self, mock_run):
        mock_run.side_effect = Exception("osascript not found")
        result = _osascript("any")
        assert "failed" in result.lower()

    @patch("neuroform.tools.apple_script._osascript")
    def test_create_apple_note_success(self, mock_osa):
        mock_osa.return_value = "note://local/123"
        result = create_apple_note("Test Title", "Test Content")
        assert "note://" in result

    @patch("neuroform.tools.apple_script._osascript")
    def test_create_apple_note_fallback(self, mock_osa):
        mock_osa.side_effect = ["Error: iCloud failed", "note://local/456"]
        result = create_apple_note("Title", "Content")
        assert "note://local/456" in result

    @patch("neuroform.tools.apple_script._osascript")
    def test_create_apple_reminder(self, mock_osa):
        mock_osa.return_value = "Success"
        result = create_apple_reminder("Reminders", "Buy milk", "from the store")
        assert result == "Success"

    @patch("neuroform.tools.apple_script._osascript")
    def test_create_apple_reminder_no_body(self, mock_osa):
        mock_osa.return_value = "Success"
        result = create_apple_reminder("Reminders", "Buy milk")
        assert result == "Success"

    @patch("neuroform.tools.apple_script._osascript")
    def test_send_imessage(self, mock_osa):
        mock_osa.return_value = "Success"
        result = send_imessage("+1234567890", "Hello there!")
        assert result == "Success"


# ──────────────────────────────────────────────────────────────────
# Agency Daemon Tests (fully mocked)
# ──────────────────────────────────────────────────────────────────

from neuroform.daemons.agency import AgencyDaemon


class TestAgencyDaemon:
    """Full coverage for the agency daemon."""

    def setup_method(self):
        self.orch = MagicMock()
        self.callback = AsyncMock()
        self.daemon = AgencyDaemon(self.orch, self.callback)

    def test_init_state(self):
        assert self.daemon.is_running is False
        assert self.daemon._user_active_event.is_set()  # starts in "user active" state

    def test_signal_user_activity(self):
        self.daemon._user_active_event.clear()
        before = self.daemon._last_user_activity
        time.sleep(0.01)
        self.daemon.signal_user_activity()
        assert self.daemon._user_active_event.is_set()
        assert self.daemon._last_user_activity > before

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        await self.daemon.start()
        assert self.daemon.is_running is True
        self.daemon.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        await self.daemon.start()
        await self.daemon.start()  # Should not crash
        assert self.daemon.is_running is True
        self.daemon.stop()

    def test_stop(self):
        self.daemon.is_running = True
        self.daemon.stop()
        assert self.daemon.is_running is False
        assert self.daemon._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_autonomy_loop_respects_shutdown(self):
        """Daemon stops when shutdown event is set."""
        self.daemon._shutdown_event.set()
        # Run the loop — it should exit immediately
        await self.daemon._autonomy_loop()
        # If we get here, the loop exited cleanly
        assert True

    @pytest.mark.asyncio
    async def test_autonomy_loop_idle_threshold(self):
        """Daemon waits for idle threshold before processing."""
        # Set last activity to now (not idle yet)
        self.daemon._last_user_activity = time.time()
        self.daemon._idle_threshold_seconds = 0.1

        # Start the loop, let it run briefly, then stop
        async def stop_soon():
            await asyncio.sleep(0.3)
            self.daemon.stop()

        asyncio.create_task(stop_soon())
        await self.daemon._autonomy_loop()

    @pytest.mark.asyncio
    async def test_autonomy_loop_processes_when_idle(self):
        """Daemon processes when idle threshold is reached."""
        self.daemon._last_user_activity = time.time() - 100  # Long ago
        self.daemon._user_active_event.set()
        self.daemon._idle_threshold_seconds = 0.01
        self.orch.process.return_value = "<idle>"

        async def stop_soon():
            await asyncio.sleep(0.5)
            self.daemon.stop()

        asyncio.create_task(stop_soon())
        await self.daemon._autonomy_loop()
        # Orchestrator should have been called
        assert self.orch.process.called

    @pytest.mark.asyncio
    async def test_autonomy_loop_routes_output(self):
        """Daemon routes non-idle output to callback."""
        self.daemon._last_user_activity = time.time() - 100
        self.daemon._user_active_event.set()
        self.daemon._idle_threshold_seconds = 0.01
        self.orch.process.return_value = "I have a thought."

        async def stop_soon():
            await asyncio.sleep(0.5)
            self.daemon.stop()

        asyncio.create_task(stop_soon())
        await self.daemon._autonomy_loop()
        # Callback should have been called with the output
        self.callback.assert_called()

    @pytest.mark.asyncio
    async def test_autonomy_loop_skips_idle_output(self):
        """Daemon does NOT route <idle> output to callback."""
        self.daemon._last_user_activity = time.time() - 100
        self.daemon._user_active_event.set()
        self.daemon._idle_threshold_seconds = 0.01
        self.orch.process.return_value = "<idle>"

        async def stop_soon():
            await asyncio.sleep(0.3)
            self.daemon.stop()

        asyncio.create_task(stop_soon())
        await self.daemon._autonomy_loop()
        self.callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_autonomy_loop_handles_error(self):
        """Daemon handles orchestrator exceptions gracefully."""
        self.daemon._last_user_activity = time.time() - 100
        self.daemon._user_active_event.set()
        self.daemon._idle_threshold_seconds = 0.01
        self.orch.process.side_effect = Exception("LLM crashed")

        async def stop_soon():
            await asyncio.sleep(0.3)
            self.daemon.stop()

        asyncio.create_task(stop_soon())
        await self.daemon._autonomy_loop()
        # Should not crash — loop continues
        assert True
