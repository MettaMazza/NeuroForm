"""
Unit + Integration Tests for Bridge Core + Discord Adapter
============================================================
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from neuroform.bridge.bridge import (
    MessageEvent,
    ResponseEvent,
    PlatformAdapter,
    BridgeCore,
)
from neuroform.bridge.discord_adapter import DiscordAdapter, DISCORD_MAX_MESSAGE_LENGTH


# ===========================================================================
# MessageEvent Tests
# ===========================================================================
class TestMessageEvent:

    def test_creation(self):
        event = MessageEvent(
            user_id="123", channel_id="456",
            content="Hello", platform="test"
        )
        assert event.user_id == "123"
        assert event.channel_id == "456"
        assert event.content == "Hello"
        assert event.platform == "test"
        assert isinstance(event.timestamp, datetime)
        assert event.metadata == {}

    def test_with_metadata(self):
        event = MessageEvent(
            user_id="123", channel_id="456",
            content="Hi", platform="discord",
            metadata={"guild_id": "789"}
        )
        assert event.metadata["guild_id"] == "789"


# ===========================================================================
# ResponseEvent Tests
# ===========================================================================
class TestResponseEvent:

    def test_creation(self):
        resp = ResponseEvent(
            content="Hello!", channel_id="456", platform="test"
        )
        assert resp.content == "Hello!"
        assert resp.reply_to_user is None

    def test_with_reply(self):
        resp = ResponseEvent(
            content="Hi!", channel_id="456",
            platform="test", reply_to_user="123"
        )
        assert resp.reply_to_user == "123"


# ===========================================================================
# BridgeCore Tests
# ===========================================================================
class TestBridgeCore:

    def test_init(self):
        bridge = BridgeCore()
        assert not bridge.is_initialized
        assert bridge.adapters == {}

    def test_initialize(self):
        bridge = BridgeCore()
        mock_kg = MagicMock()
        mock_client = MagicMock()
        bridge.initialize(mock_kg, mock_client, allowed_channels=["123"])
        assert bridge.is_initialized

    def test_channel_filter_allowed(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock(), allowed_channels=["123", "456"])
        assert bridge.is_channel_allowed("123") is True
        assert bridge.is_channel_allowed("456") is True
        assert bridge.is_channel_allowed("789") is False

    def test_channel_filter_empty_allows_all(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock(), allowed_channels=[])
        assert bridge.is_channel_allowed("any") is True

    def test_channel_filter_int_string_match(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock(), allowed_channels=[123])
        assert bridge.is_channel_allowed("123") is True

    def test_register_adapter(self):
        bridge = BridgeCore()
        adapter = MagicMock(spec=PlatformAdapter)
        adapter.platform_name = "test"
        bridge.register_adapter(adapter)
        assert "test" in bridge.adapters

    def test_process_message_not_initialized(self):
        bridge = BridgeCore()
        event = MessageEvent("u", "c", "hello", "test")
        result = bridge.process_message(event)
        assert result is None

    def test_process_message_channel_blocked(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock(), allowed_channels=["allowed_only"])
        event = MessageEvent("u", "blocked_channel", "hello", "test")
        result = bridge.process_message(event)
        assert result is None

    def test_process_message_success(self):
        bridge = BridgeCore()
        mock_client = MagicMock()
        mock_client.chat_with_memory.return_value = "Brain response"
        bridge.initialize(MagicMock(), mock_client, allowed_channels=["ch1"])

        event = MessageEvent("user1", "ch1", "Hello brain", "test")
        result = bridge.process_message(event)

        assert result is not None
        assert result.content == "Brain response"
        assert result.channel_id == "ch1"
        assert result.platform == "test"
        assert result.reply_to_user == "user1"
        mock_client.chat_with_memory.assert_called_once_with("user1", "Hello brain")

    def test_process_message_brain_error(self):
        bridge = BridgeCore()
        mock_client = MagicMock()
        mock_client.chat_with_memory.side_effect = Exception("Brain crash")
        bridge.initialize(MagicMock(), mock_client, allowed_channels=[])

        event = MessageEvent("user1", "ch1", "Hello", "test")
        result = bridge.process_message(event)

        assert result is not None
        assert "error" in result.content.lower()

    def test_process_message_with_orchestrator(self):
        bridge = BridgeCore()
        mock_orch = MagicMock()
        mock_orch.process.return_value = "Orchestrator response"
        bridge.initialize(MagicMock(), MagicMock(),
                         allowed_channels=["ch1"], orchestrator=mock_orch)

        event = MessageEvent("user1", "ch1", "Hello", "test")
        result = bridge.process_message(event)

        assert result is not None
        assert result.content == "Orchestrator response"
        mock_orch.process.assert_called_once_with("user1", "Hello")


# ===========================================================================
# DiscordAdapter Tests
# ===========================================================================
class TestDiscordAdapter:

    def test_platform_name(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)
        assert adapter.platform_name == "discord"

    def test_client_created(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)
        assert adapter.client is not None

    def test_chunk_message_short(self):
        chunks = DiscordAdapter._chunk_message("Hello world")
        assert chunks == ["Hello world"]

    def test_chunk_message_at_limit(self):
        text = "a" * DISCORD_MAX_MESSAGE_LENGTH
        chunks = DiscordAdapter._chunk_message(text)
        assert len(chunks) == 1

    def test_chunk_message_over_limit(self):
        text = "word " * 500  # ~2500 chars
        chunks = DiscordAdapter._chunk_message(text, limit=100)
        assert all(len(c) <= 100 for c in chunks)
        # Reassembled content should contain all words
        reassembled = " ".join(chunks)
        assert reassembled.count("word") >= 400

    def test_chunk_message_no_spaces(self):
        text = "x" * 3000
        chunks = DiscordAdapter._chunk_message(text, limit=1000)
        assert len(chunks) >= 3
        assert all(len(c) <= 1000 for c in chunks)

    def test_chunk_message_newline_split(self):
        text = "line1\n" * 300  # ~1800 chars
        chunks = DiscordAdapter._chunk_message(text, limit=100)
        assert all(len(c) <= 100 for c in chunks)

    @pytest.mark.asyncio
    async def test_send_response_via_channel(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)

        mock_channel = AsyncMock()
        response = ResponseEvent("Hello!", "123", "discord")
        await adapter._send_discord_response(mock_channel, response)
        mock_channel.send.assert_called_once_with("Hello!")

    @pytest.mark.asyncio
    async def test_send_response_empty(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)

        mock_channel = AsyncMock()
        response = ResponseEvent("", "123", "discord")
        await adapter._send_discord_response(mock_channel, response)
        mock_channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_response_chunked(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)

        mock_channel = AsyncMock()
        long_msg = "a " * 1500  # 3000 chars
        response = ResponseEvent(long_msg, "123", "discord")
        await adapter._send_discord_response(mock_channel, response)
        assert mock_channel.send.call_count >= 2

    @pytest.mark.asyncio
    async def test_stop(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)
        adapter._client.close = AsyncMock()
        await adapter.stop()
        adapter._client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_response_by_channel_id(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)

        mock_channel = AsyncMock()
        adapter._client.get_channel = MagicMock(return_value=mock_channel)

        response = ResponseEvent("Test", "12345", "discord")
        await adapter.send_response(response)
        adapter._client.get_channel.assert_called_once_with(12345)
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_response_channel_not_found(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)

        adapter._client.get_channel = MagicMock(return_value=None)
        response = ResponseEvent("Test", "99999", "discord")
        # Should not raise
        await adapter.send_response(response)

    @pytest.mark.asyncio
    async def test_send_discord_response_http_exception(self):
        import discord
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)

        mock_channel = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 500
        mock_channel.send.side_effect = discord.HTTPException(mock_response, "Server Error")

        response = ResponseEvent("Test", "123", "discord")
        # Should not raise, but log error
        await adapter._send_discord_response(mock_channel, response)


    @pytest.mark.asyncio
    async def test_start_calls_client_start(self):
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)
        adapter._client.start = AsyncMock()
        await adapter.start()
        adapter._client.start.assert_called_once_with("fake_token")

    def test_set_message_handler(self):
        """Test the PlatformAdapter.set_message_handler on the adapter."""
        bridge = BridgeCore()
        bridge.initialize(MagicMock(), MagicMock())
        adapter = DiscordAdapter("fake_token", bridge)
        custom_handler = MagicMock()
        adapter.set_message_handler(custom_handler)
        assert adapter._message_handler == custom_handler


# ===========================================================================
# Bridge E2E: Full message flow through brain
# ===========================================================================
class TestBridgeE2E:
    """E2E tests for the bridge processing pipeline against real Neo4j + Ollama."""

    @pytest.fixture
    def live_bridge(self):
        """Create a bridge with real brain systems."""
        import os
        from pathlib import Path
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

        from neuroform.memory.graph import KnowledgeGraph
        from neuroform.llm.ollama_client import OllamaClient
        from neuroform.memory.working_memory import WorkingMemory
        from neuroform.memory.amygdala import Amygdala

        kg = KnowledgeGraph()
        if not kg.driver:
            pytest.skip("Neo4j not available")

        kg.clear_all()
        model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

        try:
            import ollama as ollama_lib
            tags = ollama_lib.list()
            if not tags.get("models"):
                pytest.skip("Ollama not available")
        except Exception:
            pytest.skip("Ollama not available")

        wm = WorkingMemory(capacity=7)
        amygdala = Amygdala()
        client = OllamaClient(kg, model=model, working_memory=wm, amygdala=amygdala)

        bridge = BridgeCore()
        bridge.initialize(kg, client, allowed_channels=["test_channel"])

        yield bridge

        kg.clear_all()
        kg.close()

    def test_bridge_processes_message_through_brain(self, live_bridge):
        """Full e2e: MessageEvent → BridgeCore → NeuroForm Brain → ResponseEvent."""
        event = MessageEvent(
            user_id="e2e_user",
            channel_id="test_channel",
            content="My name is Alice and I work as a data scientist.",
            platform="test"
        )

        response = live_bridge.process_message(event)

        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.channel_id == "test_channel"
        assert response.platform == "test"
        print(f"\n[BRIDGE E2E RESPONSE]: {response.content}")

    def test_bridge_blocks_wrong_channel(self, live_bridge):
        """Verify channel filtering works in live bridge."""
        event = MessageEvent(
            user_id="e2e_user",
            channel_id="wrong_channel",
            content="This should be ignored",
            platform="test"
        )

        response = live_bridge.process_message(event)
        assert response is None

    def test_bridge_discord_adapter_integration(self, live_bridge):
        """Verify Discord adapter correctly registers with bridge."""
        adapter = DiscordAdapter("fake_token_for_test", live_bridge)
        live_bridge.register_adapter(adapter)
        assert "discord" in live_bridge.adapters
        assert adapter.platform_name == "discord"

