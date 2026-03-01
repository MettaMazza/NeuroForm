"""
Discord Adapter — NeuroForm Bridge Implementation
===================================================

Implements PlatformAdapter for Discord using discord.py.
Reads DISCORD_TOKEN and DISCORD_CHANNEL_ID from environment variables.
"""
import os
import logging
import asyncio
from typing import Optional

import discord
from discord import Intents

from neuroform.bridge.bridge import (
    PlatformAdapter,
    BridgeCore,
    MessageEvent,
    ResponseEvent,
)

logger = logging.getLogger(__name__)

# Discord has a 2000 character message limit
DISCORD_MAX_MESSAGE_LENGTH = 2000


class DiscordAdapter(PlatformAdapter):
    """
    Discord platform adapter for the NeuroForm bridge.

    Connects to Discord via discord.py, listens for messages in
    configured channels, routes them through BridgeCore, and
    sends responses back.
    """

    def __init__(self, token: str, bridge: BridgeCore):
        self._token = token
        self._bridge = bridge
        self._message_handler = bridge.process_message

        # Set up intents
        intents = Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._setup_events()

    @property
    def platform_name(self) -> str:
        return "discord"

    @property
    def client(self) -> discord.Client:
        return self._client

    def _setup_events(self):
        """Wire Discord events to bridge processing."""

        @self._client.event
        async def on_ready():  # pragma: no cover
            logger.info(f"Discord bot connected as {self._client.user}")
            logger.info(f"Bot ID: {self._client.user.id}")
            logger.info(f"Guilds: {[g.name for g in self._client.guilds]}")

        @self._client.event
        async def on_message(message: discord.Message):  # pragma: no cover
            # Never respond to ourselves
            if message.author == self._client.user:
                return

            # Never respond to other bots
            if message.author.bot:
                return

            # Build platform-neutral event
            event = MessageEvent(
                user_id=str(message.author.id),
                channel_id=str(message.channel.id),
                content=message.content,
                platform="discord",
                metadata={
                    "author_name": str(message.author),
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "message_id": str(message.id),
                },
            )

            # Process through bridge (runs sync brain in executor to not block)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._message_handler, event
            )

            if response:
                await self._send_discord_response(message.channel, response)

    async def _send_discord_response(self, channel, response: ResponseEvent):
        """Send a response, handling Discord's 2000-char limit."""
        content = response.content
        if not content:
            return

        # Chunk long messages
        chunks = self._chunk_message(content)
        for chunk in chunks:
            try:
                await channel.send(chunk)
            except discord.HTTPException as e:
                logger.error(f"Discord send error: {e}")

    @staticmethod
    def _chunk_message(text: str, limit: int = DISCORD_MAX_MESSAGE_LENGTH) -> list:
        """Split a message into chunks that fit Discord's character limit."""
        if len(text) <= limit:
            return [text]

        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break

            # Try to split at a newline
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                # Fall back to splitting at space
                split_at = text.rfind(" ", 0, limit)
            if split_at == -1:
                # Hard split
                split_at = limit

            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()

        return chunks

    async def start(self):
        """Start the Discord bot."""
        await self._client.start(self._token)

    async def stop(self):
        """Stop the Discord bot."""
        await self._client.close()

    async def send_response(self, response: ResponseEvent):
        """Send a response to a specific channel by ID."""
        channel = self._client.get_channel(int(response.channel_id))
        if channel:
            await self._send_discord_response(channel, response)


def run_bot():  # pragma: no cover
    """
    Entry point: load .env, initialize all brain systems,
    wire the bridge, and start the Discord bot.
    """
    from pathlib import Path

    # Load .env
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    # Validate
    token = os.environ.get("DISCORD_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")

    if not token:
        print("ERROR: DISCORD_TOKEN not set in .env")
        return

    # Initialize brain
    from neuroform.memory.graph import KnowledgeGraph
    from neuroform.brain.orchestrator import BrainOrchestrator
    from neuroform.brain.background import BackgroundScheduler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("====== NeuroForm Discord Bot ======")
    kg = KnowledgeGraph()
    if not kg.driver:
        print("ERROR: Neo4j not connected. Set NEO4J_URI/USER/PASSWORD in .env")
        return

    model = os.environ.get("OLLAMA_MODEL", "llama3")

    # Create orchestrator with all 9 brain systems
    orchestrator = BrainOrchestrator(kg, model=model)

    # Set up bridge with orchestrator
    allowed = [channel_id] if channel_id else []
    bridge = BridgeCore()
    bridge.initialize(
        kg, orchestrator.client,
        allowed_channels=allowed,
        orchestrator=orchestrator,
    )

    # Create and register Discord adapter
    adapter = DiscordAdapter(token, bridge)
    bridge.register_adapter(adapter)

    # Start background scheduler
    scheduler = BackgroundScheduler(kg, model=model,
                                    circadian=orchestrator.circadian,
                                    neuroplasticity=orchestrator.neuroplasticity)
    scheduler.start()

    print(f"Listening on channel: {channel_id or 'ALL'}")
    print(f"Model: {model}")
    print("All 9 brain systems active.")
    print("Background scheduler: dream consolidation + DMN + decay")
    print("Starting bot...")

    # Run
    asyncio.run(adapter.start())


if __name__ == "__main__":  # pragma: no cover
    run_bot()
