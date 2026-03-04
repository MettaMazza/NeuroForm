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
from discord import Intents, app_commands

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

    def __init__(self, token: str, bridge: BridgeCore,
                 orchestrator=None, kg=None, scheduler=None):
        self._token = token
        self._bridge = bridge
        self._message_handler = bridge.process_message
        self._orchestrator = orchestrator
        self._kg = kg
        self._scheduler = scheduler
        self.agency_daemon = None

        # Set up intents
        intents = Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._tree = app_commands.CommandTree(self._client)
        self._setup_events()
        self._setup_slash_commands()

    @property
    def platform_name(self) -> str:
        return "discord"

    @property
    def client(self) -> discord.Client:
        return self._client

    def _setup_slash_commands(self):
        """Register Discord slash commands."""

        @self._tree.command(name="clear", description="[Admin] Wipe all memory data and shut down for a fresh start")
        async def clear_command(interaction: discord.Interaction):  # pragma: no cover
            # Check admin permission
            owner_env = os.environ.get("DISCORD_OWNER_ID", "")
            owner_ids = [uid.strip() for uid in owner_env.split(",")] if owner_env else []
            user_id = str(interaction.user.id)

            if user_id not in owner_ids:
                await interaction.response.send_message(
                    "⛔ Permission denied. Only admins can use this command.",
                    ephemeral=True,
                )
                return

            await interaction.response.send_message(
                "🧹 **Clearing all memory data...**", ephemeral=False,
            )

            cleared = []
            import shutil
            from pathlib import Path

            # ── Step 1: Clear in-memory state FIRST (before disk) ──
            # This prevents the buffer from re-persisting on shutdown
            if self._orchestrator:
                orch = self._orchestrator
                # Context Stream — wipe in-memory buffer + compaction summaries
                if hasattr(orch, 'context_stream') and orch.context_stream:
                    orch.context_stream.clear()
                    cleared.append("ContextStream: in-memory buffer cleared")

                # Tape Machine — reset in-memory state
                if hasattr(orch, 'tape_machine') and orch.tape_machine:
                    try:
                        orch.tape_machine.buffer = []
                        orch.tape_machine.head = [0, 0, 0]
                        cleared.append("TapeMachine: in-memory state reset")
                    except Exception as e:
                        cleared.append(f"TapeMachine: error — {e}")

                # Lessons — wipe in-memory
                if hasattr(orch, 'lesson_manager') and orch.lesson_manager:
                    try:
                        orch.lesson_manager.lessons = []
                        cleared.append("Lessons: in-memory cleared")
                    except Exception as e:
                        cleared.append(f"Lessons: error — {e}")

            # ── Step 2: Clear Neo4j Knowledge Graph ──
            try:
                if self._kg and self._kg.driver:
                    deleted = self._kg.clear_all()
                    cleared.append(f"Neo4j: {deleted} nodes deleted")
            except Exception as e:
                cleared.append(f"Neo4j: error — {e}")

            # ── Step 3: Delete persistent files on disk ──
            memory_core = Path(os.getcwd()) / "memory" / "core"
            for fname in ["working_memory.jsonl", "archive_working_memory.jsonl", "lessons.json"]:
                fpath = memory_core / fname
                if fpath.exists():
                    fpath.unlink()
                    cleared.append(f"Deleted {fname}")

            # T5: TapeMachine persistent directory
            tape_dir = Path(os.getcwd()) / "memory" / "tape"
            if tape_dir.exists():
                shutil.rmtree(tape_dir)
                cleared.append("Deleted tape/ directory")

            summary = "\n".join(f"  ✅ {item}" for item in cleared)
            await interaction.followup.send(
                f"🧠 **All memory cleared:**\n{summary}\n\n"
                f"Shutting down for a fresh start... 👋",
            )

            logger.info(f"ADMIN CLEAR executed by {interaction.user} ({user_id})")

            # Close connections and shut down
            if self._scheduler:
                self._scheduler.stop()
            if self._kg:
                self._kg.close()
            await self._client.close()

    def _setup_events(self):
        """Wire Discord events to bridge processing."""

        @self._client.event
        async def on_ready():  # pragma: no cover
            # Sync slash commands to each guild (instant, unlike global sync)
            try:
                for guild in self._client.guilds:
                    self._tree.copy_global_to(guild=guild)
                    synced = await self._tree.sync(guild=guild)
                    logger.info(f"Synced {len(synced)} slash command(s) to guild '{guild.name}'")
            except Exception as e:
                logger.error(f"Failed to sync slash commands: {e}")

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

            # ── STRICT CHANNEL GUARD ──
            # Only listen in the designated chat channel. All other channels are ignored.
            allowed_chat_channel = os.environ.get("DISCORD_CHANNEL_ID")
            if allowed_chat_channel and str(message.channel.id) != str(allowed_chat_channel):
                return
                
            # IMPERATIVE: Yield autonomy immediately on user input
            if self.agency_daemon:
                self.agency_daemon.signal_user_activity()

            # Build platform-neutral event
            # Ground rule: resolve scope from channel type
            is_dm = isinstance(message.channel, discord.DMChannel)
            scope = "PRIVATE" if is_dm else "PUBLIC"

            event = MessageEvent(
                user_id=str(message.author.id),
                channel_id=str(message.channel.id),
                content=message.content,
                platform="discord",
                metadata={
                    "author_name": str(message.author),
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "message_id": str(message.id),
                    "scope": scope,
                },
            )

            # Process through bridge (runs sync brain in executor to not block)
            loop = asyncio.get_running_loop()
            async with message.channel.typing():
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
        try:
            channel_id = int(response.channel_id)
            channel = self._client.get_channel(channel_id)
            
            if not channel:
                # Fallback to API call if not in local cache (useful for secondary channels)
                channel = await self._client.fetch_channel(channel_id)
                
            if channel:
                await self._send_discord_response(channel, response)
            else:
                logger.error(f"Could not find or fetch channel with ID {channel_id}")
        except Exception as e:
            logger.error(f"Error sending Discord response to channel {response.channel_id}: {e}")


def run_bot():  # pragma: no cover
    """
    Entry point: load .env, initialize all brain systems,
    wire the bridge, and start the Discord bot.

    Kills any existing bot instances on startup to prevent double messages.
    """
    import os
    import sys
    import signal
    import subprocess
    from pathlib import Path

    # ─── Kill existing instances (prevent double messages) ─────
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*neuroform"],
            capture_output=True, text=True
        )
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str.strip():
                pid = int(pid_str.strip())
                if pid != my_pid:
                    os.kill(pid, signal.SIGKILL)
                    print(f"Killed existing bot instance (PID {pid})")
    except Exception:
        pass  # pgrep not available or no processes found

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
    from neuroform.daemons.agency import AgencyDaemon

    # --- Production Logging Setup ---
    from logging.handlers import RotatingFileHandler
    
    # Session Error Handler (Refreshes on startup, captures WARNING+)
    session_error_handler = logging.FileHandler("session_error.log", mode='w')
    session_error_handler.setLevel(logging.WARNING)
    session_error_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler("nero_bot.log", maxBytes=10*1024*1024, backupCount=5),
            session_error_handler
        ]
    )

    # Silence noisy external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("neo4j.io").setLevel(logging.WARNING)
    logging.getLogger("neo4j.pool").setLevel(logging.WARNING)
    print("====== NeuroForm Discord Bot ======")
    kg = KnowledgeGraph()
    if not kg.driver:
        print("ERROR: Neo4j not connected. Set NEO4J_URI/USER/PASSWORD in .env")
        return

    model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

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

    # Create background scheduler (before adapter, so it can be passed for /clear shutdown)
    scheduler = BackgroundScheduler(kg, model=model,
                                    circadian=orchestrator.circadian,
                                    neuroplasticity=orchestrator.neuroplasticity)

    # Create and register Discord adapter
    adapter = DiscordAdapter(token, bridge,
                             orchestrator=orchestrator, kg=kg, scheduler=scheduler)
    bridge.register_adapter(adapter)

    # Start background scheduler
    scheduler.start()
    
    # ─── Start Continuous Autonomy Loop ───
    async def autonomy_output(msg: str):
        # We output to the primary channel configured in .env
        from neuroform.bridge.bridge import ResponseEvent
        auto_channel_id = os.environ.get("DISCORD_AUTONOMY_CHANNEL_ID") or channel_id
        event = ResponseEvent(
            content=msg,
            channel_id=auto_channel_id or "", 
            platform="discord"
        )
        await adapter.send_response(event)
        
    agency = AgencyDaemon(orchestrator, autonomy_output)
    # Store reference so the adapter can ping it
    adapter.agency_daemon = agency 
    
    # We must start agency daemon inside the asyncio loop
    async def start_discord_and_daemons():
        await agency.start()
        await adapter.start()

    autonomy_channel_id = os.environ.get("DISCORD_AUTONOMY_CHANNEL_ID")
    print(f"Chat channel:     {channel_id or 'ALL'}")
    print(f"Autonomy channel: {autonomy_channel_id or channel_id or 'SAME AS CHAT'}")
    print(f"Model: {model}")
    print("All 9 brain systems active.")
    print("Continuous Agency Daemon active.")
    print("Background scheduler: dream consolidation + DMN + decay")
    print("Starting bot...")

    # Run
    asyncio.run(start_discord_and_daemons())


if __name__ == "__main__":  # pragma: no cover
    run_bot()
