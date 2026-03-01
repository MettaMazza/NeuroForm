"""
NeuroForm Bridge — Platform-Agnostic Connection Layer
======================================================

Inspired by clawbot architecture: a neutral bridge that decouples the
NeuroForm brain from any specific messaging platform. Adapters plug in
for Discord, Telegram, Slack, or any other platform.

The bridge core handles:
1. Message normalization (platform events → MessageEvent)
2. Brain processing (MessageEvent → NeuroForm → ResponseEvent)
3. Response routing (ResponseEvent → platform adapter → user)
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MessageEvent:
    """Platform-neutral representation of an incoming message."""
    user_id: str
    channel_id: str
    content: str
    platform: str  # "discord", "telegram", "slack", "cli", etc.
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseEvent:
    """Platform-neutral representation of an outgoing response."""
    content: str
    channel_id: str
    platform: str
    reply_to_user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlatformAdapter(ABC):
    """
    Abstract base for platform adapters.
    Each platform (Discord, Telegram, Slack, CLI) implements this.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier string."""
        ...

    @abstractmethod
    async def start(self):
        """Start the platform connection (login, connect websocket, etc.)."""
        ...

    @abstractmethod
    async def stop(self):
        """Gracefully stop the platform connection."""
        ...

    @abstractmethod
    async def send_response(self, response: ResponseEvent):
        """Send a response back to the platform."""
        ...

    def set_message_handler(self, handler: Callable):
        """Set the callback for incoming messages."""
        self._message_handler = handler


class BridgeCore:
    """
    The central brain-to-platform bridge.

    Receives MessageEvents from any adapter, passes them through the
    NeuroForm brain, and returns ResponseEvents.
    """

    def __init__(self):
        self._kg = None
        self._client = None
        self._adapters: Dict[str, PlatformAdapter] = {}
        self._allowed_channels: List[str] = []
        self._initialized = False

    def initialize(self, kg, client, allowed_channels: Optional[List[str]] = None):
        """
        Wire the bridge to the NeuroForm brain.

        Args:
            kg: KnowledgeGraph instance
            client: OllamaClient instance (with WorkingMemory + Amygdala)
            allowed_channels: Optional list of channel IDs to filter on
        """
        self._kg = kg
        self._client = client
        self._allowed_channels = allowed_channels or []
        self._initialized = True
        logger.info(f"Bridge initialized. Allowed channels: {self._allowed_channels}")

    def register_adapter(self, adapter: PlatformAdapter):
        """Register a platform adapter."""
        self._adapters[adapter.platform_name] = adapter
        adapter.set_message_handler(self.process_message)
        logger.info(f"Adapter registered: {adapter.platform_name}")

    def is_channel_allowed(self, channel_id: str) -> bool:
        """Check if a channel is in the allowed list (empty = allow all)."""
        if not self._allowed_channels:
            return True
        return str(channel_id) in [str(c) for c in self._allowed_channels]

    def process_message(self, event: MessageEvent) -> Optional[ResponseEvent]:
        """
        Process an incoming message through the NeuroForm brain.

        Returns a ResponseEvent or None if the message should be ignored.
        """
        if not self._initialized or not self._client:
            logger.warning("Bridge not initialized — ignoring message")
            return None

        # Channel filter
        if not self.is_channel_allowed(event.channel_id):
            logger.debug(f"Channel {event.channel_id} not allowed — ignoring")
            return None

        # Process through the brain
        try:
            reply = self._client.chat_with_memory(event.user_id, event.content)

            return ResponseEvent(
                content=reply,
                channel_id=event.channel_id,
                platform=event.platform,
                reply_to_user=event.user_id,
            )
        except Exception as e:
            logger.error(f"Bridge processing error: {e}")
            return ResponseEvent(
                content="I encountered an error processing your message.",
                channel_id=event.channel_id,
                platform=event.platform,
                reply_to_user=event.user_id,
            )

    @property
    def adapters(self) -> Dict[str, PlatformAdapter]:
        return dict(self._adapters)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
