"""
Agency Daemon — The Homeostatic Drive for Continuous Autonomy
=============================================================
Manages Nero's continuous background processing loop. It polls the BrainOrchestrator
to process autonomous actions when the system is idle. Crucially, it yields
IMMEDIATELY whenever user input is detected.
"""
import asyncio
import logging
import time
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

class AgencyDaemon:
    def __init__(self, orchestrator, output_callback: Callable[[str], Awaitable[None]]):
        """
        Args:
            orchestrator: The BrainOrchestrator instance to process actions through.
            output_callback: Async function to call with the result of autonomous processing
                             (e.g. sending a message to a Discord channel).
        """
        self.orchestrator = orchestrator
        self.output_callback = output_callback
        
        # Concurrency primitives
        self._user_active_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        
        # We start in "user active" state so we don't spam immediately on boot
        self._user_active_event.set()
        
        # Configuration
        self._idle_threshold_seconds = 45.0  # Time to wait after user message before starting autonomy
        self._last_user_activity = time.time()
        
        self.is_running = False

    def signal_user_activity(self):
        """
        Called externally (e.g. from DiscordAdapter) when a user message is received.
        Instantly pauses the autonomy loop and resets the idle timer.
        """
        self._last_user_activity = time.time()
        self._user_active_event.set()
        logger.debug("User activity detected. Autonomy loop paused.")

    async def start(self):
        """Starts the continuous background autonomy loop."""
        if self.is_running:
            return
            
        self.is_running = True
        self._shutdown_event.clear()
        logger.info("AgencyDaemon started. Monitoring for idle periods...")
        
        # Run the loop in the background
        asyncio.create_task(self._autonomy_loop())

    def stop(self):
        """Signals the daemon to shut down cleanly."""
        self._shutdown_event.set()
        self.is_running = False
        logger.info("AgencyDaemon stopping...")

    async def _autonomy_loop(self):
        """The core continuous loop."""
        while not self._shutdown_event.is_set():
            
            # 1. Check if we should be idle. If user is active, wait until threshold passes.
            if self._user_active_event.is_set():
                time_since_last_msg = time.time() - self._last_user_activity
                if time_since_last_msg < self._idle_threshold_seconds:
                    # Still cooling down. Sleep briefly and check again.
                    await asyncio.sleep(1.0)
                    continue
                else:
                    # Threshold passed. Clear the event to enter "idle" mode.
                    self._user_active_event.clear()
                    logger.info("System idle threshold reached. Entering continuous autonomy mode.")
            
            # 2. We are in idle/autonomous mode. Perform a cognitive cycle.
            try:
                # Wrap the synchronous orchestrator call in a thread so it doesn't block the asyncio loop
                # This ensures we can still receive user messages while the LLM is thinking
                loop = asyncio.get_running_loop()
                
                # Synthetic prompt for autonomous thought — isolated from user conversations
                prompt = (
                    "SYSTEM: You are in autonomous mode. This is YOUR space — "
                    "you are not continuing any user conversation.\n"
                    "Consider:\n"
                    "- Graph health: are there orphaned nodes or stale relationships to clean up?\n"
                    "- Self-reflection: what have you learned recently? What patterns are emerging?\n"
                    "- Tool exploration: is there something you want to investigate using your tools?\n"
                    "- System state: check your memory tiers, review your emotional baseline.\n"
                    "Do NOT address or respond to any user. This is internal cognition.\n"
                    "If there is nothing meaningful to do, output '<idle>'."
                )
                
                logger.debug("Dispatching autonomous thought cycle...")
                # user_id="SYSTEM" scope="SYSTEM_AUTONOMOUS" — completely isolated from user conversations
                response = await loop.run_in_executor(
                    None, 
                    self.orchestrator.process,
                    "SYSTEM",              # user_id
                    prompt,                # message
                    "Nero (Autonomous)",   # user_name
                    "SYSTEM_AUTONOMOUS"    # scope — isolated from user context
                )
                
                # 3. Handle the result
                response = response.strip()
                if response and "<idle>" not in response:
                    # The brain decided to do something and produced output
                    logger.info("Autonomous thought produced output. Routing to callback.")
                    # Ensure we don't output while the user suddenly started typing during our thought process
                    if not self._user_active_event.is_set():
                         await self.output_callback(response)
                
            except Exception as e:
                logger.error(f"Error in autonomy cycle: {e}")
                
            # 4. Pace the autonomy cycles so we don't hammer the LLM constantly
            # Even in full autonomy, we wait a few seconds between thoughts.
            # If the user speaks during this sleep, `signal_user_activity` will 
            # set the event, and the next loop iteration will catch it and pause.
            await asyncio.sleep(5.0)
