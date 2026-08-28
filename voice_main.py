"""
Hasini voice interface entrypoint.

Compatible with the current main.py interface:

    agent, registry = await create_research_agent()

The voice layer uses the same research agent, Agent_stream, and
ConversationMemory as the terminal main.py while keeping the existing
wake-word, STT, VAD, TTS, confirmation, and barge-in pipeline.
"""

from __future__ import annotations

import time
import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack

from agent_runtime import create_research_agent, Agent_stream
from services.conversation_memory import ConversationMemory
from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT

from voice.audio_capture import AudioCapture
from voice.audio_broadcaster import AudioBroadcaster
from voice.audio_queue import OrderedAudioQueue

from voice.providers.factory import (
    create_stt_provider,
    create_tts_provider,
    create_vad_provider,
    create_wakeword_provider,
)

from voice.confirmation_listener import VoiceConfirmationListener
from voice.confirmation import ConfirmationManager

from voice.command_listener import CommandListener
from voice.agent_controller import AgentController
from voice.barge_in_detector import BargeInDetector
from voice.state_machine import VoiceStateMachine
from voice.wake_listener import WakeWordListener


logger = logging.getLogger("hasini.voice")

def _configure_logging() -> None:
    """Configure application-wide logging before startup work begins."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Keep noisy third-party loggers from overwhelming the application logs.
    for logger_name in (
        "httpx",
        "httpcore",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger.info("Logging initialized.")


class HasiniStartupError(RuntimeError):
    """Raised when the voice pipeline cannot be started cleanly."""


def _build_providers() -> dict:
    """Create the voice providers used by the pipeline."""
    try:
        return {
            "wakeword": create_wakeword_provider(),
            "vad": create_vad_provider(),
            "barge_vad": create_vad_provider(),
            "stt": create_stt_provider(),
            "tts": create_tts_provider(),
        }
    except Exception as exc:
        raise HasiniStartupError(
            f"Failed to initialize voice providers: {exc}"
        ) from exc


async def run() -> int:
    """Start and run the Hasini voice interface."""

    startup_start = time.perf_counter()
    _configure_logging()

    shutdown_event = asyncio.Event()

    def _request_shutdown(sig_name: str) -> None:
        logger.info("Received %s, shutting down...", sig_name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                _request_shutdown,
                sig.name,
            )
        except (NotImplementedError, RuntimeError):
            # Windows/event-loop implementations may not support this.
            logger.debug(
                "Signal handler for %s is not supported by this event loop.",
                sig,
            )

    async with AsyncExitStack() as stack:
        try:
            # ==========================================================
            # Core audio components
            # ==========================================================
            capture = AudioCapture()
            broadcaster = AudioBroadcaster(capture)
            state_machine = VoiceStateMachine()

            # ==========================================================
            # Voice providers
            # ==========================================================
            providers = _build_providers()

            wakeword = providers["wakeword"]
            vad = providers["vad"]
            barge_vad = providers["barge_vad"]
            stt = providers["stt"]
            tts = providers["tts"]

            # ==========================================================
            # Confirmation system
            # ==========================================================
            confirmation_listener = VoiceConfirmationListener(vad=vad,stt=stt,tts=tts,)

            confirmation_manager = ConfirmationManager(
                listen_callback=confirmation_listener.listen,
                speak_callback=confirmation_listener.speak,
                timeout_seconds=15,
            )

            # ==========================================================
            # Audio playback queue
            # ==========================================================
            audio_queue = OrderedAudioQueue()

            # ==========================================================
            # Research agent
            #
            # IMPORTANT:
            # Current main.py exposes:
            #
            #     agent, registry = await create_research_agent()
            #
            # Do not pass confirmation_manager/mode here because those
            # arguments belong to the previous create_research_agent API.
            # ==========================================================
            logger.info("Creating research agent...")

            try:
                agent, registry = await create_research_agent(
                    confirmation_manager=confirmation_manager,
                    mode="voice",)
                logger.info("Creating research agent for voice mode...")

            except Exception as exc:
                raise HasiniStartupError(
                    f"Failed to create research agent: {exc}"
                ) from exc

            logger.debug("Research agent created successfully for voice interface.")

            # Keep registry available for future voice-side tool
            # inspection/control without changing the current API.
            _ = registry

            # ==========================================================
            # Conversation memory
            # ==========================================================
            memory = ConversationMemory(
                history_limit=MEMORY_HISTORY_LIMIT,
                enabled=MEMORY_ENABLED,
            )

            # ==========================================================
            # Agent controller
            # ==========================================================
            agent_controller = AgentController(
                agent=agent,
                memory=memory,
                agent_stream=Agent_stream,
                tts=tts,
                state_machine=state_machine,
                audio_queue=audio_queue,
            )

            # ==========================================================
            # Wake-word listener
            # ==========================================================
            wake_listener = WakeWordListener(
                broadcaster=broadcaster,
                wakeword=wakeword,
                state_machine=state_machine,
            )

            # ==========================================================
            # Command listener
            # ==========================================================
            command_listener = CommandListener(
                broadcaster=broadcaster,
                vad=vad,
                stt=stt,
                agent_controller=agent_controller,
                state_machine=state_machine,
            )

            # ==========================================================
            # Barge-in detector
            # ==========================================================
            barge_in_detector = BargeInDetector(
                broadcaster=broadcaster,
                vad=barge_vad,
                state_machine=state_machine,
                agent_controller=agent_controller,
            )

            # ==========================================================
            # Start components
            # ==========================================================
            async def start_and_register(
                component,
                name: str,
            ) -> None:
                try:
                    await component.start()
                except Exception as exc:
                    raise HasiniStartupError(
                        f"Failed to start {name}: {exc}"
                    ) from exc

                stack.push_async_callback(
                    _safe_stop,
                    component,
                    name,
                )

            await start_and_register(
                capture,
                "audio capture",
            )

            await start_and_register(
                broadcaster,
                "audio broadcaster",
            )

            await start_and_register(
                agent_controller,
                "agent controller",
            )

            await start_and_register(
                wake_listener,
                "wake-word listener",
            )

            await start_and_register(
                command_listener,
                "command listener",
            )

            await start_and_register(
                barge_in_detector,
                "barge-in detector",
            )
            startup_duration = time.perf_counter() - startup_start

            logger.info("Voice system initialized in %.2f seconds.",startup_duration,)

            _print_banner()

            # ==========================================================
            # Keep the process alive until Ctrl+C / SIGTERM.
            # ==========================================================
            await shutdown_event.wait()

        except HasiniStartupError as exc:
            logger.error("Startup failed: %s", exc)
            return 1

        except Exception:
            logger.exception(
                "Unexpected error during voice interface execution."
            )
            return 1

        logger.info("Shutting down...")

    logger.info("Hasini stopped.")
    return 0


async def _safe_stop(component, name: str) -> None:
    """Stop a component without preventing other components from stopping."""
    try:
        await component.stop()
    except Exception:
        logger.warning(
            "Error while stopping %s",
            name,
            exc_info=True,
        )


def _print_banner() -> None:
    logger.info("Hasini Voice Research Agent started.")
    logger.info("Say 'Hey Jarvis' to start a conversation.")
    logger.info("Say 'exit', 'quit', or 'goodbye' to end the session.")
    logger.info("Press Ctrl+C to stop.")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        exit_code = 130

    sys.exit(exit_code)
