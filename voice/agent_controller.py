import asyncio
import numpy as np

from .config import VOICE_TEST_MODE
from .audio_queue import OrderedAudioQueue
from .segmenter import ResponseSegmenter
from .test.test_response import TEST_RESPONSE
from .confirmation import ConfirmationRejected, ConfirmationTimeout

from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)


def _synthesize_blocking(
    tts,
    text,
):
    """
    Run TTS synthesis outside the asyncio event loop.
    """

    audio_parts = []

    sample_rate = None

    for audio, rate in tts.synthesize_stream(text):

        if audio is None or len(audio) == 0:
            continue

        audio_parts.append(audio)

        sample_rate = rate

    if not audio_parts:
        return None, None

    if len(audio_parts) == 1:
        final_audio = audio_parts[0]

    else:
        final_audio = np.concatenate(
            audio_parts
        )

    return final_audio, sample_rate


async def synthesize_chunk(
    tts,
    audio_queue,
    index,
    text,
    generation,
):
    """
    Synthesize one response segment.

    A chunk is accepted only if it belongs to the
    currently active response generation.
    """

    try:

        audio, sample_rate = await asyncio.to_thread(
            _synthesize_blocking,
            tts,
            text,
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Ignore TTS belonging to an old response.
        # --------------------------------------------------

        if not audio_queue.is_generation_active(
            generation
        ):

            return

        # --------------------------------------------------
        # Empty/failed synthesis still occupies the
        # sequence slot.
        # --------------------------------------------------

        if audio is None:

            audio = np.array(
                [],
                dtype=np.float32,
            )

            sample_rate = 0

        await audio_queue.put(
            index,
            audio,
            sample_rate,
            generation,
        )

    except asyncio.CancelledError:

        raise

    except Exception as e:

        print(
            f"\n❌ TTS chunk "
            f"{index + 1} error: {e}"
        )

        if audio_queue.is_generation_active(
            generation
        ):

            await audio_queue.put(
                index,
                np.array(
                    [],
                    dtype=np.float32,
                ),
                0,
                generation,
            )


class AgentController:
    """
    Owns the complete response lifecycle.

    Agent
        ↓
    ResponseSegmenter
        ↓
    TTS
        ↓
    OrderedAudioQueue
        ↓
    Speaker

    Every response gets a unique generation ID.

    When a response is cancelled, all chunks from that
    generation become invalid immediately.
    """

    def __init__(
        self,
        agent=None,
        memory=None,
        agent_stream=None,
        tts=None,
        state_machine: VoiceStateMachine = None,
        audio_queue: OrderedAudioQueue = None,
        use_test_response: bool = None,
        test_response: str = TEST_RESPONSE,
    ):

        self.agent = agent

        self.memory = memory

        self.agent_stream = agent_stream

        self.tts = tts

        self.state_machine = state_machine

        self.audio_queue = audio_queue

        self.use_test_response = use_test_response

        self.test_response = test_response

        self.running = False

        self.current_task = None

        self.tts_tasks = []

        # --------------------------------------------------
        # Response generation counter.
        # --------------------------------------------------

        self.generation = 0

    async def _get_test_stream(self):
        """
        Stream predefined test response token by token to simulate LLM streaming.
        """
        response_text = (
            self.test_response
            if self.test_response is not None
            else TEST_RESPONSE
        )

        words = response_text.split(" ")

        for i, word in enumerate(words):

            token = word + (
                " " if i < len(words) - 1 else ""
            )

            yield token

            await asyncio.sleep(0.02)

    # ============================================================
    # START
    # ============================================================

    async def start(self):

        if self.running:
            return

        self.running = True

        print(
            "🧠 Agent controller started."
        )

    # ============================================================
    # PROCESS
    # ============================================================

    async def process(
        self,
        text: str,
    ):

        if not text or not text.strip():
            return None

        if (
            self.state_machine.state
            != VoiceState.THINKING
        ):

            print(
                "⚠️ Agent request ignored. "
                f"Current state: "
                f"{self.state_machine.state.name}"
            )

            return None

        self.current_task = (
            asyncio.current_task()
        )

        self.tts_tasks.clear()

        # --------------------------------------------------
        # Create NEW response generation.
        # --------------------------------------------------

        self.generation += 1

        generation = self.generation

        segmenter = ResponseSegmenter()

        response_parts = []

        chunk_index = 0

        try:

            # ==================================================
            # RESET PLAYBACK QUEUE
            # ==================================================

            await self.audio_queue.reset(
                generation
            )

            # ==================================================
            # START PLAYBACK
            # ==================================================

            await self.audio_queue.start_playback()

            print(
                f"🧠 Thinking about: {text}"
            )

            # ==================================================
            # STREAM AGENT RESPONSE
            # ==================================================

            is_test_mode = (
                self.use_test_response
                if self.use_test_response is not None
                else VOICE_TEST_MODE
            )

            token_stream = (
                self._get_test_stream()
                if is_test_mode
                else self.agent_stream(
                    text,
                    self.agent,
                    self.memory,
                )
            )

            async for token in token_stream:

                # --------------------------------------------------
                # Ignore old generation.
                # --------------------------------------------------

                if not self.audio_queue.is_generation_active(
                    generation
                ):

                    return None

                if not token:
                    continue

                # --------------------------------------------------
                # Stop after cancellation.
                # --------------------------------------------------

                if (
                    self.state_machine.state
                    != VoiceState.THINKING
                    and self.state_machine.state
                    != VoiceState.RESPONDING
                    and self.state_machine.state
                    != VoiceState.SPEAKING
                ):

                    return None

                token = str(token)

                response_parts.append(token)

                chunk = segmenter.add(
                    token
                )

                if not chunk:
                    continue

                print(
                    f"\n🔊 Response chunk "
                    f"{chunk_index + 1}: "
                    f"{chunk}"
                )

                # --------------------------------------------------
                # THINKING → RESPONDING
                # --------------------------------------------------

                if (
                    self.state_machine.state
                    == VoiceState.THINKING
                ):

                    self.state_machine.handle_event(
                        VoiceEvent.RESPONSE_STARTED
                    )

                # --------------------------------------------------
                # Start TTS asynchronously.
                # --------------------------------------------------

                task = asyncio.create_task(
                    synthesize_chunk(
                        self.tts,
                        self.audio_queue,
                        chunk_index,
                        chunk,
                        generation,
                    )
                )

                self.tts_tasks.append(
                    task
                )

                chunk_index += 1

            # ==================================================
            # FLUSH FINAL SEGMENT
            # ==================================================

            final_chunk = segmenter.flush()

            if (
                final_chunk
                and self.audio_queue.is_generation_active(
                    generation
                )
            ):

                print(
                    f"\n🔊 Final response chunk: "
                    f"{final_chunk}"
                )

                if (
                    self.state_machine.state
                    == VoiceState.THINKING
                ):

                    self.state_machine.handle_event(
                        VoiceEvent.RESPONSE_STARTED
                    )

                task = asyncio.create_task(
                    synthesize_chunk(
                        self.tts,
                        self.audio_queue,
                        chunk_index,
                        final_chunk,
                        generation,
                    )
                )

                self.tts_tasks.append(
                    task
                )

                chunk_index += 1

            # ==================================================
            # FULL TEXT
            # ==================================================

            response = "".join(
                response_parts
            ).strip()

            print(
                f"\n🤖 Full response: "
                f"{response}"
            )

            # ==================================================
            # WAIT FOR TTS GENERATION
            # ==================================================

            if self.tts_tasks:

                await asyncio.gather(
                    *self.tts_tasks,
                    return_exceptions=True,
                )

            # ==================================================
            # CHECK GENERATION
            # ==================================================

            if not self.audio_queue.is_generation_active(
                generation
            ):

                return None

            # ==================================================
            # RESPONDING → SPEAKING
            # ==================================================

            if (
                self.state_machine.state
                == VoiceState.RESPONDING
            ):

                self.state_machine.handle_event(
                    VoiceEvent.TTS_STARTED
                )

            # ==================================================
            # WAIT FOR PLAYBACK
            # ==================================================

            await self.audio_queue.wait_until_finished(
                generation
            )

            # ==================================================
            # CHECK AGAIN
            # ==================================================

            if not self.audio_queue.is_generation_active(
                generation
            ):

                return None

            # ==================================================
            # CLOSE CURRENT PLAYBACK
            # ==================================================

            await self.audio_queue.close(
                generation
            )

            # ==================================================
            # SPEAKING → IDLE
            # ==================================================

            if (
                self.state_machine.state
                == VoiceState.SPEAKING
            ):

                self.state_machine.handle_event(
                    VoiceEvent.TTS_COMPLETE
                )

            return response

        except asyncio.CancelledError:

            raise

        except (ConfirmationRejected, ConfirmationTimeout) as e:

            print(
                f"\n🛑 Action cancelled ({e.__class__.__name__}). "
                "Stopping agent execution."
            )

            self.state_machine.reset()

            return None

        except Exception as e:

            print(
                f"\n❌ Agent response error: {e}"
            )

            self.state_machine.reset()

            return None

        finally:

            # --------------------------------------------------
            # Only clear task reference if this is still the
            # active task.
            # --------------------------------------------------

            current = asyncio.current_task()

            if self.current_task is current:

                self.current_task = None

    # ============================================================
    # CANCEL RESPONSE
    # ============================================================

    async def cancel_response(self):

        print(
            "\n🛑 Cancelling response..."
        )

        # ==================================================
        # INVALIDATE CURRENT GENERATION FIRST
        # ==================================================
        #
        # Capture current task to cancel BEFORE any await calls
        # to prevent cancelling a new command task that starts
        # during audio queue cleanup.
        # ==================================================

        task_to_cancel = self.current_task

        self.current_task = None

        cancelled_generation = self.generation

        self.generation += 1

        if (
            task_to_cancel is not None
            and not task_to_cancel.done()
        ):

            task_to_cancel.cancel()

        # --------------------------------------------------
        # Cancel Piper tasks.
        # --------------------------------------------------

        for task in list(
            self.tts_tasks
        ):

            if not task.done():

                task.cancel()

        self.tts_tasks.clear()

        # ==================================================
        # IMMEDIATE STATE TRANSITIONS
        # ==================================================
        # Perform transitions immediately so CommandListener
        # can transition to LISTENING without waiting for audio hardware.

        if self.state_machine.can_handle(
            VoiceEvent.USER_SPEECH
        ):
            self.state_machine.handle_event(
                VoiceEvent.USER_SPEECH
            )

        if self.state_machine.can_handle(
            VoiceEvent.TTS_CANCELLED
        ):
            self.state_machine.handle_event(
                VoiceEvent.TTS_CANCELLED
            )

        if self.state_machine.can_handle(
            VoiceEvent.LISTENING_STARTED
        ):
            self.state_machine.handle_event(
                VoiceEvent.LISTENING_STARTED
            )

        # ==================================================
        # CANCEL AUDIO PLAYBACK
        # ==================================================

        await self.audio_queue.cancel(
            cancelled_generation
        )

        print(
            "🎤 Response cancelled. "
            "Listening for new command."
        )

    # ============================================================
    # STOP
    # ============================================================

    async def stop(self):

        if not self.running:
            return

        self.running = False

        await self.cancel_response()

        self.current_task = None

        self.tts_tasks.clear()

        print(
            "🧠 Agent controller stopped."
        )