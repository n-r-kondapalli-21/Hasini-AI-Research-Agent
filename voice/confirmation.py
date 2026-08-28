import asyncio
import logging
import re


logger = logging.getLogger("hasini.voice.confirmation_manager")


MEDIUM_CONFIRMATIONS = {
    "yes",
    "yeah",
    "yep",
    "confirm",
    "confirmed",
    "do it",
    "go ahead",
    "proceed",
    "okay",
    "ok",
}

HIGH_CONFIRMATIONS = {
    "yes",
    "yes do it",
    "confirm",
    "confirm action",
    "do it",
    "proceed",
}

MEDIUM_REJECTIONS = {
    "no",
    "nope",
    "cancel",
    "cancel it",
    "don't",
    "do not",
    "stop",
}


def is_medium_rejection(text: str) -> bool:
    """Return True when the normalized text is a medium-risk rejection."""
    normalized = normalize_confirmation(text)
    return normalized in MEDIUM_REJECTIONS


def normalize_confirmation(text: str) -> str:
    """Normalize confirmation text for reliable exact matching."""
    if not text:
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def is_medium_confirmation(text: str) -> bool:
    """Return True when the normalized text is a valid medium confirmation."""
    normalized = normalize_confirmation(text)
    return normalized in MEDIUM_CONFIRMATIONS


def is_high_confirmation(text: str) -> bool:
    """Return True when the normalized text is a valid high confirmation."""
    normalized = normalize_confirmation(text)
    return normalized in HIGH_CONFIRMATIONS


class ConfirmationTimeout(Exception):
    """Raised when confirmation is not received within the allowed time."""


class ConfirmationRejected(Exception):
    """Raised when the user rejects or fails to provide valid confirmation."""


class ConfirmationManager:
    """
    Handles confirmation requests for permission-gated tools.

    Voice recording/STT and optional TTS callbacks are supplied by the
    voice layer so this module remains independent of microphone and
    speaker implementations.
    """

    def __init__(
        self,
        listen_callback,
        speak_callback=None,
        timeout_seconds: float = 15.0,
    ):
        if not callable(listen_callback):
            raise TypeError("listen_callback must be callable.")

        if speak_callback is not None and not callable(speak_callback):
            raise TypeError("speak_callback must be callable or None.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self.listen_callback = listen_callback
        self.speak_callback = speak_callback
        self.timeout_seconds = timeout_seconds

        logger.debug(
            "Confirmation manager initialized with timeout %.1fs.",
            timeout_seconds,
        )

    async def _speak(self, text: str) -> None:
        """Speak a confirmation message when a TTS callback is available."""
        if not self.speak_callback:
            logger.debug("No confirmation TTS callback configured.")
            return

        if not text or not text.strip():
            return

        try:
            await self.speak_callback(text)
        except asyncio.CancelledError:
            logger.debug("Confirmation speech cancelled.")
            raise
        except Exception:
            # TTS failure should not hide the confirmation flow itself.
            logger.exception("Confirmation spoken announcement failed.")

    async def wait_for_confirmation(
        self,
        tier: str,
        action_description: str,
    ) -> bool:
        """
        Wait for and validate confirmation for a permission-gated action.

        MEDIUM requires an exact recognized confirmation/rejection.
        HIGH additionally requires a confidence score of at least 0.80.
        """
        tier = str(tier).upper().strip()
        clean_action = str(action_description).rstrip(".")

        if tier not in {"MEDIUM", "HIGH"}:
            logger.error("Unsupported confirmation tier: %s", tier)
            raise ValueError(f"Unsupported confirmation tier: {tier}")

        if not clean_action:
            logger.error("Confirmation requested without an action description.")
            raise ValueError("action_description must not be empty.")

        if tier == "MEDIUM":
            logger.info("Medium-risk confirmation required: %s", action_description)

            prompt_text = (
                f"I need your confirmation to {clean_action}. "
                "Shall I proceed?"
            )
        else:
            logger.warning(
                "High-risk confirmation required: %s",
                action_description,
            )

            prompt_text = (
                f"Warning. High risk action: {clean_action}. "
                "Please clearly confirm to proceed."
            )

        # Speak prompt before starting the confirmation listener.
        await self._speak(prompt_text)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds

        while True:
            remaining = deadline - loop.time()

            if remaining <= 0:
                await self._handle_timeout()
                raise ConfirmationTimeout()

            try:
                result = await asyncio.wait_for(
                    self.listen_callback(),
                    timeout=remaining,
                )

            except asyncio.CancelledError:
                logger.debug("Confirmation request cancelled.")
                raise

            except asyncio.TimeoutError:
                await self._handle_timeout()
                raise ConfirmationTimeout()

            except Exception:
                logger.exception("Confirmation listener failed.")
                await self._reject("Cancelled.")
                raise ConfirmationRejected()

            if not result:
                logger.debug("Confirmation listener returned no result.")
                continue

            # ------------------------------------------
            # Extract transcription and confidence.
            # ------------------------------------------
            if isinstance(result, dict):
                text = result.get("text", "")
                confidence = result.get("confidence")
            else:
                text = str(result)
                confidence = None

            if not text or not str(text).strip():
                logger.debug("Empty confirmation transcription received.")
                continue

            text = normalize_confirmation(text)

            if not text:
                continue

            logger.info("Confirmation heard: %s", text)

            # ------------------------------------------
            # MEDIUM
            # ------------------------------------------
            if tier == "MEDIUM":
                if is_medium_confirmation(text):
                    logger.info("Medium-risk confirmation accepted.")
                    await self._speak("Okay.")
                    return True

                if is_medium_rejection(text):
                    logger.info("User declined the medium-risk action.")
                    await self._reject("Cancelled.")
                    raise ConfirmationRejected()

                logger.warning(
                    "Medium-risk confirmation not recognized. Action cancelled."
                )
                await self._reject("Cancelled.")
                raise ConfirmationRejected()

            # ------------------------------------------
            # HIGH
            # ------------------------------------------
            if confidence is None:
                logger.warning(
                    "High-risk confirmation has no confidence score. "
                    "Action cancelled."
                )
                await self._reject("Cancelled.")
                raise ConfirmationRejected()

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid high-risk confirmation confidence: %r. "
                    "Action cancelled.",
                    confidence,
                )
                await self._reject("Cancelled.")
                raise ConfirmationRejected()

            if confidence < 0.80:
                logger.warning(
                    "High-risk confirmation confidence too low: %.2f. "
                    "Action cancelled.",
                    confidence,
                )
                await self._reject("Cancelled.")
                raise ConfirmationRejected()

            if is_high_confirmation(text):
                logger.info(
                    "High-risk confirmation accepted with confidence %.2f.",
                    confidence,
                )
                await self._speak("Okay.")
                return True

            logger.warning(
                "Strong confirmation not recognized. Action cancelled."
            )
            await self._reject("Cancelled.")
            raise ConfirmationRejected()

    async def _handle_timeout(self) -> None:
        """Handle confirmation timeout consistently."""
        logger.warning(
            "Confirmation timed out after %.1f seconds. Action cancelled.",
            self.timeout_seconds,
        )
        await self._speak(
            "Confirmation timed out. Action cancelled."
        )

    async def _reject(self, message: str) -> None:
        """Speak a rejection/cancellation message safely."""
        await self._speak(message)
