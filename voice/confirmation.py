import asyncio
import re


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
    normalized = normalize_confirmation(text)

    return normalized in MEDIUM_REJECTIONS


def normalize_confirmation(text: str) -> str:
    text = text.lower().strip()

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
    normalized = normalize_confirmation(text)

    return (
        normalized in MEDIUM_CONFIRMATIONS
    )


def is_high_confirmation(text: str) -> bool:
    normalized = normalize_confirmation(text)

    return (
        normalized in HIGH_CONFIRMATIONS
    )


class ConfirmationTimeout(Exception):
    pass


class ConfirmationRejected(Exception):
    pass


class ConfirmationManager:
    """
    Handles confirmation requests for permission-gated tools.

    The actual voice recording/STT and optional TTS speaking functions
    are supplied by the voice layer so this module does not depend
    on a specific microphone or speaker implementation.
    """

    def __init__(
        self,
        listen_callback,
        speak_callback=None,
        timeout_seconds: float = 15.0,
    ):
        self.listen_callback = listen_callback
        self.speak_callback = speak_callback
        self.timeout_seconds = timeout_seconds

    async def _speak(self, text: str):
        if self.speak_callback:
            try:
                await self.speak_callback(text)
            except Exception as e:
                print(f"⚠️ Confirmation spoken announcement error: {e}")

    async def wait_for_confirmation(
        self,
        tier: str,
        action_description: str,
    ) -> bool:

        clean_action = str(action_description).rstrip(".")

        if tier == "MEDIUM":
            print("\n🔐 Confirmation required.")
            print(f"⚠️ {action_description}")
            print("Say yes, confirm, or do it.")

            prompt_text = f"I need your confirmation to {clean_action}. Shall I proceed?"
        else:
            print("\n🔐 Strong confirmation required.")
            print(f"⚠️ {action_description}")
            print("Please clearly confirm this action.")

            prompt_text = f"Warning. High risk action: {clean_action}. Please clearly confirm to proceed."

        # Speak prompt out loud before starting confirmation listener
        await self._speak(prompt_text)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds

        while True:
            remaining = deadline - loop.time()

            if remaining <= 0:
                print("\n⏱️ Confirmation timed out. Action cancelled.")
                await self._speak("Confirmation timed out. Action cancelled.")
                raise ConfirmationTimeout()

            try:
                result = await asyncio.wait_for(
                    self.listen_callback(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                print("\n⏱️ Confirmation timed out. Action cancelled.")
                await self._speak("Confirmation timed out. Action cancelled.")
                raise ConfirmationTimeout()

            if not result:
                continue

            # ------------------------------------------
            # Extract transcription BEFORE any checks
            # ------------------------------------------

            if isinstance(result, dict):
                text = result.get("text", "")
                confidence = result.get("confidence")
            else:
                text = str(result)
                confidence = None

            if not text or not text.strip():
                continue

            text = normalize_confirmation(text)

            if not text:
                continue

            print(f"🔐 Confirmation heard: {text}")

            # ------------------------------------------
            # MEDIUM
            # ------------------------------------------

            if tier == "MEDIUM":
                if is_medium_confirmation(text):
                    await self._speak("Okay.")
                    return True

                if is_medium_rejection(text):
                    print("🛑 User declined the action.")
                    await self._speak("Cancelled.")
                    raise ConfirmationRejected()

                print("❌ Confirmation not recognized. Action cancelled.")
                await self._speak("Cancelled.")
                raise ConfirmationRejected()

            # ------------------------------------------
            # HIGH
            # ------------------------------------------

            if confidence is None:
                print("❌ High-risk confirmation has no confidence score. Action cancelled.")
                await self._speak("Cancelled.")
                raise ConfirmationRejected()

            if confidence < 0.80:
                print(f"❌ Confirmation confidence too low ({confidence:.2f}). Action cancelled.")
                await self._speak("Cancelled.")
                raise ConfirmationRejected()

            if is_high_confirmation(text):
                await self._speak("Okay.")
                return True

            print("❌ Strong confirmation not recognized. Action cancelled.")
            await self._speak("Cancelled.")
            raise ConfirmationRejected()