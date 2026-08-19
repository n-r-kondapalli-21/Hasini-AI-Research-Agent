from collections import deque


class ConversationMemory:
    """
    Manages short-term conversation memory for the voice agent.
    """

    def __init__(self, history_limit: int = 10, enabled: bool = True):
        self.enabled = enabled
        self.history_limit = history_limit

        self.conversation_history = deque(
            maxlen=self.history_limit
        )

    def add_user_message(self, user_query: str):
        """Add a user message to the conversation history."""

        if not self.enabled:
            return

        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })

    def add_assistant_message(self, assistant_response: str):
        """Add an assistant response to the conversation history."""

        if not self.enabled:
            return

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })

    def get_history(self):
        """Return conversation history in LangChain message format."""

        if not self.enabled:
            return []

        return list(self.conversation_history)

    def clear(self):
        """Clear the current conversation."""

        self.conversation_history.clear()

    def is_empty(self):
        """Check whether conversation history is empty."""

        return len(self.conversation_history) == 0