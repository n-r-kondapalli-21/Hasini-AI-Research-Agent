import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.tools import StructuredTool

from permission_gated_tool import create_permission_gated_tool, _format_action
from voice.confirmation import ConfirmationManager, ConfirmationRejected, ConfirmationTimeout
from main import Agent_stream
from services.conversation_memory import ConversationMemory


async def test_permission_gated_tool_reraises_rejection():
    """Verify that gated tool re-raises ConfirmationRejected instead of returning a string."""

    # Mock tool
    async def dummy_fn(path: str):
        return "written"

    raw_tool = StructuredTool.from_function(
        coroutine=dummy_fn,
        name="filesystem_write_file",
        description="Write a file",
    )

    # Mock confirmation manager that raises ConfirmationRejected
    conf_mgr = MagicMock(spec=ConfirmationManager)
    conf_mgr.wait_for_confirmation = AsyncMock(side_effect=ConfirmationRejected())

    gated_tool = create_permission_gated_tool(
        tool=raw_tool,
        server_id="filesystem",
        method_name="write_file",
        confirmation_manager=conf_mgr,
    )

    try:
        await gated_tool.ainvoke({"path": "test.txt"})
        assert False, "Expected ConfirmationRejected but call succeeded"
    except ConfirmationRejected:
        pass


async def test_permission_gated_tool_reraises_timeout():
    """Verify that gated tool re-raises ConfirmationTimeout instead of returning a string."""

    async def dummy_fn(path: str):
        return "written"

    raw_tool = StructuredTool.from_function(
        coroutine=dummy_fn,
        name="filesystem_write_file",
        description="Write a file",
    )

    conf_mgr = MagicMock(spec=ConfirmationManager)
    conf_mgr.wait_for_confirmation = AsyncMock(side_effect=ConfirmationTimeout())

    gated_tool = create_permission_gated_tool(
        tool=raw_tool,
        server_id="filesystem",
        method_name="write_file",
        confirmation_manager=conf_mgr,
    )

    try:
        await gated_tool.ainvoke({"path": "test.txt"})
        assert False, "Expected ConfirmationTimeout but call succeeded"
    except ConfirmationTimeout:
        pass


async def test_agent_stream_handles_rejection_without_retry():
    """Verify Agent_stream catches ConfirmationRejected and stops execution cleanly."""

    # Mock agent stream throwing ConfirmationRejected
    async def mock_astream(*args, **kwargs):
        raise ConfirmationRejected()
        yield  # make it a generator

    agent = MagicMock()
    agent.astream = mock_astream

    memory = ConversationMemory(enabled=True)

    chunks = []
    async for chunk in Agent_stream("Write to test.txt", agent, memory):
        chunks.append(chunk)

    # Assert no chunks returned
    assert chunks == []
    # Assert assistant message was NOT added to history
    assert len(memory.get_history()) == 1  # Only user message exists
    assert memory.get_history()[0]["content"] == "Write to test.txt"


async def test_agent_stream_filters_tool_messages():
    """Verify Agent_stream yields only AIMessage/AIMessageChunk and filters out ToolMessage output."""
    from langchain_core.messages import AIMessageChunk, ToolMessageChunk

    tool_msg = ToolMessageChunk(content="[FILE]\ndef secret_code():\n    pass", tool_call_id="123")
    ai_msg1 = AIMessageChunk(content="Done. ")
    ai_msg2 = AIMessageChunk(content="I copied mine.py to mine new 2.0.py.")

    async def mock_astream(*args, **kwargs):
        yield (tool_msg, {})
        yield (ai_msg1, {})
        yield (ai_msg2, {})

    agent = MagicMock()
    agent.astream = mock_astream
    memory = ConversationMemory(enabled=True)

    chunks = []
    async for chunk in Agent_stream("Copy file", agent, memory):
        chunks.append(chunk)

    assert chunks == ["Done. ", "I copied mine.py to mine new 2.0.py."]
    assert "[FILE]" not in "".join(chunks)
    assert "def secret_code" not in "".join(chunks)


async def test_confirmation_manager_deadline_loop():
    """Verify wait_for_confirmation loops past empty speech until valid confirmation is heard."""

    responses = [
        {"text": "", "confidence": None},
        {"text": "", "confidence": None},
        {"text": "yes", "confidence": None},
    ]

    async def mock_listen():
        return responses.pop(0)

    mgr = ConfirmationManager(listen_callback=mock_listen, timeout_seconds=5.0)
    result = await mgr.wait_for_confirmation("MEDIUM", "Write to test.txt")
    assert result is True


async def test_confirmation_manager_timeout_loop():
    """Verify wait_for_confirmation times out after deadline if no speech is heard."""

    async def mock_listen():
        await asyncio.sleep(0.05)
        return {"text": "", "confidence": None}

    mgr = ConfirmationManager(listen_callback=mock_listen, timeout_seconds=0.2)
    try:
        await mgr.wait_for_confirmation("MEDIUM", "Write to test.txt")
        assert False, "Expected ConfirmationTimeout"
    except ConfirmationTimeout:
        pass


async def test_format_action():
    """Verify action descriptions are formatted concisely."""
    desc = _format_action("filesystem", "write_file", {"path": "marina.py", "content": "long file content " * 50})
    assert desc == "Create or overwrite marina.py."

    desc_move = _format_action("filesystem", "move_file", {"path": "mine.py", "new_path": "marina.py"})
    assert desc_move == "Move mine.py to marina.py."

    desc_generic = _format_action("custom_server", "do_something", {"arg1": "a" * 50})
    assert "..." in desc_generic


async def test_high_tier_confidence():
    """Verify HIGH tier requires confidence >= 0.80 and strict phrase."""
    async def mock_low_conf():
        return {"text": "confirm action", "confidence": 0.75}

    mgr = ConfirmationManager(listen_callback=mock_low_conf, timeout_seconds=5.0)
    try:
        await mgr.wait_for_confirmation("HIGH", "Delete database")
        assert False, "Expected ConfirmationRejected due to low confidence"
    except ConfirmationRejected:
        pass

    async def mock_high_conf():
        return {"text": "confirm action", "confidence": 0.92}

    mgr2 = ConfirmationManager(listen_callback=mock_high_conf, timeout_seconds=5.0)
    result = await mgr2.wait_for_confirmation("HIGH", "Delete database")
    assert result is True


async def test_voice_system_prompt():
    """Verify voice system prompt selection and strict concise output rules."""
    from system_prompt import get_system_prompt, SYSTEM_PROMPT, VOICE_SYSTEM_PROMPT

    text_prompt = get_system_prompt("text")
    voice_prompt = get_system_prompt("voice")

    assert text_prompt == SYSTEM_PROMPT
    assert voice_prompt == VOICE_SYSTEM_PROMPT
    assert "NEVER read or repeat raw MCP tool outputs" in voice_prompt
    assert "Done. I copied mine.py to mine-date.py." in voice_prompt


async def test_ack_chime_asset():
    """Verify pre-generated voice acknowledgements exist and are valid WAV files."""
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    ack_files = [
        "ack_yes.wav",
        "ack_listening.wav",
        "ack_go_ahead.wav",
        "ack_yes_listening.wav",
    ]

    import wave
    for fname in ack_files:
        fpath = os.path.join(assets_dir, fname)
        assert os.path.exists(fpath), f"Missing {fname}"
        with wave.open(fpath, "rb") as w:
            assert w.getnchannels() == 1
            assert w.getframerate() == 22050
            assert w.getnframes() > 0


async def test_copy_operation_invokes_write_tool():
    """Verify that write_file tool execution triggers MEDIUM confirmation prompt."""
    from permission_gated_tool import create_permission_gated_tool
    from langchain_core.tools import tool

    confirmation_appeared = False

    async def mock_listen():
        nonlocal confirmation_appeared
        confirmation_appeared = True
        return {"text": "yes", "confidence": None}

    conf_mgr = ConfirmationManager(listen_callback=mock_listen, timeout_seconds=5.0)

    @tool
    def write_file(path: str, content: str):
        """Write text content to file."""
        return f"Successfully wrote to {path}"

    gated_write = create_permission_gated_tool(
        tool=write_file,
        server_id="filesystem",
        method_name="write_file",
        confirmation_manager=conf_mgr,
    )

    res = await gated_write.ainvoke({"path": "mine-date.py", "content": "print('hello')"})
    assert confirmation_appeared is True, "MEDIUM confirmation for write_file did not appear!"
    assert "Successfully wrote to mine-date.py" in str(res)


async def test_readonly_request_no_write():
    """Verify that read_text_file tool execution does not trigger write tools or confirmation prompts."""
    from permission_gated_tool import create_permission_gated_tool
    from langchain_core.tools import tool

    confirmation_appeared = False

    async def mock_listen():
        nonlocal confirmation_appeared
        confirmation_appeared = True
        return {"text": "yes", "confidence": None}

    conf_mgr = ConfirmationManager(listen_callback=mock_listen, timeout_seconds=5.0)

    @tool
    def read_text_file(path: str):
        """Read text content from file."""
        return "file content here"

    gated_read = create_permission_gated_tool(
        tool=read_text_file,
        server_id="filesystem",
        method_name="read_text_file",
        confirmation_manager=conf_mgr,
    )

    res = await gated_read.ainvoke({"path": "mine.py"})
    assert confirmation_appeared is False, "Confirmation prompt appeared during read-only operation!"
    assert "file content here" in str(res)


async def test_spoken_confirmation_announcement():
    """Verify that ConfirmationManager calls speak_callback for prompt, accept, rejection, and timeout."""
    spoken_messages = []

    async def mock_speak(text: str):
        spoken_messages.append(text)

    async def mock_listen_yes():
        return {"text": "yes", "confidence": None}

    mgr = ConfirmationManager(
        listen_callback=mock_listen_yes,
        speak_callback=mock_speak,
        timeout_seconds=5.0,
    )

    res = await mgr.wait_for_confirmation("MEDIUM", "Create or overwrite norina.py.")
    assert res is True
    assert len(spoken_messages) == 2
    assert "I need your confirmation to Create or overwrite norina.py. Shall I proceed?" in spoken_messages[0]
    assert spoken_messages[1] == "Okay."

    # Test rejection announcement
    spoken_messages.clear()

    async def mock_listen_no():
        return {"text": "no", "confidence": None}

    mgr_no = ConfirmationManager(
        listen_callback=mock_listen_no,
        speak_callback=mock_speak,
        timeout_seconds=5.0,
    )
    try:
        await mgr_no.wait_for_confirmation("MEDIUM", "Delete file.txt.")
    except ConfirmationRejected:
        pass
    assert len(spoken_messages) == 2
    assert spoken_messages[1] == "Cancelled."


if __name__ == "__main__":
    asyncio.run(test_permission_gated_tool_reraises_rejection())
    asyncio.run(test_permission_gated_tool_reraises_timeout())
    asyncio.run(test_agent_stream_handles_rejection_without_retry())
    asyncio.run(test_agent_stream_filters_tool_messages())
    asyncio.run(test_confirmation_manager_deadline_loop())
    asyncio.run(test_confirmation_manager_timeout_loop())
    asyncio.run(test_format_action())
    asyncio.run(test_high_tier_confidence())
    asyncio.run(test_voice_system_prompt())
    asyncio.run(test_ack_chime_asset())
    asyncio.run(test_copy_operation_invokes_write_tool())
    asyncio.run(test_readonly_request_no_write())
    asyncio.run(test_spoken_confirmation_announcement())
    print("✅ All Phase 4 confirmation, deadline, format, confidence, voice prompt, message filtering, copy, read-only, and spoken confirmation tests passed!")
