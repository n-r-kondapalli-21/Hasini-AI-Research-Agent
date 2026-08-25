import asyncio

from langchain.agents import create_agent

from system_prompt import get_system_prompt, SYSTEM_PROMPT
from services.llm import llm
from Tools.web_search import web_tools
from mcp_servers.mcp_tools import get_mcp_tools
from permission_registry import discover_permissions
from tool_commands import handle_tool_command
from services.conversation_memory import ConversationMemory
from permission_gated_tool import create_permission_gated_tool
from permission_registry import (
    discover_permissions,
    resolve_tool_server,
)


async def create_research_agent(confirmation_manager=None, mode: str = "text"):

    mcp_client, mcp_tools = await get_mcp_tools()

    discover_permissions()

    gated_mcp_tools = []

    for tool in mcp_tools:

        server_id, method_name = resolve_tool_server(
            tool.name
        )

        gated_tool = create_permission_gated_tool(
            tool=tool,
            server_id=server_id,
            method_name=method_name,
            confirmation_manager=confirmation_manager,
        )

        gated_mcp_tools.append(gated_tool)

    all_tools = web_tools + gated_mcp_tools

    prompt = get_system_prompt(mode)

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=prompt,
    )

    return agent, mcp_client, all_tools


from langchain_core.messages import AIMessage, AIMessageChunk
from voice.confirmation import ConfirmationRejected, ConfirmationTimeout

# Phase 2 streaming function
async def Agent_stream(query, agent, memory):
    """
    Stream agent response token-by-token.

    Yields:
        Text chunks as they are generated.
    """

    memory.add_user_message(query)

    full_response = ""

    try:
        async for chunk in agent.astream(
            {"messages": memory.get_history()},
            stream_mode="messages",
        ):
            message, metadata = chunk

            if not message:
                continue

            # --------------------------------------------------
            # LangChain Message-Level Filtering:
            #
            # Only yield assistant/AI messages (AIMessage/AIMessageChunk).
            # Ignore tool messages (ToolMessage), raw tool outputs,
            # directory listings, and internal results.
            # --------------------------------------------------
            msg_type = getattr(message, "type", "")
            if not isinstance(message, (AIMessage, AIMessageChunk)) and msg_type not in {"ai", "assistant"}:
                continue

            content = getattr(message, "content", "")

            if not content:
                continue

            # Some providers can return structured content.
            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, dict):

                        text = item.get("text", "")

                        if text:
                            text_parts.append(text)

                    elif isinstance(item, str):

                        text_parts.append(item)

                content = "".join(text_parts)

            if not content:
                continue

            full_response += content

            yield content

    except ConfirmationRejected:
        print("\n🛑 Action cancelled by user.")
        return
    except ConfirmationTimeout:
        print("\n⏱️ Confirmation timed out. Action cancelled.")
        return
    finally:
        # Save the complete response only after streaming finishes successfully.
        if full_response:
            memory.add_assistant_message(full_response)


async def main():

    agent, mcp_client, all_tools = await create_research_agent()

    memory = ConversationMemory(
        history_limit=10,
        enabled=True,
    )

    print("=" * 50)
    print("🤖 AI Research Agent Started")
    print("=" * 50)

    print("Commands:")
    print("  /tools")
    print("  /tools filesystem")
    print("  /tools github")
    print("  /tools web")
    print("  /tool <tool_name>")
    print("  exit / quit")

    print("=" * 50)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle tool commands
        if handle_tool_command(
            user_input,
            all_tools,
            web_tools,
        ):
            continue

        print("\nHasini: ", end="", flush=True)

        async for chunk in Agent_stream(
            user_input,
            agent,
            memory,
        ):
            print(chunk, end="", flush=True)

        print()


if __name__ == "__main__":
    asyncio.run(main())