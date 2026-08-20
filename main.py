import asyncio

from langchain.agents import create_agent
from system_prompt import SYSTEM_PROMPT
from services.llm import llm
from Tools.web_search import web_tools
from mcp_servers.mcp_tools import get_mcp_tools
from tool_commands import handle_tool_command
from services.conversation_memory import ConversationMemory


async def create_research_agent():

    # Connect to Filesystem MCP
    mcp_client, mcp_tools = await get_mcp_tools()

    # Combine existing web tools + filesystem MCP tools
    all_tools = web_tools + mcp_tools

    agent = create_agent(model=llm,tools=all_tools,system_prompt=SYSTEM_PROMPT)

    return agent, mcp_client, all_tools

#this function provides entire responce of an llm at a time 
# async def Agent(query,agent,memory):

#     memory.add_user_message(query)

#     response = await agent.ainvoke({"messages": memory.get_history()})

#     agent_response = response["messages"][-1].content

#     memory.add_assistant_message(agent_response)

#     return agent_response

#updated phase 2 function to get llm responce as a streams 
async def Agent_stream(query, agent, memory):
    """
    Stream agent response token-by-token.

    Yields:
        Text chunks as they are generated.
    """

    memory.add_user_message(query)

    full_response = ""

    async for chunk in agent.astream(
        {"messages": memory.get_history()},
        stream_mode="messages",
    ):
        message, metadata = chunk

        if not message:
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

    # Save the complete response only after streaming finishes.
    memory.add_assistant_message(full_response)


async def main():
    agent,mcp_client,all_tools = await create_research_agent()

    memory = ConversationMemory(history_limit=10,enabled=True)

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
                print("\n Goodbye!")
                break
    
            if not user_input:
                continue

             # Handle tool commands
            if handle_tool_command(user_input,all_tools,web_tools):
                continue

    
            print("\nHasini: ", end="", flush=True)

            async for chunk in Agent_stream(
                user_input,
                agent,
                memory
            ):
                print(chunk, end="", flush=True)

            print()
                

if __name__=="__main__":
   asyncio.run(main())