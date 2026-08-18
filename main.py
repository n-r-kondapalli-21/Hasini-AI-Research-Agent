import asyncio

from langchain.agents import create_agent
from system_prompt import SYSTEM_PROMPT
from llm import llm
from Tools.web_search import web_tools
from mcp_servers.mcp_tools import get_mcp_tools
from tool_commands import handle_tool_command

history = []

async def create_research_agent():

    # Connect to Filesystem MCP
    mcp_client, mcp_tools = await get_mcp_tools()

    # Combine existing web tools + filesystem MCP tools
    all_tools = web_tools + mcp_tools

    agent = create_agent(model=llm,tools=all_tools,system_prompt=SYSTEM_PROMPT)

    return agent, mcp_client, all_tools


async def Agent(query,agent):

    history.append({"role": "user","content": query})

    response = await agent.ainvoke({"messages": history})

    agent_response = response["messages"][-1].content

    history.append({"role": "assistant","content": agent_response})

    return f"Hasini: {agent_response}"

async def main():
    agent,mcp_client,all_tools = await create_research_agent()

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

    
            response = await Agent(user_input,agent)
    
            print(response)
     

if __name__=="__main__":
   asyncio.run(main())