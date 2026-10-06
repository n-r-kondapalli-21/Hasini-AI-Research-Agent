"""
Verification script for Hasini AI Research Agent tool selection.

Verifies that the LLM correctly chooses whether to invoke `search_knowledge_base`
based on the query context, tool description, and system prompt instructions.
"""

import asyncio
import logging
import sys

from agent_runtime import create_research_agent
from config import RAG_ENABLED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_tool_rag_selection")


TEST_CASES = [
    {
        "query": "hi",
        "expect_rag": False,
        "description": "Greeting query",
    },
    {
        "query": "what is 12 * 15",
        "expect_rag": False,
        "description": "Math calculation query",
    },
    {
        "query": "latest news about AI",
        "expect_rag": False,
        "description": "Live web news query",
    },
    {
        "query": "According to my SQL notes, what is INSERT INTO?",
        "expect_rag": True,
        "description": "Explicit reference to indexed SQL notes",
    },
    {
        "query": "What does my indexed document say about Attention Is All You Need?",
        "expect_rag": True,
        "description": "Query about indexed Transformer paper",
    },
    {
        "query": "Search my knowledge base for SQL interview questions",
        "expect_rag": True,
        "description": "Explicit command to search Knowledge Base",
    },
]


async def run_test():
    logger.info("Initializing research agent for verification test...")
    agent, registry = await create_research_agent(mode="text")

    all_tools = registry.get_all_tools()
    tool_names = [t.name for t in all_tools]
    logger.info("Agent tools loaded (%d): %s", len(tool_names), tool_names)

    if "search_knowledge_base" not in tool_names:
        logger.error("`search_knowledge_base` tool is missing from registered agent tools!")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("RUNNING TOOL SELECTION VERIFICATION TEST")
    print("=" * 80 + "\n")

    results = []
    all_passed = True

    for case in TEST_CASES:
        query = case["query"]
        expect_rag = case["expect_rag"]
        desc = case["description"]

        print(f"Testing Query: '{query}' ({desc})")
        print(f"Expected RAG tool call: {expect_rag}")

        called_tools = []
        full_response = ""

        try:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="messages",
            ):
                if chunk and len(chunk) == 2:
                    msg, _meta = chunk
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            called_tools.append(tc.get("name"))
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        full_response += msg.content

        except Exception as exc:
            print(f"  ❌ Exception during execution: {exc}")
            all_passed = False
            continue

        rag_called = "search_knowledge_base" in called_tools
        passed = (rag_called == expect_rag)

        print(f"  Observed Tool Calls: {called_tools}")
        print(f"  `search_knowledge_base` called: {rag_called}")
        if passed:
            print("  STATUS: ✅ PASS\n")
        else:
            print("  STATUS: ❌ FAIL\n")
            all_passed = False

        results.append({
            "query": query,
            "desc": desc,
            "expect_rag": expect_rag,
            "rag_called": rag_called,
            "called_tools": called_tools,
            "passed": passed,
        })

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{status} | Query: '{r['query']}' | Called: {r['called_tools']}")

    if not all_passed:
        print("\nOne or more test cases failed.")
        sys.exit(1)
    else:
        print("\nAll tool selection test cases PASSED successfully!")


if __name__ == "__main__":
    asyncio.run(run_test())
