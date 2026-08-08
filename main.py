from langchain.agents import create_agent
from system_prompt import SYSTEM_PROMPT
from llm import llm
from tools import tools


history = []

agent = create_agent(model=llm,tools=tools,system_prompt=SYSTEM_PROMPT)


def Agent(query):

    history.append({"role": "user","content": query})

    response = agent.invoke({"messages": history})

    agent_response = response["messages"][-1].content

    history.append({"role": "assistant","content": agent_response})

    return f"Hasini: {agent_response}"

if __name__=="__main__":
    print("=" * 50)
    print("🤖 AI Agent Started")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\n Goodbye!")
            break

        if not user_input:
            continue

        response=Agent(user_input)

        print(response)
 


    

