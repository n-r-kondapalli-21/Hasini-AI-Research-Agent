from langchain_openai import ChatOpenAI
from config import provider, MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=provider["api_key"],
    base_url=provider["base_url"])


if __name__=="__main__":

    response = llm.invoke("Hello!")
    print(response.content)
