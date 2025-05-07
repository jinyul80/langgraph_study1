from typing import Any

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools import BraveSearch
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

# web_search_tool = BraveSearch(search_kwargs={"count": 2})
web_search_tool = TavilySearch(max_results=2)


def web_search(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"]
    if "documents" in state:  # if the route to web search in first time then give error
        documents = state["documents"]

    searcher_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [searcher_result["content"] for searcher_result in searcher_results]
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    result = web_search(state={"question": "agent memory", "documents": None})

    print(result)
