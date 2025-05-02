from dotenv import load_dotenv
import time

load_dotenv()

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langchain_community.tools import BraveSearch

from schemas import AnswerQuestion, ReviseAnswer

search_tool = BraveSearch().from_search_kwargs(search_kwargs={"count": 2})


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""

    results = []

    for query in search_queries:
        results.append(search_tool.run(query))
        time.sleep(1)

    return results


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)


if __name__ == "__main__":
    print("Tool test")

    test_queries = [
        "AI SOC startups 2023 funding rounds",
        "Darktrace post-IPO venture funding timeline",
        "CyberReason AI SOC technology stack",
        "GDPR compliance challenges in AI-driven SOCs",
    ]

    results = run_queries(test_queries)

    print(results)
