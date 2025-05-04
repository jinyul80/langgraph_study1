from dotenv import load_dotenv
import os

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chanins import revisor, first_responder
from tool_excutor import execute_tools

load_dotenv()

MAX_ITERATIONS = 2


def event_loop(state: list[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits

    if num_iterations > MAX_ITERATIONS:
        return END  # 조건 만족 시 종료
    return "execute_tools"  # 조건 미만족 시 execute_tools로 전환


builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

# 정규 엣지: draft -> execute_tools -> revise
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# 조건부 엣지: revise -> execute_tools 또는 END
builder.add_conditional_edges(
    "revise", event_loop, {"execute_tools": "execute_tools", END: END}
)

builder.set_entry_point("draft")
graph = builder.compile()


if __name__ == "__main__":
    print("Hello Reflexion")

    print(graph.get_graph().draw_mermaid())
    output_png_path = os.path.join("reflexion-graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=output_png_path)

    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )

    print(res[-1].tool_calls[0]["args"]["answer"])

    print("\nsuccess.")
