from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import START, END, StateGraph

from nodes import run_agent_reasoning_engine, TOOL_EXECUTOR
from langgraph.graph import MessagesState

AGENT_REASON = "agent_reason"
TOOLS = "tools"


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return TOOLS
    else:
        return END


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.add_node(TOOLS, TOOL_EXECUTOR)

flow.add_edge(START, AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, should_continue, {TOOLS: TOOLS, END: END})

flow.add_edge(TOOLS, AGENT_REASON)

app = flow.compile()
print(app.get_graph().draw_mermaid())
# app.get_graph().draw_mermaid_png(
#     output_file_path="graph.png", max_retries=5, retry_delay=2.0
# )

if __name__ == "__main__":
    print("Hello ReAct with LangGraph")

    res = app.invoke(
        input={
            "messages": [
                (
                    "human",
                    "서울의 날씨는 어때요? 서울의 날씨 온도에서 3배를 곱한 수를 알려주세요.",
                )
            ]
        }
    )

    for message in res["messages"]:
        message.pretty_print()
