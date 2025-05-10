from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode


from react import tools, model_with_tools
from langgraph.graph import MessagesState

load_dotenv()

TOOL_EXECUTOR = ToolNode(tools)


def run_agent_reasoning_engine(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
