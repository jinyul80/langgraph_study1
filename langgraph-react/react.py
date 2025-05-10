from typing import Any
from dotenv import load_dotenv

from langchain import hub
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    Returns the triple of the given number.

    Args:
        num (float): The number to be tripled.

    Returns:
        float: The triple of the input number.
    """

    return 3 * float(num)


tools = [TavilySearch(max_results=1), triple]

_llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash")
model_with_tools = _llm.bind_tools(tools)
