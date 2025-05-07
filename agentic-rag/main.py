import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")

    print("Current working directory: ", os.getcwd())

    question = "Langgraph에서 MCP를 사용할 수 있나요?\n\n한국어로 대답해 주세요."

    res = app.invoke(input={"question": question})

    print(res["generation"])
