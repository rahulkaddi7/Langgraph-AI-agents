from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
llm = ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)
graph.set_entry_point("chatbot")

app = graph.compile()

print(
    app.invoke({
        "messages": [HumanMessage(content="hello")]
    })
)