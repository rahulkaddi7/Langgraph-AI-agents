from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict
 
class State(TypedDict):
    text: str

def node_a(state: State): 
    return Command(
        goto="node_b", 
        update={
            "text": state["text"] + "a"
        }
    )

def node_c(state: State): 
    return Command(
        goto=END, 
        update={
            "text": state["text"] + "c"
        }
    )

def node_b(state: State): 
    return Command(
        goto="node_c", 
        update={
            "text": state["text"] + "b"
        }
    )

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

graph.set_entry_point("node_a")

app = graph.compile()

response = app.invoke({
    "text": ""
})

print(response)