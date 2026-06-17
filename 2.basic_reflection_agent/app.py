from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from dotenv import load_dotenv
load_dotenv()

from chains import generation_chain, reflection_chain

graph = MessageGraph()
critique_count = 0

REFLECT = 'reflect'
GENERATE = 'generate'

#takes state as input and appends its answer/response to state in end
def generate_node(state):
    print("\n=== GENERATE ===")
    print(state)

    response = generation_chain.invoke({
        'messages': state
    })

    print("TYPE:", type(response))
    print("RESPONSE:", response)
    print("CONTENT:", repr(response.content))
    return response

def reflect_node(state):
    latest_tweet = state[-1]
    print("\n=== REFLECT ===")
    print(state)

    response = reflection_chain.invoke({
        'messages': [
            HumanMessage(
                content=latest_tweet.text
            )
        ]
    })

    print("TYPE:", type(response))
    print("RESPONSE:", response)
    print("CONTENT:", repr(response.content))

    text = response.text()
    return [HumanMessage(content=text)]

def should_continue(state):
    if(len(state) > 6):
        return END

    return REFLECT

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# print(app.get_graph().draw_mermaid())
# app.get_graph().draw_ascii()

response = app.invoke(
    [
        HumanMessage(
            content="AI agents taking over content creation"
        )
    ]
)
print(response)