from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

#initialise sqlite and pass connection
sqlite_conn = sqlite3.connect("checkpont.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

class ChatbotState(TypedDict):
    messages: Annotated[list, add_messages]

def post_generator(state: ChatbotState)-> ChatbotState:
    response = llm.invoke(state['messages'])
    return {
        'messages': [response]
    }

def collect_feedback(state):
    user_feedback = input("How do I improve the post? : ")
    return {
        "messages": [
            HumanMessage(content=user_feedback)
        ]
    }

def human_decision(state: ChatbotState)-> ChatbotState:
    post_generated = state["messages"][-1].content
    print(post_generated)
    user_approval = input("Approve post (yes/no)?")

    if user_approval.lower() == 'yes':
        return END
    return "collect_feedback"

graph = StateGraph(ChatbotState)
graph.add_node("post_generator", post_generator)
graph.add_node("collect_feedback", collect_feedback)

graph.add_conditional_edges("post_generator", human_decision)
graph.add_edge("collect_feedback", "post_generator")
graph.set_entry_point("post_generator")

app = graph.compile(checkpointer=memory) 

#create a config with thread_id
config = {
    "configurable": {
        "thread_id": 4
    }
}

response = app.invoke({
    "messages" : [HumanMessage(content="create a linkedin post on global warming")]
}, config=config)

