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

class BasicChatbotState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatbotState)-> BasicChatbotState:
    response = llm.invoke(state['messages'])
    return {
        'messages': [response]
    }

CHATBOT = "chatbot"
graph = StateGraph(BasicChatbotState)
graph.add_node(CHATBOT, chatbot)
graph.add_edge(CHATBOT, END)
graph.set_entry_point(CHATBOT)

app = graph.compile(checkpointer= memory) 

#create a config with thread_id
config = {
    "configurable": {
        "thread_id": 2
    }
}

while True:
    user_input = input("Ask LLM: ")

    if user_input.lower() == "exit":
        break
    else:
        response = app.invoke({
            'messages': [HumanMessage(content= user_input)]
        },config=config)

        print(response["messages"][-1].content)
