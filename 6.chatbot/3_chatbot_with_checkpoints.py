from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

#initialse memory saver
memory = MemorySaver()

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

#pass the thread config with invocation
#same thread = same state
response1= app.invoke({
    'messages': [HumanMessage(content="Hi, I am rahul")]
}, config=config)

response2 = app.invoke({
    'messages': [HumanMessage(content= "whats my name?")]
},config=config)

print(response1["messages"][-1].content)
print(response2["messages"][-1].content)

print("========================== Checkpoint history ==========================")
print(app.get_state(config=config))

# while True:
#     user_input = input("Ask LLM: ")

#     if user_input.lower() == "exit":
#         break
#     else:
#         response = app.invoke({
#             'messages': [HumanMessage(content= user_input)]
#         },config=config)

#         print(response["messages"][-1].content)
