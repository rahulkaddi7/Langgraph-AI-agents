from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import AIMessage, HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

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

app = graph.compile() 

while True:
    userInput = input("Ask Groq: ")

    if userInput.lower() == 'exit':
        break
    
    result = app.invoke({
        "messages": [HumanMessage(content= userInput)]
    })
    print('Groq:', result["messages"][-1].content)