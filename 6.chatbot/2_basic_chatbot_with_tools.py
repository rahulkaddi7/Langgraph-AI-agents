from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model='llama-3.1-8b-instant')
llm = llm.bind_tools(tools)

# state
class BasicChatbotState(TypedDict):
    messages: Annotated[list, add_messages]

#chatbot functionality
def chatbot(state: BasicChatbotState)-> BasicChatbotState:
    response = llm.invoke(state['messages'])
    return {
        'messages': [response]
    }

#graph section
CHATBOT = "chatbot"
TOOLNODE = "tools"

graph = StateGraph(BasicChatbotState)
graph.add_node(CHATBOT, chatbot)
graph.add_node(TOOLNODE, ToolNode(tools))

graph.add_edge(TOOLNODE, CHATBOT)
graph.add_conditional_edges(CHATBOT, tools_condition)

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