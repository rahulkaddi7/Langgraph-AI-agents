from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
import datetime

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# agent = initialize_agent(tools=[], llm=llm, agent='zero-shot-react-description', verbose=True)

#does a google search
search_tool = TavilySearchResults()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)

    return formatted_time

tools = [search_tool, get_system_time]

agent = create_agent(
    model=llm,
    tools=tools
)

response= agent.stream(
    {
        "messages": [
            {"role": "user", "content": "When was SpaceX's last launch and how many days ago was that from this instant"}
        ]
    },
    stream_mode="values"
)

for chunk in response:
    chunk["messages"][-1].pretty_print()

