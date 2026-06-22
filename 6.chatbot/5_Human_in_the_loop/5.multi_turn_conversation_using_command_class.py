from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

class State(TypedDict): 
    linkedin_topic: str
    generated_post: Annotated[List, add_messages]
    human_feedback: Annotated[List, add_messages]

def post_generator(state: State)-> State:
    linkedin_topic = state["linkedin_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]

    prompt = f"""

        LinkedIn Topic: {linkedin_topic}
        Human Feedback: {feedback[-1] if feedback else "No feedback yet"}

        Generate a structured and well-written LinkedIn post based on the given topic.

        Consider previous human feedback to refine the reponse. 
    """
    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content writer"), 
        HumanMessage(content=prompt)
    ])

    geneated_linkedin_post = response.content

    print(f"[model_node] Generated post:\n{geneated_linkedin_post}\n")

    return {
       "generated_post": [AIMessage(content=geneated_linkedin_post)]
    }


def collect_feedback(state):
    print("\n [human_node] awaiting human feedback...")

    generated_post = state["generated_post"]

    user_feedback = interrupt("write feedback")
    if user_feedback.lower() == "done": 
        return Command(
            update= {
                "human_feedback": [HumanMessage(content="Finalised")]
            }, 
            goto="end_node"
        )
    
    return Command(
        update={
            "human_feedback": [HumanMessage(content=user_feedback)]
        }, 
        goto="post_generator"
    )

def end_node(state: State): 
    print("\n[end_node] Process finished")
    print("Final Generated Post:", state["generated_post"][-1])
    print("Final Human Feedback", state["human_feedback"])

    print(state)


graph = StateGraph(State)
graph.add_node("post_generator", post_generator)
graph.add_node("collect_feedback", collect_feedback)
graph.add_node("end_node", end_node)

graph.add_edge("post_generator", "collect_feedback")
graph.add_edge("end_node", END)
graph.set_entry_point("post_generator")

memory = MemorySaver()
app = graph.compile(checkpointer=memory) 

#create a config with thread_id
config = {
    "configurable": {
        "thread_id": 2
    }
}

linkedin_topic = input("Enter your LinkedIn topic: ")
initial_state = {
    "linkedin_topic": linkedin_topic, 
    "generated_post": [], 
    "human_feedback": []
}


for chunk in app.stream(initial_state, config=config):
    for node_id, value in chunk.items():
        #  If we reach an interrupt, continuously ask for human feedback

        if(node_id == "__interrupt__"):
            while True: 
                user_feedback = input("Provide feedback (or type 'done' when finished): ")

                # Resume the graph execution with the user's feedback
                app.invoke(Command(resume=user_feedback), config=config)

                if user_feedback.lower() == "done":
                    break