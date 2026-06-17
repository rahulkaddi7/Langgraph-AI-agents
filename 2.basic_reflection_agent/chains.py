from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a twitter techie influencer tasked to write excellent tweets.
            Generate the best twitter post for the user's request.
            If the user provides a critique, respond with a revised version of the previous tweet based on the critique.
            """
        ),
        MessagesPlaceholder(variable_name="messages") 
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a viral Twitter influencer.
            Review the tweet and provide constructive criticism.
            Focus on:
            - Virality
            - Clarity
            - Engagement
            - Length
            - Hook quality
            Give only critique and recommendations.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm