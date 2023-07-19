import os
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from chains.flight_api_chain import FlightChain
from chains.travel_planner_chain import TravelPlannerChain
from chains.qna_chain import QNAChain
from tools.car_booking_tool import BookRentalCar

from typing import Dict, Any


def chat():
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key == "":
        raise Exception("No OpenAI API key specified.")

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.5, max_tokens=300)

    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        k=10,
        return_messages=True,
    )

    tools = [
        Tool(
            name="Flight API Tool",
            func=FlightChain.run,
            description="Use this tool whenever you need to get the flight information between two locations. Input to this tool will be the user query as it is.",
        ),
        Tool(
            name="Travel Planner Tool",
            func=TravelPlannerChain.run,
            description="Use this tool whenever you need to get the travel plan or information about popular places and tourist attractions for a location. Input to this tool will the user query as it is.",
        ),
        Tool(
            name="QNA Tool",
            func=QNAChain.run,
            description="Use this tool whenever you need to get information about baggage claim or car rental services. Input to this tool will the user query as it is. User queries about car rental prices would go under this tool. User queries like 'can I rent a car' and 'do you have swift' would also go under this tool.",
        ),
        BookRentalCar(),
    ]

    PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, but Assisstant does not have any knowledge of its own.

For every input that it receives, it needs the help of a tool available to it to answer the input query."""

    agent = initialize_agent(
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        llm=llm,
        tools=tools,
        memory=conversational_memory,
        max_iterations=3,
        early_stopping_method="generate",
        max_tokens=400,
        return_intermediate_steps=True,
        agent_kwargs={"prefix": PREFIX},
    )

    while True:
        query = input("YOU: ")

        response = agent({"input": query})

        if len(response["intermediate_steps"]) == 0:
            print("BOT: " + response["output"])
        else:
            print("BOT: " + response["intermediate_steps"][0][1])


if __name__ == "__main__":
    chat()
