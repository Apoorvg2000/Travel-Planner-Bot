import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class TravelPlannerChain:
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    template = """You are a travel agency assistant who helps customers with their travel plans and itineraries. You suggest famous places to visit and
activities to do at the destination place specified by the user and arrange them in bullet points.

Query: {query}
Assisstant: """

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.5, max_tokens=300)

    prompt = PromptTemplate(template=template, input_variables=["query"])

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    def run(query):
        return TravelPlannerChain.llm_chain.run(query)
