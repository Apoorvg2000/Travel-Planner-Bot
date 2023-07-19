import os
import requests
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class FlightChain:
    openai_api_key = os.getenv("OPENAI_API_KEY")

    output_template = """Using the structured data provided, inform the user about the flights in the following manner:

Structured data: [('flight_date': '2023-06-08', 'departure': ('airport': 'Indira Gandhi International', 'scheduled': '2023-06-08T02:00:00+00:00'), 'arrival': ('airport': 'Chhatrapati Shivaji International (Sahar International)', 'scheduled': '2023-06-08T04:05:00+00:00'), 'airline': 'IndiGo'), ('flight_date': '2023-06-08', 'departure': ('airport': 'Indira Gandhi International', 'scheduled': '2023-06-08T01:50:00+00:00'), 'arrival': ('airport': 'Chhatrapati Shivaji International (Sahar International)', 'scheduled': '2023-06-08T04:10:00+00:00'), 'airline': 'Blue Dart Aviation'), ('flight_date': '2023-06-08', 'departure': ('airport': 'Indira Gandhi International', 'scheduled': '2023-06-08T11:50:00+00:00'), 'arrival': ('airport': 'Chhatrapati Shivaji International (Sahar International)', 'scheduled': '2023-06-08T14:15:00+00:00'), 'airline': 'American Airlines'), ('flight_date': '2023-06-08', 'departure': ('airport': 'Indira Gandhi International', 'scheduled': '2023-06-08T08:00:00+00:00'), 'arrival': ('airport': 'Chhatrapati Shivaji International (Sahar International)', 'scheduled': '2023-06-08T10:15:00+00:00'), 'airline': 'Air France'), ('flight_date': '2023-06-08', 'departure': ('airport': 'Indira Gandhi International', 'scheduled': '2023-06-08T05:00:00+00:00'), 'arrival': ('airport': 'Chhatrapati Shivaji International (Sahar International)', 'scheduled': '2023-06-08T07:05:00+00:00')]

Attendant: Sure here are 5 flights I found scheduled for departure from New Delhi and arrival at Mumbai -
1. Airline IndiGo scheduled for departure at 02:00 AM on 8th June 2023 from Indira Gandhi International Airport, New Delhi and scheduled for arrival at 04:05 AM on 8th June 2023 at Chhatrapati Shivaji International Airport, Mumbai.
2. Airline Blue Dart Aviation scheduled for departure at 01:50 AM on 8th June 2023 from Indira Gandhi International Airport, New Delhi and scheduled for arrival at 04:10 AM on 8th June 2023 at Chhatrapati Shivaji International Airport, Mumbai.
3. Airline American Airlines scheduled for departure at 11:50 AM on 8th June 2023 from Indira Gandhi International Airport, New Delhi and scheduled for arrival at 14:15 PM on 8th June 2023 at Chhatrapati Shivaji International Airport, Mumbai.
4. Airline Air France scheduled for departure at 08:00 AM on 8th June 2023 from Indira Gandhi International Airport, New Delhi and scheduled for arrival at 10:15 AM on 8th June 2023 at Chhatrapati Shivaji International Airport, Mumbai.
5. Airline Qantas scheduled for departure at 05:00 AM on 8th June 2023 from Indira Gandhi International Airport, New Delhi and scheduled for arrival at 07:05 AM on 8th June 2023 at Chhatrapati Shivaji International Airport, Mumbai.

Structured data: {data}
Attendant: """

    input_template = """Using the query given about fetching flight information between a departure city and an arrival city, form a string in such a way
that the string should be a comma separated list of names of cities and their iata codes where departure city should come first followed by arrival
city followed by their respective iata codes. For example, if the user says: 'Please show me flights from mumbai to new york', then 'mumbai,new york,BOM,NYC'
would be the input. If the user does not specify the departure city or the arrival city, return "unknown".

Query: {query}
AI: """

    prompt_input = PromptTemplate(template=input_template, input_variables=["query"])

    prompt_output = PromptTemplate(template=output_template, input_variables=["data"])

    llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key, max_tokens=350)

    llm_chain_input = LLMChain(llm=llm, prompt=prompt_input)

    llm_chain_output = LLMChain(llm=llm, prompt=prompt_output)

    def run(query):
        try:
            input_str = FlightChain.llm_chain_input.run(query).strip().lower()
            if input_str == "unknown":
                return "Please specify both source and destination cities."

            departure_from, arrival_at, dep_iata, arr_iata = input_str.strip().split(",")
            departure_from = departure_from.strip().lower()
            arrival_at = arrival_at.strip().lower()
            aviationstack_api_key = os.getenv("AVIATIONSTACK_API_KEY", "")
            if aviationstack_api_key == "":
                raise Exception("Error in getting Aviationstack API Key")

            url_flights = "http://api.aviationstack.com/v1/flights"
            params = {
                "access_key": aviationstack_api_key,
                "dep_iata": dep_iata,
                "arr_iata": arr_iata,
                "flight_status": "scheduled",
                "limit": 5,
            }
            api_flights = requests.get(url=url_flights, params=params).json()

            if "data" in api_flights and len(api_flights["data"]) > 0:
                api_flights = api_flights["data"]
            else:
                return "There are currently no flights for this route."

            flights_data = []

            for flight in api_flights:
                new_flight = {}
                new_flight["flight_date"] = flight["flight_date"]
                new_flight["departure"] = {}
                new_flight["departure"]["airport"] = flight["departure"]["airport"]
                new_flight["departure"]["scheduled"] = flight["departure"]["scheduled"]
                new_flight["arrival"] = {}
                new_flight["arrival"]["airport"] = flight["arrival"]["airport"]
                new_flight["arrival"]["scheduled"] = flight["arrival"]["scheduled"]
                new_flight["airline"] = flight["airline"]["name"]

                flights_data.append(new_flight)

            output_str = FlightChain.llm_chain_output.run(flights_data)
            return output_str
        except Exception as e:
            print("Internal error occured: {}".format(e))
            exit(0)
