from typing import Any
import os
import json
import random
import string
from langchain.tools import BaseTool, HumanInputRun
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class BookRentalCar(BaseTool):
    name = "Rental Car Booking"
    description = """Use this tool only when the user wants to rent or book a car. The input to this tool will be the user query as it is."""

    def _run(self, query: str) -> Any:
        response = HumanInputRun(BookRentalCar.get_user_info(query=query)).run(query)

    def _arun(self, input_str: str) -> Any:
        return NotImplementedError("This tool does not support async")

    def get_user_info(query: str):
        input_template = """Extract the car name specified by the user in the query. If the user does not specify any car name, return 'unknown'.

Query: I want to rent a car.
Assistant: unknown

Query: I want to rent baleno.
Assistant: baleno

Query: Book Santro for me.
Assistant: Santro

Query: {query}
Assistant: """

        print(query)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

        llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.5,
        )

        prompt = PromptTemplate(template=input_template, input_variables=["query"])

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        vehicle_name = llm_chain.run(query).strip().lower()

        print(vehicle_name)

        vehicle_file_path = os.path.join(os.path.dirname(__file__), "vehicles.txt")
        with open(vehicle_file_path, "r") as vehicle_file:
            vehicles = set(vehicle_file.read().split("\n"))

        if vehicle_name != "unknown":
            if vehicle_name in vehicles:
                print("Sure, I can help with that.")
            else:
                print(
                    "Sorry, we do not have {}. But you can choose from the following options: {}".format(
                        vehicle_name, ", ".join(vehicles)
                    )
                )
                vehicle_name = input("Which one would you like: ")
        else:
            print(
                "Sure, I can help with that. You can choose from the following options: {}".format(", ".join(vehicles))
            )
            vehicle_name = input("Which one would you like: ")

        db_file_path = os.path.join(os.path.dirname(__file__), "bookings.json")

        try:
            user_booking = json.load(open(db_file_path, "r"))
        except FileNotFoundError:
            with open(db_file_path, "w") as outfile:
                json.dump({}, outfile, indent=4)
            user_booking = json.load(open(db_file_path, "r"))

        user_name = input("Can you please tell me your name: ")

        user_age = input("Okay, what is your age: ")

        if int(user_age) < 18:
            return "Sorry, but we cannot rent car to a minor."

        count = 0
        while count < 3:
            user_dl_no = input("Please provide your DL No.: ")
            if BookRentalCar.isValidDLNo(user_dl_no):
                break
            else:
                count += 1
                print("Invalid DL number. Please enter correct DL number.")

        if count == 3:
            return "Maximum tries exceeded. Try again later."

        count = 0
        while count < 3:
            user_mobile_no = input("Please provide your contact info: ")
            if BookRentalCar.isValidMobile(user_mobile_no):
                break
            else:
                count += 1
                print("Invalid mobile number. Please enter correct mobile number.")

        if count == 3:
            return "Maximum tries exceeded. Try again later."

        user_days = input("For how many days do you want this vehicle: ")

        booking_id = "".join(random.choices(string.ascii_letters + string.digits, k=6))

        user_booking[booking_id] = {}
        user_booking[booking_id]["name"] = user_name
        user_booking[booking_id]["age"] = user_age
        user_booking[booking_id]["contact info"] = user_mobile_no
        user_booking[booking_id]["DL No."] = user_dl_no
        user_booking[booking_id]["days"] = user_days
        user_booking[booking_id]["car name"] = vehicle_name

        with open(db_file_path, "w") as outfile:
            json.dump(user_booking, outfile, indent=4)

        return "Great, you have booked {} for {} days. Your booking id: {}".format(vehicle_name, user_days, booking_id)

    def isValidDLNo(user_dl_no: str):
        if user_dl_no.isalnum() and len(user_dl_no) == 12:
            return True

        return False

    def isValidMobile(user_mobile_no: str):
        if len(user_mobile_no) == 10 and user_mobile_no.isnumeric():
            return True

        return False
