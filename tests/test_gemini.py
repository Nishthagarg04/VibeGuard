import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env"
    )


client = genai.Client(
    api_key=api_key
)


interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=(
        "Write a simple Python function called "
        "add_numbers that takes two numbers and "
        "returns their sum. Return only the code."
    )
)


print("Gemini response:")
print(interaction.output_text)