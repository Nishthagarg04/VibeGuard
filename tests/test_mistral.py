import os

from dotenv import load_dotenv
from mistralai.client import Mistral


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise RuntimeError(
        "MISTRAL_API_KEY is missing from .env"
    )


client = Mistral(
    api_key=api_key
)


response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": (
                "Write a simple Python function called "
                "add_numbers that takes two numbers and "
                "returns their sum. Return only the code."
            )
        }
    ]
)


text = (
    response
    .choices[0]
    .message
    .content
)


print("Mistral response:")
print(text)