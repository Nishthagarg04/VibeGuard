import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )


client = Groq(
    api_key=api_key
)


response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
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


print("GPT-OSS response:")
print(text)