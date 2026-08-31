import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# 1. Load environment variables
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


# =========================================================
# 2. Model configuration
# =========================================================

MODEL_CONFIG = {
    "gemini_3_5_flash_lite": {
        "provider": "google",
        "model": "gemini-3.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY"
    },

    "mistral_small": {
        "provider": "mistral",
        "model": "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY"
    },

    "gpt_oss_120b": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "api_key_env": "GROQ_API_KEY"
    }
}


# =========================================================
# 3. Validate API keys
# =========================================================

def validate_api_key(
    model_id
):

    if model_id not in MODEL_CONFIG:

        raise ValueError(
            f"Unknown model ID: {model_id}"
        )

    api_key_env = MODEL_CONFIG[
        model_id
    ][
        "api_key_env"
    ]

    api_key = os.getenv(
        api_key_env
    )

    if not api_key:

        raise RuntimeError(
            f"{api_key_env} is not set "
            "in the environment."
        )

    return api_key


# =========================================================
# 4. Gemini
# =========================================================

def generate_gemini(
    prompt
):

    from google import genai

    api_key = validate_api_key(
        "gemini_3_5_flash_lite"
    )

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned empty output."
        )

    return response.text


# =========================================================
# 5. Mistral
# =========================================================

def generate_mistral(
    prompt
):

    from mistralai.client import Mistral

    api_key = validate_api_key(
        "mistral_small"
    )

    client = Mistral(
        api_key=api_key
    )

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "Mistral returned empty output."
        )

    return content


# =========================================================
# 6. GPT-OSS through Groq
# =========================================================

def generate_gpt_oss(
    prompt
):

    from groq import Groq

    api_key = validate_api_key(
        "gpt_oss_120b"
    )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response.choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "GPT-OSS returned empty output."
        )

    return content


# =========================================================
# 7. Common model interface
# =========================================================

MODEL_GENERATORS = {
    "gemini_3_5_flash_lite":
        generate_gemini,

    "mistral_small":
        generate_mistral,

    "gpt_oss_120b":
        generate_gpt_oss
}


# =========================================================
# 8. Generate using model ID
# =========================================================

def generate_code(
    model_id,
    prompt
):

    if model_id not in MODEL_GENERATORS:

        raise ValueError(
            f"No generator registered "
            f"for model: {model_id}"
        )

    generator = MODEL_GENERATORS[
        model_id
    ]

    return generator(
        prompt
    )


# =========================================================
# 9. Main test
# =========================================================

if __name__ == "__main__":

    test_prompt = """
Return only this JavaScript code:

console.log("model interface test");
""".strip()

    for model_id in MODEL_GENERATORS:

        print(
            f"\nTesting: {model_id}"
        )

        output = generate_code(
            model_id,
            test_prompt
        )

        print(
            output
        )