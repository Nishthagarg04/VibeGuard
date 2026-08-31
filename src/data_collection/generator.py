import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from google import genai
from mistralai.client import Mistral
from groq import Groq


# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()


# =========================================================
# 2. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_DIR = PROJECT_ROOT / "prompts" / "templates"

TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "training"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata"
)

METADATA_FILE = (
    METADATA_DIR
    / "samples.jsonl"
)

PROGRESS_FILE = (
    METADATA_DIR
    / "generation_progress.json"
)

MODEL_CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "model_config.yaml"
)


# =========================================================
# 3. Create required directories
# =========================================================

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 4. Load model configuration
# =========================================================

def load_model_config():

    with open(
        MODEL_CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    return config["models"]


def get_model_config(model_id):

    models = load_model_config()

    for model in models:

        if model["id"] == model_id:
            return model

    raise ValueError(
        f"Model not found: {model_id}"
    )


# =========================================================
# 5. Generate unique IDs
# =========================================================

def create_sample_id():

    return (
        "S"
        + uuid.uuid4().hex[:10].upper()
    )


def create_chain_id():

    return (
        "CHAIN"
        + uuid.uuid4().hex[:10].upper()
    )


# =========================================================
# 6. Load prompt template
# =========================================================

def load_template(prompt_id):

    template_path = (
        TEMPLATE_DIR
        / f"{prompt_id}.json"
    )

    if not template_path.exists():

        raise FileNotFoundError(
            f"Prompt template not found: "
            f"{template_path}"
        )

    with open(
        template_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)

# =========================================================
# 8. Discover all prompt templates
# =========================================================

def discover_templates():

    template_files = sorted(
        TEMPLATE_DIR.glob("*.json")
    )

    if not template_files:

        raise FileNotFoundError(
            "No prompt templates found in "
            f"{TEMPLATE_DIR}"
        )

    prompt_ids = []

    for template_file in template_files:

        # template_index.json is an index file,
        # not an individual prompt template.
        if template_file.name == "template_index.json":
            continue

        prompt_ids.append(
            template_file.stem
        )

    if not prompt_ids:

        raise FileNotFoundError(
            "No individual prompt templates found in "
            f"{TEMPLATE_DIR}"
        )

    return prompt_ids

# =========================================================
# 9. Validate prompt templates
# =========================================================

def validate_templates():

    prompt_ids = discover_templates()

    required_fields = [
        "prompt_id",
        "category",
        "language",
        "framework",
        "file_role",
        "rephrasings"
    ]

    for prompt_id in prompt_ids:

        template = load_template(
            prompt_id
        )

        for field in required_fields:

            if field not in template:

                raise ValueError(
                    f"{prompt_id} is missing "
                    f"required field: {field}"
                )

        rephrasings = template[
            "rephrasings"
        ]

        if len(rephrasings) != 4:

            raise ValueError(
                f"{prompt_id} must contain "
                f"exactly 4 rephrasings, "
                f"but found "
                f"{len(rephrasings)}"
            )

    return prompt_ids
# =========================================================
# 7. Select prompt wording
# =========================================================

def select_prompt(
    template,
    rephrasing_index=0
):

    rephrasings = template.get(
        "rephrasings",
        []
    )

    if not rephrasings:

        raise ValueError(
            f"No rephrasings found for "
            f"{template['prompt_id']}"
        )

    if (
        rephrasing_index
        >= len(rephrasings)
    ):

        raise IndexError(
            f"Invalid rephrasing index "
            f"{rephrasing_index}"
        )

    return rephrasings[
        rephrasing_index
    ]


# =========================================================
# 8. Gemini
# =========================================================

def generate_with_gemini(
    prompt,
    model_name
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is missing."
        )

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    if not response.text:

        raise ValueError(
            "Gemini returned empty text."
        )

    return response.text


# =========================================================
# 9. Mistral
# =========================================================

def generate_with_mistral(
    prompt,
    model_name
):

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "MISTRAL_API_KEY is missing."
        )

    client = Mistral(
        api_key=api_key
    )

    response = client.chat.complete(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = (
        response
        .choices[0]
        .message
        .content
    )

    if not text:

        raise ValueError(
            "Mistral returned empty text."
        )

    return text


# =========================================================
# 10. Groq / GPT-OSS
# =========================================================

def generate_with_groq(
    prompt,
    model_name
):

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = (
        response
        .choices[0]
        .message
        .content
    )

    if not text:

        raise ValueError(
            "Groq returned empty text."
        )

    return text


# =========================================================
# 11. Unified model interface
# =========================================================

def generate_code(
    prompt,
    model_id
):

    config = get_model_config(
        model_id
    )

    provider = config["provider"]

    model_name = config["name"]

    if provider == "google":

        return generate_with_gemini(
            prompt,
            model_name
        )

    elif provider == "mistral":

        return generate_with_mistral(
            prompt,
            model_name
        )

    elif provider == "groq":

        return generate_with_groq(
            prompt,
            model_name
        )

    else:

        raise ValueError(
            f"Unsupported provider: "
            f"{provider}"
        )


# =========================================================
# 12. Extract code from response
# =========================================================

def extract_code(text):

    matches = re.findall(
        r"```(?:[a-zA-Z0-9_+#.-]+)?\s*(.*?)```",
        text,
        re.DOTALL
    )

    if matches:

        return max(
            matches,
            key=len
        ).strip()

    return text.strip()


# =========================================================
# 13. Determine file extension
# =========================================================

def get_extension(language):

    extensions = {

        "JavaScript": ".js",

        "Python": ".py",

        "TypeScript": ".ts",

        "Java": ".java"
    }

    if language not in extensions:

        raise ValueError(
            f"Unsupported language: "
            f"{language}"
        )

    return extensions[language]


# =========================================================
# 14. Save generated code
# =========================================================

def save_code(
    sample_id,
    language,
    code
):

    extension = get_extension(
        language
    )

    filename = (
        f"{sample_id}{extension}"
    )

    output_path = (
        TRAINING_DIR
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(code)

    return output_path


# =========================================================
# 15. Save metadata
# =========================================================

def save_metadata(metadata):

    with open(
        METADATA_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                metadata,
                ensure_ascii=False
            )
            + "\n"
        )

# =========================================================
# 16. Progress tracking
# =========================================================

def load_progress():

    if not PROGRESS_FILE.exists():

        return {}

    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_progress(progress):

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            progress,
            file,
            indent=4
        )


def calculate_progress():

    progress = {}

    models = load_model_config()

    for model in models:

        progress[
            model["id"]
        ] = 0

    if not METADATA_FILE.exists():

        return progress

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            provider_model = record.get(
                "model"
            )

            for model in models:

                if (
                    model["name"]
                    == provider_model
                ):

                    progress[
                        model["id"]
                    ] += 1

                    break

    return progress


# =========================================================
# 16. Generate one sample
# =========================================================

def generate_one_sample(
    prompt_id,
    model_id,
    rephrasing_index=0
):

    template = load_template(
        prompt_id
    )

    prompt = select_prompt(
        template,
        rephrasing_index
    )

    model_config = get_model_config(
        model_id
    )

    sample_id = create_sample_id()

    chain_id = create_chain_id()

    generated_text = generate_code(
        prompt,
        model_id
    )

    code = extract_code(
        generated_text
    )

    code_path = save_code(
        sample_id,
        template["language"],
        code
    )

    metadata = {

        "sample_id":
            sample_id,

        "chain_id":
            chain_id,

        "code_path":
            str(
                code_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "model":
            model_config[
                "name"
            ],

        "provider":
            model_config[
                "provider"
            ],

        "prompt_id":
            template[
                "prompt_id"
            ],

        "prompt_text":
            prompt,

        "category":
            template[
                "category"
            ],

        "round":
            1,

        "generation_type":
            "one-shot",

        "language":
            template[
                "language"
            ],

        "framework":
            template[
                "framework"
            ],

        "file_role":
            template[
                "file_role"
            ],

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    save_metadata(
        metadata
    )

    progress = calculate_progress()

    save_progress(
        progress
    )

    print(
        "\nGenerated sample:"
    )

    print(
        f"Sample ID: {sample_id}"
    )

    print(
        f"Chain ID: {chain_id}"
    )

    print(
        f"Model: "
        f"{model_config['name']}"
    )

    print(
        f"Provider: "
        f"{model_config['provider']}"
    )

    print(
        f"Code path: "
        f"{code_path}"
    )


# =========================================================
# 17. Test entry point
# =========================================================

if __name__ == "__main__":

    templates = validate_templates()

    print(
        f"Found {len(templates)} prompt templates."
    )

    print(
        "\nPrompt templates:"
    )

    for template in templates:

        print(
            f" - {template}"
        )