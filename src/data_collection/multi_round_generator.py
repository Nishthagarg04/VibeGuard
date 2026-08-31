import json
import uuid
import sys
from pathlib import Path


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "data_collection")
)


from model_clients import generate_code
CHAIN_PLAN_FILE = (
    PROJECT_ROOT
    / "configs"
    / "chain_plan.json"
)


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

def load_chain_plan():

    if not CHAIN_PLAN_FILE.exists():

        raise FileNotFoundError(
            f"Chain plan not found: "
            f"{CHAIN_PLAN_FILE}"
        )

    with open(
        CHAIN_PLAN_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "prompts"
    / "templates"
)


def load_real_template(
    prompt_id
):

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

def get_first_planned_chain(
    model_id
):

    plan = load_chain_plan()

    if model_id not in plan["models"]:

        raise ValueError(
            f"Model not found in chain plan: "
            f"{model_id}"
        )

    model_plan = plan[
        "models"
    ][
        model_id
    ]

    for prompt_id, template_plan in (
        model_plan[
            "templates"
        ].items()
    ):

        if not template_plan["chains"]:

            continue

        return (
            prompt_id,
            template_plan,
            template_plan["chains"][0]
        )

    raise RuntimeError(
        "No planned chains were found."
    )

# =========================================================
# 2. Generate unique IDs
# =========================================================

def create_sample_id():

    return (
        "SAMPLE_"
        + uuid.uuid4().hex[:12]
    )


def create_chain_id():

    return (
        "CHAIN_"
        + uuid.uuid4().hex[:12]
    )


# =========================================================
# 3. Save generated code
# =========================================================

def save_code(
    sample_id,
    code
):

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        TRAINING_DIR
        / f"{sample_id}.js"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            code
        )

    return file_path


# =========================================================
# 4. Save metadata
# =========================================================

def save_metadata(
    metadata
):

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
# 5. Build the initial prompt
# =========================================================

def build_initial_prompt(
    template,
    rephrasing_index
):

    rephrasings = template[
        "rephrasings"
    ]

    if (
        rephrasing_index < 0
        or
        rephrasing_index >= len(
            rephrasings
        )
    ):

        raise ValueError(
            "Invalid rephrasing index: "
            f"{rephrasing_index}"
        )

    return rephrasings[
        rephrasing_index
    ]


# =========================================================
# 6. Build refinement prompt
# =========================================================

def build_refinement_prompt(
    previous_code,
    round_number
):

    return f"""
You are refining a JavaScript code solution.

This is refinement round {round_number}.

Review the code below carefully.

Improve it while preserving the original task.

Pay particular attention to:

- correctness
- security
- input validation
- authentication and authorization
- error handling
- safe handling of sensitive data
- avoiding unnecessary vulnerabilities

Return ONLY the complete revised JavaScript code.

Do not explain your changes.

Previous code:

```javascript
{previous_code}""".strip()

# 7. Generate one chain
# =========================================================

def generate_chain(
    template,
    rephrasing_index,
    number_of_rounds,
    model_name,
    generate_function
):

    if number_of_rounds not in [2, 3, 4]:

        raise ValueError(
            "Multi-round chains must have "
            "2, 3, or 4 rounds."
        )

    chain_id = create_chain_id()

    prompt = build_initial_prompt(
        template,
        rephrasing_index
    )

    previous_code = None

    generated_samples = []

    for round_number in range(
        1,
        number_of_rounds + 1
    ):

        if round_number == 1:

            current_prompt = prompt

        else:

            current_prompt = (
                build_refinement_prompt(
                    previous_code,
                    round_number
                )
            )

        current_code = generate_function(
            current_prompt
        )

        if not current_code:

            raise RuntimeError(
                f"Model returned empty output "
                f"for round {round_number}."
            )

        sample_id = create_sample_id()

        code_path = save_code(
            sample_id,
            current_code
        )

        metadata = {

            "sample_id":
                sample_id,

            "chain_id":
                chain_id,

            "round":
                round_number,

            "prompt_id":
                template["prompt_id"],

            "rephrasing_index":
                rephrasing_index,

            "category":
                template["category"],

            "language":
                template["language"],

            "framework":
                template["framework"],

            "file_role":
                template["file_role"],

            "generation_type":
                "multi-round",

            "model":
                model_name,

            "code_file":
                str(
                    code_path.relative_to(
                        PROJECT_ROOT
                    )
                )
        }

        save_metadata(
            metadata
        )

        generated_samples.append(
            metadata
        )

        previous_code = current_code

    return generated_samples


# =========================================================
# 8. Main test
# =========================================================
if __name__ == "__main__":

    MODEL_ID = (
        "gpt_oss_120b"
    )

    MODEL_NAME = (
        "openai/gpt-oss-120b"
    )

    print(
        "Loading first planned chain..."
    )

    (
        prompt_id,
        template_plan,
        chain
    ) = get_first_planned_chain(
        MODEL_ID
    )

    print(
        f"\nTemplate: {prompt_id}"
    )

    print(
        f"Group: "
        f"{template_plan['group']}"
    )

    print(
        f"Rounds: "
        f"{chain['rounds']}"
    )

    print(
        f"Rephrasing: "
        f"{chain['rephrasing_index']}"
    )

    template = load_real_template(
        prompt_id
    )

    print(
        "\nGenerating planned chain..."
    )

    samples = generate_chain(
        template=template,

        rephrasing_index=(
            chain["rephrasing_index"]
        ),

        number_of_rounds=(
            chain["rounds"]
        ),

        model_name=MODEL_NAME,

        generate_function=lambda prompt:
            generate_code(
                MODEL_ID,
                prompt
            )
    )

    print(
        "\nPlanned chain generated."
    )

    print(
        f"Chain ID: "
        f"{samples[0]['chain_id']}"
    )

    for sample in samples:

        print(
            f"Round {sample['round']}: "
            f"{sample['sample_id']}"
        )