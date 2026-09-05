import json
import uuid
import re
from pathlib import Path

from .model_clients import generate_code


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
# Project root
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


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

def get_all_planned_chains(
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

    planned_chains = []

    for prompt_id, template_plan in (
        model_plan["templates"].items()
    ):

        for chain in template_plan["chains"]:

            planned_chains.append(
                {
                    "prompt_id": prompt_id,
                    "template_plan": template_plan,
                    "chain": chain
                }
            )

    if not planned_chains:

        raise RuntimeError(
            f"No planned chains were found "
            f"for model: {model_id}"
        )

    return planned_chains
# =========================================================
# 2. Generate unique IDs
# =========================================================

def create_sample_id():

    return (
        "SAMPLE_"
        + uuid.uuid4().hex[:12]
    )


def create_chain_id(model_id,prompt_id,chain_number):

    return (
        f"CHAIN_{model_id}_{prompt_id}_{chain_number:03d}"
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

def load_existing_metadata():

    if not METADATA_FILE.exists():

        return []

    records = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records

def get_existing_rounds(
    chain_id,
    metadata_records
):

    return {
        record["round"]
        for record in metadata_records
        if record.get("chain_id") == chain_id
    }

def get_next_round(
    chain_id,
    total_rounds,
    metadata_records
):

    existing_rounds = get_existing_rounds(
        chain_id,
        metadata_records
    )

    for round_number in range(
        1,
        total_rounds + 1
    ):

        if round_number not in existing_rounds:

            return round_number

    return None

def validate_existing_rounds(
    chain_id,
    total_rounds,
    metadata_records
):

    existing_rounds = get_existing_rounds(
        chain_id,
        metadata_records
    )

    for round_number in existing_rounds:

        if (
            round_number < 1
            or
            round_number > total_rounds
        ):

            raise RuntimeError(
                f"Invalid existing round "
                f"{round_number} for chain "
                f"{chain_id}. "
                f"Expected rounds 1-{total_rounds}."
            )

    if existing_rounds:

        expected_rounds = set(
            range(
                1,
                max(existing_rounds) + 1
            )
        )

        if existing_rounds != expected_rounds:

            raise RuntimeError(
                f"Missing earlier round in "
                f"chain {chain_id}. "
                f"Existing rounds: "
                f"{sorted(existing_rounds)}"
            )

def load_code(code_file):

    code_path = PROJECT_ROOT / code_file

    if not code_path.exists():

        raise FileNotFoundError(
            f"Code file referenced by metadata "
            f"does not exist: {code_path}"
        )

    with open(
        code_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()

def get_existing_round_metadata(
    chain_id,
    round_number,
    metadata_records
):

    for record in metadata_records:

        if (
            record.get("chain_id") == chain_id
            and
            record.get("round") == round_number
        ):

            return record

    return None

def get_resume_state(
    chain_id,
    total_rounds,
    metadata_records
):

    validate_existing_rounds(
        chain_id,
        total_rounds,
        metadata_records
    )

    next_round = get_next_round(
        chain_id,
        total_rounds,
        metadata_records
    )

    if next_round is None:

        return None, None

    if next_round == 1:

        return 1, None

    previous_round = (
        next_round - 1
    )

    previous_metadata = (
        get_existing_round_metadata(
            chain_id,
            previous_round,
            metadata_records
        )
    )

    if previous_metadata is None:

        raise RuntimeError(
            f"Previous round {previous_round} "
            f"metadata not found for chain "
            f"{chain_id}."
        )

    previous_code = load_code(
        previous_metadata["code_file"]
    )

    return next_round, previous_code

def generate_resumable_chain(
    template,
    rephrasing_index,
    number_of_rounds,
    model_name,
    generate_function,
    chain_id,
    metadata_records
):

    start_round, previous_code = get_resume_state(
        chain_id,
        number_of_rounds,
        metadata_records
    )

    if start_round is None:

        print(
            f"Chain already complete: {chain_id}"
        )

        return []

    print(
        f"Starting chain {chain_id} "
        f"from round {start_round}."
    )

    return generate_chain(
        template=template,
        rephrasing_index=rephrasing_index,
        number_of_rounds=number_of_rounds,
        model_name=model_name,
        generate_function=generate_function,
        chain_id=chain_id,
        start_round=start_round,
        previous_code=previous_code
    )

def run_batch(
    model_id,
    model_name,
    generate_function,
    limit=None
):

    planned_chains = get_all_planned_chains(
        model_id
    )

    if limit is not None:

        if limit <= 0:

            raise ValueError(
                "Batch limit must be greater than 0."
            )

        planned_chains = planned_chains[:limit]

    metadata_records = (
        load_existing_metadata()
    )

    generated_samples = []

    for planned in planned_chains:

        prompt_id = planned[
            "prompt_id"
        ]

        chain = planned[
            "chain"
        ]

        chain_id = create_chain_id(
            model_id,
            prompt_id,
            chain["chain_number"]
        )

        print(
            f"\nProcessing chain: {chain_id}"
        )

        template = load_real_template(
            prompt_id
        )

        samples = generate_resumable_chain(
            template=template,

            rephrasing_index=(
                chain["rephrasing_index"]
            ),

            number_of_rounds=(
                chain["rounds"]
            ),

            model_name=model_name,

            generate_function=generate_function,

            chain_id=chain_id,

            metadata_records=metadata_records
        )

        if samples:

            generated_samples.extend(
                samples
            )

            metadata_records.extend(
                samples
            )

    return generated_samples

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
{previous_code}
```""".strip()

# 7. Generate one chain
# =========================================================

def generate_chain(
    template,
    rephrasing_index,
    number_of_rounds,
    model_name,
    generate_function,
    chain_id,
    start_round=1,
    previous_code=None
):

    if number_of_rounds not in [2, 3, 4]:

        raise ValueError(
            "Multi-round chains must have "
            "2, 3, or 4 rounds."
        )

    prompt = build_initial_prompt(
        template,
        rephrasing_index
    )

    generated_samples = []

    for round_number in range(
    start_round,
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

        current_output = generate_function(
            current_prompt
        )

        if not current_output:

            raise RuntimeError(
                f"Model returned empty output "
                f"for round {round_number}."
            )

        current_code = extract_code(
            current_output
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
        "\nRunning first 3 planned chains..."
    )

    samples = run_batch(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        generate_function=lambda prompt:
            generate_code(
                MODEL_ID,
                prompt
            ),
        limit=3
    )

    print(
        "\nBatch generation complete."
    )

    if samples:

        print(
            f"New samples generated: "
            f"{len(samples)}"
        )

        for sample in samples:

            print(
                f"Chain {sample['chain_id']} "
                f"| Round {sample['round']} "
                f"| Sample {sample['sample_id']}"
            )

    else:

        print(
            "No new samples generated."
        )
