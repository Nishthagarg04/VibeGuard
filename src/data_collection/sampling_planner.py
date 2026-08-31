import json
from pathlib import Path


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "prompts"
    / "templates"
)

PLAN_FILE = (
    PROJECT_ROOT
    / "configs"
    / "sampling_plan.json"
)


# =========================================================
# 2. Experiment configuration
# =========================================================

SAMPLES_PER_MODEL = 1000

ONE_SHOT_PER_MODEL = 500

MULTI_ROUND_PER_MODEL = 500

TEMPLATES_PER_MODEL = 20

SAMPLES_PER_TEMPLATE = 50

ONE_SHOT_PER_TEMPLATE = 25

MULTI_ROUND_PER_TEMPLATE = 25


# =========================================================
# 3. Multi-round distribution
# =========================================================

MULTI_ROUND_DISTRIBUTION = {
    "2": 200,
    "3": 180,
    "4": 120
}


# =========================================================
# 4. Discover prompt templates
# =========================================================

def discover_templates():

    template_files = sorted(
        TEMPLATE_DIR.glob("*.json")
    )

    templates = []

    for template_file in template_files:

        if template_file.name == "template_index.json":
            continue

        templates.append(
            template_file.stem
        )

    return templates


# =========================================================
# 5. Validate experiment configuration
# =========================================================

def validate_configuration(
    templates
):

    if len(templates) != TEMPLATES_PER_MODEL:

        raise ValueError(
            f"Expected "
            f"{TEMPLATES_PER_MODEL} templates, "
            f"but found "
            f"{len(templates)}."
        )

    if (
        ONE_SHOT_PER_MODEL
        + MULTI_ROUND_PER_MODEL
        != SAMPLES_PER_MODEL
    ):

        raise ValueError(
            "One-shot and multi-round "
            "totals do not equal the "
            "per-model sample target."
        )

    if (
        ONE_SHOT_PER_TEMPLATE
        + MULTI_ROUND_PER_TEMPLATE
        != SAMPLES_PER_TEMPLATE
    ):

        raise ValueError(
            "One-shot and multi-round "
            "template allocations do not "
            "equal the template target."
        )

    if (
        SAMPLES_PER_TEMPLATE
        * TEMPLATES_PER_MODEL
        != SAMPLES_PER_MODEL
    ):

        raise ValueError(
            "Template allocation does not "
            "equal the per-model target."
        )

    multi_round_total = sum(
        MULTI_ROUND_DISTRIBUTION.values()
    )

    if (
        multi_round_total
        != MULTI_ROUND_PER_MODEL
    ):

        raise ValueError(
            "Multi-round distribution does "
            "not equal the multi-round "
            "sample target."
        )

    for rounds, sample_count in (
        MULTI_ROUND_DISTRIBUTION.items()
    ):

        if sample_count % int(rounds) != 0:

            raise ValueError(
                f"{sample_count} samples cannot "
                f"be evenly divided into "
                f"{rounds}-round chains."
            )


# =========================================================
# 6. Build one-template allocation
# =========================================================

def build_template_plan(
    prompt_id
):

    return {
        "prompt_id": prompt_id,

        "one_shot": {
            "target": ONE_SHOT_PER_TEMPLATE,
            "generated": 0,
            "remaining": ONE_SHOT_PER_TEMPLATE
        },

        "multi_round": {
            "target": MULTI_ROUND_PER_TEMPLATE,
            "generated": 0,
            "remaining": MULTI_ROUND_PER_TEMPLATE
        }
    }


# =========================================================
# 7. Build complete sampling plan
# =========================================================

def build_plan():

    templates = discover_templates()

    validate_configuration(
        templates
    )

    models = [
        {
            "id": "gemini_3_5_flash_lite",
            "name": "gemini-3.5-flash-lite",
            "provider": "google"
        },
        {
            "id": "mistral_small",
            "name": "mistral-small-latest",
            "provider": "mistral"
        },
        {
            "id": "gpt_oss_120b",
            "name": "openai/gpt-oss-120b",
            "provider": "groq"
        }
    ]

    plan = {

        "experiment": {
            "target_samples_per_model":
                SAMPLES_PER_MODEL,

            "one_shot_per_model":
                ONE_SHOT_PER_MODEL,

            "multi_round_per_model":
                MULTI_ROUND_PER_MODEL,

            "samples_per_template":
                SAMPLES_PER_TEMPLATE
        },

        "multi_round_distribution":
            MULTI_ROUND_DISTRIBUTION,

        "models": {}
    }

    for model in models:

        model_id = model["id"]

        plan["models"][model_id] = {

            "name":
                model["name"],

            "provider":
                model["provider"],

            "target":
                SAMPLES_PER_MODEL,

            "generated":
                0,

            "remaining":
                SAMPLES_PER_MODEL,

            "one_shot": {
                "target":
                    ONE_SHOT_PER_MODEL,

                "generated":
                    0,

                "remaining":
                    ONE_SHOT_PER_MODEL
            },

            "multi_round": {
                "target":
                    MULTI_ROUND_PER_MODEL,

                "generated":
                    0,

                "remaining":
                    MULTI_ROUND_PER_MODEL
            },

            "templates": {

                prompt_id:
                    build_template_plan(
                        prompt_id
                    )

                for prompt_id in templates
            }
        }

    return plan


# =========================================================
# 8. Save plan
# =========================================================

def save_plan(plan):

    PLAN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        PLAN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            plan,
            file,
            indent=4
        )


# =========================================================
# 9. Main
# =========================================================

if __name__ == "__main__":

    plan = build_plan()

    save_plan(plan)

    print(
        "Sampling plan created successfully."
    )

    print(
        f"Templates: "
        f"{len(plan['models']['gemini_3_5_flash_lite']['templates'])}"
    )

    print(
        "Samples per model: "
        f"{SAMPLES_PER_MODEL}"
    )

    print(
        "One-shot per model: "
        f"{ONE_SHOT_PER_MODEL}"
    )

    print(
        "Multi-round per model: "
        f"{MULTI_ROUND_PER_MODEL}"
    )

    print(
        "Sampling plan saved to:"
    )

    print(
        PLAN_FILE
    )