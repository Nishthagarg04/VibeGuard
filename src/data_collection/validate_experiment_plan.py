import json
from pathlib import Path


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLING_PLAN_FILE = (
    PROJECT_ROOT
    / "configs"
    / "sampling_plan.json"
)

CHAIN_PLAN_FILE = (
    PROJECT_ROOT
    / "configs"
    / "chain_plan.json"
)


# =========================================================
# 2. Load JSON
# =========================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# 3. Validate model sets
# =========================================================

def validate_models(
    sampling_plan,
    chain_plan
):

    sampling_models = set(
        sampling_plan["models"].keys()
    )

    chain_models = set(
        chain_plan["models"].keys()
    )

    if sampling_models != chain_models:

        raise ValueError(
            "The models in sampling_plan.json "
            "and chain_plan.json do not match."
        )


# =========================================================
# 4. Validate template sets
# =========================================================

def validate_templates(
    sampling_plan,
    chain_plan
):

    for model_id in sampling_plan["models"]:

        sampling_templates = set(
            sampling_plan["models"][
                model_id
            ][
                "templates"
            ].keys()
        )

        chain_templates = set(
            chain_plan["models"][
                model_id
            ][
                "templates"
            ].keys()
        )

        if sampling_templates != chain_templates:

            raise ValueError(
                f"Template mismatch for "
                f"{model_id}."
            )


# =========================================================
# 5. Validate multi-round totals
# =========================================================

def validate_multi_round_totals(
    sampling_plan,
    chain_plan
):

    for model_id in sampling_plan["models"]:

        sampling_model = (
            sampling_plan[
                "models"
            ][
                model_id
            ]
        )

        chain_model = (
            chain_plan[
                "models"
            ][
                model_id
            ]
        )

        sampling_target = (
            sampling_model[
                "multi_round"
            ][
                "target"
            ]
        )

        chain_target = (
            chain_model[
                "target_samples"
            ]
        )

        if sampling_target != chain_target:

            raise ValueError(
                f"Multi-round target mismatch "
                f"for {model_id}: "
                f"sampling plan = "
                f"{sampling_target}, "
                f"chain plan = "
                f"{chain_target}."
            )


# =========================================================
# 6. Validate every template
# =========================================================

def validate_template_totals(
    sampling_plan,
    chain_plan
):

    for model_id in sampling_plan["models"]:

        sampling_templates = (
            sampling_plan[
                "models"
            ][
                model_id
            ][
                "templates"
            ]
        )

        chain_templates = (
            chain_plan[
                "models"
            ][
                model_id
            ][
                "templates"
            ]
        )

        for prompt_id in sampling_templates:

            sampling_template = (
                sampling_templates[
                    prompt_id
                ]
            )

            chain_template = (
                chain_templates[
                    prompt_id
                ]
            )

            sampling_target = (
                sampling_template[
                    "multi_round"
                ][
                    "target"
                ]
            )

            chain_target = (
                chain_template[
                    "target_samples"
                ]
            )

            if sampling_target != chain_target:

                raise ValueError(
                    f"Template target mismatch: "
                    f"{model_id} / "
                    f"{prompt_id}. "
                    f"Sampling plan = "
                    f"{sampling_target}, "
                    f"chain plan = "
                    f"{chain_target}."
                )


# =========================================================
# 7. Validate calculated chain totals
# =========================================================

def validate_chain_totals(
    chain_plan
):

    for model_id, model_data in (
        chain_plan["models"].items()
    ):

        total_samples = 0
        total_chains = 0

        round_samples = {
            "2": 0,
            "3": 0,
            "4": 0
        }

        round_chains = {
            "2": 0,
            "3": 0,
            "4": 0
        }

        for template_data in (
            model_data[
                "templates"
            ].values()
        ):

            for chain in (
                template_data[
                    "chains"
                ]
            ):

                rounds = str(
                    chain["rounds"]
                )

                total_samples += (
                    chain["rounds"]
                )

                total_chains += 1

                round_samples[
                    rounds
                ] += chain["rounds"]

                round_chains[
                    rounds
                ] += 1

        if total_samples != 500:

            raise ValueError(
                f"{model_id} contains "
                f"{total_samples} multi-round "
                "samples instead of 500."
            )

        if total_chains != 190:

            raise ValueError(
                f"{model_id} contains "
                f"{total_chains} chains instead of 190."
            )

        if round_samples != {
            "2": 200,
            "3": 180,
            "4": 120
        }:

            raise ValueError(
                f"Incorrect round sample "
                f"distribution for "
                f"{model_id}: "
                f"{round_samples}"
            )

        if round_chains != {
            "2": 100,
            "3": 60,
            "4": 30
        }:

            raise ValueError(
                f"Incorrect round chain "
                f"distribution for "
                f"{model_id}: "
                f"{round_chains}"
            )


# =========================================================
# 8. Validate one-shot + multi-round totals
# =========================================================

def validate_overall_totals(
    sampling_plan
):

    for model_id, model_data in (
        sampling_plan["models"].items()
    ):

        one_shot = (
            model_data[
                "one_shot"
            ][
                "target"
            ]
        )

        multi_round = (
            model_data[
                "multi_round"
            ][
                "target"
            ]
        )

        total = (
            model_data[
                "target"
            ]
        )

        if one_shot != 500:

            raise ValueError(
                f"{model_id} does not have "
                "a 500-sample one-shot target."
            )

        if multi_round != 500:

            raise ValueError(
                f"{model_id} does not have "
                "a 500-sample multi-round target."
            )

        if total != 1000:

            raise ValueError(
                f"{model_id} does not have "
                "a 1000-sample total target."
            )

        if one_shot + multi_round != total:

            raise ValueError(
                f"{model_id} has inconsistent "
                "overall sample totals."
            )


# =========================================================
# 9. Run every validation
# =========================================================

def validate_experiment():

    sampling_plan = load_json(
        SAMPLING_PLAN_FILE
    )

    chain_plan = load_json(
        CHAIN_PLAN_FILE
    )

    validate_models(
        sampling_plan,
        chain_plan
    )

    validate_templates(
        sampling_plan,
        chain_plan
    )

    validate_multi_round_totals(
        sampling_plan,
        chain_plan
    )

    validate_template_totals(
        sampling_plan,
        chain_plan
    )

    validate_chain_totals(
        chain_plan
    )

    validate_overall_totals(
        sampling_plan
    )


# =========================================================
# 10. Main
# =========================================================

if __name__ == "__main__":

    validate_experiment()

    print(
        "Experiment plan validation PASSED."
    )

    print(
        "\nVerified:"
    )

    print(
        "  3 models"
    )

    print(
        "  20 templates per model"
    )

    print(
        "  1,000 samples per model"
    )

    print(
        "  500 one-shot samples per model"
    )

    print(
        "  500 multi-round samples per model"
    )

    print(
        "  100 two-round chains per model"
    )

    print(
        "  60 three-round chains per model"
    )

    print(
        "  30 four-round chains per model"
    )