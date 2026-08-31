import json
from pathlib import Path


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata"
    / "samples.jsonl"
)

PLAN_FILE = (
    PROJECT_ROOT
    / "configs"
    / "sampling_plan.json"
)


# =========================================================
# 2. Load JSON file
# =========================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# 3. Load all existing metadata records
# =========================================================

def load_metadata():

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


# =========================================================
# 4. Find model ID from model name
# =========================================================

def find_model_id(
    plan,
    model_name
):

    for model_id, model_data in (
        plan["models"].items()
    ):

        if (
            model_data["name"]
            == model_name
        ):

            return model_id

    return None


# =========================================================
# 5. Synchronize plan with metadata
# =========================================================

def synchronize_plan():

    plan = load_json(
        PLAN_FILE
    )

    records = load_metadata()

    # -----------------------------------------------------
    # Reset all generated counters
    # -----------------------------------------------------

    for model_id, model_data in (
        plan["models"].items()
    ):

        model_data["generated"] = 0

        model_data["remaining"] = (
            model_data["target"]
        )

        model_data[
            "one_shot"
        ]["generated"] = 0

        model_data[
            "one_shot"
        ]["remaining"] = (
            model_data[
                "one_shot"
            ]["target"]
        )

        model_data[
            "multi_round"
        ]["generated"] = 0

        model_data[
            "multi_round"
        ]["remaining"] = (
            model_data[
                "multi_round"
            ]["target"]
        )

        for template_data in (
            model_data[
                "templates"
            ].values()
        ):

            template_data[
                "one_shot"
            ]["generated"] = 0

            template_data[
                "one_shot"
            ]["remaining"] = (
                template_data[
                    "one_shot"
                ]["target"]
            )

            template_data[
                "multi_round"
            ]["generated"] = 0

            template_data[
                "multi_round"
            ]["remaining"] = (
                template_data[
                    "multi_round"
                ]["target"]
            )

    # -----------------------------------------------------
    # Process every existing sample
    # -----------------------------------------------------

    for record in records:

        model_name = record.get(
            "model"
        )

        prompt_id = record.get(
            "prompt_id"
        )

        generation_type = record.get(
            "generation_type"
        )

        model_id = find_model_id(
            plan,
            model_name
        )

        if model_id is None:

            print(
                "WARNING: Model not found "
                f"in plan: {model_name}"
            )

            continue

        model_data = plan[
            "models"
        ][
            model_id
        ]

        # -------------------------------------------------
        # Update model-level count
        # -------------------------------------------------

        model_data[
            "generated"
        ] += 1

        model_data[
            "remaining"
        ] = (
            model_data["target"]
            - model_data["generated"]
        )

        # -------------------------------------------------
        # Determine generation mode
        # -------------------------------------------------

        if generation_type == "one-shot":

            mode = "one_shot"

        elif generation_type == "multi-round":

            mode = "multi_round"

        else:

            print(
                "WARNING: Unknown generation "
                f"type: {generation_type}"
            )

            continue

        # -------------------------------------------------
        # Update mode-level count
        # -------------------------------------------------

        model_data[
            mode
        ]["generated"] += 1

        model_data[
            mode
        ]["remaining"] = (
            model_data[
                mode
            ]["target"]
            - model_data[
                mode
            ]["generated"]
        )

        # -------------------------------------------------
        # Update template-level count
        # -------------------------------------------------

        if prompt_id not in (
            model_data[
                "templates"
            ]
        ):

            print(
                "WARNING: Template not found "
                f"in plan: {prompt_id}"
            )

            continue

        template_data = model_data[
            "templates"
        ][
            prompt_id
        ]

        template_data[
            mode
        ]["generated"] += 1

        template_data[
            mode
        ]["remaining"] = (
            template_data[
                mode
            ]["target"]
            - template_data[
                mode
            ]["generated"]
        )

    return plan


# =========================================================
# 6. Save synchronized plan
# =========================================================

def save_plan(plan):

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
# 7. Print summary
# =========================================================

def print_summary(plan):

    print(
        "\nSampling plan synchronized."
    )

    print(
        "\nCurrent progress:"
    )

    for model_id, model_data in (
        plan["models"].items()
    ):

        print(
            f"\n{model_data['name']}"
        )

        print(
            f"  Total: "
            f"{model_data['generated']} / "
            f"{model_data['target']}"
        )

        print(
            f"  One-shot: "
            f"{model_data['one_shot']['generated']} / "
            f"{model_data['one_shot']['target']}"
        )

        print(
            f"  Multi-round: "
            f"{model_data['multi_round']['generated']} / "
            f"{model_data['multi_round']['target']}"
        )


# =========================================================
# 8. Main
# =========================================================

if __name__ == "__main__":

    plan = synchronize_plan()

    save_plan(
        plan
    )

    print_summary(
        plan
    )