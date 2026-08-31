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
    / "chain_plan.json"
)


# =========================================================
# 2. Experiment targets
# =========================================================

TEMPLATE_COUNT = 20

MULTI_ROUND_SAMPLES_PER_TEMPLATE = 25

GROUP_A = {
    "2": 4,
    "3": 3,
    "4": 2
}

GROUP_B = {
    "2": 6,
    "3": 3,
    "4": 1
}


# =========================================================
# 3. Model configuration
# =========================================================

MODELS = [
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


# =========================================================
# 4. Discover actual templates
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

    if len(templates) != TEMPLATE_COUNT:

        raise ValueError(
            f"Expected {TEMPLATE_COUNT} templates, "
            f"but found {len(templates)}."
        )

    return templates


# =========================================================
# 5. Assign Group A / Group B to models
# =========================================================

def get_group(
    model_id,
    template_position
):

    # Positions are zero-based internally.
    position = template_position + 1

    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    if model_id == "gemini_3_5_flash_lite":

        if position <= 10:
            return "A"

        return "B"

    # -----------------------------------------------------
    # Mistral
    # -----------------------------------------------------

    if model_id == "mistral_small":

        if position <= 10:
            return "B"

        return "A"

    # -----------------------------------------------------
    # GPT-OSS
    # -----------------------------------------------------

    if model_id == "gpt_oss_120b":

        if position % 2 == 1:
            return "A"

        return "B"

    raise ValueError(
        f"Unknown model: {model_id}"
    )


# =========================================================
# 6. Get chain distribution for a group
# =========================================================

def get_group_distribution(
    group
):

    if group == "A":
        return GROUP_A.copy()

    if group == "B":
        return GROUP_B.copy()

    raise ValueError(
        f"Unknown group: {group}"
    )


# =========================================================
# 7. Calculate samples from chain distribution
# =========================================================

def calculate_samples(
    distribution
):

    return (
        (2 * distribution["2"])
        + (3 * distribution["3"])
        + (4 * distribution["4"])
    )


# =========================================================
# 8. Assign chains to rephrasings
# =========================================================

def assign_rephrasings(
    chains
):

    # Track the number of generated samples
    # assigned to each rephrasing.
    rephrasing_samples = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    # Track how many chains each rephrasing has.
    rephrasing_chains = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    # Sort longer chains first so that the
    # sample totals remain as balanced as possible.
    chains = sorted(
        chains,
        key=lambda chain: (
            -chain["rounds"],
            chain["chain_number"]
        )
    )

    for chain in chains:

        # Find the rephrasing with the fewest
        # currently assigned samples.
        selected_rephrasing = min(
            rephrasing_samples,
            key=lambda index: (
                rephrasing_samples[index],
                rephrasing_chains[index],
                index
            )
        )

        chain[
            "rephrasing_index"
        ] = selected_rephrasing

        rephrasing_samples[
            selected_rephrasing
        ] += chain["rounds"]

        rephrasing_chains[
            selected_rephrasing
        ] += 1

    return chains


# =========================================================
# 9. Build chains for one template
# =========================================================

def build_template_chains(
    prompt_id,
    group
):

    distribution = (
        get_group_distribution(group)
    )

    chains = []

    chain_number = 1

    for rounds in ["2", "3", "4"]:

        count = distribution[rounds]

        for _ in range(count):

            chains.append(
                {
                    "chain_number": chain_number,
                    "rounds": int(rounds),
                    "rephrasing_index": None
                }
            )

            chain_number += 1

    chains = assign_rephrasings(
        chains
    )

    return {
        "prompt_id": prompt_id,
        "group": group,
        "target_samples": (
            MULTI_ROUND_SAMPLES_PER_TEMPLATE
        ),
        "target_chains": len(chains),
        "chains": chains
    }


# =========================================================
# 10. Build plan for one model
# =========================================================

def build_model_plan(
    model,
    templates
):

    model_plan = {
        "name": model["name"],
        "provider": model["provider"],
        "target_samples": 500,
        "target_chains": 190,
        "templates": {}
    }

    total_samples = 0
    total_chains = 0

    for index, prompt_id in enumerate(
        templates
    ):

        group = get_group(
            model["id"],
            index
        )

        template_plan = (
            build_template_chains(
                prompt_id,
                group
            )
        )

        model_plan[
            "templates"
        ][
            prompt_id
        ] = template_plan

        total_samples += (
            template_plan[
                "target_samples"
            ]
        )

        total_chains += (
            template_plan[
                "target_chains"
            ]
        )

    model_plan[
        "calculated_samples"
    ] = total_samples

    model_plan[
        "calculated_chains"
    ] = total_chains

    return model_plan


# =========================================================
# 11. Build complete plan
# =========================================================

def build_plan():

    templates = discover_templates()

    plan = {
        "experiment": {
            "templates": TEMPLATE_COUNT,
            "multi_round_samples_per_model": 500,
            "multi_round_chains_per_model": 190,

            "chain_distribution": {
                "2_round_samples": 200,
                "3_round_samples": 180,
                "4_round_samples": 120
            }
        },

        "models": {}
    }

    for model in MODELS:

        plan[
            "models"
        ][
            model["id"]
        ] = build_model_plan(
            model,
            templates
        )

    return plan


# =========================================================
# 12. Validate template allocation
# =========================================================

def validate_template(
    template_plan
):

    samples = 0
    chain_counts = {
        "2": 0,
        "3": 0,
        "4": 0
    }

    for chain in (
        template_plan["chains"]
    ):

        rounds = str(
            chain["rounds"]
        )

        if rounds not in chain_counts:

            raise ValueError(
                "Invalid round count: "
                f"{rounds}"
            )

        chain_counts[
            rounds
        ] += 1

        samples += chain[
            "rounds"
        ]

        if chain[
            "rephrasing_index"
        ] not in [0, 1, 2, 3]:

            raise ValueError(
                "Every chain must have a "
                "valid rephrasing index."
            )

    if samples != 25:

        raise ValueError(
            f"{template_plan['prompt_id']} "
            f"has {samples} samples instead of 25."
        )

    expected_chains = (
    GROUP_A
    if template_plan["group"] == "A"
    else GROUP_B
)

    expected_chain_count = sum(
      expected_chains.values()
    )

    actual_chain_count = (
       chain_counts["2"]
       + chain_counts["3"]
       + chain_counts["4"]
    )

    if actual_chain_count != expected_chain_count:

       raise ValueError(
           f"{template_plan['prompt_id']} "
           f"should contain "
           f"{expected_chain_count} chains, "
           f"but found "
           f"{actual_chain_count}."
       )

    return chain_counts


# =========================================================
# 13. Validate complete model
# =========================================================

def validate_model(
    model_plan
):

    total_samples = 0
    total_chains = 0

    chain_counts = {
        "2": 0,
        "3": 0,
        "4": 0
    }

    for template_plan in (
        model_plan[
            "templates"
        ].values()
    ):

        template_counts = (
            validate_template(
                template_plan
            )
        )

        total_samples += (
            template_plan[
                "target_samples"
            ]
        )

        total_chains += (
            template_plan[
                "target_chains"
            ]
        )

        for rounds in chain_counts:

            chain_counts[
                rounds
            ] += template_counts[
                rounds
            ]

    if total_samples != 500:

        raise ValueError(
            "Model must contain exactly "
            f"500 multi-round samples, "
            f"found {total_samples}."
        )

    if total_chains != 190:

        raise ValueError(
            "Model must contain exactly "
            f"190 chains, "
            f"found {total_chains}."
        )

    if (
        chain_counts["2"] != 100
        or chain_counts["3"] != 60
        or chain_counts["4"] != 30
    ):

        raise ValueError(
            "Incorrect global chain "
            f"distribution: {chain_counts}"
        )

    return chain_counts


# =========================================================
# 14. Validate complete experiment
# =========================================================

def validate_plan(
    plan
):

    expected_models = {
        model["id"]
        for model in MODELS
    }

    actual_models = set(
        plan["models"].keys()
    )

    if actual_models != expected_models:

        raise ValueError(
            "Model set does not match "
            "the experiment configuration."
        )

    for model_id, model_plan in (
        plan["models"].items()
    ):

        chain_counts = validate_model(
            model_plan
        )

        print(
            f"{model_plan['name']}: "
            f"200 two-round samples, "
            f"180 three-round samples, "
            f"120 four-round samples"
        )

        print(
            f"  Chains: "
            f"2R={chain_counts['2']}, "
            f"3R={chain_counts['3']}, "
            f"4R={chain_counts['4']}"
        )


# =========================================================
# 15. Save plan
# =========================================================

def save_plan(
    plan
):

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
# 16. Main
# =========================================================

if __name__ == "__main__":

    plan = build_plan()

    validate_plan(
        plan
    )

    save_plan(
        plan
    )

    print(
        "\nMulti-round chain plan "
        "created successfully."
    )

    print(
        f"Templates: {TEMPLATE_COUNT}"
    )

    print(
        "Multi-round samples/model: 500"
    )

    print(
        "Multi-round chains/model: 190"
    )

    print(
        "\nSaved to:"
    )

    print(
        PLAN_FILE
    )