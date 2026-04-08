import json
import os
from collections import defaultdict

# ==============================================================================
# 1. Define the Mapping Dictionary & Weights
# ==============================================================================
# The linear combination weights for Match 1, Match 2, and Match 3.
# Tweak these numbers to give more/less power to the top choices!
MATCH_WEIGHTS = [0.70, 0.20, 0.10]

DEFENSE_MAPPING = {
    # Level 1: Action Defenses
    "passive aggression": 1,
    "help-rejecting complaining": 1,
    "acting out": 1,
    # Level 2: Major Image-Distorting Defenses
    "splitting": 2,
    "projective identification": 2,
    # Level 3: Disavowal Defenses
    "denial": 3,
    "rationalization": 3,
    "projection": 3,
    "autistic fantasy": 3,
    # Level 4: Minor Image-Distorting Defenses
    "devaluation": 4,
    "idealization": 4,
    "omnipotence": 4,
    # Level 5: Neurotic Defenses
    "repression": 5,
    "dissociation": 5,
    "reaction formation": 5,
    "displacement": 5,
    # Level 6: Obsessional Defenses
    "isolation of affect": 6,
    "intellectualization": 6,
    "undoing": 6,
    # Level 7: High-Adaptive (Mature) Defenses
    "affiliation": 7,
    "altruism": 7,
    "anticipation": 7,
    "humor": 7,
    "self-assertion": 7,
    "self-observation": 7,
    "sublimation": 7,
    "suppression": 7,
    # Edges cases / Level 0 and 8
    "no defense": 0,
    "no defense / neutral utterance": 0,
    "neutral utterance": 0,
    "unclear": 8,
    "needs more information": 8,
}


def normalize_name(name: str) -> str:
    """Helper function to clean up LLM string outputs for dictionary matching."""
    if not name:
        return ""
    return name.lower().strip()


# ==============================================================================
# 2. Main Processing Function
# ==============================================================================
def map_sublevels_to_labels(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_count = 0
    missing_mappings = defaultdict(list)

    for item in data:
        extracted_evidence = item.get("extracted_dmrs_evidence", [])

        if not extracted_evidence:
            continue

        # A dictionary to hold the weighted vote tally for this specific conversation turn
        level_scores = defaultdict(float)

        # Process each piece of extracted evidence
        for idx, evidence in enumerate(extracted_evidence):
            sublevel = evidence.get("sublevel_name", "")
            clean_sublevel = normalize_name(sublevel)

            level = DEFENSE_MAPPING.get(clean_sublevel)

            if level is not None:
                evidence["defense_level"] = level

                # Grab the weight based on the item's rank (index 0, 1, or 2)
                # If there are somehow more than 3 items, default to a weight of 1.0
                weight = MATCH_WEIGHTS[idx] if idx < len(MATCH_WEIGHTS) else 1.0

                # Add the vote weight to that specific defense level's total score
                level_scores[level] += weight
            else:
                evidence["defense_level"] = None
                identifier = item.get("text", str(item)[:100])
                missing_mappings[sublevel].append(identifier)

        # Determine the final label using the weighted scores
        if level_scores:
            # max() automatically finds the level with the highest weighted score
            # In the rare event of a perfect tie, it naturally favors the one that was processed first (higher rank)
            winning_level = max(level_scores, key=level_scores.get)

            item["predicted_defense_level"] = winning_level
            item["level_scores"] = dict(
                level_scores
            )  # Save the math breakdown in the JSON!
        else:
            item["predicted_defense_level"] = None
            item["level_scores"] = {}

        processed_count += 1

    # Save the updated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\nSuccessfully mapped and weighted levels for {processed_count} items.")
    print(f"Data saved to {output_file}")

    if missing_mappings:
        print(
            "\nWARNING: The LLM generated these unrecognized sublabels which could not be mapped:"
        )
        for missing, locations in missing_mappings.items():
            loc_str = " | ".join(str(loc).replace("\n", " ") for loc in locations[:2])
            more_str = (
                f" (...and {len(locations) - 2} more)" if len(locations) > 2 else ""
            )
            print(f" - '{missing}' -> Found in: {loc_str}{more_str}")


if __name__ == "__main__":
    # Specify your input and output file names here
    INPUT_JSON = "extracted_evidence_train_data_finalall.json"
    OUTPUT_JSON = "final_mapped_data_with_weights_directvote.json"

    map_sublevels_to_labels(INPUT_JSON, OUTPUT_JSON)
