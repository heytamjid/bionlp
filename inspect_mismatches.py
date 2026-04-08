import json
import os


def separate_mismatched_predictions(input_file: str):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    not_all_3_match_list = []
    all_3_different_list = []

    for item in data:
        extracted_evidence = item.get("extracted_dmrs_evidence", [])

        if not extracted_evidence:
            continue

        mapped_levels = [
            evidence.get("defense_level") for evidence in extracted_evidence
        ]

        # We only care about items that have exactly 3 predictions
        if len(mapped_levels) == 3:
            unique_len = len(set(mapped_levels))

            # If they don't all match (i.e. unique levels > 1)
            if unique_len > 1:
                not_all_3_match_list.append(item)

            # If all 3 are completely different (i.e. unique levels == 3)
            if unique_len == 3:
                all_3_different_list.append(item)

    # Save results
    output_not_all_3 = "inspection_not_all_3_match.json"
    output_all_diff = "inspection_all_3_different.json"

    with open(output_not_all_3, "w", encoding="utf-8") as f:
        json.dump(not_all_3_match_list, f, indent=4)

    with open(output_all_diff, "w", encoding="utf-8") as f:
        json.dump(all_3_different_list, f, indent=4)

    print("\nExtraction Complete:")
    print(
        f"- Saved {len(not_all_3_match_list)} items where 'Not all 3 match' to {output_not_all_3}"
    )
    print(
        f"- Saved {len(all_3_different_list)} items where 'All 3 different' to {output_all_diff}"
    )


if __name__ == "__main__":
    # Ensure this matches the output of your directvoting.py script
    # so we have access to the mapped "defense_level" on each evidence item
    INPUT_JSON = "final_mapped_data_with_weights_directvote.json"

    separate_mismatched_predictions(INPUT_JSON)
