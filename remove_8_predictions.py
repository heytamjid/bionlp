import json


def remove_eights(filepath="raw_predictions_not_all_3.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    initial_count = len(data)

    # Keep only the items that do NOT have a predicted_label of 8
    # (Or you could keep them if they are genuinely 8 and not just parsing errors,
    # but based on the context, we want to re-infer all 8s)
    filtered_data = [item for item in data if item.get("predicted_label") != 8]

    final_count = len(filtered_data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Loaded {initial_count} predictions.")
    print(f"Removed {initial_count - final_count} predictions with label 8.")
    print(f"Saved {final_count} remaining predictions back to {filepath}.")


if __name__ == "__main__":
    remove_eights()
