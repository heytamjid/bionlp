import json
import os


def merge_predictions_filtered(base_file, resolved_file, filter_file, output_file):
    print(f"Loading filter file from {filter_file}...")
    with open(filter_file, "r", encoding="utf-8") as f:
        filter_data = json.load(f)

    # Extract the IDs that we are actually allowed to update
    target_ids = {item["id"] for item in filter_data if "id" in item}
    print(f"Found {len(target_ids)} target IDs to update.")

    print(f"Loading resolved predictions from {resolved_file}...")
    with open(resolved_file, "r", encoding="utf-8") as f:
        resolved_data = json.load(f)

    # Create a mapping of id -> resolved label ONLY for the target IDs
    resolved_map = {}
    for item in resolved_data:
        sample_id = item.get("id")
        if sample_id in target_ids:
            resolved_map[sample_id] = item["label"]

    print(f"Found {len(resolved_map)} resolved predictions that match the target IDs.")

    print(f"Loading base predictions from {base_file}...")
    with open(base_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    updated_count = 0

    # Iterate through the base data and update the label if it exists in the filtered resolved map
    for item in base_data:
        sample_id = item.get("id")
        if sample_id in resolved_map:
            # Replace with the resolved label
            item["label"] = resolved_map[sample_id]
            updated_count += 1

    # Save the merged result
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(base_data, f, indent=4)

    print("\n✅ Filtered Merge Complete!")
    print(f"Total samples in base file: {len(base_data)}")
    print(f"Actual samples updated:     {updated_count}")
    print(f"Merged output saved to:     {output_file}")


if __name__ == "__main__":
    BASE_FILE = "prediction_geminituned_tracedmrs.json"
    RESOLVED_FILE = "prediction_not_all_3.json"
    FILTER_FILE = "inspection_all_3_different.json"
    OUTPUT_FILE = "geminiflashtunedproresolve_all3_diff.json"

    if not os.path.exists(BASE_FILE):
        print(f"Error: {BASE_FILE} not found.")
    elif not os.path.exists(RESOLVED_FILE):
        print(f"Error: {RESOLVED_FILE} not found.")
    elif not os.path.exists(FILTER_FILE):
        print(f"Error: {FILTER_FILE} not found.")
    else:
        merge_predictions_filtered(BASE_FILE, RESOLVED_FILE, FILTER_FILE, OUTPUT_FILE)
