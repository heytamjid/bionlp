import json
import os


def merge_predictions(base_file, resolved_file, output_file):
    print(f"Loading base predictions from {base_file}...")
    with open(base_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    print(f"Loading resolved predictions from {resolved_file}...")
    with open(resolved_file, "r", encoding="utf-8") as f:
        resolved_data = json.load(f)

    # Create a mapping of id -> resolved label
    resolved_map = {item["id"]: item["label"] for item in resolved_data}

    updated_count = 0

    # Iterate through the base data and update the label if it exists in the resolved map
    for item in base_data:
        sample_id = item.get("id")
        if sample_id in resolved_map:
            # Replace with the resolved label
            item["label"] = resolved_map[sample_id]
            updated_count += 1

    # Save the merged result
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(base_data, f, indent=4)

    print("\n✅ Merge Complete!")
    print(f"Total samples in base file: {len(base_data)}")
    print(f"Resolved samples provided:  {len(resolved_data)}")
    print(f"Actual samples updated:     {updated_count}")
    print(f"Merged output saved to:     {output_file}")


if __name__ == "__main__":
    BASE_FILE = "prediction_geminituned_tracedmrs.json"
    RESOLVED_FILE = "prediction_not_all_3.json"
    OUTPUT_FILE = "geminiflashtunedproresolve.json"

    if not os.path.exists(BASE_FILE):
        print(f"Error: {BASE_FILE} not found.")
    elif not os.path.exists(RESOLVED_FILE):
        print(f"Error: {RESOLVED_FILE} not found.")
    else:
        merge_predictions(BASE_FILE, RESOLVED_FILE, OUTPUT_FILE)
