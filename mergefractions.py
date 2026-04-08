import os
import json
import glob


def merge_fractional_jsons(
    fraction_dir="fraction",
    output_file="extracted_evidence_train_data_finalall.json",
    max_test_id=472,
):
    # Ensure the directory exists to avoid errors
    if not os.path.exists(fraction_dir):
        print(
            f"Error: The directory '{fraction_dir}/' does not exist. Please create it and add your JSONs."
        )
        return

    # Find all JSON files in the fraction directory
    json_files = glob.glob(os.path.join(fraction_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in '{fraction_dir}/'.")
        return

    print(
        f"Found {len(json_files)} JSON files in '{fraction_dir}/'. Starting merge...\n"
    )

    merged_data = {}
    duplicate_count = 0

    # 1. Merge files and handle duplicates
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                item_id = item.get("id")

                if not item_id:
                    print(
                        f"Warning: Found an item without an 'id' in {file_path}. Skipping."
                    )
                    continue

                if item_id in merged_data:
                    print(
                        f"Notice: Duplicate found for '{item_id}' (in {os.path.basename(file_path)}). Keeping the first version encountered."
                    )
                    duplicate_count += 1
                else:
                    merged_data[item_id] = item

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 2. Check for missing test cases
    # Generate the expected IDs: "test_00000" to "test_00472"
    expected_ids = {f"test_{str(i).zfill(5)}" for i in range(max_test_id + 1)}
    found_ids = set(merged_data.keys())

    missing_ids = expected_ids - found_ids

    print("\n" + "=" * 40)
    print("MERGE SUMMARY")
    print("=" * 40)
    print(f"Total unique items merged: {len(merged_data)}")
    print(f"Total duplicates ignored: {duplicate_count}")

    if missing_ids:
        print(
            f"\nWARNING: Missing {len(missing_ids)} test cases between test_00000 and test_{str(max_test_id).zfill(5)}:"
        )
        # Sort the missing IDs to make them easy to read
        for missing_id in sorted(missing_ids):
            print(f" - {missing_id}")
    else:
        print(
            f"\nPerfect! All test cases from test_00000 to test_{str(max_test_id).zfill(5)} are accounted for."
        )

    # 3. Sort the final data by ID and save
    # This ensures your final JSON is beautifully organized, regardless of what order the files were read
    final_list = sorted(merged_data.values(), key=lambda x: x.get("id", ""))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=4)

    print(f"\nSuccessfully saved final merged dataset to: {output_file}")


if __name__ == "__main__":
    # You can change the folder name or max ID here if needed later
    merge_fractional_jsons(
        fraction_dir="fraction",
        output_file="extracted_evidence_train_data_finalall.json",
        max_test_id=472,
    )
