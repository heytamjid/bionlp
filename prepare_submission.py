import json
import os


def prepare_submission(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading mapped data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    submission = []

    for item in data:
        # Extract the 'id' (defaulting to empty string if missing)
        item_id = item.get("id", "")

        # Get the predicted defense level. If it's None or missing, default to 0
        predicted_level = item.get("predicted_defense_level")
        label = int(predicted_level) if predicted_level is not None else 0

        submission.append({"id": item_id, "label": label})

    print(f"Writing {len(submission)} predictions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    # Define input and output files
    INPUT_JSON = "final_mapped_data_with_weights_directvote.json"
    OUTPUT_JSON = "prediction.json"

    prepare_submission(INPUT_JSON, OUTPUT_JSON)
