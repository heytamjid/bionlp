import json
import os


def format_for_submission(
    input_json_path="annotated_train_data.json", output_json_path="prediction.json"
):
    """
    Reads the predicted data and formats it into the exact schema required for submission.
    """
    if not os.path.exists(input_json_path):
        print(f"Error: {input_json_path} not found.")
        return

    # Load the processed dataset with predictions
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = []

    # Extract only the `id` and the final predicted `label` (integer tier)
    for item in data:
        # Based on your gemini_predict.py, the integer level is stored in predicted_defense_level
        # Fallback to 0 if something is missing
        predicted_level = item.get("predicted_defense_level", 0)

        predictions.append({"id": item["id"], "label": int(predicted_level)})

    # Save the predictions for submission
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(
        f"Successfully saved {len(predictions)} predictions to {output_json_path} for submission."
    )


if __name__ == "__main__":
    # You can change the input file here if your predictions are saved in a different file (e.g., test.json output)
    format_for_submission("annotated_train_data.json", "prediction.json")
