import os
import json
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Define the Structured Output Schema
# ==============================================================================
class DMRSItemMatch(BaseModel):
    dmrs_q_item: str = Field(
        description="The exact DMRS-Q Item number and text from the handbook (e.g., 'ITEM 45: At times when expressing...')."
    )
    match_justification: str = Field(
        description="Detailed explanation of why this specific DMRS-Q item matches the conversation based on the handbook's definition, scenario, examples, and distinctions."
    )
    sublevel_name: str = Field(
        description="The specific defense mechanism name from the handbook (e.g., 'Passive Aggression', 'Rationalization', 'Devaluation' etc.). Do NOT use the 0-8 level numbers."
    )


class ExtractionResult(BaseModel):
    top_3_matches: list[DMRSItemMatch] = Field(
        description="Exactly 3 most matching DMRS-Q items and their details."
    )


# ==============================================================================
# 2. Setup the Instructions & Load Handbook
# ==============================================================================
# Make sure the markdown file is in the same directory
try:
    with open("handbook_stipped.md", "r", encoding="utf-8") as f:
        HANDBOOK_TEXT = f.read()
except FileNotFoundError:
    print(
        "Error: Could not find 'STIPPED Psychological Defense Mechanism Coding Handbook.md'."
    )
    exit(1)

SYSTEM_INSTRUCTION = f"""
You are an expert clinical psychologist and data annotator. 

Below is the Psychological Defense Mechanism Coding Handbook:
==================================================================
{HANDBOOK_TEXT}
==================================================================

YOUR TASK:
You will be provided with a dialogue and a 'current_text_to_classify'. 
You must SCAN THROUGH THE HANDBOOK and pick out the MOST MATCHING 3 DMRS-Q items for the given text.

CRITICAL INSTRUCTIONS:
1. FOCUS EXPLICITLY ON THE DMRS-Q ITEMS. Compare the dialogue's scenario against the specific behaviors described in the DMRS-Q items.
2. Identify the corresponding defense mechanism (such as 'Splitting', 'Acting Out', 'Humor', 'Repression').
3. Base your selections strictly on the Definitions, Given Scenarios, Examples, and Distinctions provided in the handbook.
5. Provide exactly 3 matches. If one defense is blatantly obvious, provide the top 3 DMRS-Q items that best capture the nuances of the interaction.
"""

MODEL_ID = "gemini-3.1-pro-preview"


# ==============================================================================
# 3. Main Extraction Function
# ==============================================================================
def extract_dmrs_evidence(input_json_path: str, output_json_path: str):
    # Initialize the client. Adjust project/location or remove vertexai=True
    # depending on your specific authentication setup (e.g., API key vs Vertex).
    client = genai.Client(
        vertexai=True, project="project-ade5f3ce-e086-4d4c-91c", location="global"
    )

    # Load dataset
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # REVERSE THE DATA LIST HERE
    data = data[::-1]

    # Support resuming from existing output file
    results = []
    processed_ids = set()
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_ids = {item.get("id") for item in results if "id" in item}
            print(
                f"Resuming from {output_json_path}. {len(processed_ids)} items already processed."
            )
        except Exception as e:
            print(f"Warning: Could not load existing progress: {e}. Starting fresh.")

    # Process the dataset
    new_items_processed = 0
    for item in data:
        item_id = item.get("id")
        if item_id in processed_ids:
            continue

        print(f"Processing Dialogue ID: {item_id}")

        # Format the user prompt
        dialogue_lines = []
        for turn in item.get("dialogue", []):
            speaker = turn.get("speaker", "").capitalize()
            text = turn.get("text", "")
            dialogue_lines.append(f"{speaker}: {text}")

        formatted_dialogue = "\n".join(dialogue_lines)
        target_text = item.get("current_text", "")

        user_prompt = (
            f"DIALOGUE:\n{formatted_dialogue}\n\n"
            f"TEXT TO ANALYZE:\n{target_text}\n\n"
            f"TASK: Based on the handbook provided in your system instructions, extract the top 3 DMRS-Q items that best match the TEXT TO ANALYZE in the context of the DIALOGUE."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Call the model
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                        temperature=0.1,  # Low temperature for analytical extraction
                    ),
                )

                # Parse the JSON response
                prediction = json.loads(response.text)

                # Store the results back into the item dictionary
                item["extracted_dmrs_evidence"] = prediction["top_3_matches"]

                results.append(item)
                processed_ids.add(item_id)
                new_items_processed += 1

                print(
                    f"Successfully extracted {len(prediction['top_3_matches'])} items."
                )
                break  # Exit the retry loop upon successful execution

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {item_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Sleep for 2 seconds before retrying
                else:
                    print(f"Giving up on {item_id} after {max_retries} attempts.")

        # Checkpoint: Save every 5 newly processed items
        if new_items_processed > 0 and new_items_processed % 5 == 0:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"--- Checkpoint saved: {len(results)} total items processed ---")

    # Save the final results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        print(f"Successfully saved extracted evidence data to {output_json_path}")


if __name__ == "__main__":
    # Example usage:
    # Ensure your input JSON is structured properly (list of dicts with 'id', 'dialogue', 'current_text')
    extract_dmrs_evidence(
        "input_data/test.json", "extracted_evidence_train_data_REVERSE.json"
    )
