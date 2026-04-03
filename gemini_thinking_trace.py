import os
import json
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Define the Structured Output Schema (Forces Chain-of-Thought)
# ==============================================================================

LABEL_DESCRIPTIONS = """
LABEL REFERENCE (Defense Mechanism Rating Scale Tiers):
  0 = No Defense / Neutral Utterance  — simple, undefended statement; no psychological distortion
  1 = Action Defense Level            — Acting Out / Help-Rejecting Complaining / Passive Aggression
  2 = Major Image-distorting Defense  — Splitting / Projective Identification
  3 = Disavowal Defense Level         — Denial / Projection / Rationalization / Autistic Fantasy
  4 = Minor Image-distorting Defense  — Devaluation / Idealization / Omnipotence
  5 = Neurotic Defense Level          — Displacement / Dissociation / Reaction Formation / Repression
  6 = Obsessional Defense Level       — Intellectualization / Isolation of Affects / Undoing
  7 = Highly Adaptive Defense Level   — Affiliation / Altruism / Anticipation / Humor /
                                        Self-Assertion / Self-Observation / Sublimation / Suppression
  8 = Need More Information           — Evidence suggests a defense but is insufficient to confirm any tier
"""


class ClinicalReasoning(BaseModel):
    context_trigger: str = Field(
        description="Description of the stressor in the dialogue."
    )
    psychological_goal: str = Field(
        description="What the speaker is trying to achieve/avoid."
    )
    handbook_alignment: str = Field(
        description="Specific evidence from the handbook that justifies the label."
    )
    differential_diagnosis: str = Field(
        description="Why this isn't a higher or lower-level defense."
    )


class DefensePrediction(BaseModel):
    clinical_reasoning: ClinicalReasoning
    defense_level: int = Field(description="The numeric defense level (0-8).")
    label: str = Field(description="The specific name of the defense mechanism used.")


# ==============================================================================
# 2. Setup the System Prompt & Few-Shot Examples
# ==============================================================================
with open(
    "Psychological Defense Mechanism Coding Handbook.md", "r", encoding="utf-8"
) as f:
    HANDBOOK_TEXT = f.read()

# Few-shot examples guide the model's reasoning style.
with open("few_shot_examples.txt", "r", encoding="utf-8") as f:
    FEW_SHOT_EXAMPLES = f.read()

SYSTEM_INSTRUCTION = f"""
You are an expert clinical psychologist and data annotator. Your task is to analyze dialogues and generate the exact clinical reasoning (thought trace) that perfectly justifies the PROVIDED psychological defense mechanism level for the 'current_text_to_classify', based on the Defense Mechanisms Rating Scales (DMRS) hierarchy.

The correct true label level will be given to you in the prompt. You must explain *why* it is the correct level.
{LABEL_DESCRIPTIONS}

Your classification reasoning must be grounded in the DMRS hierarchy given below. This following comprehensive HANDBOOK serves as your core classifying guideline:
{HANDBOOK_TEXT}

Here are some examples of how to reason through the task:
{FEW_SHOT_EXAMPLES}

CORE INSTRUCTIONS:
1. Primacy of Context: Always read the preceding dialogue to understand what triggered the 'current_text_to_classify'.
2. Function-Oriented: Ask yourself, "What psychological goal is the speaker trying to achieve?"
3. Handbook Grounded: Match the behavior to the specific criteria in the DMRS Handbook. Reason through why specific criteria are met pointing towards the provided correct label.
4. Hierarchical Integrity: You must maintain hierarchical integrity — explicitly reason through why the classification does not drift into higher or lower levels by verifying that all exclusionary criteria for the selected level are met.
5. Emulate the Correct Path: Provide the thought process as if you independently arrived at the provided correct label.
8. Output strict JSON matching the requested schema, ensuring your output 'defense_level' matches the provided correct level.
"""

# The model to use. Context caching is supported on Gemini 2.5 Pro Preview.
MODEL_ID = "gemini-3.1-pro-preview"

# Minimum token count required for context caching to be cost-effective.
# Gemini enforces a minimum of 32,768 tokens for cached content.
CACHE_TTL = "7200s"  # Cache lives for 2 hour; adjust as needed.


# ==============================================================================
# 3. Main Annotation Function
# ==============================================================================
def annotate_dataset(input_json_path: str, output_json_path: str):
    # Initialize the client for Vertex AI.
    # We explicitly provide the project_id found in your service account JSON.
    client = genai.Client(
        vertexai=True, project="project-ade5f3ce-e086-4d4c-91c", location="global"
    )

    # Load your dataset
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    print("Creating Context Cache for the Handbook...")

    # Log the exact system instructions being sent to the cache for inspection
    with open(
        "inspection_system_instruction_log.txt", "w", encoding="utf-8"
    ) as log_file:
        log_file.write("========== SYSTEM INSTRUCTION (CACHED) ==========\n")
        log_file.write(
            "system_instruction: You are an expert clinical psychologist and data annotator classifying dialogues based on the Defense Mechanisms Rating Scales (DMRS).\n\n"
        )
        log_file.write("contents:\n")
        log_file.write(SYSTEM_INSTRUCTION + "\n")
        log_file.write("=================================================\n")

    # AS ALREADY CREATED, USING VIA cacheRef
    # try:
    #     # Pass the massive SYSTEM_INSTRUCTION string into contents to cache it,
    #     # and set a concise system_instruction for the model's persona.
    #     cache = client.caches.create(
    #         model=MODEL_ID,
    #         config=types.CreateCachedContentConfig(
    #             system_instruction="You are an expert clinical psychologist and data annotator classifying dialogues based on the Defense Mechanisms Rating Scales (DMRS).",
    #             contents=[SYSTEM_INSTRUCTION],
    #             display_name="dmrs-handbook-cache",
    #             ttl=CACHE_TTL,
    #         ),
    #     )
    #     print(f"Cache created successfully: {cache.name}")
    # except Exception as e:
    #     print(f"Failed to create cache: {e}")
    #     return

    # or if alrady has
    cacheRef = "projects/942972453935/locations/global/cachedContents/5700403225157435392"  # cache.name
    print("\n Your Cache REF ISSSSS \n")
    print(cacheRef)

    # Process the dataset
    new_items_processed = 0
    for item in data:
        item_id = item.get("id")
        if item_id in processed_ids:
            continue

        print(f"Processing Dialogue ID: {item_id}")

        # Format the user prompt iteratively
        dialogue_lines = []
        for turn in item.get("dialogue", []):
            speaker = turn.get("speaker", "").capitalize()
            text = turn.get("text", "")
            dialogue_lines.append(f"{speaker}: {text}")

        formatted_dialogue = "\n".join(dialogue_lines)

        user_prompt = (
            f"{formatted_dialogue}\n\n"
            f"current_text_to_classify: {item.get('current_text', '')}\n\n"
            f"CORRECT_DEFENSE_LEVEL: {item.get('label')}\n"
            f"Your task is to generate the ClinicalReasoning (thinking trace) that correctly concludes that the defense level is {item.get('label')}."
        )

        # Log the exact prompt being sent to the model for inspection
        with open("inspection_prompt_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"========== PROMPT FOR {item_id} ==========\n")
            log_file.write(user_prompt + "\n")
            log_file.write("==============================================\n\n")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Call the model using the cached content
                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cacheRef,  # cache.name,  # Reference the active cache here
                        response_mime_type="application/json",
                        response_schema=DefensePrediction,
                        temperature=0.1,  # Low temperature for classification consistency
                        thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.HIGH
                        ),
                    ),
                )

                # The SDK automatically parses the JSON if it matches the schema
                prediction = json.loads(response.text)

                item["predicted_label"] = prediction["label"]
                item["predicted_defense_level"] = prediction["defense_level"]
                item["clinical_reasoning"] = prediction["clinical_reasoning"]

                # Simple raw dump of everything the model sent back, converted to a cleaned dictionary
                parts_dump = []
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        try:
                            # Safely attempt to convert the Part object to a dict to see its keys
                            part_dict = part.model_dump(exclude_none=True)
                        except Exception:
                            # Fallback if it's not a standard Pydantic model
                            part_dict = vars(part)

                        # If this part is just the final text payload, skip it so we don't duplicate the JSON
                        if part_dict.get("text") is not None:
                            continue

                        parts_dump.append(part_dict)

                # Store the dump directly as a JSON object list instead of raw bracket strings
                item["model_thinking_dump"] = parts_dump

                results.append(item)
                processed_ids.add(item_id)
                new_items_processed += 1

                print(
                    f"Predicted Label: {prediction['label']} (Defense Level: {prediction['defense_level']})\n"
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
        print(f"Successfully saved annotated data to {output_json_path}")

    # ==========================================================================
    # CLEANUP: Delete the cache to prevent ongoing storage charges
    # ==========================================================================
    try:
        client.caches.delete(name=cacheRef)
        print("Context cache deleted successfully.")
    except Exception as e:
        print(
            f"Warning: Failed to delete cache. You may need to delete it manually via the API or Google AI Studio. Error: {e}"
        )


if __name__ == "__main__":
    # Example usage:
    annotate_dataset("input_data/train.json", "annotated_train_data.json")
