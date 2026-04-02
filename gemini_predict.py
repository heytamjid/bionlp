import os
import json
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


class DefensePrediction(BaseModel):
    # The model MUST generate this reasoning string BEFORE it predicts the label.
    reasoning: str = Field(
        description="Step-by-step reasoning based on the DMRS handbook. "
        "1. Analyze context. 2. Identify psychological function (evading pain, discharging emotion, etc.). "
        "3. Evaluate against DMRS hierarchy. 4. Determine final defense."
    )
    label: int = Field(
        description=(
            "The integer label of the defense mechanism tier. Strictly output a number for this field. Choose from:\n"
            "  0 = No Defense / Neutral Utterance\n"
            "  1 = Action Defense Level (Acting Out / Help-Rejecting Complaining / Passive Aggression)\n"
            "  2 = Major Image-distorting Defense Level (Splitting / Projective Identification)\n"
            "  3 = Disavowal Defense Level (Denial / Projection / Rationalization / Autistic Fantasy)\n"
            "  4 = Minor Image-distorting Defense Level (Devaluation / Idealization / Omnipotence)\n"
            "  5 = Neurotic Defense Level (Displacement / Dissociation / Reaction Formation / Repression)\n"
            "  6 = Obsessional Defense Level (Intellectualization / Isolation of Affects / Undoing)\n"
            "  7 = Highly Adaptive Defense Level (Affiliation / Altruism / Anticipation / Humor / "
            "Self-Assertion / Self-Observation / Sublimation / Suppression)\n"
            "  8 = Need More Information (evidence suggests a defense but insufficient to confirm any tier)"
        )
    )


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
You are an expert clinical psychologist and data annotator. Your task is to analyze dialogues and classify the psychological defense mechanism used in the 'current_text' based on the Defense Mechanisms Rating Scales (DMRS) hierarchy.

You must assign exactly one label from the list below:
{LABEL_DESCRIPTIONS}

Here is the comprehensive handbook you must follow:
{HANDBOOK_TEXT}

Here are some examples of how to reason through the task:
{FEW_SHOT_EXAMPLES}

CORE INSTRUCTIONS:
1. Primacy of Context: Always read the preceding dialogue to understand what triggered the 'current_text'.
2. Function-Oriented: Ask yourself, "What psychological goal is the speaker trying to achieve?"
3. Distinguish Emotion from Defense: Saying "I am sad" is Level 0. A defense requires distortion, avoidance, or transformation.
4. Always pick the single most accurate label (0–8) from the LABEL REFERENCE above.
5. Output strict JSON matching the requested schema. Reason step-by-step before selecting the label.
"""

# The model to use. Context caching is supported on Gemini 2.5 Pro Preview.
MODEL_ID = "gemini-3.1-pro-preview"

# Minimum token count required for context caching to be cost-effective.
# Gemini enforces a minimum of 32,768 tokens for cached content.
CACHE_TTL = "7200s"  # Cache lives for 1 hour; adjust as needed.


# ==============================================================================
# 3. Main Annotation Function
# ==============================================================================
def annotate_dataset(input_json_path: str, output_json_path: str):
    # Initialize the client. Ensure GEMINI_API_KEY is in your environment variables.
    client = genai.Client()

    # Load your dataset
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    print("Creating Context Cache for the Handbook...")
    try:
        # Pass the massive SYSTEM_INSTRUCTION string into contents to cache it,
        # and set a concise system_instruction for the model's persona.
        cache = client.caches.create(
            model="gemini-3.1-pro-preview",
            config=types.CreateCachedContentConfig(
                system_instruction="You are an expert clinical psychologist and data annotator classifying dialogues based on the Defense Mechanisms Rating Scales (DMRS).",
                contents=[SYSTEM_INSTRUCTION],
                display_name="dmrs-handbook-cache",
            ),
        )
        print(f"Cache created successfully: {cache.name}")
    except Exception as e:
        print(f"Failed to create cache: {e}")
        return

    # Process the dataset
    for item in data:
        print(f"Processing Dialogue ID: {item.get('id')}")

        # Format the user prompt exactly as it appears in the JSON
        user_prompt = f"""
        Dialogue History:
        {json.dumps(item.get('dialogue', []), indent=2)}
        
        Target Utterance to Classify:
        "{item.get('current_text', '')}"
        """

        try:
            # Call the model using the cached content
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache.name,  # Reference the active cache here
                    response_mime_type="application/json",
                    response_schema=DefensePrediction,
                    temperature=0.1,  # Low temperature for classification consistency
                ),
            )

            # The SDK automatically parses the JSON if it matches the schema
            prediction = json.loads(response.text)

            item["predicted_label"] = prediction["label"]
            item["model_reasoning"] = prediction["reasoning"]
            results.append(item)

            print(f"Predicted Label: {prediction['label']}\n")

        except Exception as e:
            print(f"Error processing {item.get('id')}: {e}")

    # Save the results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        print(f"Successfully saved annotated data to {output_json_path}")

    # ==========================================================================
    # CLEANUP: Delete the cache to prevent ongoing storage charges
    # ==========================================================================
    try:
        client.caches.delete(name=cache.name)
        print("Context cache deleted successfully.")
    except Exception as e:
        print(
            f"Warning: Failed to delete cache. You may need to delete it manually via the API or Google AI Studio. Error: {e}"
        )


if __name__ == "__main__":
    # Example usage:
    # annotate_dataset("train_data.json", "annotated_train_data.json")
    pass
