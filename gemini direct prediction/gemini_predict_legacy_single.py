import os
import json
import time
import argparse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Keep the same label descriptions and schema for consistency
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


MODEL_ID = "gemini-3.1-pro-preview"


def build_prompt(handbook_text: str, few_shot: str, item: dict) -> str:
    # Build a single long prompt that contains handbook, few-shot examples and the dialogue
    # The format follows the original script: system instruction + formatted dialogue + current_text_to_classify
    system_instruction = f"""
You are an expert clinical psychologist and data annotator. Your task is to analyze dialogues and classify the psychological defense mechanism used in the 'current_text_to_classify' based on the Defense Mechanisms Rating Scales (DMRS) hierarchy.

You must assign exactly one label from the list below:
{LABEL_DESCRIPTIONS}

This comprehensive HANDBOOK serves as your core classifying guideline:
{handbook_text}

Here are some examples of how to reason through the task:
{few_shot}

CORE INSTRUCTIONS:
1. Primacy of Context: Always read the preceding dialogue to understand what triggered the 'current_text_to_classify'.
2. Function-Oriented: Ask yourself, "What psychological goal is the speaker trying to achieve?"
3. Handbook Grounded: Match the behavior to the specific criteria in the DMRS Handbook. Reason through why specific criteria are met.
4. You must maintain hierarchical integrity — explicitly reason through why the classification does not drift into higher (more adaptive) or lower (more pathological) levels by verifying that all exclusionary criteria for the selected level are met.
5. Distinguish Emotion from Defense: Saying "I am sad" is Level 0. A defense requires distortion, avoidance, or transformation.
6. Always pick the single most accurate label (0–8) from the LABEL REFERENCE above.
7. Output strict JSON matching the requested schema (clinical_reasoning, defense_level, label) as a top-level JSON object.
"""

    # Format dialogue
    dialogue_lines = []
    for turn in item.get("dialogue", []):
        speaker = turn.get("speaker", "").capitalize()
        text = turn.get("text", "")
        dialogue_lines.append(f"{speaker}: {text}")

    formatted_dialogue = "\n".join(dialogue_lines)

    user_prompt = (
        f"{system_instruction}\n\n"
        f"DIALOGUE:\n{formatted_dialogue}\n\n"
        f"current_text_to_classify: {item.get('current_text', '')}\n\n"
        f'Return a single JSON object matching the schema: {{"clinical_reasoning":{{...}}, "defense_level":<int>, "label":<str>}}\n'
    )

    return user_prompt


def save_json_safe(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def extract_json_from_text(text: str):
    # Try direct load first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first JSON object in text
    import re

    m = re.search(r"\{(?:[^{}]|(?R))*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-case DMRS prediction (no caching)."
    )
    parser.add_argument(
        "--item",
        required=True,
        help="Path to a JSON file containing a single item (dict) to classify.",
    )
    parser.add_argument(
        "--handbook",
        default="Psychological Defense Mechanism Coding Handbook.md",
        help="Path to the handbook file to inline into prompt.",
    )
    parser.add_argument(
        "--fewshot",
        default="few_shot_examples.txt",
        help="Path to few-shot examples to inline into prompt.",
    )
    parser.add_argument(
        "--out_prefix", default="single_prediction", help="Output filename prefix"
    )
    args = parser.parse_args()

    # Load item
    if not os.path.exists(args.item):
        print("Item file not found:", args.item)
        return
    with open(args.item, "r", encoding="utf-8") as f:
        item = json.load(f)

    # Load handbook and few-shot if available; include raw text even if missing warn
    handbook_text = ""
    few_shot = ""
    if os.path.exists(args.handbook):
        with open(args.handbook, "r", encoding="utf-8") as f:
            handbook_text = f.read()
    else:
        print(
            f"Warning: Handbook file not found at {args.handbook}. Proceeding without it."
        )

    if os.path.exists(args.fewshot):
        with open(args.fewshot, "r", encoding="utf-8") as f:
            few_shot = f.read()
    else:
        print(
            f"Warning: Few-shot examples file not found at {args.fewshot}. Proceeding without it."
        )

    prompt = build_prompt(handbook_text, few_shot, item)

    # Initialize client
    client = genai.Client()

    # Call model without using caches
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DefensePrediction,
                temperature=0.1,
            ),
        )

        raw_text = getattr(response, "text", None) or (
            response.candidates[0].content.parts[0].text
            if response.candidates and response.candidates[0].content.parts
            else ""
        )

    except Exception as e:
        print("Model call failed:", e)
        return

    # Save raw output
    raw_path = f"{args.out_prefix}_singleRAW.json"
    try:
        # try to dump response object as dict if possible
        raw_dict = None
        try:
            raw_dict = (
                response.model_dump() if hasattr(response, "model_dump") else None
            )
        except Exception:
            raw_dict = None

        if raw_dict:
            save_json_safe(raw_path, raw_dict)
        else:
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
    except Exception as e:
        print("Failed to save raw output:", e)

    # Try to parse JSON reasoning from the returned text
    parsed = extract_json_from_text(raw_text)
    out_path = f"{args.out_prefix}_single.json"
    if parsed:
        save_json_safe(out_path, parsed)
        print("Parsed JSON saved to:", out_path)
    else:
        # If parsing fails, save raw_text also as fallback
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print("Could not parse JSON from model text; raw text saved to:", out_path)

    print("Raw output saved to:", raw_path)


if __name__ == "__main__":
    main()
