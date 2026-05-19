import os
import json
import argparse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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


def build_prompt_from_item(item, handbook_text="", few_shot=""):
    dialogue_lines = []
    for turn in item.get("dialogue", []):
        speaker = turn.get("speaker", "").capitalize()
        text = turn.get("text", "")
        dialogue_lines.append(f"{speaker}: {text}")
    formatted_dialogue = "\n".join(dialogue_lines)

    system_instruction = f"""
You are an expert clinical psychologist and data annotator. Your task is to analyze dialogues and classify the psychological defense mechanism used in the 'current_text_to_classify' based on the Defense Mechanisms Rating Scales (DMRS) hierarchy.

You must assign exactly one label from the list below:
{LABEL_DESCRIPTIONS}

Your classification must be grounded in the DMRS hierarchy given below. This following comprehensive HANDBOOK serves as your core classifying guideline:
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
7. Output strict JSON matching the requested schema.
"""

    prompt = (
        system_instruction
        + "DIALOGUE:\n"
        + formatted_dialogue
        + "\n\ncurrent_text_to_classify: "
        + item.get("current_text", "")
    )
    return prompt


def try_parse_json_from_text(text):
    try:
        return json.loads(text)
    except Exception:
        import re

        m = re.search(r"\{(?:[^{}]|(?R))*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def main():
    # Set Google Cloud credentials to use the provided service account
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        r"c:\Users\USER\Desktop\DONTTOUCH\project-ade5f3ce-e086-4d4c-91c-1e8d1c34cf7e.json"
    )

    parser = argparse.ArgumentParser(
        description="Call Vertex AI endpoint for single-case DMRS prediction using genai SDK."
    )
    parser.add_argument("--item", required=True, help="Path to single item JSON")
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Gemini Model ID to use (e.g. gemini-3.1-pro-preview or gemini-1.5-pro-preview-0409)",
    )
    parser.add_argument(
        "--project_id",
        default="project-ade5f3ce-e086-4d4c-91c",
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Vertex AI location (default global for 3.1)",
    )
    parser.add_argument(
        "--handbook", default="", help="Path to handbook file (optional)"
    )
    parser.add_argument(
        "--fewshot", default="", help="Path to few-shot examples (optional)"
    )
    parser.add_argument(
        "--out_prefix", default="vertex_single", help="Output filename prefix"
    )
    args = parser.parse_args()

    with open(args.item, "r", encoding="utf-8") as f:
        item = json.load(f)

    handbook_text = ""
    few_shot = ""
    if args.handbook and os.path.exists(args.handbook):
        with open(args.handbook, "r", encoding="utf-8") as h:
            handbook_text = h.read()
    if args.fewshot and os.path.exists(args.fewshot):
        with open(args.fewshot, "r", encoding="utf-8") as fh:
            few_shot = fh.read()

    prompt = build_prompt_from_item(item, handbook_text, few_shot)

    # Initialize the genai client using Vertex AI
    client = genai.Client(
        vertexai=True, project=args.project_id, location=args.location
    )

    print(f"Calling Gemini model '{args.model}' in {args.location}...")
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DefensePrediction,
                temperature=0.2,
            ),
        )
    except Exception as e:
        print("Vertex predict failed:", e)
        return

    raw_path = f"{args.out_prefix}_singleRAW.json"
    raw_response = {
        "text": response.text,
    }
    with open(raw_path, "w", encoding="utf-8") as rf:
        json.dump(raw_response, rf, indent=2, ensure_ascii=False)

    parsed = try_parse_json_from_text(response.text)

    out_path = f"{args.out_prefix}_single.json"
    if parsed:
        with open(out_path, "w", encoding="utf-8") as outf:
            json.dump(parsed, outf, indent=2, ensure_ascii=False)
        print("Parsed JSON saved to:", out_path)
    else:
        print("Could not parse JSON. Raw text saved to RAW file.")


if __name__ == "__main__":
    main()
