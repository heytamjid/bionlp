import json
import os
import sys


def build_prompt(handbook_text: str, few_shot: str, item: dict) -> str:
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


def main():
    if len(sys.argv) < 2:
        print("Usage: save_prompt_for_item.py <path_to_tmp_item_json> [out_path]")
        sys.exit(1)

    item_path = sys.argv[1]
    out_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "single_test_prompt.txt"
        )
    )

    if not os.path.exists(item_path):
        print("Item file not found:", item_path)
        sys.exit(2)

    with open(item_path, "r", encoding="utf-8") as f:
        item = json.load(f)

    # attempt to load handbook and few-shot if present
    repo_root = os.path.dirname(os.path.dirname(__file__))
    handbook_path = os.path.join(
        repo_root,
        "gemini direct prediction",
        "Psychological Defense Mechanism Coding Handbook.md",
    )
    few_shot_path = os.path.join(
        repo_root, "gemini direct prediction", "few_shot_examples.txt"
    )

    handbook_text = ""
    few_shot = ""
    if os.path.exists(handbook_path):
        with open(handbook_path, "r", encoding="utf-8") as h:
            handbook_text = h.read()
    if os.path.exists(few_shot_path):
        with open(few_shot_path, "r", encoding="utf-8") as f:
            few_shot = f.read()

    prompt = build_prompt(handbook_text, few_shot, item)
    with open(out_path, "w", encoding="utf-8") as o:
        o.write(prompt)

    print("Saved prompt to:", out_path)


if __name__ == "__main__":
    main()
