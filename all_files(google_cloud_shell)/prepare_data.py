import json
import os
from google.cloud import storage

# ── Configuration ─────────────────────────────────────────────────────────────
BUCKET_NAME = os.environ["BUCKET_NAME"]
PROJECT_ID  = os.environ["PROJECT_ID"]

INPUT_TRAIN = "train.json"          # your local JSON files
INPUT_VAL   = "val.json"

OUTPUT_TRAIN = "train_gemini.jsonl"
OUTPUT_VAL   = "val_gemini.jsonl"

GCS_TRAIN_PATH = "datasets/train/train_gemini.jsonl"
GCS_VAL_PATH   = "datasets/val/val_gemini.jsonl"
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert clinical psychologist analyzing conversational data. Your task is to evaluate the psychological defense mechanism used in a target utterance based on the Defense Mechanisms Rating Scales (DMRS).

LABEL REFERENCE (Defense Mechanism Rating Scale Tiers):
  0 = No Defense / Neutral Utterance  — simple, undefended statement; no psychological distortion - Functional utterances that maintain conversational flow without engaging conflict.
  1 = Action Defense Level            — Acting Out / Help-Rejecting Complaining / Passive Aggression - Distress is released by acting on the environment instead of reflecting.
  2 = Major Image-distorting Defense  — Splitting / Projective Identification - Reduces anxiety via all-good/all-bad distortions of self or other.
  3 = Disavowal Defense Level         — Denial / Projection / Rationalization / Autistic Fantasy - Rejects threatening reality by denying, excusing, blaming, or fantasizing.
  4 = Minor Image-distorting Defense  — Devaluation / Idealization / Omnipotence - Softer distortions temporarily inflate or deflate self-esteem.
  5 = Neurotic Defense Level          — Displacement / Dissociation / Reaction Formation / Repression - Keeps unacceptable motives out of awareness; feelings surface indirectly.
  6 = Obsessional Defense Level       — Intellectualization / Isolation of Affects / Undoing - Uses excessive logic or symbolic acts to separate feelings from events.
  7 = Highly Adaptive Defense Level   — Affiliation / Altruism / Anticipation / Humor / Self-Assertion / Self-Observation / Sublimation / Suppression - Mature coping that integrates emotion and thought to channel affect constructively.
  8 = Need More Information           — Evidence suggests a defense but is insufficient to confirm any tier - Label used when an utterance is too ambiguous or lacks context.

Follow these analytical steps:
1. Context   : Analyze the preceding dialogue to understand what triggered this utterance.
2. Function  : Identify the psychological goal the speaker is trying to achieve or avoid.
3. Grounding : Match the behavior to specific DMRS criteria.
4. Hierarchy : Verify that exclusionary criteria for higher/lower levels are met.

First provide your clinical reasoning trace, then output the defense level (0–8).\
"""

SPEAKER_MAP = {
    "seeker":    "Seeker",
    "supporter": "Supporter",
}


def build_user_message(sample: dict) -> str:
    """
    Build the formatted user message from a data sample.

    Excludes the final turn that matches current_text (the target utterance)
    and formats the preceding dialogue as context.
    """
    dialogue     = sample["dialogue"]
    current_text = sample["current_text"].strip()

    # Find the index of the target turn (last occurrence of current_text)
    target_idx = None
    for i in range(len(dialogue) - 1, -1, -1):
        if dialogue[i]["text"].strip() == current_text:
            target_idx = i
            break

    # Use all turns before the target as context
    context_turns = dialogue[:target_idx] if target_idx is not None else dialogue[:-1]

    # Format dialogue block
    dialogue_lines = []
    for turn in context_turns:
        speaker = SPEAKER_MAP.get(turn["speaker"].lower(), turn["speaker"].capitalize())
        dialogue_lines.append(f"{speaker}: {turn['text']}")

    dialogue_block = "\n".join(dialogue_lines)

    user_message = (
        "## Dialogue Context\n\n"
        f"{dialogue_block}\n\n"
        f"## Target Utterance To Classify: {current_text} ..."
    )
    return user_message


def build_assistant_response(sample: dict) -> str:
    """
    Build the structured assistant response from the clinical_reasoning fields.
    """
    cr    = sample.get("clinical_reasoning", {})
    level = sample.get("predicted_defense_level", sample.get("label", "?"))

    context_trigger      = cr.get("context_trigger", "").strip()
    psychological_goal   = cr.get("psychological_goal", "").strip()
    handbook_alignment   = cr.get("handbook_alignment", "").strip()
    differential_diagnosis = cr.get("differential_diagnosis", "").strip()

    response = (
        f"Context Trigger: {context_trigger}\n\n"
        f"Psychological Goal: {psychological_goal}\n\n"
        f"Handbook Alignment: {handbook_alignment}\n\n"
        f"Differential Diagnosis: {differential_diagnosis}\n\n"
        f"Defense Level: {level}"
    )
    return response


def convert_to_gemini_format(input_file: str, output_file: str) -> None:
    """
    Convert the dataset JSON file to Gemini supervised fine-tuning JSONL format.

    Each line in the output is a JSON object with:
      - systemInstruction: the DMRS system prompt
      - contents: [user turn, model turn]
    """
    print(f"Reading {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both a bare list and a dict with a key like "data"
    if isinstance(data, dict):
        data = next(iter(data.values()))

    skipped = 0
    converted = []

    for sample in data:
        # Skip samples with missing or empty clinical_reasoning
        cr = sample.get("clinical_reasoning", {})
        if not cr or not cr.get("context_trigger"):
            skipped += 1
            continue

        user_text  = build_user_message(sample)
        model_text = build_assistant_response(sample)

        record = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {"role": "user",  "parts": [{"text": user_text}]},
                {"role": "model", "parts": [{"text": model_text}]},
            ],
        }
        converted.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Converted {len(converted)} samples  (skipped {skipped} missing clinical_reasoning)")
    print(f"  Saved → {output_file}")


def upload_to_gcs(local_file: str, destination_blob_name: str) -> None:
    print(f"Uploading {local_file} → gs://{BUCKET_NAME}/{destination_blob_name} ...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob   = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print("  Upload complete.")


def main() -> None:
    convert_to_gemini_format(INPUT_TRAIN, OUTPUT_TRAIN)
    upload_to_gcs(OUTPUT_TRAIN, GCS_TRAIN_PATH)

    convert_to_gemini_format(INPUT_VAL, OUTPUT_VAL)
    upload_to_gcs(OUTPUT_VAL, GCS_VAL_PATH)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()