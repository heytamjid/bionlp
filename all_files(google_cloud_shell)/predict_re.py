import argparse
import json
import os
import re
import time
import random
from tqdm import tqdm
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")

if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=REGION)
else:
    vertexai.init(location=REGION)
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

SPEAKER_MAP = {"seeker": "Seeker", "supporter": "Supporter"}


def build_user_message(sample: dict) -> str:
    dialogue = sample.get("dialogue", [])
    current_text = sample.get("current_text", "").strip()

    target_idx = None
    for i in range(len(dialogue) - 1, -1, -1):
        if dialogue[i].get("text", "").strip() == current_text:
            target_idx = i
            break

    context_turns = dialogue[:target_idx] if target_idx is not None else dialogue[:-1]
    lines = [
        f"{SPEAKER_MAP.get(t.get('speaker', '').lower(), t.get('speaker', '').capitalize())}: {t.get('text', '')}"
        for t in context_turns
    ]

    return (
        "## Dialogue Context\n\n"
        + "\n".join(lines)
        + f"\n\n## Target Utterance To Classify: {current_text} ..."
    )


def parse_predicted_level(response_text: str) -> int:
    matches = re.findall(
        r"Defense\s+Level\s*[:\-]\s*([0-8])", response_text, re.IGNORECASE
    )
    if matches:
        return int(matches[-1])
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"[0-8]", line):
            return int(line)
    return 8


def reinfer_zeros(endpoint_name: str, raw_input: str, submit_input: str):
    print(f"Loading existing predictions from: {raw_input}")

    with open(raw_input, "r", encoding="utf-8") as f:
        all_raw_data = json.load(f)

    # Separate labels that are 0 from the rest
    to_reinfer = []
    retained_data = []

    for item in all_raw_data:
        if item.get("predicted_label") == 8:
            to_reinfer.append(item)
        else:
            retained_data.append(item)

    print(f"Total samples: {len(all_raw_data)}")
    print(f"Samples to keep: {len(retained_data)}")
    print(f"Samples to re-infer (Label 0): {len(to_reinfer)}")

    if not to_reinfer:
        print("No samples with label 0 found. Exiting.")
        return

    print(f"\nInitializing model: {endpoint_name}")
    model = GenerativeModel(endpoint_name, system_instruction=SYSTEM_PROMPT)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    generation_config = GenerationConfig(temperature=0.1, max_output_tokens=1024)

    raw_results_re = []
    submit_results_re = []

    # ── RE-INFER LOGIC ────────────────────────────────────────────────────────
    for index, sample in tqdm(enumerate(to_reinfer), total=len(to_reinfer)):
        sample_id = sample.get("id", f"re_test_{index}")
        user_msg = build_user_message(sample)

        max_retries = 6
        base_delay = 5

        success = False
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    user_msg,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                response_text = response.text
                predicted_label = parse_predicted_level(response_text)

                # Update sample
                sample["model_reasoning"] = response_text
                sample["predicted_label"] = predicted_label

                raw_results_re.append(sample)
                submit_results_re.append({"id": sample_id, "label": predicted_label})

                success = True
                break

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Resource exhausted" in error_msg:
                    if attempt < max_retries - 1:
                        sleep_time = (base_delay * (2**attempt)) + random.uniform(0, 2)
                        tqdm.write(
                            f"\n[429 Quota Hit] Pausing {sleep_time:.1f}s before retrying {sample_id}..."
                        )
                        time.sleep(sleep_time)
                    else:
                        tqdm.write(
                            f"\n[FAILED] Sample {sample_id} failed after {max_retries} attempts: {e}"
                        )
                else:
                    tqdm.write(f"\n[ERROR] Non-quota error on sample {sample_id}: {e}")
                    break

        if not success:
            sample["model_reasoning"] = f"ERROR: {str(e)}"
            sample["predicted_label"] = 8
            raw_results_re.append(sample)
            submit_results_re.append({"id": sample_id, "label": 8})

        time.sleep(0.5)

    # ── SAVE INTERMEDIATE FILES ───────────────────────────────────────────────
    raw_re_file = "raw_predictions_re.json"
    submit_re_file = "predictions_re.json"

    with open(raw_re_file, "w", encoding="utf-8") as f:
        json.dump(raw_results_re, f, indent=4)
    with open(submit_re_file, "w", encoding="utf-8") as f:
        json.dump(submit_results_re, f, indent=4)

    print(f"\n✅ Re-inference complete!")
    print(f"📄 Re-inferred raw traces saved to: {raw_re_file}")
    print(f"🎯 Re-inferred submission file saved to: {submit_re_file}")

    # ── MERGE AND SORT ────────────────────────────────────────────────────────
    print("\nMerging newly inferred data back into main datasets...")

    final_raw = retained_data + raw_results_re

    # Sort strictly by ID (e.g., test_00120)
    final_raw_sorted = sorted(final_raw, key=lambda x: x.get("id", ""))

    # Create the submission array from the sorted raw array
    final_submit_sorted = [
        {"id": item["id"], "label": item["predicted_label"]}
        for item in final_raw_sorted
    ]

    with open(raw_input, "w", encoding="utf-8") as f:
        json.dump(final_raw_sorted, f, indent=4)

    with open(submit_input, "w", encoding="utf-8") as f:
        json.dump(final_submit_sorted, f, indent=4)

    print(f"✅ Merged and sorted successfully!")
    print(f"Overwrote {raw_input} and {submit_input} with the complete, updated lists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        type=str,
        default="projects/project-ade5f3ce-e086-4d4c-91c/locations/us-central1/endpoints/8182003133711384576",
        help="Your fine-tuned endpoint resource name",
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="raw_predictions.json",
        help="The main raw predictions file to read from and overwrite",
    )
    parser.add_argument(
        "--submit",
        type=str,
        default="prediction.json",
        help="The main clean submission JSON to overwrite",
    )
    args = parser.parse_args()

    reinfer_zeros(args.endpoint, args.raw, args.submit)
