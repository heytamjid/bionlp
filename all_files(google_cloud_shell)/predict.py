# import argparse
# import json
# import os
# import re
# import time
# from tqdm import tqdm
# import vertexai
# from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold

# # ── Configuration ─────────────────────────────────────────────────────────────
# PROJECT_ID = os.environ.get("PROJECT_ID")
# REGION     = os.environ.get("REGION", "us-central1") # Change if your endpoint is in another region

# if PROJECT_ID:
#     vertexai.init(project=PROJECT_ID, location=REGION)
# else:
#     # If running in Cloud Shell where project is already set by gcloud
#     vertexai.init(location=REGION)
# # ──────────────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """\
# You are an expert clinical psychologist analyzing conversational data. Your task is to evaluate the psychological defense mechanism used in a target utterance based on the Defense Mechanisms Rating Scales (DMRS).

# LABEL REFERENCE (Defense Mechanism Rating Scale Tiers):
#   0 = No Defense / Neutral Utterance  — simple, undefended statement; no psychological distortion - Functional utterances that maintain conversational flow without engaging conflict.
#   1 = Action Defense Level            — Acting Out / Help-Rejecting Complaining / Passive Aggression - Distress is released by acting on the environment instead of reflecting.
#   2 = Major Image-distorting Defense  — Splitting / Projective Identification - Reduces anxiety via all-good/all-bad distortions of self or other.
#   3 = Disavowal Defense Level         — Denial / Projection / Rationalization / Autistic Fantasy - Rejects threatening reality by denying, excusing, blaming, or fantasizing.
#   4 = Minor Image-distorting Defense  — Devaluation / Idealization / Omnipotence - Softer distortions temporarily inflate or deflate self-esteem.
#   5 = Neurotic Defense Level          — Displacement / Dissociation / Reaction Formation / Repression - Keeps unacceptable motives out of awareness; feelings surface indirectly.
#   6 = Obsessional Defense Level       — Intellectualization / Isolation of Affects / Undoing - Uses excessive logic or symbolic acts to separate feelings from events.
#   7 = Highly Adaptive Defense Level   — Affiliation / Altruism / Anticipation / Humor / Self-Assertion / Self-Observation / Sublimation / Suppression - Mature coping that integrates emotion and thought to channel affect constructively.
#   8 = Need More Information           — Evidence suggests a defense but is insufficient to confirm any tier - Label used when an utterance is too ambiguous or lacks context.

# Follow these analytical steps:
# 1. Context   : Analyze the preceding dialogue to understand what triggered this utterance.
# 2. Function  : Identify the psychological goal the speaker is trying to achieve or avoid.
# 3. Grounding : Match the behavior to specific DMRS criteria.
# 4. Hierarchy : Verify that exclusionary criteria for higher/lower levels are met.

# First provide your clinical reasoning trace, then output the defense level (0–8).\
# """

# SPEAKER_MAP = {"seeker": "Seeker", "supporter": "Supporter"}

# def build_user_message(sample: dict) -> str:
#     dialogue = sample.get("dialogue", [])
#     current_text = sample.get("current_text", "").strip()

#     target_idx = None
#     for i in range(len(dialogue) - 1, -1, -1):
#         if dialogue[i].get("text", "").strip() == current_text:
#             target_idx = i
#             break

#     context_turns = dialogue[:target_idx] if target_idx is not None else dialogue[:-1]
#     lines = [
#         f"{SPEAKER_MAP.get(t.get('speaker', '').lower(), t.get('speaker', '').capitalize())}: {t.get('text', '')}"
#         for t in context_turns
#     ]

#     return (
#         "## Dialogue Context\n\n"
#         + "\n".join(lines)
#         + f"\n\n## Target Utterance To Classify: {current_text} ..."
#     )

# def parse_predicted_level(response_text: str) -> int:
#     """
#     Extract the integer from the last 'Defense Level: X' line in the response.
#     Returns 8 (Need More Info) as a safe fallback if parsing fails.
#     """
#     matches = re.findall(r"Defense\s+Level\s*[:\-]\s*([0-8])", response_text, re.IGNORECASE)
#     if matches:
#         return int(matches[-1])
#     # Fallback: last standalone digit 0-8 on its own line
#     lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
#     for line in reversed(lines):
#         if re.fullmatch(r"[0-8]", line):
#             return int(line)
#     return 8 # Fallback class

# def predict(endpoint_name: str, test_file: str, raw_output: str, submission_output: str):
#     print(f"Loading fine-tuned model: {endpoint_name}")

#     model = GenerativeModel(
#         endpoint_name,
#         # system_instruction=SYSTEM_PROMPT
#     )

#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     }

#     generation_config = GenerationConfig(temperature=0.1, max_output_tokens=1024)

#     with open(test_file, 'r', encoding='utf-8') as f:
#         test_data = json.load(f)

#     print(f"Loaded {len(test_data)} examples from {test_file}.")

#     raw_results = []
#     submission_results = []

#     for index, sample in tqdm(enumerate(test_data), total=len(test_data)):
#         user_msg = build_user_message(sample)
#         sample_id = sample.get("id", f"test_{index}")

#         # --- PRINT EXACT PROMPT FOR THE FIRST SAMPLE ---
#         if index == 0:
#             print("\n" + "="*60)
#             print("🔍 SNEAK PEEK: WHAT THE MODEL EXACTLY SEES FOR SAMPLE 1")
#             print("="*60)
#             print("\n[SYSTEM INSTRUCTION]\n")
#             print(SYSTEM_PROMPT)
#             print("\n" + "-"*60)
#             print("\n[USER MESSAGE]\n")
#             print(user_msg)
#             print("\n" + "="*60 + "\n")
#         # -----------------------------------------------

#         try:
#             response = model.generate_content(
#                 user_msg,
#                 generation_config=generation_config,
#                 safety_settings=safety_settings
#             )

#             response_text = response.text
#             predicted_label = parse_predicted_level(response_text)

#             # Save raw trace
#             sample["model_reasoning"] = response_text
#             sample["predicted_label"] = predicted_label
#             raw_results.append(sample)

#             # Save clean submission format
#             submission_results.append({
#                 "id": sample_id,
#                 "label": predicted_label
#             })

#         except Exception as e:
#             print(f"\nError predicting sample {sample_id}: {e}")
#             sample["model_reasoning"] = f"ERROR: {str(e)}"
#             sample["predicted_label"] = 8 # Fallback
#             raw_results.append(sample)
#             submission_results.append({
#                 "id": sample_id,
#                 "label": 8
#             })
#             time.sleep(2) # Backoff if quota hit

#         time.sleep(0.3) # Prevent rate limits

#     # Save outputs
#     with open(raw_output, 'w', encoding='utf-8') as f:
#         json.dump(raw_results, f, indent=4)

#     with open(submission_output, 'w', encoding='utf-8') as f:
#         json.dump(submission_results, f, indent=4)

#     print(f"\n✅ Predictions complete!")
#     print(f"📄 Raw traces saved to: {raw_output}")
#     print(f"🎯 Submission file saved to: {submission_output}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--endpoint", type=str, default="projects/project-ade5f3ce-e086-4d4c-91c/locations/us-central1/endpoints/8182003133711384576", help="Your fine-tuned endpoint resource name")
#     parser.add_argument("--test", type=str, default="test.json", help="Path to your blind test data JSON")
#     parser.add_argument("--raw_out", type=str, default="raw_predictions.json", help="Where to save the full reasoning traces")
#     parser.add_argument("--submit_out", type=str, default="prediction.json", help="Where to save the clean submission JSON")
#     args = parser.parse_args()

#     predict(args.endpoint, args.test, args.raw_out, args.submit_out)

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
REGION = os.environ.get(
    "REGION", "us-central1"
)  # Change if your endpoint is in another region

if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=REGION)
else:
    # If running in Cloud Shell where project is already set by gcloud
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


def predict(
    endpoint_name: str, test_file: str, raw_output: str, submission_output: str
):
    print(f"Loading fine-tuned model: {endpoint_name}")

    model = GenerativeModel(endpoint_name, system_instruction=SYSTEM_PROMPT)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    generation_config = GenerationConfig(temperature=0.1, max_output_tokens=1024)

    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} examples from {test_file}.")

    # ── RESUME / MERGE LOGIC ─────────────────────────────────────────────
    existing_results = {}
    if os.path.exists(raw_output):
        print(
            f"Found existing {raw_output}. Loading to merge and skip successful predictions..."
        )
        try:
            with open(raw_output, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                for item in saved_data:
                    existing_results[item.get("id")] = item
        except json.JSONDecodeError:
            print("Warning: Existing raw_output is corrupted. Starting fresh.")
    # ──────────────────────────────────────────────────────────────────────

    raw_results = []
    submission_results = []

    for index, sample in tqdm(enumerate(test_data), total=len(test_data)):
        sample_id = sample.get("id", f"test_{index}")

        # Check if we already processed this successfully
        existing_entry = existing_results.get(sample_id)
        if existing_entry and "ERROR:" not in existing_entry.get("model_reasoning", ""):
            # Skip generation, just append existing data
            raw_results.append(existing_entry)
            submission_results.append(
                {"id": sample_id, "label": existing_entry.get("predicted_label", 8)}
            )
            continue

        user_msg = build_user_message(sample)

        # --- EXPONENTIAL BACKOFF RETRY LOGIC ---
        max_retries = 6
        base_delay = 5  # seconds

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

                sample["model_reasoning"] = response_text
                sample["predicted_label"] = predicted_label
                raw_results.append(sample)

                submission_results.append({"id": sample_id, "label": predicted_label})

                success = True
                break  # Exit the retry loop on success

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Resource exhausted" in error_msg:
                    if attempt < max_retries - 1:
                        # Exponential backoff + jitter
                        sleep_time = (base_delay * (2**attempt)) + random.uniform(0, 2)
                        tqdm.write(
                            f"\n[429 Quota Hit] Pausing for {sleep_time:.1f}s before retrying sample {sample_id}..."
                        )
                        time.sleep(sleep_time)
                    else:
                        tqdm.write(
                            f"\n[FAILED] Sample {sample_id} failed after {max_retries} attempts: {e}"
                        )
                else:
                    tqdm.write(f"\n[ERROR] Non-quota error on sample {sample_id}: {e}")
                    break  # Don't retry for non-429 errors (like prompt blocked by safety settings)

        # If it completely failed after retries
        if not success:
            sample["model_reasoning"] = f"ERROR: {str(e)}"
            sample["predicted_label"] = 8  # Fallback
            raw_results.append(sample)
            submission_results.append({"id": sample_id, "label": 8})

        # ── CHECKPOINT SAVE every 5 samples ──────────────────────────────
        if (index + 1) % 5 == 0:
            with open(raw_output, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=4)
            tqdm.write(f"[Checkpoint] Saved {len(raw_results)} results to {raw_output}")
        # ─────────────────────────────────────────────────────────────────
        # Standard safety delay between successful requests
        time.sleep(0.5)

    # Save outputs
    with open(raw_output, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=4)

    with open(submission_output, "w", encoding="utf-8") as f:
        json.dump(submission_results, f, indent=4)

    print(f"\n✅ Predictions complete and merged!")
    print(f"📄 Raw traces saved to: {raw_output}")
    print(f"🎯 Submission file saved to: {submission_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        type=str,
        default="projects/project-ade5f3ce-e086-4d4c-91c/locations/us-central1/endpoints/8182003133711384576",
        help="Your fine-tuned endpoint resource name",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="test.json",
        help="Path to your blind test data JSON",
    )
    parser.add_argument(
        "--raw_out",
        type=str,
        default="raw_predictions.json",
        help="Where to save the full reasoning traces",
    )
    parser.add_argument(
        "--submit_out",
        type=str,
        default="prediction.json",
        help="Where to save the clean submission JSON",
    )
    args = parser.parse_args()

    predict(args.endpoint, args.test, args.raw_out, args.submit_out)
