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


# --- UPDATE SIGNATURE ---
# --- UPDATE SIGNATURE ---
def predict(
    endpoint_name: str,
    test_file: str,
    raw_output: str,
    submission_output: str,
    handbook_file: str,
    prompt_dump: str,
):
    print(f"Loading fine-tuned model: {endpoint_name}")

    # --- ADD THIS BLOCK TO READ AND INJECT THE HANDBOOK ---
    handbook_content = ""
    if os.path.exists(handbook_file):
        with open(handbook_file, "r", encoding="utf-8") as f:
            handbook_content = f.read()
        print(f"Loaded handbook from {handbook_file}")
    else:
        print(
            f"Warning: Handbook file '{handbook_file}' not found. Proceeding without it."
        )

    # Combine the base system prompt with the handbook content
    augmented_system_prompt = (
        SYSTEM_PROMPT + "\n\n### DMRS REFERENCE HANDBOOK ###\n" + handbook_content
    )
    # ------------------------------------------------------

    model = GenerativeModel(endpoint_name, system_instruction=augmented_system_prompt)

    # --- ADD THIS BLOCK: INITIALIZE DUMP FILE WITH SYSTEM PROMPT ---
    if prompt_dump:
        with open(prompt_dump, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("SYSTEM PROMPT (Sent at model initialization)\n")
            f.write("=" * 80 + "\n")
            f.write(augmented_system_prompt + "\n\n")
        print(f"Logging prompts to {prompt_dump}")
    # ---------------------------------------------------------------

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

        # --- ADD THIS BLOCK: APPEND USER MESSAGE TO DUMP FILE ---
        if prompt_dump:
            with open(prompt_dump, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"USER MESSAGE FOR SAMPLE ID: {sample_id}\n")
                f.write("=" * 80 + "\n")
                f.write(user_msg + "\n\n")
        # --------------------------------------------------------

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

    # --- ADD THIS ARGUMENT ---
    parser.add_argument(
        "--handbook",
        type=str,
        default="handbook.md",
        help="Path to the DMRS handbook markdown file",
    )

    # --- ADD THIS ARGUMENT ---
    parser.add_argument(
        "--prompt_dump",
        type=str,
        default="prompt_dump.txt",
        help="Optional text file to dump the exact prompts sent to the model",
    )
    # -------------------------
    args = parser.parse_args()

    predict(
        args.endpoint,
        args.test,
        args.raw_out,
        args.submit_out,
        args.handbook,
        args.prompt_dump,
    )
