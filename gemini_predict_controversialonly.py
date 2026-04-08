import argparse
import json
import os
import re
import time
import random
from tqdm import tqdm
from google import genai
from google.genai import types

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID", "project-ade5f3ce-e086-4d4c-91c")
REGION = os.environ.get(
    "REGION", "global"  # Switch to global for new SDK and preview models
)
# ──────────────────────────────────────────────────────────────────────────────


def build_system_prompt(controversial_levels: set) -> str:
    base_prompt = """\
You are an expert clinical psychologist analyzing conversational data. Your task is to evaluate the psychological defense mechanism used in a target utterance based on the Defense Mechanisms Rating Scales (DMRS).

You will be acting as an expert adjudicator. The given utterance has been pre-analyzed and mapping models have provided an initial selection of 2 or 3 potential (but differing) defense items, along with their reasoning ("Retrieved DMRS Evidence").

YOUR TASK LIES IN BROADER EVALUATION: You have been given the broad handbook sections for these specific controversial labels, as well as the handbook sections for Level 0 (No Defense) and Level 8 (Unclear/More need inof). You are NOT bound exclusively to evaluating just the specific items/reasoning in the "Retrieved DMRS Evidence". Use the pre-analyzed reasoning as an initial consideration, but evaluate the target utterance against the ENTIRE provided handbook sections for these levels to verdict WHICH LEVEL ultimately applies best.

CRITICAL RULES:
1. Grounding: You will be provided with the exact broad handbook sections for the controversial levels in question, PLUS Level 0 and Level 8. Base your decision on comparing the target utterance against these full handbook sections. The "Retrieved DMRS Evidence" should guide your thinking, but your final verdict must reflect the broader definitions and criteria from the handbook text provided.
2. Multiple Defenses: What if a single sentence contains multiple defense mechanisms?
   * Principle: Choose the most central, dominant defense mechanism. If it's truly impossible to distinguish and multiple defenses are prominent, consider using 8 (Unclear / Needs More Information) and make a note. If multiple defenses are present, the one from less mature is typically chosen as the primary label.
3. Non-Defense or Ambiguity: If the utterance does not contain a defense mechanism, label it 0 (No Defense / Neutral). In case of severe ambiguity or if none of the provided controversial options legitimately fit the broad handbook criteria, label it 8 (Unclear / Needs More Information).

Follow these analytical steps:
1. Context: Analyze the preceding dialogue to understand what triggered this utterance.
2. Function: Identify the psychological goal the speaker is trying to achieve or avoid.
3. Evaluation: Rigorously evaluate the target utterance using the broader DMRS criteria from the provided handbook snippets, considering the pre-analyzed "Retrieved DMRS Evidence" as helpful initial context.
4. Verdict: Select the single BEST fitting level from among the proposed controversial ones, OR choose 0 (No Defense) or 8 (Unclear).

First provide your clinical reasoning trace evaluating the broad criteria, then output the final selected defense level (0-8).
"""

    handbook_content = "\n\n### RELEVANT HANDBOOK SECTIONS FOR THIS ADJUDICATION:\n"
    handbook_dir = "fraction_handbook"

    loaded_any = False
    levels_to_load = set(controversial_levels)
    levels_to_load.update([0, 8])
    for level in sorted(list(levels_to_load)):
        if level is None or level == "None":
            continue
        try:
            with open(
                os.path.join(handbook_dir, f"handbook_{level}.md"),
                "r",
                encoding="utf-8",
            ) as f:
                handbook_content += f"\n--- LEVEL {level} HANDBOOK ---\n"
                handbook_content += f.read() + "\n"
                loaded_any = True
        except FileNotFoundError:
            pass

    if loaded_any:
        return base_prompt + handbook_content
    return base_prompt


SPEAKER_MAP = {"seeker": "Seeker", "supporter": "Supporter"}


def build_user_message(sample: dict, evidence_keys: list = None) -> str:
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

    msg = (
        "## Dialogue Context\n\n"
        + "\n".join(lines)
        + f"\n\n## Target Utterance To Classify: {current_text}"
    )

    if evidence_keys and "extracted_dmrs_evidence" in sample:
        evidence_list = sample["extracted_dmrs_evidence"]
        if evidence_list:
            msg += "\n\n## Retrieved DMRS Evidence (Top Matches)\n"
            msg += "The following items were retrieved as potential matches. Evaluate them to determine which (IF ANY) applies best:\n"
            for ev in evidence_list:
                msg += "\n"
                for k in evidence_keys:
                    if k in ev:
                        msg += f"- {k}: {ev[k]}\n"

    return msg


def parse_predicted_level(response_text: str) -> int:
    # 1. Look for explicit concluding statements
    matches = re.findall(
        r"(?:defense\s+level|final\s+selected|verdict)[^\d]*([0-8])",
        response_text,
        re.IGNORECASE,
    )
    if matches:
        return int(matches[-1])

    # 2. Fallback: check from the bottom up for a standalone digit
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        # Remove markdown chars, punctuation, and whitespace
        clean_line = re.sub(r"[\*\#\-\.\,\s]", "", line)
        if re.fullmatch(r"[0-8]", clean_line):
            return int(clean_line)

    return 8


# --- UPDATE SIGNATURE ---
def predict(
    endpoint_name: str,
    test_file: str,
    raw_output: str,
    submission_output: str,
    evidence_keys_str: str,
    prompt_dump: str,
):
    print(f"Loading model: {endpoint_name}")

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

    evidence_keys = (
        [k.strip() for k in evidence_keys_str.split(",")] if evidence_keys_str else []
    )

    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} examples from {test_file}.")

    # --- ADD THIS BLOCK: INITIALIZE DUMP FILE WITH HEADER ---
    if prompt_dump:
        with open(prompt_dump, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PROMPT DUMP START\n")
            f.write("=" * 80 + "\n\n")
        print(f"Logging prompts to {prompt_dump}")
    # ---------------------------------------------------------------

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

        # Extract the controversial levels
        controversial_levels = set()
        evidence_list = sample.get("extracted_dmrs_evidence", [])
        for ev in evidence_list:
            lvl = ev.get("defense_level")
            if lvl is not None:
                controversial_levels.add(lvl)

        dynamic_system_prompt = build_system_prompt(controversial_levels)

        generation_config = types.GenerateContentConfig(
            system_instruction=dynamic_system_prompt,
            temperature=0.1,
            max_output_tokens=8192,
            safety_settings=safety_settings,
        )

        user_msg = build_user_message(sample, evidence_keys)

        # --- ADD THIS BLOCK: APPEND SYSTEM AND USER MESSAGE TO DUMP FILE ---
        if prompt_dump:
            with open(prompt_dump, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"SYSTEM PROMPT + USER MESSAGE FOR SAMPLE ID: {sample_id}\n")
                f.write("=" * 80 + "\n")
                f.write("--- SYSTEM PROMPT ---\n")
                f.write(dynamic_system_prompt + "\n")
                f.write("\n--- USER MESSAGE ---\n")
                f.write(user_msg + "\n\n")
        # --------------------------------------------------------

        # --- EXPONENTIAL BACKOFF RETRY LOGIC ---
        max_retries = 6
        base_delay = 5  # seconds

        success = False
        error_msg = ""
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=endpoint_name,
                    contents=user_msg,
                    config=generation_config,
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
            sample["model_reasoning"] = f"ERROR: {error_msg}"
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
        default="gemini-3.1-pro-preview",
        help="Your fine-tuned endpoint resource name or base model name",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="inspection_not_all_3_match.json",
        help="Path to your blind test data JSON",
    )
    parser.add_argument(
        "--raw_out",
        type=str,
        default="raw_predictions_not_all_3.json",
        help="Where to save the full reasoning traces",
    )
    parser.add_argument(
        "--submit_out",
        type=str,
        default="prediction_not_all_3.json",
        help="Where to save the clean submission JSON",
    )

    # --- ADD THIS ARGUMENT ---
    parser.add_argument(
        "--evidence_keys",
        type=str,
        default="dmrs_q_item,match_justification,sublevel_name,defense_level",
        help="Comma-separated list of keys to extract from retrieved evidence and append to the prompt. Example: 'dmrs_q_item,match_justification'",
    )

    # --- ADD THIS ARGUMENT ---
    parser.add_argument(
        "--prompt_dump",
        type=str,
        default="prompt_dump_not_all_3.txt",
        help="Optional text file to dump the exact prompts sent to the model",
    )
    # -------------------------
    args = parser.parse_args()

    predict(
        args.endpoint,
        args.test,
        args.raw_out,
        args.submit_out,
        args.evidence_keys,
        args.prompt_dump,
    )
