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
FEW_SHOT_EXAMPLES = """
Example 1: No Defense / Neutral Utterance (Level 0)
Input Dialogue:
- supporter: "Have you spoken to a professional? Is this something that you are able to do through your insurance or work?"
- seeker: "I'm to embarassed to seek professional help I think i need time to control my emotions first"
- supporter: "I completely understand. I have been in situations at work that I couldn't resolve due to the exact same reasons."
- seeker: "Thanks for understanding"
Current Text: "Thanks for understanding"

Expected Output:
{
  "reasoning": "1. Context: The supporter validates the seeker's feelings of embarrassment about seeking professional help. 2. Function: The seeker responds with a standard expression of gratitude to acknowledge the validation. 3. Hierarchy: According to the handbook, a simple 'thank you' is a social nicety used to maintain conversational flow and does not engage with emotional conflict or psychological distortion. 4. Final: This is a purely functional No Defense / Neutral Utterance.",
  "label": 0
}

Example 2: Action Defenses (Level 1)
Input Dialogue:
- seeker: "I'm to embarassed to seek professional help I think i need time to control my emotions first"
- supporter: "I completely understand. I have been in situations at work that I couldn't resolve due to the exact same reasons."
- seeker: "Thanks for understanding"
- supporter: "Is there a way that you could seek help at work without anyone finding out?"
- seeker: "I might be, but I'm not really close with any coworkers. I work primarily remote and when I entered my current job COVID happened and I didnt really get the chance to get to know anyone all that well"
Current Text: "I might be, but I'm not really close with any coworkers. I work primarily remote and when I entered my current job COVID happened and I didnt really get the chance to get to know anyone all that well"

Expected Output:
{
  "reasoning": "1. Context: The supporter offers a practical suggestion for finding help discreetly at work. 2. Function: The seeker ostensibly considers the help ('I might be') but immediately lists obstacles to avoid taking action. 3. Hierarchy: The seeker brings up circumstances (remote work, COVID, not knowing coworkers) to systematically reject the supporter's suggestion without engaging in problem-solving. This matches Help-Rejecting Complaining. 4. Final: Help-Rejecting Complaining falls under the Action Defenses Level.",
  "label": 1
}

Example 3: Disavowal Defenses (Level 3)
Input Dialogue:
- supporter: "It is okay to feel confused and perhaps even angry about this. Have you considered talking to him and asking him straight out what his intentions are?"
- seeker: "yes but i'm afraid what i'll hear"
- supporter: "I realise that this must be difficult but isn't it best to get things out into the open?"
- seeker: "it is but hes already hurt me before. i feel i'll find out sooner or later anyways. that way i can save face"
Current Text: "it is but hes already hurt me before. i feel i'll find out sooner or later anyways. that way i can save face"

Expected Output:
{
  "reasoning": "1. Context: The supporter advises confronting the person directly about their intentions to resolve the confusion. 2. Function: The seeker wants to avoid the anxiety and potential pain of a direct confrontation. 3. Hierarchy: Instead of fully confronting the fear, the seeker devises a self-serving explanation to justify their inaction ('I'll find out sooner or later anyways', 'that way i can save face'). Devising reassuring but incorrect explanations to avoid a threatening reality is Rationalization. 4. Final: Rationalization is a Disavowal Defense.",
  "label": 3
}

Example 4: Obsessional Defenses (Level 6)
Input Dialogue:
- supporter: "I realise that this must be difficult but isn't it best to get things out into the open?"
- seeker: "it is but hes already hurt me before. i feel i'll find out sooner or later anyways. that way i can save face"
- supporter: "I think it might be best for you to make a clean break. It sounds like you are being manipulated."
- seeker: "possibly. i'd prob be better being away from the situation"
Current Text: "possibly. i'd prob be better being away from the situation"

Expected Output:
{
  "reasoning": "1. Context: The supporter points out a highly distressing emotional reality: the seeker is being manipulated. 2. Function: The seeker manages the anxiety of this realization by keeping it emotionally distant. 3. Hierarchy: Instead of expressing anger, shock, or pain about being manipulated, the seeker responds with a detached, clinical, and hypothetical assessment ('possibly. i'd prob be better being away'). This reflects Isolation of Affect, where the cognitive idea is present but the emotional component is stripped away. 4. Final: Isolation of Affect is an Obsessional Defense.",
  "label": 6
}


Example 5: Minor Image-distorting Defenses (Level 4)
Input Dialogue:
- supporter: "Certainly you would be. I know that it is hard but you will feel better in the long run."
- seeker: "I think I would. I tried to break away once but they asked me to come back i feel there are benefits to it also"
- supporter: "You need to be aware that that can and will happen. Ignore it. What sort of benefits do you feel that there are?"
- seeker: "it gives me an upper edge and more clout"
Current Text: "it gives me an upper edge and more clout"

Expected Output:
{
  "reasoning": "1. Context: The supporter is urging the seeker to ignore attempts to pull them back into a confusing/manipulative relationship. 2. Function: The seeker is managing feelings of powerlessness and vulnerability in this dynamic. 3. Hierarchy: The seeker responds to the emotional conflict by emphasizing their own superiority, advantage, and 'clout' to prop up their self-esteem. Responding to stressors by acting superior or emphasizing personal power over the situation aligns with Omnipotence. 4. Final: Omnipotence is a Minor Image-Distorting Defense.",
  "label": 4
}

Example 6: Obsessional Defenses (Level 6)
Input Dialogue:
- seeker: "I think I would. I tried to break away once but they asked me to come back i feel there are benefits to it also"
- supporter: "You need to be aware that that can and will happen. Ignore it. What sort of benefits do you feel that there are?"
- seeker: "it gives me an upper edge and more clout"
- supporter: "In what way does it do that?"
- seeker: "being on the board gives me perks"
Current Text: "being on the board gives me perks"

Expected Output:
{
  "reasoning": "1. Context: The supporter questions the seeker on how staying in this manipulative dynamic actually gives them an edge. 2. Function: The seeker avoids the painful emotional reality of being used by focusing on transactional logic. 3. Hierarchy: Instead of exploring the emotional distress of the relationship, the seeker uses generalization and logical abstraction ('gives me perks') to keep their feelings distant. This excessive use of logic to avoid disturbing feelings is Intellectualization. 4. Final: Intellectualization belongs to the Obsessional Defenses Level.",
  "label": 6
}

Example 7: Disavowal Defenses (Level 3)
Input Dialogue:
- supporter: "Let me get this straight - you do have feelings for him but you are confused and you think that there's an element of mutual manipulation."
- seeker: "yes"
- supporter: "You are definitely not the only person to feel this way!"
- seeker: "it's a difficult situation. i think it will become clearer the deeper i get into it. thing is he is the head of the board and holds the power. what it comes down to is im wondering if he has feelings for me"
Current Text: "it's a difficult situation. i think it will become clearer the deeper i get into it. thing is he is the head of the board and holds the power. what it comes down to is im wondering if he has feelings for me"

Expected Output:
{
  "reasoning": "1. Context: The supporter validates that the seeker is in a mutually manipulative and confusing situation. 2. Function: The seeker wants to evade the unpleasant fact that the relationship is toxic. 3. Hierarchy: The seeker devises a reassuring explanation to justify staying ('it will become clearer the deeper i get into it') and fixates on his power and potential feelings rather than the reality of the manipulation. Devising self-serving, incorrect explanations to cover up real subjective reasons is Rationalization. 4. Final: Rationalization is a Disavowal Defense.",
  "label": 3
}

Example 8: Disavowal Defenses (Level 3)
Input Dialogue:
- supporter: "I feel that you are paying too much attention to your position on the board."
- seeker: "i feel it gives me an advantage i want at the moment."
- supporter: "But that is rather manipulative of you."
- seeker: "if you were me you would forget about him and the board"
Current Text: "if you were me you would forget about him and the board"

Expected Output:
{
  "reasoning": "1. Context: The supporter directly confronts the seeker, stating that the seeker's behavior is manipulative. 2. Function: The seeker is attempting to evade the guilt and shame of being called out for manipulation. 3. Hierarchy: Instead of acknowledging their own fault or exploring the feelings of guilt, the seeker deflects responsibility outward, putting the focus entirely on what the supporter would do ('if you were me'). This evasion of responsibility by shifting the narrative aligns with Rationalization (making excuses to avoid taking responsibility). 4. Final: This falls under the Disavowal Defense Level.",
  "label": 3
}
"""

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
# 3. Context Cache Setup
# ==============================================================================
def create_or_get_cache(client: genai.Client) -> str:
    """
    Creates a context cache containing the large, static system instruction
    (handbook + few-shot examples). Returns the cache resource name.

    The cache is billed at a reduced rate and avoids re-sending the full
    handbook on every single API call, cutting costs dramatically.
    """
    print("Creating context cache for system instruction (handbook + examples)...")
    cache = client.caches.create(
        model=MODEL_ID,
        config=types.CreateCachedContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            ttl=CACHE_TTL,
            display_name="psydef_handbook_cache",  # Human-readable label
        ),
    )
    print(f"Cache created: {cache.name}  (expires in {CACHE_TTL})\n")
    return cache.name


# ==============================================================================
# 4. Main Annotation Function
# ==============================================================================
def annotate_dataset(input_json_path: str, output_json_path: str):
    # Initialize the client. Ensure GEMINI_API_KEY is in your environment variables.
    client = genai.Client()

    # ------------------------------------------------------------------
    # Build the context cache ONCE before the annotation loop.
    # Every subsequent generate_content call references the cache name
    # instead of re-sending the full system instruction (~32k+ tokens).
    # ------------------------------------------------------------------
    cache_name = create_or_get_cache(client)

    # Load your dataset
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data:
        print(f"Processing Dialogue ID: {item.get('id')}")

        # Format the user prompt — only the small, dynamic part per item
        user_prompt = f"""
        Dialogue History:
        {json.dumps(item.get('dialogue', []), indent=2)}

        Target Utterance to Classify:
        "{item.get('current_text', '')}"

        Refer to the LABEL REFERENCE in your instructions and output the correct integer label (0–8).
        """

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    # Reference the pre-built cache; the handbook is NOT re-sent.
                    cached_content=cache_name,
                    response_mime_type="application/json",
                    response_schema=DefensePrediction,
                    temperature=0.1,  # Low temperature for classification consistency
                ),
            )

            prediction = json.loads(response.text)

            item["predicted_label"] = prediction["label"]
            item["model_reasoning"] = prediction["reasoning"]
            results.append(item)

            print(f"Predicted Label: {prediction['label']}\n")

        except Exception as e:
            print(f"Error processing {item.get('id')}: {e}")
            # Preserve the item without a prediction so no data is silently dropped
            item["predicted_label"] = None
            item["model_reasoning"] = f"ERROR: {str(e)}"
            results.append(item)

    # Save the results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Successfully saved annotated data to {output_json_path}")


if __name__ == "__main__":
    # Example usage:
    # annotate_dataset("train_data.json", "annotated_train_data.json")
    pass
