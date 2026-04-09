import argparse
import json
import os
import re
import time

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID = os.environ["PROJECT_ID"]
REGION     = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)
# ──────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {
    0: "No Defense",
    1: "Action",
    2: "Major Image-distorting",
    3: "Disavowal",
    4: "Minor Image-distorting",
    5: "Neurotic",
    6: "Obsessional",
    7: "Highly Adaptive",
    8: "Need More Info",
}

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


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_user_message(sample: dict) -> str:
    dialogue     = sample["dialogue"]
    current_text = sample["current_text"].strip()

    target_idx = None
    for i in range(len(dialogue) - 1, -1, -1):
        if dialogue[i]["text"].strip() == current_text:
            target_idx = i
            break

    context_turns = dialogue[:target_idx] if target_idx is not None else dialogue[:-1]
    lines = [
        f"{SPEAKER_MAP.get(t['speaker'].lower(), t['speaker'].capitalize())}: {t['text']}"
        for t in context_turns
    ]

    return (
        "## Dialogue Context\n\n"
        + "\n".join(lines)
        + f"\n\n## Target Utterance To Classify: {current_text} ..."
    )


def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next(iter(data.values()))
    return [s for s in data if s.get("clinical_reasoning", {}).get("context_trigger")]


def parse_predicted_level(response_text: str) -> int | None:
    """
    Extract the integer from the last 'Defense Level: X' line in the response.
    Returns None if no valid label is found.
    """
    matches = re.findall(r"Defense\s+Level\s*[:\-]\s*([0-8])", response_text, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    # Fallback: last standalone digit 0-8 on its own line
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"[0-8]", line):
            return int(line)
    return None


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    model_name: str,
    test_file: str,
    max_samples: int | None = None,
    output_json: str = "results.json",
    sleep_between: float = 0.3,
) -> list:
    print(f"\nEvaluating: {model_name}")
    samples = load_dataset(test_file)
    if max_samples:
        samples = samples[:max_samples]
    print(f"Samples: {len(samples)}")

    model = GenerativeModel(
        model_name,
        system_instruction=SYSTEM_PROMPT,
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    generation_config = GenerationConfig(temperature=0.1, max_output_tokens=1024)

    results = []
    parse_failures = 0

    for sample in tqdm(samples):
        user_msg   = build_user_message(sample)
        true_label = int(sample.get("label", sample.get("predicted_defense_level", -1)))

        try:
            response = model.generate_content(
                user_msg,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            response_text  = response.text
            predicted_label = parse_predicted_level(response_text)

            if predicted_label is None:
                parse_failures += 1

            results.append({
                "id":               sample.get("id"),
                "dialogue_id":      sample.get("dialogue_id"),
                "true_label":       true_label,
                "predicted_label":  predicted_label,
                "true_label_name":  LABEL_NAMES.get(true_label, str(true_label)),
                "pred_label_name":  LABEL_NAMES.get(predicted_label, "parse_error"),
                "response_text":    response_text,
            })

        except Exception as e:
            print(f"\nError on {sample.get('id', '?')}: {e}")
            results.append({
                "id":              sample.get("id"),
                "dialogue_id":     sample.get("dialogue_id"),
                "true_label":      true_label,
                "predicted_label": None,
                "error":           str(e),
            })
            time.sleep(2)

        time.sleep(sleep_between)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {output_json}")
    print(f"Parse failures: {parse_failures}/{len(results)}")
    return results


# ── Metrics & plots ───────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    valid = [r for r in results if r.get("predicted_label") is not None]
    if not valid:
        print("No valid predictions to score.")
        return {}

    y_true = [r["true_label"]      for r in valid]
    y_pred = [r["predicted_label"] for r in valid]

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    labels_present = sorted(set(y_true) | set(y_pred))
    label_names    = [LABEL_NAMES.get(l, str(l)) for l in labels_present]

    report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "n_samples":    len(results),
        "n_valid":      len(valid),
        "n_parse_fail": len(results) - len(valid),
        "accuracy":     round(acc, 4),
        "macro_f1":     round(macro_f1, 4),
        "weighted_f1":  round(weighted_f1, 4),
        "per_class":    report,
    }

    print("\n─── Results Summary ───────────────────────────────────────")
    print(f"  Samples evaluated : {len(valid)} / {len(results)}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Weighted F1       : {weighted_f1:.4f}")
    print("\n" + classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=label_names,
        zero_division=0,
    ))

    return metrics


def plot_confusion_matrix(results: list, title: str, filename: str) -> None:
    valid  = [r for r in results if r.get("predicted_label") is not None]
    y_true = [r["true_label"]      for r in valid]
    y_pred = [r["predicted_label"] for r in valid]

    labels = sorted(set(y_true) | set(y_pred))
    names  = [f"{l}\n{LABEL_NAMES[l]}" for l in labels]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize for colour scaling, keep raw counts as annotations
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall per class")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)

    thresh = cm_norm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color=color)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → plots/{filename}")


def plot_per_class_f1(results: list, title: str, filename: str) -> None:
    valid  = [r for r in results if r.get("predicted_label") is not None]
    y_true = [r["true_label"]      for r in valid]
    y_pred = [r["predicted_label"] for r in valid]

    labels    = sorted(set(y_true))
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    names     = [f"L{l}: {LABEL_NAMES[l]}" for l in labels]

    colors = ["#4C9BE8" if s >= 0.5 else "#E8834C" for s in f1_scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1_scores, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title(title, fontsize=11)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, f1_scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}", dpi=150)
    plt.close()
    print(f"Per-class F1 chart saved → plots/{filename}")


def compare_results(baseline_file: str, tuned_file: str) -> None:
    with open(baseline_file, "r") as f:
        baseline = json.load(f)
    with open(tuned_file, "r") as f:
        tuned = json.load(f)

    def _metrics(results):
        valid  = [r for r in results if r.get("predicted_label") is not None]
        y_true = [r["true_label"]      for r in valid]
        y_pred = [r["predicted_label"] for r in valid]
        return {
            "accuracy":     accuracy_score(y_true, y_pred),
            "macro_f1":     f1_score(y_true, y_pred, average="macro",    zero_division=0),
            "weighted_f1":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    bm = _metrics(baseline)
    tm = _metrics(tuned)

    print("\n─── Baseline vs Fine-tuned ────────────────────────────────")
    for k in bm:
        diff = tm[k] - bm[k]
        sign = "+" if diff >= 0 else ""
        print(f"  {k:<15}: baseline={bm[k]:.4f}  tuned={tm[k]:.4f}  ({sign}{diff:.4f})")

    # Bar chart comparison
    metrics = list(bm.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, [bm[m] for m in metrics], width, label="Baseline", color="#7DA3C9")
    ax.bar(x + width/2, [tm[m] for m in metrics], width, label="Fine-tuned", color="#4C9BE8")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Fine-tuned — Defense Mechanism Classification")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/comparison.png", dpi=150)
    plt.close()
    print("Comparison bar chart saved → plots/comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Gemini model on the PSYDEFCONV defense mechanism task."
    )
    parser.add_argument("--model",    type=str, required=True,
                        help="Model name or fine-tuned endpoint resource name")
    parser.add_argument("--test",     type=str, default="test.json",
                        help="Path to test JSON file")
    parser.add_argument("--max",      type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--output",   type=str, default="results.json",
                        help="Path to save per-sample results JSON")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a previous results.json for comparison")
    parser.add_argument("--sleep",    type=float, default=0.3,
                        help="Seconds to sleep between API calls (default: 0.3)")
    args = parser.parse_args()

    # Run inference
    results = evaluate(
        model_name=args.model,
        test_file=args.test,
        max_samples=args.max,
        output_json=args.output,
        sleep_between=args.sleep,
    )

    # Compute and print metrics
    metrics = compute_metrics(results)

    # Save metrics separately for easy comparison
    metrics_file = args.output.replace(".json", "_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_file}")

    # Plots
    tag = os.path.splitext(os.path.basename(args.output))[0]
    plot_confusion_matrix(results, f"Confusion Matrix — {args.model}", f"{tag}_cm.png")
    plot_per_class_f1(results,     f"Per-class F1 — {args.model}",     f"{tag}_f1.png")

    # Optional comparison
    if args.baseline:
        compare_results(args.baseline, args.output)


if __name__ == "__main__":
    main()