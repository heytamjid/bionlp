import json
import os
import argparse


def normalize_dialogue_item(item):
    # returns a string block for a single item
    parts = []
    id_ = item.get("id") or item.get("example_id") or ""
    dialogue = item.get("dialogue")
    # header
    header = f"=== {id_} ==="
    parts.append(header)

    if isinstance(dialogue, list) and len(dialogue) > 0:
        for turn in dialogue:
            # expect dicts with 'speaker' and 'text'
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "").strip().lower()
            text = turn.get("text", "")
            # Preserve text exactly; only normalize speaker label
            if speaker == "supporter":
                parts.append(f"Supporter: {text}")
            elif speaker == "seeker":
                parts.append(f"Seeker: {text}")
            else:
                # Unknown speaker: output raw speaker value followed by text
                label = turn.get("speaker", "")
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(f"Unknown: {text}")
    else:
        # no dialogue list, fall back to current_text if present
        current = item.get("current_text")
        if current is not None:
            # Write as Seeker by default
            parts.append(f"Seeker: {current}")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize dialogues to 'Supporter:' and 'Seeker:' lines without changing dialogue characters."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "gemini direct prediction",
            "annotated_train_data.json",
        ),
        help="Path to input JSON file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "normalized_dialogues.txt"
        ),
        help="Path to output text file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        return

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = []
    for item in data:
        if not isinstance(item, dict):
            continue
        block = normalize_dialogue_item(item)
        blocks.append(block)

    out_text = "\n\n".join(blocks)
    with open(args.output, "w", encoding="utf-8") as out:
        out.write(out_text)

    print(f"Wrote {len(blocks)} dialogues to: {args.output}")


if __name__ == "__main__":
    main()
