import json
import os
import re


def parse_label(val):
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        v = val.strip()
        # try exact int
        try:
            return int(v)
        except Exception:
            pass
        # extract first number
        m = re.search(r"(\d+)", v)
        if m:
            return int(m.group(1))
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    paths = {
        "test": os.path.join(root, "test_label.json"),
        "annot": os.path.join(
            root, "gemini direct prediction", "annotated_train_data.json"
        ),
        "approach": os.path.join(root, "approach5.json"),
        "raw": os.path.join(
            root, "all_files(google_cloud_shell)", "raw_predictions.json"
        ),
    }

    data_test = load_json(paths["test"]) if os.path.exists(paths["test"]) else []
    data_annot = load_json(paths["annot"]) if os.path.exists(paths["annot"]) else []
    data_approach = (
        load_json(paths["approach"]) if os.path.exists(paths["approach"]) else []
    )
    data_raw = load_json(paths["raw"]) if os.path.exists(paths["raw"]) else []

    n = min(len(data_test), len(data_annot), len(data_approach), len(data_raw))

    results = {str(i): {} for i in range(n)}

    cat1 = []  # zero shot failed
    cat2 = []  # voting failed
    cat3 = []  # reasoning failed
    cat4 = []  # all three failed and SFT right
    cat5 = []  # voting and reasoning wrong but SFT right
    cat6 = []  # all three candidate items wrong but SFT right

    for i in range(n):
        gold = data_test[i].get("label") if isinstance(data_test[i], dict) else None

        ann = data_annot[i] if isinstance(data_annot[i], dict) else {}
        zero = parse_label(ann.get("predicted_defense_level"))
        item_id = ann.get("id") or ann.get("example_id")

        app = data_approach[i] if isinstance(data_approach[i], dict) else {}
        voting = parse_label(app.get("predicted_defense_level"))
        reasoning = parse_label(app.get("predicted_label"))

        rawp = data_raw[i] if isinstance(data_raw[i], dict) else {}
        sft = parse_label(rawp.get("predicted_label"))

        # candidate items
        candidates = []
        evid = (
            app.get("extracted_dmrs_evidence")
            or app.get("extracted_dmrs_evidence_list")
            or []
        )
        if isinstance(evid, list):
            for e in evid:
                if isinstance(e, dict):
                    candidates.append(parse_label(e.get("defense_level")))

        # comparisons
        zero_fail = (zero is None) or (gold is None) or (zero != gold)
        voting_fail = (voting is None) or (gold is None) or (voting != gold)
        reasoning_fail = (reasoning is None) or (gold is None) or (reasoning != gold)
        sft_ok = (sft is not None) and (gold is not None) and (sft == gold)

        # cat1
        if zero_fail:
            cat1.append((i, item_id, gold, zero, voting, reasoning, sft, candidates))
        if voting_fail:
            cat2.append((i, item_id, gold, zero, voting, reasoning, sft, candidates))
        if reasoning_fail:
            cat3.append((i, item_id, gold, zero, voting, reasoning, sft, candidates))

        if zero_fail and voting_fail and reasoning_fail and sft_ok:
            cat4.append((i, item_id, gold, zero, voting, reasoning, sft, candidates))

        if voting_fail and reasoning_fail and sft_ok:
            cat5.append((i, item_id, gold, zero, voting, reasoning, sft, candidates))

        # all three candidate items wrong (need at least 3 candidates)
        if len(candidates) >= 3 and sft_ok:
            first_three = candidates[:3]
            all_three_wrong = all(
                (c is None) or (gold is None) or (c != gold) for c in first_three
            )
            if all_three_wrong:
                cat6.append(
                    (i, item_id, gold, zero, voting, reasoning, sft, first_three)
                )

    out_path = os.path.join(root, "failure_cases.txt")
    with open(out_path, "w", encoding="utf-8") as out:

        def write_cat(name, items):
            out.write(f"=== {name} ({len(items)}) ===\n")
            for t in items:
                idx, item_id, gold, zero, voting, reasoning, sft, candidates = t
                out.write(
                    f"{idx}\t{item_id}\tgold={gold}\tzero={zero}\tvoting={voting}\treasoning={reasoning}\tsft={sft}\tcandidates={candidates}\n"
                )
            out.write("\n")

        write_cat("zero_shot_failed", cat1)
        write_cat("voting_failed", cat2)
        write_cat("reasoning_failed", cat3)
        write_cat("all_three_failed_and_sft_right", cat4)
        write_cat("voting_and_reasoning_wrong_but_sft_right", cat5)
        write_cat("three_candidates_wrong_but_sft_right", cat6)

    print("Wrote results to:", out_path)
    print(
        "Counts:",
        {
            "cat1": len(cat1),
            "cat2": len(cat2),
            "cat3": len(cat3),
            "cat4": len(cat4),
            "cat5": len(cat5),
            "cat6": len(cat6),
        },
    )


if __name__ == "__main__":
    main()
